import padertorch as pt
from torch.nn import functional as F
from padertorch.modules.recurrent import StatefulLSTM
from padertorch.ops.mappings import ACTIVATION_FN_MAP

import numpy as np
from paderbox.transform.module_stft import _biorthogonal_window_fastest
from scipy.signal import hamming
import torch
from einops import rearrange
from padertorch.contrib.je.modules.conv import Pad, Trim


class Encoder(pt.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L=40, N=257, stride=20, get_ipd_features=False):
        super().__init__()
        # Hyper-parameter
        self.L, self.N, self.stride = L, N, stride

        # Components
        # 50% overlap
        self.conv1d_U = torch.nn.Conv1d(1, N, kernel_size=L,
                                        stride=stride, bias=False)
        self.pad = Pad()
        self.get_ipd_features = get_ipd_features

    def forward(self, mixture):
        """
        Args:
            mixture: [C, T], C is channel size, T is #samples
        Returns:
            mixture_w: [C, Fr, N], where Fr = (T-L)/(L/2)+1 = 2T/L-1
        """
        pad_size = self.L - 1 - ((mixture.shape[-1] - 1) % self.stride)
        mixture = torch.unsqueeze(mixture, 1) # [C, 1, T]
        mixture = self.pad(mixture, pad_size)

        encoder_out = F.relu(self.conv1d_U(mixture)).permute(0,2,1)  # [C, Fr, N]
        return encoder_out


def get_stft_kernel(L, N):
    h_window = hamming(L)
    real = np.array([
        [np.cos(-1 * n * 2 * np.pi / (N - 2) * k) * h_window[k]
         for k in range(L)] for n in range(N // 2)
    ])
    imag = np.array([
        [np.sin(-1 * n * 2 * np.pi / (N - 2) * k) * h_window[k]
         for k in range(L)] for n in range(N // 2)
    ])
    kernel = np.concatenate([real, imag], axis=0)
    # print(kernel.shape)
    return torch.from_numpy(kernel).float().unsqueeze(dim=1)


class STFTEncoder(pt.Module):
    def __init__(self, L=40, N=514, stride=20, get_ipd_features=False):
        super().__init__()
        self.L, self.N, self.stride = L, N, stride
        self.get_ipd_features = get_ipd_features

        self.kernel_STFT = get_stft_kernel(L, N)
        self.pad = Pad()


    def forward(self, mixture):
        """
        Args:
            mixture: [C, T], C is channel size, T is #samples
        Returns:
            mixture_w: [C, Fr, N], where Fr = (T-L)/(L/2)+1 = 2T/L-1
        """
        pad_size = self.L - 1 - ((mixture.shape[-1] - 1) % self.stride)
        mixture = torch.unsqueeze(mixture, 1) # [C, 1, T]
        mixture = self.pad(mixture, pad_size)
        weights = self.kernel_STFT.to(mixture.device)
        encoder_out = F.conv1d(mixture, weight=weights,
                       stride=self.stride).permute(0,2,1)
        C = encoder_out.shape[0]
        return encoder_out


class Decoder(pt.Module):
    def __init__(self, L=40, N=257, stride=20, remove_ipd_features=False,
                 beamforming=False, griffon_lim=False):
        super().__init__()
        # Hyper-parameter
        self.N, self.L, self.stride = N, L, stride
        self.griffon_lim = griffon_lim
        # Components
        self.basis_signals = torch.nn.ConvTranspose1d(
            N, 1, kernel_size=L, stride=stride, bias=False)
        self.remove_ipd_features = remove_ipd_features
        self.cut = Trim()
        self.beamforming = beamforming

    def forward(self, mixture_w, est_mask, num_samples):
        """
        Args:
            mixture_w: [C, F, N]
            est_mask: [C, K , F, N]
        Returns:
            est_source: [C, K, T]
        """
        # D = W * M
        if self.remove_ipd_features:
            mixture_w = mixture_w[..., :self.N]
        source_w = torch.unsqueeze(mixture_w, dim=1) * est_mask  # [C, K, F, N]
        # S = DV
        C, K, F, N = source_w.shape
        source_w = rearrange(source_w, 'c k f n -> (c k) n f')
        est_source = self.basis_signals(source_w)[:, 0]  # [C, K, F*L]
        size = est_source.shape[-1] - num_samples
        est_source = rearrange(est_source, '(c k) t -> c k t', c=C, k=K)
        return self.cut(est_source, size=size)


class IstftDecoder(pt.Module):
    def __init__(self, L=40, N=514, stride=20):
        super().__init__()
        # Hyper-parameter
        self.N, self.L, self.stride = N, L, stride

        h_window = hamming(self.L)
        h_window = _biorthogonal_window_fastest(h_window, self.stride) / (
                    self.L - 2)
        final_r = np.array([
            [np.cos(1 * f * 2 * np.pi / (self.N- 2) * n) * h_window[n]
             for n in range(self.L)] for f in range(self.N - 2)
        ])

        kernel_iSTFT_r = torch.from_numpy(final_r).float()
        self.kernel_iSTFT_r = torch.unsqueeze(kernel_iSTFT_r, dim=1)

        final_i = np.array([
            [np.sin(-1 * f * 2 * np.pi / (self.N - 2) * n) * h_window[n]
             for n in range(self.L)] for f in range(self.N - 2)
        ])
        kernel_iSTFT_i = torch.from_numpy(final_i).float()
        self.kernel_iSTFT_i = torch.unsqueeze(kernel_iSTFT_i, dim=1)
        self.cut = Trim()

    def forward(self, mixture_w, est_mask, num_samples):
        """
        Args:
            mixture_w: [C, Fr, N]
            est_mask: [C, K , Fr, N]
        Returns:
            est_source: [C, K, T]
        """
        # D = W * M
        if self.remove_ipd_features:
            mixture_w = mixture_w[..., :self.N]
        source_w = torch.unsqueeze(mixture_w, dim=1) * est_mask  # [C, K, Fr, N]
        C, K, _, _ = source_w.shape
        source_w = rearrange(source_w, 'c k fr n -> (c k) n fr')
        speaker_real, speaker_imag = torch.split(source_w, 257, dim=-2)
        speaker_real = torch.cat(
            [speaker_real, speaker_real[:, 1:-1].flip(1)], dim=1)
        speaker_imag = torch.cat(
            [speaker_imag, -speaker_imag[:, 1:-1].flip(1)], dim=1)
        kernel_real = self.kernel_iSTFT_r.to(speaker_real.device)
        est_source_real = F.conv_transpose1d(speaker_real, weight=kernel_real,
                                             stride=self.stride)
        kernel_imag = self.kernel_iSTFT_i.to(speaker_imag.device)
        est_source_imag = F.conv_transpose1d(speaker_imag, kernel_imag,
                                             stride=self.stride)
        est_source = est_source_real + est_source_imag
        est_source = est_source[:, 0]
        est_source = rearrange(est_source, '(c k) t -> c k t', c=C, k=K)

        size = est_source.shape[-1] - num_samples
        return self.cut(est_source, size=size)


class TasnetBaseline(pt.Module):

    def __init__(self, N=257, num_layers=4, num_units=300,
                 recurrent_dropout=0.2, use_log: bool = True, num_spks=2,
                 activation='softmax'):
        super().__init__()
        assert num_layers % 2 == 0
        self.num_units = num_units
        self.num_layers = num_layers
        self.fbins = N
        self.norm = torch.nn.LayerNorm(N, eps=1e-10)
        self.lstm = torch.nn.ModuleList()
        input_size = N
        self.activation = ACTIVATION_FN_MAP[activation]()
        for idx in range(num_layers // 2):

            self.lstm.append(
                StatefulLSTM(input_size=input_size,
                             num_layers=2,
                             hidden_size=num_units, dropout=recurrent_dropout,
                             bidirectional=True)
            )
            if idx == 0:
                input_size = N + 2 * num_units
            else:
                input_size = 4 * num_units
        self.linear = torch.nn.Linear(input_size, num_spks*N)
        self.batch_norm = torch.nn.BatchNorm1d(num_spks*N)
        self.use_log = use_log

    def forward(self, x):
        num_channels = x[0].shape[0]
        h = [obs_single_channel for obs in x for obs_single_channel in obs]
        h = [self.norm(inputs) for inputs in h]
        h_packed = pt.ops.pack_module.pack_sequence(h)

        h = 0
        h_data = h_old = h_packed.data
        for idx in range(self.num_layers // 2):
            if not idx == 0:
                h_data = torch.cat([h, h_old], dim=-1)
                h_old = h
            h_packed = pt.pack_module.PackedSequence(h_data, h_packed.batch_sizes)
            del self.lstm[idx].states
            h = self.lstm[idx](h_packed).data
        h_data = torch.cat([h, h_old], dim=-1)
        h_data = self.linear(h_data)
        h_data = self.batch_norm(h_data)
        h_data = torch.nn.Sigmoid()(h_data)

        h_packed = pt.pack_module.PackedSequence(h_data, h_packed.batch_sizes)

        out = pt.ops.pack_module.pad_packed_sequence(
            h_packed, batch_first=True)[0]
        out = rearrange(out, '(c b) t f -> b c t f', c=num_channels)
        out = torch.stack(torch.split(out, self.fbins, dim=-1), dim=2) # b c k t f

        return [h[..., :x[idx].shape[1], :] for idx, h in enumerate(out)]


def si_loss(estimate, target, eps=1e-10):
    ###### expects estimate and target as shape of (CxKxT)

    th = torch
    x = estimate
    s = target
    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - th.mean(x, dim=-1, keepdim=True)
    s_zm = s - th.mean(s, dim=-1, keepdim=True)
    t = th.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    return -20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def beta_log_mse_loss(estimate, target, eps=1e-10):

    th = torch
    x = estimate
    s = target
    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - th.mean(x, dim=-1, keepdim=True)
    s_zm = s - th.mean(s, dim=-1, keepdim=True)
    t = l2norm(s_zm, keepdim=True) ** 2* x_zm / (th.sum(
        x_zm * s_zm, dim=-1, keepdim=True) + eps)
    return 20 * th.log10(l2norm(s_zm - t) + eps)

def log_mse_loss(estimate, target, eps=1e-10):
    th = torch
    x = estimate
    s = target
    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - th.mean(x, dim=-1, keepdim=True)
    s_zm = s - th.mean(s, dim=-1, keepdim=True)
    return 10 * th.log10(torch.sum((s_zm - x_zm)**2, axis=-1))