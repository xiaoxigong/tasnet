from padertorch import Model
import torch
import padertorch as pt
from padertorch.summary import mask_to_image, stft_to_image
from padertorch.modules.mask_estimator import MaskKeys as K
import numpy as np
from einops import rearrange
from desecting_tasnet.tasnet import Encoder, Decoder, si_loss, beta_log_mse_loss, log_mse_loss
from desecting_tasnet.tasnet import TasnetBaseline


class MaskKeys:
    SPEECH_MASK_PRED = 'speech_mask_prediction'
    SPATIAL_FEATURE = 'spatial_feature'
    NOISE_MASK_PRED = 'noise_mask_prediction'
    SPEECH_MASK_LOGITS = 'speech_mask_logits'
    NOISE_MASK_LOGITS = 'noise_mask_logits'
    SPEECH_MASK_TARGET = 'speech_mask_target'
    NOISE_MASK_TARGET = 'noise_mask_target'
    OBSERVATION_STFT = 'observation_stft'
    OBSERVATION_ABS = 'observation_abs'
    SPEECH_PREDICTION_STFT = 'speech_prediction_stft'
    SPEECH_PREDICTION = 'speech_prediction'
    COS_PHASE_DIFFERENCE = 'cos_phase_difference'
    SPEECH_ABS = 'speech_abs'
    EMBEDDINGS = 'embeddings'
    SPEECH_FEATURES = 'speech_features'


M_K = MaskKeys


class TasnetTransformer:

    def __init__(self, num_speaker=2, target='source'):
        self.num_speaker = num_speaker
        self.target = target

    def inv(self, signal):
        return signal

    def __call__(self, example):
        import collections
        if isinstance(example, (list, tuple, collections.Generator)):
            return [self.transform(ex) for ex in example]
        else:
            return self.transform(example)

    def maybe_add_channel(self, signal):
        if signal.ndim == 3:
            signal = signal.swapaxes(0,1)
        if signal.ndim > 1 and signal.shape[0] > self.num_speaker:
            return signal
        else:
            return np.expand_dims(signal, axis=0)

    def transform(self, example):
        num_samp = example[K.NUM_SAMPLES]
        example[M_K.OBSERVATION_ABS] = self.maybe_add_channel(
            example[K.OBSERVATION]).astype(np.float32)
        num_channels = example[M_K.OBSERVATION_ABS].shape[0]
        if self.target == 'early':
            example[K.SPEECH_SOURCE] = self.maybe_add_channel(
                example[K.SPEECH_REVERBERATION_DIRECT]).astype(np.float32)
        elif self.target == 'image':
            example[K.SPEECH_SOURCE] = self.maybe_add_channel(
                example[K.SPEECH_IMAGE]).astype(np.float32)
        elif self.target == 'source':
            example[K.SPEECH_SOURCE] = self.maybe_add_channel(
                example[K.SPEECH_SOURCE]).astype(np.float32)
        if K.SPEECH_SOURCE in example:
            example[K.SPEECH_SOURCE] = example[K.SPEECH_SOURCE][..., :num_samp]
        assert example[K.SPEECH_SOURCE].shape[-1] == example[K.NUM_SAMPLES], (
            example[K.SPEECH_SOURCE].shape, example[K.NUM_SAMPLES])
        # example.pop(K.SPEECH_IMAGE, {})
        # example.pop(K.NOISE_IMAGE, {})
        return example


class ModelTemplate(Model):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['separator'] = dict(factory=TasnetBaseline)
        config['transformer'] = dict(factory=TasnetTransformer)

    def __init__(self, separator, encoder=None, decoder=None,
                 transformer=None, sample_rate=8000, load_ckpt=None):
        super().__init__()
        self.separator = separator
        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer
        self.sample_rate = sample_rate
        self.load_ckpt = load_ckpt

    def load_ckeckpoint(self):
        if not self.load_ckpt is None:
            state_dict = torch.load(self.load_ckpt)
            self.load_state_dict(state_dict['model'])

    def forward(self, inputs):
        out = dict()
        obs_in = inputs[M_K.OBSERVATION_ABS]
        if self.encoder is not None:
            observations = [self.encoder(obs) for obs in obs_in]
        else:
            observations = obs_in
        out[M_K.SPEECH_FEATURES] = observations
        if M_K.SPATIAL_FEATURE in inputs:
            observations = [torch.cat(
                [torch.log(observations[idx] + 1e-10), spatial_ft], dim=-1
            ) for idx, spatial_ft in enumerate(inputs[M_K.SPATIAL_FEATURE])]

        mask_prediction = self.separator(observations)

        out[M_K.SPEECH_MASK_PRED] = mask_prediction
        if self.decoder is not None:
            decoder_out = [
                self.decoder(obs, mask_prediction[idx],
                             inputs[K.NUM_SAMPLES][idx])
                for idx, obs in enumerate(observations)
            ]
            out[M_K.SPEECH_PREDICTION] = decoder_out

        return out

    def review(self, inputs, outputs):
        """
                :param batch: dict of lists
                :param output: output of the forward step
                :return:
                """
        losses = self.get_losses(inputs, outputs)
        if not self.training:
            scalars = self.get_scalars(inputs, outputs)
        else:
            scalars = dict()
        scalars.update(losses)
        return dict(loss=losses.pop('loss'),
                    scalars=scalars,
                    audios=self.get_audios(inputs, outputs),
                    images=self.get_images(inputs, outputs))

    def get_scalars(self, inputs, outputs):
        from pb_bss.evaluation import mir_eval_sources
        scalars = dict()

        def mir_eval_torch(source, prediction):
            if isinstance(source, torch.Tensor):
                source = source.detach().cpu().numpy()
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach().cpu().numpy()
            check_all_zero = np.abs(np.sum(prediction, axis=-1))
            if all(np.abs(check_all_zero) > 0):
                return mir_eval_sources(source, prediction)[0]
            else:
                print(f'SDR is set to zero since one estimated'
                      f'speech signal was all zero {check_all_zero}')
                return 0

        speech_prediction = None
        if M_K.SPEECH_PREDICTION in outputs:
            speech_prediction = maybe_remove_channel(
                outputs[M_K.SPEECH_PREDICTION][0])
        elif M_K.SPEECH_PREDICTION_STFT in outputs:
            speech_prediction = maybe_remove_channel(self.transformer.inv(
                outputs[M_K.SPEECH_PREDICTION_STFT][0])
            )[..., :inputs[K.NUM_SAMPLES][0]]

        if K.SPEECH_SOURCE in inputs and speech_prediction is not None:
            source = maybe_remove_channel(inputs['speech_source'][0])
            scalars['sdr'] = mir_eval_torch(source, speech_prediction)
        elif K.SPEECH_IMAGE in inputs and speech_prediction is not None:
            source = maybe_remove_channel(inputs['speech_image'][0])
            scalars['sdr'] = mir_eval_torch(source, speech_prediction)
        return scalars

    def get_losses(self, inputs, outputs):
        raise NotImplementedError

    def get_images(self, batch, output):
        images = dict()
        if M_K.SPEECH_MASK_PRED in output:
            speech_mask = output[M_K.SPEECH_MASK_PRED][0][0]
            if self.encoder is not None:
                observation = output[M_K.SPEECH_FEATURES][0]
            else:
                observation = batch[M_K.OBSERVATION_ABS][0]
            for idx, mask in enumerate(speech_mask):
                images[f'speech_mask_{idx}'] = mask_to_image(speech_mask[idx], True)
            images['observed_stft'] = stft_to_image(observation, True)

        if M_K.NOISE_MASK_PRED in output:
            noise_mask = output[M_K.NOISE_MASK_PRED][0]
            images['noise_mask'] = mask_to_image(noise_mask, True)
        if batch is not None and M_K.SPEECH_MASK_TARGET in batch:
            target_mask = batch[M_K.SPEECH_MASK_TARGET][0][0]
            for idx, mask in enumerate(target_mask):
                images[f'speech_mask_target_{idx}'] = mask_to_image(
                target_mask[idx], True)
            if M_K.NOISE_MASK_TARGET in batch:
                images['noise_mask_target'] = mask_to_image(
                    batch[M_K.NOISE_MASK_TARGET][0], True)
        return images

    # ToDo: add scalar review

    def get_audios(self, inputs, outputs):

        def norm_and_maybe_remove_channel(audio, exp_dim=2):
            if isinstance(audio, torch.Tensor):
                audio = audio / (torch.max(torch.abs(audio), dim=-1, keepdim=True)[0] + 1e-10)
            else:
                audio = audio / (np.max(np.abs(audio), axis=-1, keepdims=True) + 1e-10)
            return maybe_remove_channel(audio, exp_dim)

        audio_dict = dict()
        if K.OBSERVATION in inputs:
            audio_dict.update({K.OBSERVATION: (norm_and_maybe_remove_channel(
                inputs[K.OBSERVATION][0], exp_dim=1), self.sample_rate)})
        if K.SPEECH_IMAGE in inputs:
            audio_dict.update({
                K.SPEECH_IMAGE: (norm_and_maybe_remove_channel(
                    inputs[K.SPEECH_IMAGE][0])[0], self.sample_rate)
            })
        if K.SPEECH_SOURCE in inputs:
            audio_dict.update({
                K.SPEECH_SOURCE: (norm_and_maybe_remove_channel(
                    inputs[K.SPEECH_SOURCE][0])[0], self.sample_rate)
            })
        if M_K.SPEECH_PREDICTION in outputs:
            audio_dict.update({
                M_K.SPEECH_PREDICTION: (norm_and_maybe_remove_channel(
                    outputs[M_K.SPEECH_PREDICTION][0])[0], self.sample_rate)
            })
        elif M_K.SPEECH_PREDICTION_STFT in outputs:
            speech_prediction = self.transformer.inv(outputs[M_K.SPEECH_PREDICTION_STFT][0])
            audio_dict.update({
                M_K.SPEECH_PREDICTION: (norm_and_maybe_remove_channel(
                    speech_prediction)[0], self.sample_rate)
            })
        return audio_dict


def maybe_remove_channel(signal, exp_dim=2, ref_channel=0):
    if isinstance(signal, np.ndarray):
        ndim = signal.ndim
    else:
        ndim = signal.dim()
    if ndim == exp_dim + 1:
        assert signal.shape[0] < 20, f'The first dim is supposed to be the ' \
            f'channel dimension, however the shape is {signal.shape}'
        return signal[ref_channel]
    elif ndim <= exp_dim:
        return signal
    else:
        raise ValueError(f'Either the signal has ndim {exp_dim} or'
                         f' {exp_dim +1}', signal.shape)


class TasnetModel(ModelTemplate):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['encoder'] = dict(factory=Encoder)
        config['separator'] = dict(factory=TasnetBaseline,
                                   N=config['encoder']['N'])
        config['transformer'] = dict(factory=TasnetTransformer)
        config['decoder'] = dict(factory=Decoder, N=config['encoder']['N'],
                                 L=config['encoder']['L'],
                                 stride=config['encoder']['stride'])

    def __init__(self, separator, encoder=None, decoder=None,
                 transformer=None, sample_rate=8000, load_ckpt=None,
                 loss_fn='si_snr'):
        super().__init__(separator, encoder, decoder, transformer,
                         sample_rate, load_ckpt)
        self.load_ckeckpoint()
        if loss_fn == 'si_snr':
            self.loss_fn = si_loss
        elif loss_fn == 'beta_log_mse':
            self.loss_fn = beta_log_mse_loss
        elif loss_fn == 'log_mse':
            self.loss_fn = log_mse_loss
        else:
            raise ValueError(loss_fn)

    def get_losses(self, inputs, outputs):
        tasnet_loss = list()
        for speech_source, speech_pred in zip(
            inputs[K.SPEECH_SOURCE],
            outputs[M_K.SPEECH_PREDICTION]
        ):
            if speech_source.shape[0] == 1:
                num_channels = speech_pred.shape[0]
                speech_source = speech_source.expand(
                    (num_channels, *speech_source.shape[1:]))
            if speech_pred.shape[0] == 1:
                speech_source = speech_source[:1]
            tasnet_loss.append(pt.ops.loss.pit_loss(
                rearrange(speech_pred[:, :2], 'c k t -> c k t'),
                rearrange(speech_source, 'c k t -> c k t'),
                loss_fn=self.loss_fn))
        return dict(loss=sum(tasnet_loss))