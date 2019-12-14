"""
Advanced training script for a mask_estimator. Uses sacred and configurable
to create a config, instantiate the model and Trainer and write everything
to a model file.
May be called as follows:
python -m padertorch.contrib.examples.mask_estimator.advanced_train -F $STORAGE_ROOT/name/of/model


"""
from pathlib import Path
import numpy as np

import sacred
from paderbox.database.merl_mixtures import MerlMixtures
from paderbox.database.keys import *
from paderbox.io import dump_json
from paderbox.utils.nested import deflatten
from padertorch.configurable import config_to_instance
from padertorch.contrib.jensheit.data import SequenceProvider
from desecting_tasnet.model import TasnetModel,TasnetTransformer
from padertorch.configurable import recursive_class_to_str
from padertorch.contrib.jensheit.utils import get_experiment_name
from padertorch.contrib.jensheit.utils import compare_configs
from desecting_tasnet.tasnet import TasnetBaseline
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred.utils import apply_backspaces_and_linefeeds
import os

model_dir = Path(os.environ['MODEL_DIR']) / 'test'
ex = sacred.Experiment('Train Mask Estimator')

ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    model_class = TasnetModel
    max_it = int(2e5)
    trainer_opts = deflatten({
        'model.factory': model_class,
        'model.separator.factory': TasnetBaseline,
        'optimizer.factory': Adam,
        'stop_trigger': (max_it, 'iteration'),
        'summary_trigger': (250, 'iteration'),
        'checkpoint_trigger': (1000, 'iteration'),
        'virtual_minibatch_size': 2
    })
    provider_opts = deflatten({
        'database.factory': MerlMixtures,
        'transform.factory': TasnetTransformer,
        'batch_size': 5,
        'audio_keys': [OBSERVATION, SPEECH_SOURCE]
    })
    trainer_opts['model']['transformer'] = provider_opts['transform']

    storage_dir = None
    add_name = None
    if storage_dir is None:
        ex_name = get_experiment_name(trainer_opts['model'],
                                      submodel='separator')
        if add_name is not None:
            ex_name += f'_{add_name}'
        observer = sacred.observers.FileStorageObserver.create(
            str(model_dir / ex_name))
        storage_dir = observer.basedir
    else:
        sacred.observers.FileStorageObserver.create(storage_dir)
    trainer_opts['storage_dir'] = storage_dir

    if (Path(storage_dir) / 'init.json').exists():
        trainer_opts, provider_opts = compare_configs(
            storage_dir, trainer_opts, provider_opts)

    Trainer.get_config(
        trainer_opts
    )
    SequenceProvider.get_config(
        provider_opts
    )
    debug=False
    validate_checkpoint = 'ckpt_latest.pth'
    validation_length = 1000  # number of examples taken from the validation iterator
    validation_kwargs = dict(
        metric='sdr',
        maximize=True,
        max_checkpoints=1
    )

@ex.named_config
def time_segments():
    provider_opts = {'time_segments': 32000,
                     'batch_size': 8}
    trainer_opts = {'max_trigger': (int(2e5), 'iteration'),
                    'virtual_minibatch_size': 2}



@ex.capture
def initialize_trainer_provider(task, trainer_opts, provider_opts, _run):


    storage_dir = Path(trainer_opts['storage_dir'])
    if (storage_dir / 'init.json').exists():
        assert task in ['restart', 'validate'], task
    elif task in ['train', 'create_checkpoint']:
        dump_json(dict(trainer_opts=recursive_class_to_str(trainer_opts),
                       provider_opts=recursive_class_to_str(provider_opts)),
                  storage_dir / 'init.json')
    else:
        raise ValueError(task, storage_dir)
    sacred.commands.print_config(_run)

    trainer = Trainer.from_config(trainer_opts)
    assert isinstance(trainer, Trainer)
    provider = config_to_instance(provider_opts)
    return trainer, provider


@ex.command
def restart(validation_length):
    trainer, provider = initialize_trainer_provider(task='restart')
    train_iterator = provider.get_train_iterator()
    eval_iterator = provider.get_eval_iterator(
        num_examples=validation_length
    )
    trainer.load_checkpoint()
    trainer.test_run(train_iterator, eval_iterator)
    trainer.register_validation_hook(eval_iterator, metric='sdr',
                                     maximize=True)
    trainer.train(train_iterator, resume=True)


# @ex.command
# def validate(_config):
#     import os
#     from pt_bss.evaluate import evaluate_masks
#     from functools import partial
#     from paderbox.io import dump_json
#     import torch
#     trainer, provider = initialize_trainer_provider(task='validate')
#     storage_dir = trainer.storage_dir
#     checkpoint_dir = trainer.checkpoint_dir / _config['validate_checkpoint']
#     json_name = '_'.join(['results', *checkpoint_dir.name.split('.')[0].split('_')[1:]]) + '.json'
#     assert not (storage_dir / json_name).exists(), (
#         f'model_dir has already bin evaluatet, {storage_dir}')
#     checkpoint_dict = torch.load(str(checkpoint_dir), map_location='cpu')
#     trainer.load_state_dict(checkpoint_dict)
#     trainer.model.cpu()
#     trainer.model.eval()
#     predict_iterator = provider.get_predict_iterator()
#     evaluation_json = dict(sdr=dict(), pesq=dict())
#     provider.opts.multichannel = True
#     batch_size = 1
#     provider.opts.batch_size = batch_size
#     # with ThreadPoolExecutor(os.cpu_count()) as executor:
#     for example_id, sdr, pesq in map(partial(
#             evaluate_masks, model=trainer.model,
#             stft=provider.transform.stft
#     ), predict_iterator):
#         evaluation_json['sdr'][example_id] = sdr
#         evaluation_json['pesq'][example_id] = pesq
#     evaluation_json['pesq_mean'] = np.mean(
#         [value for value in evaluation_json['pesq'].values()], axis=0)
#     evaluation_json['sdr'] = np.mean(
#         [value for value in evaluation_json['sdr'].values()], axis=0)
#     dump_json(evaluation_json, storage_dir / json_name)


@ex.command
def create_checkpoint(_config):
    # This may be useful to merge to separatly trained models into one
    raise NotImplementedError


@ex.automain
def train(debug, validation_length):
    trainer, provider = initialize_trainer_provider(task='train')
    train_iterator = provider.get_train_iterator()
    if debug:
        eval_iterator = provider.get_eval_iterator(
            num_examples=2
        )
    else:
        eval_iterator = provider.get_eval_iterator(
            num_examples=validation_length
    )
    trainer.register_validation_hook(eval_iterator, metric='sdr',
                                     maximize=True)
    trainer.train(train_iterator)
