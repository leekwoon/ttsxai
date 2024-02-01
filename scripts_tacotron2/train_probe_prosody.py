import os
import time
import argparse
import pandas as pd

import ttsxai
from ttsxai.utils.utils import set_seed
from ttsxai.probe.train import train_probe
from ttsxai.probe.dataset import ProbeDataset
from ttsxai.probe.linear_probe import LinearProbe
from ttsxai.logging import logger, setup_logger


def parse_arguments(defaults):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    for key, value in defaults.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    return vars(parser.parse_args())


def setup_logdir(base_logdir, df_dir, label, seed):
    """Setup and return log directory."""
    setting1 = os.path.basename(os.path.dirname(df_dir))
    setting2 = os.path.basename(df_dir).split('.')[0]
    logdir = os.path.join(base_logdir, setting1, setting2, label, time.strftime('%Y_%m_%d_%H_%M_%S'), 'seed_' + str(seed))
    return logdir


def main():
    variant = dict(
        base_logdir=os.path.join(ttsxai.PACKAGE_DIR, 'logs/probe_prosody'),
        df_dir=os.path.join(ttsxai.PACKAGE_DIR, "data_df/LJSpeech/tacotron2_waveglow"),
        seed=0,
        lambda_l1=0.001,
        lambda_l2=0.001,
        batch_size=256,
        learning_rate=0.001,
        use_gpu=True,
        label='duration'
    )

    variant.update(parse_arguments(variant))
    set_seed(variant['seed'])
    logdir = setup_logdir(variant['base_logdir'], variant['df_dir'], variant['label'], variant['seed'])
    setup_logger(variant=variant, log_dir=logdir)

    """
    TODO: normalizer
    """

    """
    Debug
    """
    train_df = pd.read_pickle(os.path.join(variant['df_dir'], 'train_activation_df.pkl'))
    val_df = pd.read_pickle(os.path.join(variant['df_dir'], 'val_activation_df.pkl'))

    train_dataset = ProbeDataset(train_df, input_columns='activations', label_column=variant['label'])
    val_dataset = ProbeDataset(val_df, input_columns='activations', label_column=variant['label'])

    print(f'train dataset size: {len(train_dataset)}')
    print(f'val dataset size: {len(val_dataset)}')

    input_dim = len(train_dataset[0][0])
    probe = LinearProbe(input_dim, 1)

    train_probe(
        probe=probe,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_type='regression',
        lambda_l1=variant['lambda_l1'],
        lambda_l2=variant['lambda_l2'],
        batch_size=variant['batch_size'],
        learning_rate=variant['learning_rate'],
        use_gpu=variant['use_gpu'],
        logger=logger,
        logdir=logdir
    )


if __name__ == "__main__":
    main()