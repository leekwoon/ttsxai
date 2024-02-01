import os
import sys
import shutil
import random
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import numpy as np

import torch

import ttsxai


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def read_ljs_metadata(mode):
    assert mode in ['train', 'test', 'val']
    path = os.path.join(
        os.path.dirname(ttsxai.__file__), 
        'filelists', 
        f'ljs_audio_text_{mode}_filelist.txt'
    )
    text_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            text_id, text = split[0], split[-1]
            text_dict[text_id] = text.strip()
    return text_dict


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confirm_and_delete_directory(directory):
    """Check if a directory exists and ask the user to confirm its deletion."""
    if os.path.exists(directory):
        response = input(f"The directory '{directory}' already exists. Delete it? [y/n]: ")
        if response.lower() == 'y':
            shutil.rmtree(directory)  # Delete the directory
            print(f"Deleted the directory: {directory}")
        else:
            print("Program exiting without deleting the directory.")
            sys.exit()  # Exit the program
