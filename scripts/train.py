import os
import glob
import numpy as np
from tqdm import tqdm

import torch


def create_tensors(
    source_tokens,
    target_tokens,
    activations,
    task_type="classification",
    dtype=None,
):
    """
    Method to pre-process loaded datasets into tensors that can be used to train
    probes and perform analyis on. The input tokens are represented as list of
    sentences, where each sentence is a list of tokens. Each token also has
    an associated label. All tokens from all sentences are flattened into one
    dimension in the returned tensors. The returned tensors will thus have
    ``total_num_tokens`` rows.

    Parameters
    ----------
    source_tokens: list of arrays
        List of sentences, where each sentence is a list of tokens. 
        e.g., [array(['M', 'ER0', 'IY1', ...], array(['JH', 'AH1', 'S', ...])
    target_tokens: list of arrays
        It depends on task_type. For example, in classification task, 
        articulatory_features can be used as targets.
        e.g., [['Bilabial', 'Vowel', 'Vowel', ...], [...]]
    activations : list of dictionary
        e.g., activations[0] has
        {'conv_0': ..., 'conv_1': ...}
        where each key has *sentence representations*, where each *sentence representation*
        is a numpy matrix of shape.
    task_type : str
        Either "classification" or "regression", indicate the kind of task that
        is being probed.
    dtype : str, optional
        None if the dtype of the activation tensor should be the same dtype as in the activations input
        e.g. 'float16' or 'float32' to enforce half-precision or full-precision floats

    """
    assert (
        task_type == "classification" or task_type == "regression"
    ), "Invalid model type"
    num_tokens = count_tokens(source_tokens)
    print("Number of tokens: ", num_tokens)

    # Concatenate activations
    concatenated_activations = [np.concatenate(list(d.values()), axis=1) for d in activations]
    num_neurons = concatenated_activations[0].shape[1]

    if task_type == "classification":
        label2idx = tok2idx(target_tokens)

    # print("length of source dictionary: ", len(src2idx))
    if task_type == "classification":
        print("length of target dictionary: ", len(label2idx))

    if dtype == None:
        dtype = concatenated_activations[0].dtype
    X = np.zeros((num_tokens, num_neurons), dtype=dtype)
    if task_type == "classification":
        y = np.zeros((num_tokens,), dtype=np.int)
    else:
        y = np.zeros((num_tokens,), dtype=np.float32)

    example_set = set()

    idx = 0
    for instance_idx, instance in enumerate(target_tokens):
        for token_idx, _ in enumerate(instance):
            if idx < num_tokens:
                X[idx] = concatenated_activations[instance_idx][token_idx, :]

            example_set.add(source_tokens[instance_idx][token_idx])
            if task_type == "classification":
                current_target_token = target_tokens[instance_idx][token_idx]
                y[idx] = label2idx[current_target_token]
            elif task_type == "regression":
                y[idx] = float(target_tokens[instance_idx][token_idx])

            idx += 1

    print(idx)
    print("Total instances: %d" % (num_tokens))
    print(list(example_set)[:20])

    print("Number of samples: ", X.shape[0])

    if task_type == "classification":
        return X, y
    return X, y


def count_tokens(source):
    """
    Utility function to count the total number of tokens in a dataset.
    """
    return sum([len(t) for t in source])

def tok2idx(tokens):
    """
    Utility function to generate unique indices for a set of tokens.
    """
    uniq_tokens = set().union(*tokens)
    return {p: idx for idx, p in enumerate(uniq_tokens)}


data_activation_dir = "/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow"
npz_files = glob.glob(os.path.join(data_activation_dir, '*.npz'))


all_phonesymbols = []
all_articulatory_features = []
all_activations = []
for file in tqdm(npz_files):
    data_dict = np.load(file, allow_pickle=True)
    activations = data_dict['activations'].item()
    phonesymbols = data_dict['phonesymbols']
    articulatory_features = data_dict['articulatory_features']

    all_phonesymbols.append(phonesymbols)
    all_articulatory_features.append(articulatory_features)
    all_activations.append(activations)


X, y = create_tensors(
    all_phonesymbols,
    all_articulatory_features,
    all_activations,
    task_type='classification'
)