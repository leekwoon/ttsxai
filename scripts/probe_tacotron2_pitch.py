import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch

import neurox.interpretation.utils as utils
import neurox.interpretation.ablation as ablation
import neurox.interpretation.linear_probe as linear_probe

from ttsxai.utils.utils import read_ljs_metadata


log_dir = '/nas/users/dahye/kw/tts/ttsxai/logs/probe_tacotron2_pitch'
data_activation_dir = "/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow"


def load_npz_file(file_path):
    data_dict = np.load(file_path, allow_pickle=True)
    phonesymbols = list(data_dict['phonesymbols'])
    pitchs = list(data_dict['pitch'])
    articulatory_features = list(data_dict['articulatory_features'])
    activations = data_dict['activations'].item()
    return phonesymbols, pitchs, articulatory_features, activations


def load_npz_files(npz_files):
    tokens = {'source': [], 'target': []}
    dict_activations = []
    for file in tqdm(npz_files):
        data_dict = np.load(file, allow_pickle=True)
        tokens['source'].append(list(data_dict['phonesymbols']))
        tokens['target'].append(list(data_dict['pitch']))
        tokens['articulatory_features'].append(list(data_dict['articulatory_features']))
        dict_activations.append(data_dict['activations'].item())
    return tokens, dict_activations


def parallel_load_npz_files(npz_files, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(load_npz_file, npz_files), total=len(npz_files)))

    tokens = {'source': [], 'target': [], 'articulatory_features': []}
    dict_activations = []
    for phonesymbols, pitchs, articulatory_features, activations in results:
        tokens['source'].append(phonesymbols)
        tokens['target'].append(pitchs)
        # meta data for used for filtering out
        tokens['articulatory_features'].append(articulatory_features)
        dict_activations.append(activations)
    return tokens, dict_activations


def prepare_data(mode='train'):
    # Dictionary keys to filter
    keys_to_filter = read_ljs_metadata(mode=mode)

    # List to store filtered paths
    npz_files = []

    # Iterate over all files in the directory
    for file in os.listdir(data_activation_dir):
        # Check only for .npz files
        if file.endswith('.npz'):
            # Extract the identifier part from the file name (e.g., 'LJ037-0213')
            identifier = file.split('.')[0]

            # If this identifier is included in the dictionary keys, add to the list
            if identifier in keys_to_filter:
                full_path = os.path.join(data_activation_dir, file)
                npz_files.append(full_path)

    tokens, dict_activations = parallel_load_npz_files(npz_files)

    # Concatenate activations
    activations = [np.concatenate(list(d.values()), axis=1) for d in dict_activations]

    # filtering
    ignore_tags = ['Space', 'Punctuation']

    filtered_source_tokens = []
    filtered_target_tokens = []
    filtered_activations = []

    for source_sentence, target_sentence, articulatory_feature, activation in zip(tokens['source'], tokens['target'], tokens['articulatory_features'], activations):
        filtered_source_sentence = []
        filtered_target_sentence = []
        filtered_activation = []
        for source_token, target_token, af, a in zip(source_sentence, target_sentence, articulatory_feature, activation):
            if af not in ignore_tags:
                filtered_source_sentence.append(source_token)
                filtered_target_sentence.append(target_token)
                # if target_token == 0:
                #     print(source_token, af)
                filtered_activation.append(a)
        filtered_source_tokens.append(np.array(filtered_source_sentence))
        filtered_target_tokens.append(np.array(filtered_target_sentence))
        filtered_activations.append(np.array(filtered_activation))

    tokens['source'] = filtered_source_tokens
    tokens['target'] = filtered_target_tokens
    activations = filtered_activations

    X, y, mapping = utils.create_tensors(tokens, activations, 'NN', task_type='regression')
    src2idx, idx2src = mapping

    def get_neuronidx2name(d):
        mapping = {}
        current_start_index = 0
        for layer_name, activations in d.items():
            # Calculate the end index for this activation
            end_index = current_start_index + activations.shape[1] - 1
            for i in range(current_start_index, end_index + 1):
                mapping[i] = f'{layer_name}__{i - current_start_index}'
            current_start_index = end_index + 1
        return mapping

    neuronidx2name = get_neuronidx2name(dict_activations[0])

    np.savez(os.path.join(log_dir, f'data/{mode}_data.npz'),
        X=X,
        y=y,
        src2idx=src2idx, 
        idx2src=idx2src,
        neuronidx2name=neuronidx2name
    )


def main():
    prepare_data('train')
    prepare_data('test')
    # probe()
    # probe_layerwise()
    # probe_selected()


if __name__ == "__main__":
    main()