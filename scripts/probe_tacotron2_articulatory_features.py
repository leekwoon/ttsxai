import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch

import neurox.interpretation.utils as utils
import neurox.interpretation.ablation as ablation
import neurox.interpretation.linear_probe as linear_probe

from ttsxai.utils.utils import read_ljs_metadata


log_dir = '/nas/users/dahye/kw/tts/ttsxai/logs/probe_tacotron2_articulatory_features'
data_activation_dir = "/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow"


def load_npz_file(file_path):
    data_dict = np.load(file_path, allow_pickle=True)
    phonesymbols = list(data_dict['phonesymbols'])
    articulatory_features = list(data_dict['articulatory_features'])
    activations = data_dict['activations'].item()
    return phonesymbols, articulatory_features, activations

def load_npz_files(npz_files):
    tokens = {'source': [], 'target': []}
    dict_activations = []
    for file in tqdm(npz_files):
        data_dict = np.load(file, allow_pickle=True)
        tokens['source'].append(list(data_dict['phonesymbols']))
        tokens['target'].append(list(data_dict['articulatory_features']))
        dict_activations.append(data_dict['activations'].item())
    return tokens, dict_activations

def parallel_load_npz_files(npz_files, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(load_npz_file, npz_files), total=len(npz_files)))

    tokens = {'source': [], 'target': []}
    dict_activations = []
    for phonesymbols, articulatory_features, activations in results:
        tokens['source'].append(phonesymbols)
        tokens['target'].append(articulatory_features)
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

    for source_sentence, target_sentence, activation in zip(tokens['source'], tokens['target'], activations):
        filtered_source_sentence = []
        filtered_target_sentence = []
        filtered_activation = []
        for source_token, target_token, a in zip(source_sentence, target_sentence, activation):
            if target_token not in ignore_tags:
                filtered_source_sentence.append(source_token)
                filtered_target_sentence.append(target_token)
                filtered_activation.append(a)
        filtered_source_tokens.append(np.array(filtered_source_sentence))
        filtered_target_tokens.append(np.array(filtered_target_sentence))
        filtered_activations.append(np.array(filtered_activation))

    tokens['source'] = filtered_source_tokens
    tokens['target'] = filtered_target_tokens
    activations = filtered_activations

    X, y, mapping = utils.create_tensors(tokens, activations, 'NN')
    label2idx, idx2label, src2idx, idx2src = mapping

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
        label2idx=label2idx,
        idx2label=idx2label,
        src2idx=src2idx, 
        idx2src=idx2src,
        neuronidx2name=neuronidx2name
    )


def probe():
    data_dict = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X = data_dict['X']
    y = data_dict['y']
    label2idx = data_dict['label2idx'].item()
    idx2label = data_dict['idx2label'].item()
    src2idx = data_dict['src2idx'].item()
    idx2src = data_dict['idx2src'].item()
    neuronidx2name = data_dict['neuronidx2name'].item()

    probe = linear_probe.train_logistic_regression_probe(
        X, y, lambda_l1=0.001, lambda_l2=0.001, num_epochs=5)
    torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'probe.pth'))

    scores = linear_probe.evaluate_probe(probe, X, y, idx_to_class=idx2label)
    np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)


def probe_layerwise():
    data_dict = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X = data_dict['X']
    y = data_dict['y']
    label2idx = data_dict['label2idx'].item()
    idx2label = data_dict['idx2label'].item()
    src2idx = data_dict['src2idx'].item()
    idx2src = data_dict['idx2src'].item()
    neuronidx2name = data_dict['neuronidx2name'].item()
    
    for k in [0, 1, 2, 3]:
        layer_k_X = ablation.filter_activations_by_layers(X, [k], 4)
        probe_layer_k = linear_probe.train_logistic_regression_probe(
            layer_k_X, y, lambda_l1=0.001, lambda_l2=0.001, num_epochs=5
        )
        torch.save(
            probe_layer_k.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_layer_{k}.pth'))

        scores = linear_probe.evaluate_probe(probe_layer_k, layer_k_X, y, idx_to_class=idx2label)
        np.savez(os.path.join(log_dir, 'scores', f'probe_layer_{k}.npz'), **scores)


def probe_selected():
    data_dict = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X = data_dict['X']
    y = data_dict['y']
    label2idx = data_dict['label2idx'].item()
    idx2label = data_dict['idx2label'].item()
    src2idx = data_dict['src2idx'].item()
    idx2src = data_dict['idx2src'].item()
    neuronidx2name = data_dict['neuronidx2name'].item()

    # load pre-trained probe
    probe = linear_probe.train_logistic_regression_probe(
        X, y, lambda_l1=0.001, lambda_l2=0.001,
        num_epochs=0)
    probe.load_state_dict(
        torch.load(os.path.join(log_dir, 'models', 'probe.pth')))

    for n in range(1, 31):
        ordering, cutoffs = linear_probe.get_neuron_ordering(probe, label2idx)
        X_selected_n = ablation.filter_activations_keep_neurons(X, ordering[:n])
        probe_selected_n = linear_probe.train_logistic_regression_probe(X_selected_n, y, lambda_l1=0.001, lambda_l2=0.001,
            num_epochs=3)
        torch.save(
            probe_selected_n.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_selected_{n}.pth'))

        scores = linear_probe.evaluate_probe(probe_selected_n, X_selected_n, y, idx_to_class=idx2label)
        np.savez(os.path.join(log_dir, 'scores', f'probe_selected_{n}.npz'), **scores)


# def evaluate_all():



def main():
    prepare_data('train')
    prepare_data('test')
    # probe()
    # probe_layerwise()
    # probe_selected()


if __name__ == "__main__":
    main()