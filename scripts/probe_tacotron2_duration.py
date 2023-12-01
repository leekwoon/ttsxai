import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F

import neurox.interpretation.utils as utils
import neurox.interpretation.ablation as ablation
import neurox.interpretation.linear_probe as linear_probe

from ttsxai.utils.utils import read_ljs_metadata


log_dir = '/nas/users/dahye/kw/tts/ttsxai/logs/probe_tacotron2_duration'
data_activation_dir = "/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow"


class MlpProbe(nn.Module):
    """Torch model for a multi-layer perceptron (MLP)"""

    def __init__(self, input_size, num_classes, hidden_size1=256, hidden_size2=256):
        """Initialize an MLP model"""
        super(MlpProbe, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        """Run a forward pass on the model"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
        
def get_neuron_ordering_for_linear_regression(probe, search_stride=100):
    """
    Get neuron ordering for a specific class from a trained probe.

    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    search_stride : int, optional
        Number of steps to divide the weight mass percentage

    Returns
    -------
    neuron_ordering : numpy.ndarray
        Array of neurons ordered by their importance for the specified class
    """
    # class_idx = class_to_idx[class_name]
    weights = list(probe.parameters())[0].data.cpu().numpy()
    abs_weights = np.abs(weights[0])

    neuron_orderings = []
    for p in range(1, search_stride + 1):
        percentage = p / search_stride
        total_mass = np.sum(abs_weights)
        sorted_idx = np.argsort(abs_weights)[::-1]  # Sort in descending order
        cum_sums = np.cumsum(abs_weights[sorted_idx])
        selected_neurons = sorted_idx[cum_sums <= total_mass * percentage]
        neuron_orderings.extend(selected_neurons)

    # Remove duplicates while preserving order
    neuron_ordering = list(dict.fromkeys(neuron_orderings))

    return np.array(neuron_ordering)



def load_npz_file(file_path):
    data_dict = np.load(file_path, allow_pickle=True)
    phonesymbols = list(data_dict['phonesymbols'])
    durations = list(data_dict['duration'])
    articulatory_features = list(data_dict['articulatory_features'])
    activations = data_dict['activations'].item()
    return phonesymbols, durations, articulatory_features, activations

def load_npz_files(npz_files):
    tokens = {'source': [], 'target': [], 'articulatory_features': []}
    dict_activations = []
    for file in tqdm(npz_files):
        data_dict = np.load(file, allow_pickle=True)
        tokens['source'].append(list(data_dict['phonesymbols']))
        tokens['target'].append(list(data_dict['duration']))
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
    for phonesymbols, durations, articulatory_features, activations in results:
        tokens['source'].append(phonesymbols)
        tokens['target'].append(durations)
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


def probe():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    probe = linear_probe.LinearProbe(X_train.shape[1], 1)

    scores = {'train': [], 'test': []}
    for epoch in range(1):
        probe = linear_probe.train_linear_regression_probe(
            X_train, y_train, lambda_l1=0.001, lambda_l2=0.001, num_epochs=1, probe=probe)
    
        scores_train = linear_probe.evaluate_probe(probe, X_train, y_train, metric='mse')
        print( '[* train score]', scores_train)
        scores_test = linear_probe.evaluate_probe(probe, X_test, y_test, metric='mse')
        print( '[* test score]', scores_test)

        scores['train'].append(scores_train['__OVERALL__'])
        scores['test'].append(scores_test['__OVERALL__'])

    torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'probe.pth'))
    np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)


def probe_noreg(): # No regularization
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    probe = linear_probe.LinearProbe(X_train.shape[1], 1)

    scores = {'train': [], 'test': []}
    for epoch in range(1):
        probe = linear_probe.train_linear_regression_probe(
            X_train, y_train, lambda_l1=0., lambda_l2=0., num_epochs=1, probe=probe)
    
        scores_train = linear_probe.evaluate_probe(probe, X_train, y_train, metric='mse')
        print( '[* train score]', scores_train)
        scores_test = linear_probe.evaluate_probe(probe, X_test, y_test, metric='mse')
        print( '[* test score]', scores_test)

        scores['train'].append(scores_train['__OVERALL__'])
        scores['test'].append(scores_test['__OVERALL__'])

    torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'probe_noreg.pth'))
    np.savez(os.path.join(log_dir, 'scores', 'probe_noreg.npz'), **scores)


def probe_layerwise():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # data_dict = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    # X = data_dict['X']
    # y = data_dict['y']
    # label2idx = data_dict['label2idx'].item()
    # idx2label = data_dict['idx2label'].item()
    # src2idx = data_dict['src2idx'].item()
    # idx2src = data_dict['idx2src'].item()
    # neuronidx2name = data_dict['neuronidx2name'].item()
    
    for k in [0, 1, 2, 3]:
        layer_k_X_train = ablation.filter_activations_by_layers(X_train, [k], 4)
        layer_k_X_test = ablation.filter_activations_by_layers(X_test, [k], 4)

        probe_layer_k = linear_probe.LinearProbe(layer_k_X_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(3):
            probe_layer_k = linear_probe.train_linear_regression_probe(
                layer_k_X_train, y_train, lambda_l1=0.001, lambda_l2=0.001, num_epochs=1, probe=probe_layer_k)
        
            scores_train = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        # torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'probe.pth'))
        torch.save(
            probe_layer_k.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_layer_{k}.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'probe_layer_{k}.npz'), **scores)


def probe_layerwise_noreg():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # data_dict = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    # X = data_dict['X']
    # y = data_dict['y']
    # label2idx = data_dict['label2idx'].item()
    # idx2label = data_dict['idx2label'].item()
    # src2idx = data_dict['src2idx'].item()
    # idx2src = data_dict['idx2src'].item()
    # neuronidx2name = data_dict['neuronidx2name'].item()
    
    for k in [0, 1, 2, 3]:
        layer_k_X_train = ablation.filter_activations_by_layers(X_train, [k], 4)
        layer_k_X_test = ablation.filter_activations_by_layers(X_test, [k], 4)

        probe_layer_k = linear_probe.LinearProbe(layer_k_X_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(3):
            probe_layer_k = linear_probe.train_linear_regression_probe(
                layer_k_X_train, y_train, lambda_l1=0., lambda_l2=0., num_epochs=1, probe=probe_layer_k)
        
            scores_train = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        # torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'probe.pth'))
        torch.save(
            probe_layer_k.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_layer_{k}_noreg.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'probe_layer_{k}_noreg.npz'), **scores)


def probe_selected():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # load pre-trained probe
    probe = linear_probe.LinearProbe(X_train.shape[1], 1)
    probe.load_state_dict(
        torch.load(os.path.join(log_dir, 'models', 'probe.pth')))
    
    for n in range(1, 101 + 1, 5):
        ordering = get_neuron_ordering_for_linear_regression(probe)

        X_selected_n_train = ablation.filter_activations_keep_neurons(X_train, ordering[:n])
        X_selected_n_test = ablation.filter_activations_keep_neurons(X_test, ordering[:n])

        probe_selected_n = linear_probe.LinearProbe(X_selected_n_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(1):
            probe_selected_n = linear_probe.train_linear_regression_probe(
                X_selected_n_train, y_train, lambda_l1=0.001, lambda_l2=0.001, num_epochs=1, probe=probe_selected_n)
        
            scores_train = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        torch.save(
            probe_selected_n.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_selected_{n}.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'probe_selected_{n}.npz'), **scores)


def probe_selected_bottom():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # load pre-trained probe
    probe = linear_probe.LinearProbe(X_train.shape[1], 1)
    probe.load_state_dict(
        torch.load(os.path.join(log_dir, 'models', 'probe.pth')))
    
    for n in range(1, 101 + 1, 5):
        ordering = get_neuron_ordering_for_linear_regression(probe)
        # reverse order!!!!!!!!
        ordering = ordering[::-1]

        X_selected_n_train = ablation.filter_activations_keep_neurons(X_train, ordering[:n])
        X_selected_n_test = ablation.filter_activations_keep_neurons(X_test, ordering[:n])

        probe_selected_n = linear_probe.LinearProbe(X_selected_n_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(1):
            probe_selected_n = linear_probe.train_linear_regression_probe(
                X_selected_n_train, y_train, lambda_l1=0.001, lambda_l2=0.001, num_epochs=1, probe=probe_selected_n)
        
            scores_train = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        torch.save(
            probe_selected_n.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_selected_bottom_{n}.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'probe_selected_bottom_{n}.npz'), **scores)


def probe_selected_noreg():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # load pre-trained probe
    probe = linear_probe.LinearProbe(X_train.shape[1], 1)
    probe.load_state_dict(
        torch.load(os.path.join(log_dir, 'models', 'probe_noreg.pth')))
    
    for n in range(1, 101 + 1, 5):
        ordering = get_neuron_ordering_for_linear_regression(probe)

        X_selected_n_train = ablation.filter_activations_keep_neurons(X_train, ordering[:n])
        X_selected_n_test = ablation.filter_activations_keep_neurons(X_test, ordering[:n])

        probe_selected_n = linear_probe.LinearProbe(X_selected_n_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(1):
            probe_selected_n = linear_probe.train_linear_regression_probe(
                X_selected_n_train, y_train, lambda_l1=0., lambda_l2=0., num_epochs=1, probe=probe_selected_n)
        
            scores_train = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        torch.save(
            probe_selected_n.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_selected_{n}_noreg.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'probe_selected_{n}_noreg.npz'), **scores)


def probe_selected_bottom_noreg():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # load pre-trained probe
    probe = linear_probe.LinearProbe(X_train.shape[1], 1)
    probe.load_state_dict(
        torch.load(os.path.join(log_dir, 'models', 'probe_noreg.pth')))
    
    for n in range(1, 101 + 1, 5):
        ordering = get_neuron_ordering_for_linear_regression(probe)
        # reverse order !!
        ordering = ordering[::-1]

        X_selected_n_train = ablation.filter_activations_keep_neurons(X_train, ordering[:n])
        X_selected_n_test = ablation.filter_activations_keep_neurons(X_test, ordering[:n])

        probe_selected_n = linear_probe.LinearProbe(X_selected_n_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(1):
            probe_selected_n = linear_probe.train_linear_regression_probe(
                X_selected_n_train, y_train, lambda_l1=0., lambda_l2=0., num_epochs=1, probe=probe_selected_n)
        
            scores_train = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_selected_n, X_selected_n_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        torch.save(
            probe_selected_n.state_dict(), 
            os.path.join(log_dir, 'models', f'probe_selected_bottom_{n}_noreg.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'probe_selected_bottom_{n}_noreg.npz'), **scores)




##################################
# From this ... MLP.
##################################


def mlpprobe():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # probe = linear_probe.LinearProbe(X_train.shape[1], 1)
    probe = MlpProbe(X_train.shape[1], 1)
    
    scores = {'train': [], 'test': []}
    for epoch in range(3):
        probe = linear_probe.train_linear_regression_probe(
            X_train, y_train, lambda_l1=0.001, lambda_l2=0.001, num_epochs=1, probe=probe)
    
        scores_train = linear_probe.evaluate_probe(probe, X_train, y_train, metric='mse')
        print( '[* train score]', scores_train)
        scores_test = linear_probe.evaluate_probe(probe, X_test, y_test, metric='mse')
        print( '[* test score]', scores_test)

        scores['train'].append(scores_train['__OVERALL__'])
        scores['test'].append(scores_test['__OVERALL__'])

    torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'mlpprobe.pth'))
    np.savez(os.path.join(log_dir, 'scores', 'mlpprobe.npz'), **scores)


def mlpprobe_noreg():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # probe = linear_probe.LinearProbe(X_train.shape[1], 1)
    probe = MlpProbe(X_train.shape[1], 1)
    
    scores = {'train': [], 'test': []}
    for epoch in range(1):
        probe = linear_probe.train_linear_regression_probe(
            X_train, y_train, lambda_l1=0., lambda_l2=0., num_epochs=1, probe=probe)
    
        scores_train = linear_probe.evaluate_probe(probe, X_train, y_train, metric='mse')
        print( '[* train score]', scores_train)
        scores_test = linear_probe.evaluate_probe(probe, X_test, y_test, metric='mse')
        print( '[* test score]', scores_test)

        scores['train'].append(scores_train['__OVERALL__'])
        scores['test'].append(scores_test['__OVERALL__'])

    torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'mlpprobe_noreg.pth'))
    np.savez(os.path.join(log_dir, 'scores', 'mlpprobe_noreg.npz'), **scores)


def mlpprobe_layerwise():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # data_dict = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    # X = data_dict['X']
    # y = data_dict['y']
    # label2idx = data_dict['label2idx'].item()
    # idx2label = data_dict['idx2label'].item()
    # src2idx = data_dict['src2idx'].item()
    # idx2src = data_dict['idx2src'].item()
    # neuronidx2name = data_dict['neuronidx2name'].item()
    
    for k in [0, 1, 2, 3]:
        layer_k_X_train = ablation.filter_activations_by_layers(X_train, [k], 4)
        layer_k_X_test = ablation.filter_activations_by_layers(X_test, [k], 4)

        probe_layer_k = MlpProbe(layer_k_X_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(3):
            probe_layer_k = linear_probe.train_linear_regression_probe(
                layer_k_X_train, y_train, lambda_l1=0.001, lambda_l2=0.001, num_epochs=1, probe=probe_layer_k)
        
            scores_train = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        # torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'probe.pth'))
        torch.save(
            probe_layer_k.state_dict(), 
            os.path.join(log_dir, 'models', f'mlpprobe_layer_{k}.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'mlpprobe_layer_{k}.npz'), **scores)


def mlpprobe_layerwise_noreg():
    data_dict_train = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    data_dict_test = np.load(os.path.join(log_dir, 'data', 'test_data.npz'), allow_pickle=True)
    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # ignore zero duration
    X_train = X_train[y_train > 0]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0]
    y_test = y_test[y_test > 0]

    # we predict log duration
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # data_dict = np.load(os.path.join(log_dir, 'data', 'train_data.npz'), allow_pickle=True)
    # X = data_dict['X']
    # y = data_dict['y']
    # label2idx = data_dict['label2idx'].item()
    # idx2label = data_dict['idx2label'].item()
    # src2idx = data_dict['src2idx'].item()
    # idx2src = data_dict['idx2src'].item()
    # neuronidx2name = data_dict['neuronidx2name'].item()
    
    for k in [0, 1, 2, 3]:
        layer_k_X_train = ablation.filter_activations_by_layers(X_train, [k], 4)
        layer_k_X_test = ablation.filter_activations_by_layers(X_test, [k], 4)

        probe_layer_k = MlpProbe(layer_k_X_train.shape[1], 1)

        scores = {'train': [], 'test': []}
        for epoch in range(1):
            probe_layer_k = linear_probe.train_linear_regression_probe(
                layer_k_X_train, y_train, lambda_l1=0., lambda_l2=0., num_epochs=1, probe=probe_layer_k)
        
            scores_train = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_train, y_train, metric='mse')
            print( '[* train score]', scores_train)
            scores_test = linear_probe.evaluate_probe(probe_layer_k, layer_k_X_test, y_test, metric='mse')
            print( '[* test score]', scores_test)

            scores['train'].append(scores_train['__OVERALL__'])
            scores['test'].append(scores_test['__OVERALL__'])

        # torch.save(probe.state_dict(), os.path.join(log_dir, 'models', 'probe.pth'))
        torch.save(
            probe_layer_k.state_dict(), 
            os.path.join(log_dir, 'models', f'mlpprobe_layer_{k}_noreg.pth'))
        # np.savez(os.path.join(log_dir, 'scores', 'probe.npz'), **scores)
        np.savez(os.path.join(log_dir, 'scores', f'mlpprobe_layer_{k}_noreg.npz'), **scores)




def main():
    # prepare_data('train')
    # prepare_data('test')
    # probe()
    # probe_layerwise()
    # probe_selected()
    # probe_selected_bottom()

    # probe_noreg()
    # probe_layerwise_noreg()
    # probe_selected_noreg()
    probe_selected_bottom_noreg()


    """
    mlp
    """
    # mlpprobe()
    # mlpprobe_noreg()
    # mlpprobe_layerwise()
    # mlpprobe_layerwise_noreg()

if __name__ == "__main__":
    main()