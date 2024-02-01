import os
import re
import glob
import numpy as np
import argparse
import cloudpickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.multiprocessing import Pool, set_start_method

from ttsxai.utils.utils import read_ljs_metadata


try:
    set_start_method('spawn')
except RuntimeError:
    pass


def activations2array(activations):
    if type(activations) == dict:
        return np.concatenate(list(activations.values()), axis=1)
    elif type(activations) == np.ndarray:
        return activations
    else:
        raise NotImplementedError
    

def process_pkl_file(pkl_file, word_data_queue):
    with open(pkl_file, "rb") as file:
        data_dict = cloudpickle.load(file)

    text = data_dict['text']
    # wave = data_dict['wave']
    phonesymbols = data_dict['phonesymbols']
    activations = activations2array(data_dict['activations'])
    duration = data_dict['duration']
    pitch = data_dict['pitch']
    energy = data_dict['energy']

    words = []
    for w in re.split(r"([,;.\-\?\!\s+])", text.replace('.', '')):
        if w.lower() not in ['.', ',', '”','“', '"', "'", ';', '-', '(', ')', ':', '?', '!', ' ', '']:
            words.append(w.lower())

    word_index_boundaries = []

    s = 0 
    for index, phone in enumerate(phonesymbols):
        if phone == " ":
            word_index_boundaries.append([s, e])
            s = index + 1
        elif phone not in ['.', ',', '”', '“', '"', "'", ';', '-', '(', ')', ':', '?', '!', ' ', '']:
            e = index
    word_index_boundaries.append([s, e])

    assert(len(words) == len(word_index_boundaries))

    # for idx, word_index_boundary in enumerate(word_index_boundaries):
    #     s, e = word_index_boundary
    #     print(words[idx], s, e, phonesymbols[s:e+1])

    local_word_data = {}

    for idx, word_index_boundary in enumerate(word_index_boundaries):
        # For tacotron2, ignore alignment error case which has num_frames = 1000
        if sum(duration) == 1000:
            print('ignore sum(duration) == 1000', pkl_file)
            continue

        s, e = word_index_boundary

        if words[idx] == 'have':
            print(words[idx], s, e, phonesymbols[s:e+1])
            print(pkl_file)
            print('-----')

        if words[idx] not in local_word_data:
            local_word_data[words[idx]] = {
                'duration': [], 'pitch': [], 'energy': [], 
                'activations': [], 'word_index_boundary': [], 'pkl_file': []
            }
        local_word_data[words[idx]]['duration'].append(np.sum(duration[s:e+1]))
        local_word_data[words[idx]]['pitch'].append(np.mean(pitch[s:e+1]))
        local_word_data[words[idx]]['energy'].append(np.mean(energy[s:e+1]))
        local_word_data[words[idx]]['activations'].append(activations[s:e+1, :])
        local_word_data[words[idx]]['word_index_boundary'].append([s, e])
        local_word_data[words[idx]]['pkl_file'].append(pkl_file)

    word_data_queue.put(local_word_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_activation_dir", 
                        default="/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow", 
                        help="Directory containing the pkl activation files")
    parser.add_argument("--data_analysis_dir", 
                        default="/nas/users/dahye/kw/tts/ttsxai/data_analysis/analyze_word_prosody", 
                        help="Directory to save the analyzed word data")
    parser.add_argument("--num_processes", type=int, default=4, 
                        help="Number of processes for parallel processing")
    args = parser.parse_args()

    if not os.path.exists(args.data_analysis_dir):
        os.makedirs(args.data_analysis_dir)

    # Dictionary keys to filter
    keys_to_filter = read_ljs_metadata(mode='train').keys()

    # List to store filtered paths
    pkl_files = []

    # Iterate over all files in the directory
    for file in os.listdir(args.data_activation_dir):
        # Check only for .pkl files
        if file.endswith('.pkl'):
            # Extract the identifier part from the file name (e.g., 'LJ037-0213')
            identifier = file.split('.')[0]

            # If this identifier is included in the dictionary keys, add to the list
            if identifier in keys_to_filter:
                full_path = os.path.join(args.data_activation_dir, file)
                pkl_files.append(full_path)

    # debug
    # pkl_files = pkl_files[:1]
                
    from multiprocessing import Manager
    manager = Manager()
    word_data_queue = manager.Queue()

    # Parallel processing to compute word_data
    with Pool(processes=args.num_processes) as pool:
        _ = pool.starmap(process_pkl_file, tqdm([(f, word_data_queue) for f in pkl_files], 
                         total=len(pkl_files), 
                         desc="Processing pkl files"))

    # Aggregate results from all processes
    word_data = {}
    while not word_data_queue.empty():
        local_word_data = word_data_queue.get()
        for t, data in local_word_data.items():
            if t not in word_data:
                word_data[t] = {
                    'duration': [], 'pitch': [], 'energy': [], 
                    'activations': [], 'word_index_boundary': [], 'pkl_file': []
                }
            word_data[t]['duration'].extend(data['duration'])
            word_data[t]['pitch'].extend(data['pitch'])
            word_data[t]['energy'].extend(data['energy'])
            word_data[t]['activations'].extend(data['activations'])
            word_data[t]['word_index_boundary'].extend(data['word_index_boundary'])
            word_data[t]['pkl_file'].extend(data['pkl_file'])

    cloudpickle.dump(word_data, open(os.path.join(args.data_analysis_dir, 'word_data.pkl'), 'wb'))


if __name__ == "__main__":
    main()