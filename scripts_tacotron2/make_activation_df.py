import os
import re
import argparse
import numpy as np
import pandas as pd
import cloudpickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import ttsxai
from ttsxai.utils.utils import read_ljs_metadata


def activations2array(activations):
    if type(activations) == dict:
        return np.concatenate(list(activations.values()), axis=1)
    elif type(activations) == np.ndarray:
        return activations
    else:
        raise NotImplementedError


def load_pkl_file(pkl_file):
    data_dict = np.load(pkl_file, allow_pickle=True)
    with open(pkl_file, "rb") as file:
        data_dict = cloudpickle.load(file)

    text = data_dict['text']
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

    data = {
        'phone': [], 'duration': [], 'pitch': [], 'energy': [], 
        'word': [], 'phone_loc': [], 'phone_index': [], 'activations': [], 'pkl_file': []
    }
    for idx, word_index_boundary in enumerate(word_index_boundaries):
        # For tacotron2, ignore alignment error case which has num_frames = 1000
        if sum(duration) == 1000:
            print('ignore sum(duration) == 1000', pkl_file)
            continue

        s, e = word_index_boundary

        for p_idx in range(s, e + 1):
            phone = phonesymbols[p_idx]
            data['phone'].append(phonesymbols[p_idx])
            data['duration'].append(duration[p_idx])
            data['pitch'].append(pitch[p_idx])
            data['energy'].append(energy[p_idx])
            data['activations'].append(activations[p_idx, :])
            data['phone_index'].append(p_idx)
            data['word'].append(words[idx])
            data['pkl_file'].append(pkl_file)

            if idx == 0:
                phone_loc = 'start'
            elif idx == len(word_index_boundaries) - 1:
                phone_loc = 'end'
            else:
                phone_loc = 'middle'
            data['phone_loc'].append(phone_loc)

    return data


def parallel_load_pkl_files(pkl_files, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(load_pkl_file, pkl_files), total=len(pkl_files)))

    all_data = {}
    for data in results:
        for key in data.keys():
            if key not in all_data:
                all_data[key] = []

            all_data[key].extend(data[key])
   
    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_activation_dir", 
                        default=os.path.join(
                            ttsxai.PACKAGE_DIR,
                            "data_activation/LJSpeech/tacotron2_waveglow"), 
                        help="Directory containing the pkl activation files")
    parser.add_argument("--save_dir", 
                        default=os.path.join(
                            ttsxai.PACKAGE_DIR, 
                            "data_df/LJSpeech/tacotron2_waveglow"), 
                        help="Directory to save the df")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for mode in ['train', 'test', 'val']:
        # Dictionary keys to filter
        keys_to_filter = read_ljs_metadata(mode=mode).keys()

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

        data = parallel_load_pkl_files(pkl_files)
        df = pd.DataFrame(data)
        # save
        df.to_pickle(os.path.join(args.save_dir, f'{mode}_activation_df.pkl'))


if __name__ == "__main__":
    main()