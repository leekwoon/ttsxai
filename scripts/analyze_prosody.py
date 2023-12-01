import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.multiprocessing import Pool, set_start_method


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
    

def process_npz_file(npz_file, phone_data_queue):
    data_dict = np.load(npz_file, allow_pickle=True)
    token = data_dict['token']
    duration = data_dict['duration']
    pitch = data_dict['pitch']
    energy = data_dict['energy']
    activations = activations2array(data_dict['activations'].item())
    
    local_phone_data = {}

    for t, d, p, e in zip(token, duration, pitch, energy):
        # For tacotron2, ignore alignment error case which has num_frames = 1000
        if sum(duration) == 1000:
            continue

        # Convert the key to a string for npz serialization
        t_str = str(t)
        if t_str not in local_phone_data:
            local_phone_data[t_str] = {'duration': [], 'pitch': [], 'energy': [], 'activations': [], 'npz_file': []}
        local_phone_data[t_str]['duration'].append(d)
        local_phone_data[t_str]['pitch'].append(p)
        local_phone_data[t_str]['energy'].append(e)
        local_phone_data[t_str]['activations'].append(a)
        local_phone_data[t_str]['npz_file'].append(npz_file)

    phone_data_queue.put(local_phone_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_activation_dir", 
                        default="/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow", 
                        help="Directory containing the npz activation files")
    parser.add_argument("--data_analysis_dir", 
                        default="/nas/users/dahye/kw/tts/ttsxai/data_analysis/analyze_prosody", 
                        help="Directory to save the analyzed phone data")
    parser.add_argument("--num_processes", type=int, default=4, 
                        help="Number of processes for parallel processing")
    args = parser.parse_args()

    if not os.path.exists(args.data_analysis_dir):
        os.makedirs(args.data_analysis_dir)

    npz_files = glob.glob(os.path.join(args.data_activation_dir, '*.npz'))
    from multiprocessing import Manager
    manager = Manager()
    phone_data_queue = manager.Queue()

    # Parallel processing to compute phone_data
    with Pool(processes=args.num_processes) as pool:
        _ = pool.starmap(process_npz_file, tqdm([(f, phone_data_queue) for f in npz_files], 
                         total=len(npz_files), 
                         desc="Processing npz files"))

    # Aggregate results from all processes
    phone_data = {}
    while not phone_data_queue.empty():
        local_phone_data = phone_data_queue.get()
        for t, data in local_phone_data.items():
            if t not in phone_data:
                phone_data[t] = {'duration': [], 'pitch': [], 'energy': [], 'activations': [], 'npz_file': []}
            phone_data[t]['duration'].extend(data['duration'])
            phone_data[t]['pitch'].extend(data['pitch'])
            phone_data[t]['energy'].extend(data['energy'])
            phone_data[t]['activations'].extend(data['activations'])
            phone_data[t]['npz_file'].extend(data['npz_file'])

    with open(os.path.join(args.data_analysis_dir, 'phone_data.npz'), 'wb') as file:
        np.savez(file, **phone_data)


if __name__ == "__main__":
    main()