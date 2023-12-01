import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch.multiprocessing import Pool, set_start_method

from ttsxai.articulatory_features import get_articulatory_features_for_phoneme

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def process_files(args, sub_files):
    for file in tqdm(sub_files, desc=f"Processing in Process-{os.getpid()}", position=0, leave=True):
        data_dict = np.load(file, allow_pickle=True)
        phonesymbols = data_dict['phonesymbols']
        
        articulatory_features = get_articulatory_features_for_phoneme(phonesymbols)

        # Update the original data_dict with the new data
        data_dict = dict(data_dict)
        data_dict['articulatory_features'] = articulatory_features

        # Save the updated data_dict back to the same npz file
        np.savez(file, **data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_activation_dir", default="/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow", 
                        help="Directory containing activation data (.npz files)")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for parallel processing")
    args = parser.parse_args()

    npz_files = glob.glob(os.path.join(args.data_activation_dir, '*.npz'))

    # Create chunks of the file list for each process
    chunk_size = len(npz_files) // args.num_processes
    chunks = [npz_files[i:i + chunk_size] for i in range(0, len(npz_files), chunk_size)]

    with Pool(processes=args.num_processes) as pool:
        pool.starmap(process_files, [(args, chunk) for chunk in chunks])


