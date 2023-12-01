import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method

from ttsxai.interfaces.prosody_interface import ProsodyInterface
from ttsxai.interfaces.tts_interface import get_text2mel, get_mel2wave

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def process_files(args, sub_files):
    text2mel_local = get_text2mel(args.text2mel_type, args.device)

    prosody = ProsodyInterface(text2mel_local.sampling_rate, text2mel_local.hop_length)

    for file in tqdm(sub_files, desc=f"Processing in Process-{os.getpid()}", position=0, leave=True):
        data_dict = np.load(file, allow_pickle=True)
        wave = data_dict['wave']
        duration = data_dict['duration']
        prosody_dict = prosody(wave, duration)

        pitch = prosody_dict['pitch']
        energy = prosody_dict['energy']

        assert len(duration) == len(pitch) == len(energy)

        # Update the original data_dict with the new data
        data_dict = dict(data_dict)
        data_dict['pitch'] = pitch
        data_dict['energy'] = energy

        # Save the updated data_dict back to the same npz file
        np.savez(file, **data_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_activation_dir", default="/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech/tacotron2_waveglow", 
                        help="Directory containing activation data (.npz files)")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for parallel processing")
    args = parser.parse_args()

    args.text2mel_type, _ = os.path.basename(args.data_activation_dir).split('_')
    # cpu: We will not use tts in this script.
    args.device = torch.device('cpu') 

    npz_files = glob.glob(os.path.join(args.data_activation_dir, '*.npz'))

    # Create chunks of the file list for each process
    chunk_size = len(npz_files) // args.num_processes
    chunks = [npz_files[i:i + chunk_size] for i in range(0, len(npz_files), chunk_size)]

    with Pool(processes=args.num_processes) as pool:
        pool.starmap(process_files, [(args, chunk) for chunk in chunks])