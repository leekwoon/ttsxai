import os
import argparse
import torch
import numpy as np
import cloudpickle
from tqdm import tqdm
from scipy.io import wavfile

from torch.multiprocessing import Pool, set_start_method

# from ttsxai.utils.utils import read_ljs_metafile
from ttsxai.utils.utils import read_ljs_metadata
from ttsxai.interfaces.prosody_interface import ProsodyInterface
from ttsxai.interfaces.tts_interface import TTSInterface, get_text2mel, get_mel2wave


try:
    set_start_method('spawn')
except RuntimeError:
    pass


def process_texts(args, sub_dict):
    text2mel_local = get_text2mel(args.text2mel_type, args.device)
    mel2wave_local = get_mel2wave(args.mel2wave_type, args.device)

    tts_local = TTSInterface(
        device=args.device,
        text2mel=text2mel_local,
        mel2wave=mel2wave_local 
    ).to(args.device)

    prosody_local = ProsodyInterface(text2mel_local.sampling_rate, text2mel_local.hop_length)

    for file_id, text in tqdm(sub_dict.items(), desc=f"Processing in Process-{os.getpid()}", position=0, leave=True):
        tts_dict = tts_local.forward(text)

        # add prosody information
        prosody_dict = prosody_local(tts_dict['wave'], tts_dict['duration'])
        assert len(tts_dict['duration']) == len(prosody_dict['pitch']) == len(prosody_dict['energy'])
        tts_dict['pitch'] = prosody_dict['pitch']
        tts_dict['energy'] = prosody_dict['energy']

        wavfile.write(
            os.path.join(args.save_dir, f"{file_id}.wav"),
            tts_local.sampling_rate,
            tts_dict['wave']
        )
        cloudpickle.dump(tts_dict, open(os.path.join(args.save_dir, f"{file_id}.pkl"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/nas/users/dahye/kw/tts/ttsxai/data/LJSpeech", help="Directory of the metadata.csv file")
    parser.add_argument("--save_dir", default="/nas/users/dahye/kw/tts/ttsxai/data_activation/LJSpeech", help="Directory to save the activation data")
    parser.add_argument("--text2mel_type", default="tacotron2", help="Type of text2mel model")
    parser.add_argument("--mel2wave_type", default="waveglow", help="Type of mel2wave model")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes for parallel processing")
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, f'{args.text2mel_type}_{args.mel2wave_type}')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ljs_metadata_train = read_ljs_metadata(mode='train')
    
    # Create chunks of the dictionary for each process
    chunk_size = len(ljs_metadata_train) // args.num_processes
    chunks = [dict(list(ljs_metadata_train.items())[i:i + chunk_size]) for i in range(0, len(ljs_metadata_train), chunk_size)]

    with Pool(processes=args.num_processes) as pool:
        pool.starmap(process_texts, [(args, chunk) for chunk in chunks])