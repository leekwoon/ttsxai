# For ground truth. (original waves files with gt alginment.)
import sys
sys.path.append('/nas/users/dahye/kw/tts/github_download/FastSpeech2')
from matplotlib import pyplot as plt

import os
import yaml
from tqdm import tqdm

import librosa
import librosa.display
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

from preprocessor.preprocessor import Preprocessor

from tacotron2.text import _symbol_to_id


config = yaml.load(open('/nas/users/dahye/kw/tts/github_download/FastSpeech2/config/LJSpeech/preprocess.yaml', "r"), Loader=yaml.FullLoader)

in_dir = config["path"]["corpus_path"]
out_dir = config["path"]["raw_path"]
sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
cleaners = config["preprocessing"]["text"]["text_cleaners"]
speaker = "LJSpeech"

preprocessor = Preprocessor(config)
self = preprocessor
self.in_dir = os.path.join('/nas/users/dahye/kw/tts/github_download/FastSpeech2', self.in_dir)
self.out_dir = os.path.join('/nas/users/dahye/kw/tts/github_download/FastSpeech2', self.out_dir)

print("Processing Data ...")
out = list()
n_frames = 0
pitch_scaler = StandardScaler()
energy_scaler = StandardScaler()

# Compute pitch, energy, duration, and mel-spectrogram
speakers = {}
phone_data_gt = {}
for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
    speakers[speaker] = i
    for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
        if ".wav" not in wav_name:
            continue

        basename = wav_name.split(".")[0]

        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        # if np.sum(pitch != 0) <= 1:
        #     return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        # NOTE: for debuging at frame level
        pitch_ori = pitch.copy()

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        for ph, d, p, e in zip(phone, duration, pitch, energy):
            if ph == "sp" or ph == "spn" or ph == "sil":
                continue

            if ph not in _symbol_to_id.keys():
                t = _symbol_to_id['@'+ph]
            else:
                t = _symbol_to_id[ph]
            # Convert the key to a string for npz serialization
            t_str = str(t)
            if t_str not in phone_data_gt:
                phone_data_gt[t_str] = {'duration': [], 'pitch': [], 'energy': []}
            phone_data_gt[t_str]['duration'].append(d)
            phone_data_gt[t_str]['pitch'].append(p)
            phone_data_gt[t_str]['energy'].append(e)

data_analysis_dir = '/nas/users/dahye/kw/tts/ttsxai/data_analysis/analyze_prosody'

if not os.path.exists(data_analysis_dir):
    os.makedirs(data_analysis_dir)

with open(os.path.join(data_analysis_dir, 'phone_data_gt.npz'), 'wb') as file:
    np.savez(file, **phone_data_gt)
