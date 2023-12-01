import os
import re
import yaml
import numpy as np
from g2p_en import G2p
from string import punctuation

import torch
import torch.nn as nn

import fastspeech2
from fastspeech2.model import FastSpeech2
from fastspeech2.text import text_to_sequence, _id_to_symbol

from ttsxai import PACKAGE_DIR, PRETRAINED_MODELS_DIR


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phone = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phone
    return lexicon


lexicon = read_lexicon(os.path.join(
    PACKAGE_DIR, 'src', 'fastspeech2',
    'lexicon/librispeech-lexicon.txt'
))
g2p = G2p()


class FastSpeech2Wrapper(nn.Module):
    """Module for inference
    """
    # def __init__(self, device):
    def __init__(self, device):
        super().__init__()

        fastspeech2_dir = os.path.dirname(fastspeech2.__file__)
        preprocess_config_path = os.path.join(
            fastspeech2_dir, 'config/LJSpeech/preprocess.yaml'
        )
        preprocess_config = yaml.load(
            open(preprocess_config_path, "r"), Loader=yaml.FullLoader
        )
        preprocess_config['path']['preprocessed_path'] = os.path.join(
            fastspeech2_dir, preprocess_config['path']['preprocessed_path']
        )

        model_config_path = os.path.join(
            fastspeech2_dir, 'config/LJSpeech/model.yaml'
        )
        model_config = yaml.load(
            open(model_config_path, "r"), Loader=yaml.FullLoader
        )

        train_config_path = os.path.join(
            fastspeech2_dir,
            'config/LJSpeech/train.yaml'
        )
        train_config = yaml.load(
            open(train_config_path, "r"), Loader=yaml.FullLoader
        )

        ckpt_path = os.path.join(
            PRETRAINED_MODELS_DIR,
            'fastspeech2',
            train_config["path"]["ckpt_path"],
            "900000.pth.tar",
        )
        ckpt = torch.load(ckpt_path)

        self.device = device
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

        self.model = FastSpeech2(preprocess_config, model_config)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.requires_grad_ = False

        self.activations = {}  # Dictionary to store activations
        self.hooks = []  # List to store hook references

        # Add hooks for each FFTBlock in encoder
        for idx, block in enumerate(self.model.encoder.layer_stack):
            hook = block.register_forward_hook(self._save_activation(f'fftblock_{idx}'))
            self.hooks.append(hook)

    def _save_activation(self, name):
        def hook(module, input, output):
            # You might want to choose what exactly you want to save from output.
            # Here, I'm assuming you want to save the output tensor directly.
            self.activations[name] = output[0] # save only enc_output
        return hook

    @torch.no_grad()
    def forward(
        self, 
        text, 
        pitch_control=1.0,
        energy_control=1.0,
        duration_control=1.0
    ):
        phone = text2phone(text)
        token = phone2token(phone)

        # inference with batch_size = 1
        tokens = torch.tensor(token[None]).long().to(self.device)
        # We treat only 1 speaker in LJ Speech datasets
        speakers = torch.tensor(np.array([0])).long().to(self.device)
        src_lens = torch.tensor(np.array([len(token)])).long().to(self.device)
        max_src_len = torch.max(src_lens).item()

        output = self.model(
            speakers, tokens, src_lens, max_src_len,
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control
        )

        mel = output[1][0].transpose(0, 1).cpu().numpy()
        log_duration = output[4][0].cpu().numpy()
        duration = np.round(np.exp(log_duration) - 1).astype(np.int64)
        assert np.sum(duration) == mel.shape[1]

        info = {
            'text': text,
            'phone': phone,
            'phonesymbols': token2phonesymbols(token),
            'token': token,
            'duration': duration,
        }
        for k, v in self.activations.items():
            self.activations[k] = v[0].cpu().numpy()
        info['activations'] = self.activations
        
        return mel, info


def text2phone(text):
    text = text.rstrip(punctuation)
    phone = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    debug = []
    for w in words:
        if w.lower() in lexicon:
            phone += lexicon[w.lower()]
            debug.append(lexicon[w.lower()])
        else:
            phone += list(filter(lambda p: p != " ", g2p(w)))
            debug.append(filter(lambda p: p != " ", g2p(w)))
    phone = "{" + "}{".join(phone) + "}"
    phone = re.sub(r"\{[^\w\s]?\}", "{sp}", phone)
    phone = phone.replace("}{", " ")
    return phone


def phone2token(phone):
    token = text_to_sequence(phone, ["english_cleaners"])
    return np.array(token) 


def text2token(text):
    token = text_to_sequence(text2phone(text), ["english_cleaners"])
    return np.array(token) 


def token2phonesymbols(token):
    return [_id_to_symbol[t].replace('@', '') for t in token]