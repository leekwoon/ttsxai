import os
import re
import numpy as np
from g2p_en import G2p

import torch
import torch.nn as nn
from torch.nn import functional as F

from tacotron2.text import text_to_sequence, _id_to_symbol
from tacotron2.train import load_model
from tacotron2.hparams import create_hparams
from flowtron.text import HETERONYMS, _apostrophe
from flowtron.text.acronyms import cmudict

from ttsxai import PRETRAINED_MODELS_DIR


g2p = G2p()


class Tacotron2Wrapper(nn.Module):
    def __init__(self, device):
        super().__init__()

        hparams = create_hparams()

        ckpt_path = os.path.join(
            PRETRAINED_MODELS_DIR, "tacotron2/tacotron2_ljs_statedict.pt"
        )

        self.device = device
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length

        self.model = load_model(hparams)
        self.model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.model.to(device).eval()
        self.duration_calculator = DurationCalculator()

        self.hooks = []  # List to store hook references

        self.modified_activations = {}  # Dictionary to store modified activations

    def set_modified_activations(self, modified_activations):
        """Set modified activations to use in forward pass."""
        self.modified_activations = modified_activations

    @torch.no_grad()
    def forward(self, text, alignment=None):
        activations = {}  # Dictionary to store activations

        phone = text2phone(text)
        token = phone2token(phone)
        tokens = torch.tensor(token[None]).long().to(self.device)

        # === inference start ===
        embedded_inputs = self.model.embedding(tokens).transpose(1, 2)
        ####### encoder_outputs = self.encoder.inference(embedded_inputs)
        x = embedded_inputs
        for i, conv in enumerate(self.model.encoder.convolutions):
            if f'conv_{i}' in self.modified_activations.keys():
                # print(i, 'hihi')
                x = self.modified_activations[f'conv_{i}'].transpose(1, 2)
            else:
                x = F.dropout(F.relu(conv(x)), 0.5, self.model.encoder.training)
                activations[f'conv_{i}'] = x.transpose(1, 2)

            # print(x.shape)
            # print('no')            

        x = x.transpose(1, 2)

        self.model.encoder.lstm.flatten_parameters()
        outputs, _ = self.model.encoder.lstm(x)
        if 'lstm' in self.modified_activations.keys():
            print('### modify lstm')
            outputs = self.modified_activations['lstm']

        activations['lstm'] = outputs
        # print(outputs)

        encoder_outputs = outputs
        print('encoder_outputs.shape=', encoder_outputs.shape)
        # encoder_outputs_modified = torch.cat(
        #     (encoder_outputs[:, :6, :], encoder_outputs[:, 7:, :]), dim=1
        # )
        # encoder_outputs = encoder_outputs_modified
        # print('encoder_outputs.shape=', encoder_outputs.shape)

        if alignment is None:
            mel_outputs, gate_outputs, alignments = self.model.decoder.inference(
                encoder_outputs)
        else:
            rhythm = torch.tensor(alignment[None]).to(self.device)
            rhythm = rhythm.permute(1, 0, 2)
            mel_outputs, gate_outputs, alignments = self.model.decoder.inference_noattention(
                encoder_outputs, rhythm)       

        mel_outputs_postnet = self.model.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.model.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        _, mel_outputs_postnet, _, alignments = outputs
        # _, mel_outputs_postnet, _, alignments = self.model.inference(tokens)
        # === inference end ===

        mel = mel_outputs_postnet[0].cpu().numpy() 
        print(mel.shape)
        duration, _ = self.duration_calculator(alignments[0])
        duration = duration.cpu().numpy()
        assert np.sum(duration) == mel.shape[1]
        
        info = {
            'text': text,
            'phone': phone,
            'phonesymbols': token2phonesymbols(token),
            'token': token,
            'duration': duration ,
            'alignment': alignments[0].cpu().numpy() 
        }
        for k, v in activations.items():
            activations[k] = v[0].cpu().numpy()
        info['activations'] = activations.copy()

        return mel, info


# code from: https://github.com/espnet/espnet/blob/a4e69bb7772c87af9ec4e7930d6551970d88fd0f/espnet2/tts/utils/duration_calculator.py#L13
class DurationCalculator(torch.nn.Module):
    """Duration calculator module."""

    @torch.no_grad()
    def forward(self, att_ws):
        """Convert attention weight to durations.

        Args:
            att_ws (Tesnor): Attention weight tensor (T_feats, T_text) or
                (#layers, #heads, T_feats, T_text).

        Returns:
            LongTensor: Duration of each input (T_text,).
            Tensor: Focus rate value.

        """
        duration = self._calculate_duration(att_ws)
        focus_rate = self._calculate_focus_rete(att_ws)

        return duration, focus_rate

    @staticmethod
    def _calculate_focus_rete(att_ws):
        if len(att_ws.shape) == 2:
            # tacotron 2 case -> (T_feats, T_text)
            return att_ws.max(dim=-1)[0].mean()
        elif len(att_ws.shape) == 4:
            # transformer case -> (#layers, #heads, T_feats, T_text)
            return att_ws.max(dim=-1)[0].mean(dim=-1).max()
        else:
            raise ValueError("att_ws should be 2 or 4 dimensional tensor.")

    @staticmethod
    def _calculate_duration(att_ws):
        if len(att_ws.shape) == 2:
            # tacotron 2 case -> (T_feats, T_text)
            pass
        elif len(att_ws.shape) == 4:
            # transformer case -> (#layers, #heads, T_feats, T_text)
            # get the most diagonal head according to focus rate
            att_ws = torch.cat(
                [att_w for att_w in att_ws], dim=0
            )  # (#heads * #layers, T_feats, T_text)
            diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1)  # (#heads * #layers,)
            diagonal_head_idx = diagonal_scores.argmax()
            att_ws = att_ws[diagonal_head_idx]  # (T_feats, T_text)
        else:
            raise ValueError("att_ws should be 2 or 4 dimensional tensor.")
        # calculate duration from 2d attention weight
        durations = torch.stack(
            [att_ws.argmax(-1).eq(i).sum() for i in range(att_ws.shape[1])]
        )
        return durations.view(-1)
    

def get_arpabet(word, cmudict, index=0):
    re_start_punc = r"\A\W+"
    re_end_punc = r"\W+\Z"

    start_symbols = re.findall(re_start_punc, word)
    if len(start_symbols):
        start_symbols = start_symbols[0]
        word = word[len(start_symbols):]
    else:
        start_symbols = ''

    end_symbols = re.findall(re_end_punc, word)
    if len(end_symbols):
        end_symbols = end_symbols[0]
        word = word[:-len(end_symbols)]
    else:
        end_symbols = ''

    arpabet_suffix = ''
    if _apostrophe.match(word) is not None and word.lower() != "it's" and word.lower()[-1] == 's':
        word = word[:-2]
        arpabet_suffix = ' Z'
    arpabet = None if word.lower() in HETERONYMS else cmudict.lookup(word)

    if arpabet is not None:
        return start_symbols + '{%s}' % (arpabet[index] + arpabet_suffix) + end_symbols
    else:
        return start_symbols + '{%s}' % (' '.join(g2p(word))) + end_symbols
    

def text2phone(text):
    text = text.replace('-', ' ')
    words = re.findall(r'\S*\{.*?\}\S*|\S+', text)
    phone = ' '.join([get_arpabet(word, cmudict) for word in words]) 
    return phone


def phone2token(phone):
    token = text_to_sequence(phone, cleaner_names=["english_cleaners"])
    return np.array(token)


def text2token(text):
    token = text_to_sequence(text2phone(text), cleaner_names=["english_cleaners"])
    return np.array(token)


def token2phonesymbols(token):
    return [_id_to_symbol[t].replace('@', '') for t in token]