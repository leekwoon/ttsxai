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

        self.activations = {}  # Dictionary to store activations
        self.hooks = []  # List to store hook references
        
        # Add hooks for each conv layer
        for idx, layer in enumerate(self.model.encoder.convolutions):
            hook = layer.register_forward_hook(self._save_activation(f'conv_{idx}'))
            self.hooks.append(hook)
        
        # Add hook for LSTM
        hook = self.model.encoder.lstm.register_forward_hook(self._save_activation('lstm'))
        self.hooks.append(hook)

        self.modified_activations = {}  # Dictionary to store modified activations

    def _save_activation(self, name):
        def hook(module, input, output):
            # if name in self.modified_activations:
            #     if name.startswith('conv'):  # For conv layers
            #         print(name, output.shape) # torch.Size([1, 512, 27])
            #         # return output
            #         return self.modified_activations[name]
            #     else:
            #         print(name, output[0].shape) # torch.Size([1, 27, 512])
            #         return output

            if name.startswith('conv'):  # For conv layers
                self.activations[name] = F.relu(output.transpose(1, 2))#.cpu().numpy()
                # Override with modified activations
                if name in self.modified_activations:
                    return self.modified_activations[name].transpose(1, 2)
            else:  # For LSTM
                # We're interested in the output tensor, not the hidden states
                self.activations[name] = output[0] # .cpu().numpy()
                if name in self.modified_activations:
                    return (self.modified_activations[name], output[1])
        return hook

    def set_modified_activations(self, modified_activations):
        """Set modified activations to use in forward pass."""
        self.modified_activations = modified_activations

    @torch.no_grad()
    def forward(self, text):
        phone = text2phone(text)
        token = phone2token(phone)
        tokens = torch.tensor(token[None]).long().to(self.device)

        # === inference start ===
        embedded_inputs = self.model.embedding(tokens).transpose(1, 2)
        ####### encoder_outputs = self.encoder.inference(embedded_inputs)
        x = embedded_inputs
        for conv in self.model.encoder.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.model.encoder.training)

        x = x.transpose(1, 2)

        self.model.encoder.lstm.flatten_parameters()
        outputs, _ = self.model.encoder.lstm(x)
        if self.modified_activations:
            outputs = self.modified_activations['lstm']
        encoder_outputs = outputs

        mel_outputs, gate_outputs, alignments = self.model.decoder.inference(
            encoder_outputs)

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
        for k, v in self.activations.items():
            self.activations[k] = v[0].cpu().numpy()
        info['activations'] = self.activations

        return mel, info


    # @torch.no_grad()
    # def forward(self, text):
    #     phone = text2phone(text)
    #     token = phone2token(phone)
    #     tokens = torch.tensor(token[None]).long().to(self.device)
    #     _, mel_outputs_postnet, _, alignments = self.model.inference(tokens)
    #     mel = mel_outputs_postnet[0].cpu().numpy() 
    #     duration, _ = self.duration_calculator(alignments[0])
    #     duration = duration.cpu().numpy()
    #     assert np.sum(duration) == mel.shape[1]
        
    #     info = {
    #         'text': text,
    #         'phone': phone,
    #         'phonesymbols': token2phonesymbols(token),
    #         'token': token,
    #         'duration': duration ,
    #         'alignment': alignments[0].cpu().numpy() 
    #     }
    #     for k, v in self.activations.items():
    #         self.activations[k] = v[0].cpu().numpy()
    #     info['activations'] = self.activations

    #     return mel, info


# def activations2array(activations):
#     if type(activations) == dict:
#         return np.concatenate(list(activations.values()), axis=1)
#     elif type(activations) == np.ndarray:
#         return activations
#     else:
#         raise NotImplementedError
    

# def activations2dict(activations, layer_shapes={'conv_0': 512, 'conv_1': 512, 'conv_2': 512, 'lstm': 512}):
#     if type(activations) == dict:
#         return activations
    
#     if activations.shape[1] != sum(layer_shapes.values()):
#         raise ValueError("Total size of layers does not match the array width.")
    
#     activations = {}
#     current_index = 0
#     for layer_name, shape in layer_shapes.items():
#         activations[layer_name] = activations[:, current_index:current_index + shape]
#         current_index += shape
    
#     return activations


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