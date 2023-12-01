import torch
import torch.nn as nn


class TTSInterface(nn.Module):
    def __init__(
        self,
        device,
        text2mel,
        mel2wave
    ):
        super().__init__()

        self.device = device
        self.text2mel = text2mel
        self.mel2wave = mel2wave

        self.sampling_rate = self.text2mel.sampling_rate
        self.hop_length = self.text2mel.hop_length

    @torch.no_grad()
    def forward(self, text):
        mel, text2mel_info = self.text2mel(text) # (freq_dim, time_dim)
        wave = self.mel2wave(mel) # (1, ...)
        length = mel.shape[1] * self.hop_length
        wave = wave[:length]

        output_dict = {
            'wave': wave,
            'mel': mel,
            **text2mel_info
        }

        return output_dict


def get_text2mel(text2mel_type, device):
    if text2mel_type == 'tacotron2':
        from ttsxai.models.text2mel.tacotron2 import Tacotron2Wrapper
        text2mel = Tacotron2Wrapper(device)
    elif text2mel_type == 'fastspeech2':
        from ttsxai.models.text2mel.fastspeech2 import FastSpeech2Wrapper
        text2mel = FastSpeech2Wrapper(device)
    else:
        raise NotImplementedError    
    return text2mel


def get_mel2wave(mel2wave_type, device):
    if mel2wave_type == 'waveglow':
        from ttsxai.models.mel2wave.waveglow import WaveGlowWrapper
        mel2wave = WaveGlowWrapper(device)
    elif mel2wave_type == 'hifigan':
        from ttsxai.models.mel2wave.hifigan import HiFiGANWrapper
        mel2wave = HiFiGANWrapper(device)
    elif mel2wave_type == 'melgan':
        from ttsxai.models.mel2wave.melgan import MelGANWrapper
        mel2wave = MelGANWrapper(device)
    else:
        raise NotImplementedError
    return mel2wave