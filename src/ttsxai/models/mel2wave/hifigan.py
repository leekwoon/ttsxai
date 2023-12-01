import os
import json
import yaml

import torch
import torch.nn as nn

import fastspeech2
from fastspeech2 import hifigan

from ttsxai import PRETRAINED_MODELS_DIR
from ttsxai.utils import utils


class HiFiGANWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()

        fastspeech2_dir = os.path.dirname(fastspeech2.__file__)
        with open(os.path.join(fastspeech2_dir, "hifigan/config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        
        ckpt_path = os.path.join(
            PRETRAINED_MODELS_DIR, "hifigan/generator_LJSpeech.pth.tar"
        )
        ckpt = torch.load(ckpt_path)

        self.device = device
        self.model = hifigan.Generator(config)
        self.model.load_state_dict(ckpt["generator"])
        self.model.eval()
        with utils.suppress_output():
            self.model.remove_weight_norm()
        self.model.to(device)

    @torch.no_grad()
    def forward(self, mel):
        mel = torch.tensor(mel[None, :]).float().to(self.device)
        waves = self.model(mel)
        return waves[0, 0].cpu().numpy() 


