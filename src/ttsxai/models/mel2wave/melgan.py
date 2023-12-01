import os
import json
import numpy as np

import torch
import torch.nn as nn

import fastspeech2
from fastspeech2 import hifigan


class MelGANWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        fastspeech2_dir = os.path.dirname(fastspeech2.__file__)
        with open(os.path.join(fastspeech2_dir, "hifigan/config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
                
        self.device = device
        self.model = torch.hub.load(
            "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
        )
        self.model.mel2wav.eval()
        self.model.mel2wav.to(device)

    @torch.no_grad()
    def forward(self, mel):
        mel = torch.tensor(mel[None, :]).float().to(self.device)
        waves = self.model.inverse(mel / np.log(10))
        return waves[0].cpu().numpy() 


