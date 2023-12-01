import os

import torch
import torch.nn as nn

from waveglow.glow import WaveGlow
from waveglow.denoiser import Denoiser

from ttsxai import PRETRAINED_MODELS_DIR


class WaveGlowWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        ckpt_path = os.path.join(
            PRETRAINED_MODELS_DIR, "waveglow/waveglow_statedict.pt"
        )

        self.device = device
        self.model = WaveGlow(
            n_mel_channels=80, n_flows=12, n_group=8, n_early_every=4, 
            n_early_size=2, WN_config={"n_layers": 8, "n_channels": 256, "kernel_size": 3})
        self.model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.model.to(device).eval()

        self.denoiser = Denoiser(self.model).to(device).eval()

    @torch.no_grad()
    def forward(self, mel):
        mel = torch.tensor(mel[None, :]).float().to(self.device)
        waves = self.model.infer(mel, sigma=0.8)
        waves = self.denoiser(waves, 0.01)[:, 0]
        return waves[0].cpu().numpy() 


