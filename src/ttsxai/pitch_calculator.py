# modify code from: https://github.com/DigitalPhonetics/IMS-Toucan/blob/fc1c940012e21bbc18e194c1b4c43ceb6c1b7caf/TrainingInterfaces/Text_to_Spectrogram/FastSpeech2/PitchCalculator.py#L16
import math

import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d


class PitchCalculatorWrapper(object):
    def __init__(self, sampling_rate, hop_length):

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        self.parsel = Parselmouth(
            fs=sampling_rate, hop_length=hop_length, use_continuous_f0=False)

    def __call__(self, wave):
        if isinstance(wave, np.ndarray):
            # Convert numpy array to torch tensor
            wave = torch.from_numpy(wave)
        pitch = self.parsel._calculate_f0(wave)
        
        num_frames = int(wave.shape[0] / self.hop_length)
        pitch = self.parsel._adjust_num_frames(pitch, num_frames)
        pitch = pitch.cpu().numpy()

        # perform linear interpolation
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))
        return pitch

    def average_by_duration(self, pitch, duration):
        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                pitch[i] = np.mean(pitch[pos : pos + d])
            else:
                pitch[i] = 0
            pos += d
        pitch = pitch[: len(duration)]
        return pitch


class Parselmouth(torch.nn.Module):
    """
    F0 estimation with Parselmouth https://parselmouth.readthedocs.io/en/stable/index.html
    """

    def __init__(self, fs=16000, n_fft=1024, hop_length=256, f0min=40, f0max=600, use_token_averaged_f0=True,
                 use_continuous_f0=False, use_log_f0=False, reduction_factor=1):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_period = 1000 * hop_length / fs
        self.f0min = f0min
        self.f0max = f0max
        self.use_token_averaged_f0 = use_token_averaged_f0
        self.use_continuous_f0 = use_continuous_f0
        self.use_log_f0 = use_log_f0
        if use_token_averaged_f0:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

    def output_size(self):
        return 1

    def get_parameters(self):
        return dict(fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, f0min=self.f0min, f0max=self.f0max,
                    use_token_averaged_f0=self.use_token_averaged_f0, use_continuous_f0=self.use_continuous_f0, use_log_f0=self.use_log_f0,
                    reduction_factor=self.reduction_factor)

    def forward(self, input_waves, input_waves_lengths=None, feats_lengths=None, durations=None,
                durations_lengths=None, norm_by_average=True, text=None):

        # F0 extraction
        pitch = self._calculate_f0(input_waves[0])

        # Adjust length to match with the mel-spectrogram
        pitch = self._adjust_num_frames(pitch, feats_lengths[0]).view(-1)

        pitch = self._average_by_duration(pitch, durations[0], text).view(-1)
        pitch_lengths = durations_lengths

        if norm_by_average:
            average = pitch[pitch != 0.0].mean()
            pitch = pitch / average

        # Return with the shape (B, T, 1)
        return pitch.unsqueeze(-1), pitch_lengths

    def _calculate_f0(self, input):
        x = input.cpu().numpy().astype(np.double)
        snd = parselmouth.Sound(values=x, sampling_frequency=self.fs)
        f0 = snd.to_pitch(time_step=self.hop_length / self.fs, pitch_floor=self.f0min, pitch_ceiling=self.f0max).selected_array['frequency']
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return input.new_tensor(f0.reshape(-1), dtype=torch.float)

    @staticmethod
    def _adjust_num_frames(x, num_frames):
        if num_frames > len(x):
            # x = F.pad(x, (math.ceil((num_frames - len(x)) / 2), math.floor((num_frames - len(x)) / 2)))

            # Calculate padding lengths
            left_pad = math.ceil((num_frames - len(x)) / 2)
            right_pad = math.floor((num_frames - len(x)) / 2)

            # Get the value for front padding from the first element of x
            front_value = x[0].item()
            # Get the value for back padding from the last element of x
            back_value = x[-1].item()

            # Pad the front with the front_value
            x = F.pad(x, (left_pad, 0), value=front_value)
            # Pad the back with the back_value
            x = F.pad(x, (0, right_pad), value=back_value)
        elif num_frames < len(x):
            x = x[:num_frames]
        return x

