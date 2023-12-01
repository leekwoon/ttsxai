from ttsxai.pitch_calculator import PitchCalculatorWrapper
from ttsxai.energy_calculator import EnergyCalculatorWrapper


class ProsodyInterface(object):
    def __init__(self, sampling_rate, hop_length):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        self.pitch_calculator = PitchCalculatorWrapper(sampling_rate, hop_length)
        self.energy_calculator = EnergyCalculatorWrapper(sampling_rate, hop_length)

    def __call__(self, wave, duration=None):
        pitch = self.pitch_calculator(wave)
        if duration is not None:
            pitch = self.pitch_calculator.average_by_duration(pitch, duration)
        energy = self.energy_calculator(wave)
        if duration is not None:
            energy = self.energy_calculator.average_by_duration(energy, duration)

        return {
            'pitch': pitch,
            'energy': energy
        }