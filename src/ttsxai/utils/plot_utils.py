import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

# from ttsxai.constants import HOP_LENGTH


def plot_audio(audio, sr, ax=None):
    """
    Plots the waveform of an audio signal.

    Args:
        audio (np.array): 1D array containing the audio samples.
        sr (int): Sampling rate of the audio signal.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. 
                                             If None, a new figure and axis will be created. 
                                             Default is None.

    Usage:
        plot_audio(audio_data, 22050)
    """
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 3))
    librosa.display.waveshow(audio, ax=ax, sr=sr)


def plot_spectrogram(audio, sr, hop_length, ax=None, f0=None, 
                     spec_type='spectrogram', cmap='coolwarm', 
                     x_axis='time', y_axis='log'):
    """
    Plots the spectrogram or mel-spectrogram of the given audio.
    
    Args:
        audio (np.array): Audio data.
        sr (int): Sampling rate of the audio.
        f0 (np.array): Pitch contour.
        ax (matplotlib.axes.Axes, optional): Axes on which to plot. 
            Creates a new one if None.
        spec_type (str, optional): Type of spectrogram ('spectrogram' or 'mel_spectrogram').
        cmap (str, optional): Color map to use for the spectrogram.
        x_axis (str, optional): X-axis type.
        y_axis (str, optional): Y-axis type.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    else:
        fig = ax.figure

    audio = audio.copy()[:-1] # hack

    if spec_type == 'spectrogram':
        S = librosa.stft(audio, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    elif spec_type == 'mel_spectrogram':
        S = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
    else:
        raise NotImplementedError(
            f"'{spec_type}' is not a supported spec_type. "
            "Choose from 'spectrogram' or 'mel_spectrogram'."
        )

    if f0 is not None:  
        times = librosa.times_like(S, sr=sr, hop_length=hop_length)
        ax.plot(times, f0, linewidth=3, color='cyan', label='f0')
        ax.legend(loc='upper right')

    im = librosa.display.specshow(S_db, y_axis=y_axis, x_axis=x_axis, 
                             ax=ax, cmap=cmap, sr=sr,
                             hop_length=hop_length)
    fig.colorbar(im, format='%+2.0f dB')


def plot_phonesymbols(text, phonesymbols, duration, hop_length, sr, ax):
    def compute_cumulative_sums(duration):
        out = [0]
        for d in duration:
            out.append(d + out[-1])
        return out

    def compute_centers(cumulative_sums):
        centers = []
        for index, _ in enumerate(cumulative_sums):
            if index + 1 < len(cumulative_sums):
                centers.append((cumulative_sums[index] + cumulative_sums[index + 1]) / 2)
        return centers

    duration_time = duration * hop_length / sr
    duration_splits = compute_cumulative_sums(duration_time)
    phone_xticks = compute_centers(duration_splits)
    ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.5)
    ax.xaxis.grid(True, which='minor')
    ax.set_xticks(phone_xticks, minor=False)
    ax.set_xticklabels(phonesymbols)
    ax.xaxis.label.set_visible(False)


    word_boundaries = list()
    for index, word_boundary in enumerate(phonesymbols):
        if word_boundary == " ":
            word_boundaries.append(phone_xticks[index])
            
    # support: e.g., tacotron2
    # not support: e.g., fastspeech2
    if word_boundaries: 
        prev_word_boundary = 0
        word_xticks = list()
        for word_boundary in word_boundaries:
            word_xticks.append((word_boundary + prev_word_boundary) / 2)
            prev_word_boundary = word_boundary
        word_xticks.append((duration_splits[-1] + prev_word_boundary) / 2)
        ax.vlines(x=word_boundaries, colors="orange", linestyles="solid", ymin=0.0, ymax=8000, linewidth=1.2)
        secondary_ax = ax.secondary_xaxis('bottom')
        secondary_ax.tick_params(axis="x", direction="out", pad=24)
        secondary_ax.set_xticks(word_xticks, minor=False)
        secondary_ax.set_xticklabels(text.split())
        secondary_ax.tick_params(axis='x', colors='orange')
        secondary_ax.xaxis.label.set_color('orange')