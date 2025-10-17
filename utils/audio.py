"""
Description:
    This script contains a collection of functions designed to handle various
    audio processing.
"""

import random
import numpy as np
import soxr
import soundfile
import torch
import torchaudio

from pathlib import Path

def stft(x, fft_size, hop_size, win_length, window, use_complex=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(x, fft_size, hop_size, win_length, window.to(x.device),
                        return_complex=True)

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(torch.clamp(
            x_stft.real ** 2 + x_stft.imag ** 2, min=1e-7, max=1e3)).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3) # [B, 2, T, F]
        return res

def audio_volume_normalize(audio, coeff=0.2):
    """
    Normalize the volume of an audio signal.

    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(
            temp[-1], 1e-3
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]  # Length of the filtered array

    # If there are fewer than or equal to 10 significant values, return the audio without further processing
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level, clamping the scale factor between 0.1 and 10
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value in the audio does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def load_audio(
    adfile: Path,
    sampling_rate: int = None,
    length: int = None,
    volume_normalize: bool = False,
    segment_duration: int = None,
):
    r"""Load audio file with target sampling rate and lsength

    Args:
        adfile (Path): path to audio file.
        sampling_rate (int, optional): target sampling rate. Defaults to None.
        length (int, optional): target audio length. Defaults to None.
        volume_normalize (bool, optional): whether perform volume normalization. Defaults to False.
        segment_duration (int): random select a segment with duration of {segment_duration}s.
                                Defualt to None which means the whole audio will be used.

    Returns:
        audio (np.ndarray): audio
    """
    audio, sr = soundfile.read(adfile)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # print('audio', adfile, audio)
    audio = np.array(audio.squeeze())

    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate

    if segment_duration is not None:
        seg_length = int(sr * segment_duration)
        audio = random_select_audio_segment(audio, seg_length)

    # Audio volume normalize
    if volume_normalize:
        audio = audio_volume_normalize(audio)
    # check the audio length
    if length is not None:
        assert abs(audio.shape[0] - length) < 1000, (
            f"{adfile} Audio length is {audio.shape[0]}, but target length is {length}"
        )
        if audio.shape[0] > length:
            audio = audio[:length]
        else:
            audio = np.pad(audio, (0, int(length - audio.shape[0])))
    return audio


def random_select_audio_segment(audio: np.ndarray, length: int):
    """get an audio segment given the length

    Args:
        audio (np.ndarray):
        length (int): audio length = sampling_rate * duration
    """
    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]


def audio_highpass_filter(audio, sample_rate, highpass_cutoff_freq):
    """apply highpass fileter to audio
    
    Args:
        audio (np.ndarray):
        sample_rate (ind):
        highpass_cutoff_freq (int):
    """
    if audio is None:
        return None
    
    audio = torchaudio.functional.highpass_biquad(
        torch.from_numpy(audio), sample_rate, cutoff_freq=highpass_cutoff_freq
    )
    return audio.numpy()