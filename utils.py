from albumentations.core.transforms_interface import DualTransform, BasicTransform
import albumentations as A
import random
import os
import librosa
import numpy as np
import torchaudio
import torch
import matplotlib.pyplot as plt

class AudioTransform(BasicTransform):
    """Transform for Audio task"""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

class NoiseInjection(AudioTransform):
    """It simply add some random value into data by using numpy"""
    def __init__(self, always_apply=False, p=0.5):
        super(NoiseInjection, self).__init__(always_apply, p)
    
    def apply(self, data, noise_levels=(0, 0.005), **params):
        sound, sr = data

        noise_level = np.random.uniform(*noise_levels)
        
        noise = np.random.randn(len(sound),).astype(np.float32)

        augmented_sound = np.add(sound, noise_level * noise)
        
        # Cast back to same data type
        augmented_sound = augmented_sound.astype(np.float32)
        augmented_sound = np.clip(augmented_sound, -0.9, 0.9)


        return (augmented_sound, sr)

class ShiftingTime(AudioTransform):
    """Shifting time axis"""
    def __init__(self, always_apply=False, p=0.5):
        super(ShiftingTime, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        sound, sr = data

        shift_max = np.random.randint(2)

        if shift_max == 0:
            shift = 0
        else:
            shift = np.random.randint(5000,30000)

        direction = np.random.randint(0,2)
        if direction == 1:
            shift = -shift

        augmented_sound = np.roll(sound, shift)

        return augmented_sound, sr


class PitchShift(AudioTransform):
    """Shifting time axis"""
    def __init__(self, always_apply=False, p=0.5):
        super(PitchShift, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        sound, sr = data

        n_steps = np.random.randint(-10, 10)
        augmented_sound = librosa.effects.pitch_shift(sound, sr, n_steps)

        return augmented_sound, sr

class MelSpectrogram(AudioTransform):
    """Shifting time axis"""
    def __init__(self, parameters, always_apply=False, p=0.5):
        super(MelSpectrogram, self).__init__(always_apply, p)
        self.parameters = parameters
    
    def apply(self, data, **params):
        sound, sr = data

        melspec = librosa.feature.melspectrogram(sound, sr=sr, **self.parameters)
        melspec = librosa.power_to_db(melspec**2,ref=np.max)
        melspec = melspec.astype(np.float32)

        return melspec, sr


class AudioManipulation:

    def __init__(self, num_channel, max_ms):
        self.num_channel = num_channel
        self.max_ms = max_ms

    def open_tranform_data(self, file_audio):
        data = self.open(file_audio)
        # data = self.rechannel(data, self.num_channel)
        
        data = self.pad_trunc(data, self.max_ms)
        return data

    def open(self, path_file, sr=22050):

        sig, sr = torchaudio.load(path_file)
        sig = sig[0].numpy()

        return (sig, sr) # (sig, sr

    def rechannel(self, aud, new_channel):
        sig, sr = aud
        sig = sig[:,0]

        return (sig, sr)

    def pad_trunc(self, aud, max_ms):
        sig, sr = aud

        sig_len  = sig.shape[0]
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            sig = sig[:max_len]


        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.random.randn(pad_begin_len).astype(sig.dtype) * np.random.uniform(0,0.0005)
            pad_end = np.random.randn(pad_end_len).astype(sig.dtype) * np.random.uniform(0,0.0005)
            sig = np.concatenate([pad_begin, sig, pad_end])

        return (sig, sr)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
#   plt.show(block=False)

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
#   plt.show(block=False)