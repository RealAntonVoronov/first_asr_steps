import librosa
import numpy as np
import torch
from torch import Tensor

from src.augmentations.base import AugmentationBase


class BackgroundNoise(AugmentationBase):
    def __init__(self, noise_level: int = 20, shift: int = -5):
        filenames = [librosa.ex(name) for name in ['nutcracker', 'choice', 'pistachio']] # these are longer than 20 secs
        self.samples = [librosa.load(filename) for filename in filenames]  # List of tuples: (sample, sr)
        self.noise_level = torch.Tensor([noise_level])  # [0, 40]

    def __call__(self, wav: Tensor):
        noise, sr = np.random.choice(self.samples)
        noise = torch.from_numpy(noise)

        noize_energy = torch.norm(noise)
        audio_energy = torch.norm(wav)
        alpha = (audio_energy / noize_energy) * torch.pow(10, -self.noise_level / 20)

        clipped_wav = wav[..., :noise.shape[0]]
        augumented_wav = clipped_wav + alpha * noise

        return torch.clamp(augumented_wav, -1, 1)
