import librosa
import torch
from torch import Tensor

from src.augmentations.base import AugmentationBase


class Stretch(AugmentationBase):
    def __init__(self, stretching_coef: int = 2):
        self.stretching_coef = stretching_coef

    def __call__(self, wav: Tensor):
        stretched_wav = librosa.effects.time_stretch(wav.numpy().squeeze(), self.stretching_coef)
        return torch.from_numpy(stretched_wav)
    