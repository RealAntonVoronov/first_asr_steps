import librosa
import torch
from torch import Tensor

from src.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, sr: int = 16000, shift: int = -5):
        self.shift = shift
        self.sr = sr
        
    def __call__(self, wav: Tensor):
        pitch_shifted_wav = librosa.effects.pitch_shift(wav.numpy().squeeze(), self.sr, self.shift)
        return torch.from_numpy(pitch_shifted_wav)
