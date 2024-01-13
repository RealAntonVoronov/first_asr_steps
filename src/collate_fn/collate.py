import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(batch: List[dict]):
    """
    Collate and pad fields in dataset items
    input: {
            "audio": torch.tensor [bs, sr * duration],
            "spectrogram": torch.tensor [bs, n_feature, n_time],
            "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "text": data_dict["text"],
            "text_encoded": self.text_encoder.encode(data_dict["text"]),
            "audio_path": audio_path,
        }
    """
    batch_size = len(batch)

    lens = torch.zeros(batch_size, 3)
    duration, text, audio_path = [], [], []
    for i, item in enumerate(batch):
        lens[i] = torch.LongTensor([item['audio'].size(1), item['spectrogram'].size(2), item['text_encoded'].size(1)])
        duration.append(item['duration'])
        text.append(item['text'])
        audio_path.append(item['audio_path'])

    max_audio_len, max_spec_time, max_encoded_len = lens.max(dim=0).values
    spectrogram_length = lens[:, 1].long()
    text_encoded_length = lens[:, 2].long()
    duration = torch.tensor(duration)

    audio = torch.zeros(batch_size, int(max_audio_len))
    spectrogram = torch.zeros(batch_size, batch[0]['spectrogram'].size(1), int(max_spec_time))
    text_encoded = torch.zeros(batch_size, int(max_encoded_len), dtype=int)

    for i, item in enumerate(batch):
        audio[i, :item['audio'].size(1)] = item['audio']
        spectrogram[i, :, :item['spectrogram'].size(2)] = item['spectrogram']
        text_encoded[i, :item['text_encoded'].size(1)] = item['text_encoded']

    return {
        "audio": audio,
        "spectrogram": spectrogram,
        "spectrogram_length": spectrogram_length,
        "duration": duration,
        "text": text,
        "text_encoded": text_encoded,
        "text_encoded_length": text_encoded_length,
        "audio_path": audio_path,
    }
