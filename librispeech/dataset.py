import os

import torch
from torch.utils.data import Dataset
from torchaudio.datasets.librispeech import LIBRISPEECH

from general.torch.spectrogram import get_spectrogram

path_to_data_folder = os.path.normpath(r'D:\Users\lVavrek\research\data')

class LibrispeechSpectrogramDataset(LIBRISPEECH):
    def __init__(self, root=path_to_data_folder, download=True, transform=None, train=True):
        super().__init__(root=root, download=download, url='train-clean-100' if train is True else 'test-clean')
        self.transform = transform

    def __getitem__(self, idx):
        # waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id
        data = super().__getitem__(idx)

        sample = data[0] # waveform
        sample_rate = data[1] #sample_rate

        sample = get_spectrogram(sample, sample_rate)

        if self.transform:
            return self.transform(sample), torch.tensor(1, dtype=torch.long)

        # sample = torch.stack([sample, sample, sample], dim=1)

        return sample, torch.tensor(1, dtype=torch.long)
