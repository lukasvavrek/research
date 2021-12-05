import os
import re

import torch
import torchaudio
from torch.utils.data import Dataset

from general.torch.spectrogram import get_spectrogram

class PcGitaTorchDataset(Dataset):
    base_path = 'D:/Users/lVavrek/research/data/pcgita/original/'
    output_path = 'D:/Users/lVavrek/research/data/pcgita/processed/'

    def __init__(self, transform=None, train=True):
        self.transform = transform
        self.train = train

        self.samples = self.get_sample_list()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(sample_path)

        if self.transform:
            sample = get_spectrogram(waveform)
            # crucial to call self.transform on sample
            return self.transform(sample), torch.tensor(label, dtype=torch.long)

        # raise Exception("There was no transform method provided!")
        sample = get_spectrogram(waveform)
        sample = torch.stack([sample, sample, sample], dim=1)

        return sample, torch.tensor(label, dtype=torch.long)

    def get_sample_list(self):
        vowels = ['A']
        folds = 1 # this should be correct as data accross forlds are duplicated
        sets = ['train'] if self.train is True else ['test']
        targets = [0, 1]

        file_paths = []

        for vowel in vowels:
            for fold in range(folds):
                for dataset in sets:
                    for target in targets:
                        file_paths += self.load_subjects(vowel, str(fold), dataset, str(target))

        return file_paths

    # A/k_0/test/0
    def load_subjects(self, vowel, fold, dataset, target):
        path = os.path.join(self.base_path, vowel, "k_"+fold, dataset, target)
        
        file_paths = []

        for subject_path in os.listdir(path):
            file_path = self.load_subject(os.path.join(path, subject_path), vowel, fold, dataset, target)
            file_paths.append((file_path, int(target))) # converting back to int

        return file_paths

    # A/k_0/test/0/0004/
    def load_subject(self, path, vowel, fold, dataset, target):
        files = os.listdir(path)
        
        regex = '.*{}.wav$'
        
        subject_data = [
            [file for file in files if re.match(regex.format(1), file)][0],
            [file for file in files if re.match(regex.format(2), file)][0],
            [file for file in files if re.match(regex.format(3), file)][0]
        ]
        
        # we only utilize normal level (for now)
        return os.path.join(path, subject_data[1]).replace('\\', '/') # ugly hack
