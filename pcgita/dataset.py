import os
import re
from typing import Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from general.torch.spectrogram import get_spectrogram

from PIL.Image import Image

class PcGitaTorchDataset(Dataset):
    root = 'D:/Users/lVavrek/research/data/'  # only applicable for Helios

    base_path: str
    output_path: str

    def __init__(self, transform=None, train=True, root=None):
        if root is not None:
            root = root

        self.base_path = os.path.join(root, 'pcgita', 'original')
        self.output_path = os.path.join(root, 'pcgita', 'processed')

        self.transform = transform
        self.train = train

        self.samples = self.get_sample_list()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image, torch.Tensor]:
        sample_path, label = self.samples[idx]

        waveform: torch.Tensor
        sample_rate: int 

        waveform, sample_rate = torchaudio.load(sample_path) # type: ignore

        sample = get_spectrogram(waveform, sample_rate, 30, 200)

        if self.transform:
            # crucial to call self.transform on sample
            return self.transform(sample), torch.tensor(label, dtype=torch.long)

        # raise Exception("There was no transform method provided!")

        return sample, torch.tensor(label, dtype=torch.long)

    def get_sample_list(self) -> list[Tuple[str, int]]:
        """Load dataset (path) from the filesystem based on the hardcoded filters

        Returns:
            list[Tuple[str, int]]: list of tuples of paths and targets to subject data
        """
        vowels = ['A']
        folds = 1 # this should be correct as data accross forlds are duplicated
        sets = ['train'] if self.train is True else ['test']
        targets = [0, 1]

        file_paths = []

        for vowel in vowels:
            for fold in range(folds):
                for dataset in sets:
                    for target in targets:
                        file_paths += self.load_subjects(vowel, fold, dataset, target)

        return file_paths

    # A/k_0/test/0
    def load_subjects(self, vowel: str, fold: int, dataset: str, target: int) -> list[Tuple[str, int]]:
        """Load multiple subject information (paths) from the file system

        Args:
            vowel (str): a|i|u|...
            fold (int): fold index number
            dataset (_type_): train|test
            target (int): 0|1 - pathological or healthy

        Returns:
            list[Tuple[str, int]]: list of tuples of paths and targets to subject data
        """
        path = os.path.join(self.base_path, vowel, "k_"+str(fold), dataset, str(target))
        
        file_paths = []

        for subject_path in os.listdir(path):
            file_path = self.load_subject(os.path.join(path, subject_path))
            file_paths.append((file_path, target))

        return file_paths

    # A/k_0/test/0/0004/
    def load_subject(self, path: str) -> str:
        """Load subject information (path) from the file system

        Args:
            path (str): a path to a subject root folder

        Returns:
            str: a path to a picked .wav file containing subject data
        """
        files = os.listdir(path)
        
        regex = '.*{}.wav$'
        
        # loading all 'levels' recorded by a given subject
        subject_data = [
            [file for file in files if re.match(regex.format(1), file)][0],
            [file for file in files if re.match(regex.format(2), file)][0],
            [file for file in files if re.match(regex.format(3), file)][0]
        ]
        
        # we only utilize normal level (for now)
        subject_data_to_use = subject_data[1]

        return os.path.join(path, subject_data_to_use).replace('\\', '/') # ugly hack
