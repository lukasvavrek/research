# https://pytorch.org/audio/stable/_modules/torchaudio/datasets/librispeech.html#LIBRISPEECH

# this script is pretty much a copy of SIM CLR playground.ipynb
# I created it to try avoid those bugs:
# https://github.com/PyTorchLightning/lightning-bolts/issues/640
# https://github.com/PyTorchLightning/lightning-bolts/issues/642

import os
import torch
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.transforms import Spectrogram, AmplitudeToDB # try using mel spectrogram as well
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

path_to_data_folder = os.path.normpath(r'D:\Users\lVavrek\research\data')

def get_spectrogram(audio_data):
    n_fft = 1024
    win_length = 45
    hop_length = 512

    transform = Spectrogram(
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_length,
        center = True,
        pad_mode = 'reflect',
        power = 2.0
    )
    
    spectrogram = transform.forward(audio_data[0]) # audio is first

    return spectrogram[:, :, :45]

class LibrispeechSpectrogramDataset(LIBRISPEECH):
    def __init__(self, root=path_to_data_folder, download=True, transform=None, train=True):
        super().__init__(root=root, download=download, url='train-clean-100' if train is True else 'test-clean')
        if transform:
            self.transform = transform
        else:
            self.transform = get_spectrogram

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        if self.transform:
            sample = get_spectrogram(sample)
            sample = torch.stack([sample, sample, sample], dim=1)

        return sample, torch.tensor(1, dtype=torch.long)

if __name__ == '__main__':
    librispeech = LIBRISPEECH(root=path_to_data_folder, download=True)

    batch_size = 64 
    num_workers = 1

    train_dataset = LibrispeechSpectrogramDataset(transform=SimCLRTrainDataTransform(), train=True)
    val_dataset = LibrispeechSpectrogramDataset(transform=SimCLREvalDataTransform(), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    model = SimCLR(gpus=1, num_samples=len(train_dataset), batch_size=batch_size, dataset='imagenet')
    trainer = Trainer(gpus=1)
    trainer.fit(model, train_loader, test_loader)
