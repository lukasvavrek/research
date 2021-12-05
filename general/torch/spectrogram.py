import sys

import numpy as np

import torch
import torchaudio
from torchvision import transforms

from skimage.util import img_as_ubyte

transform_spectra = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)), 
    # transforms.RandomVerticalFlip(1)
])

def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

def normalize_nd(tensor):
    tensor_minusmean = tensor - tensor.mean()    
    return tensor_minusmean/np.absolute(tensor_minusmean).max()

def make0min(tensornd):
    tensor = tensornd.numpy()
    res = np.where(tensor == 0, 1E-19 , tensor)
    return torch.from_numpy(res)

def spectrogrameToImage(waveform):
    specgram = torchaudio.transforms.Spectrogram(
        n_fft=400,
        win_length=None,
        hop_length=None, 
        pad=0,
        window_fn=torch.hann_window,
        power=2, 
        normalized=True, 
        wkwargs=None)(waveform)
    specgram = make0min(specgram)
    specgram = specgram.log2()[0,:,:].numpy()
    
    np.set_printoptions(linewidth=300)
    np.set_printoptions(threshold=sys.maxsize)

    specgram= normalize_nd(specgram)
    specgram = img_as_ubyte(specgram)
    specgramImage = transform_spectra(specgram)
    return specgramImage

# skip 100ms and take 200ms
def get_spectrogram(audio_data, sample_rate):
    audio_data = select(audio_data, sample_rate, 200, 400)
    waveform = normalize(audio_data)
    spec = spectrogrameToImage(waveform)
    spec = spec.convert('RGB')
    
    return spec

# skip and take
def select(audio_data, sample_rate, skip_ms, take_ms):
   sr_ms = int(sample_rate / 1000)
   return audio_data[:, sr_ms * skip_ms : sr_ms * skip_ms + sr_ms * take_ms]
   #return torch.take(audio_data, torch.tensor(filter))
   #frames_to_skip = min(sr_ms * msTime, len(audio_data))
   #return audio_data[frames_to_skip:]
