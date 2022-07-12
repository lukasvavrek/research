import sys

import numpy as np
from numpy import ndarray

from torch import Tensor, from_numpy, hann_window
from torchaudio.transforms import Spectrogram
from torchvision.transforms import Compose, ToPILImage, Resize

from skimage.util import img_as_ubyte
from PIL.Image import Image

transform_spectra = Compose([
    ToPILImage(),
    Resize((224,224)), 
    # transforms.RandomVerticalFlip(1)
])

def normalize(tensor: Tensor) -> Tensor:
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

def normalize_nd(tensor: ndarray) -> ndarray:
    tensor_minusmean = tensor - tensor.mean()    
    return tensor_minusmean/np.absolute(tensor_minusmean).max()

def make0min(tensornd: Tensor) -> Tensor:
    tensor = tensornd.numpy()
    res = np.where(tensor == 0, 1E-19 , tensor)
    return from_numpy(res)

def spectrogrameToImage(waveform: Tensor) -> Image:
    spectrogram = Spectrogram(
        n_fft=400,
        win_length=None,
        hop_length=None, 
        pad=0,
        window_fn=hann_window,
        power=2, 
        normalized=True, 
        wkwargs=None)(waveform)

    spectrogram = make0min(spectrogram)
    specgram: ndarray = spectrogram.log2()[0,:,:].numpy()
    
    np.set_printoptions(linewidth=300)
    np.set_printoptions(threshold=sys.maxsize)

    specgram = normalize_nd(specgram)
    specgram = img_as_ubyte(specgram)

    specgramImage: Image = transform_spectra(specgram) # type: ignore

    return specgramImage  

# skip 100ms and take 200ms
def get_spectrogram(audio_data: Tensor, sample_rate: int, skip: int = 200, take: int = 400) -> Image:
    audio_data = select(audio_data, sample_rate, skip, take)
    waveform = normalize(audio_data)
    spec = spectrogrameToImage(waveform)
    spec = spec.convert('RGB')
    
    return spec

# skip and take
def select(audio_data: Tensor, sample_rate: int, skip_ms: int, take_ms: int) -> Tensor:
   sr_ms = int(sample_rate / 1000)
   return audio_data[:, sr_ms * skip_ms : sr_ms * skip_ms + sr_ms * take_ms]
   #return torch.take(audio_data, torch.tensor(filter))
   #frames_to_skip = min(sr_ms * msTime, len(audio_data))
   #return audio_data[frames_to_skip:]
