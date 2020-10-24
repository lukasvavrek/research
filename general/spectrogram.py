import librosa
import numpy as np

class SpectrogramLoader:
    cache = {}

    config = {
        "sr": 22050,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128*2
    }

    def load_spectrogram(self, file):
        if file in self.cache:
            return self.cache[file]
        
        y, _ = librosa.load(file)
        y, _ = librosa.effects.trim(y)
        S = librosa.feature.melspectrogram(
            y, 
            sr=self.config["sr"],
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            n_mels=self.config["n_mels"])
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        #S_DB = np.abs(S_DB / 255)
        
        self.cache[file] = S_DB
        
        return S_DB
