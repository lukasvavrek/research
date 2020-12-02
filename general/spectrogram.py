import librosa
import numpy as np

class SpectrogramLoader:
    cache = {}

    # https://librosa.org/doc/latest/generated/librosa.stft.html#librosa.stft
    config = {
        "sr": 22050,        # sampling rate
        "n_fft": 2048,      # length of the FFT window
        "hop_length": 512 ,  # number of samples between successive frames
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
        #S_DB = self.scale_minmax(S_DB, 0, 255).astype(np.uint8)
        #S_DB = np.flip(S_DB, axis=0) # put low frequencies at the bottom in image
        #S_DB = 255-S_DB

        self.cache[file] = S_DB
        
        return S_DB

    def scale_minmax(self, X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled
