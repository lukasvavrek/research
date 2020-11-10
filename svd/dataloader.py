import os
import re
import random
from sklearn.model_selection import train_test_split

from general.data import Data

class SVDDataLoader:
    MAX_WIDTH = 60
    base_path = '../../SB-dataset/'

    def __init__(self, spectrogram_loader):
        self.spectrogram_loader = spectrogram_loader

    # SB-dataset/p/1040
    def load_subject(self, path):
        # get 3 wavs and return them as RGB spectrogram
        files = os.listdir(path)

        regex = '.*-{}_n.wav$'

        subject_data = [
            [file for file in files if re.match(regex.format('a'), file)][0],
            [file for file in files if re.match(regex.format('i'), file)][0],
            [file for file in files if re.match(regex.format('u'), file)][0]
        ]
        
        subject_data = map(lambda f: os.path.join(path, f), subject_data)
        subject_data = map(lambda f: self.spectrogram_loader.load_spectrogram(f), subject_data)
        
        return list(subject_data)

    # TODO: refactor into base class
    def should_skip_subject(self, subject, min_width=MAX_WIDTH):
        if any(s.shape[1] < min_width for s in subject):
            return True
        return False

    # SB-dataset/p/
    def load_subjects(self, path):
        subjects = []
        
        for subject_path in os.listdir(path):
            subject = self.load_subject(os.path.join(path, subject_path))
            if not self.should_skip_subject(subject):
                subjects.append(subject)
            
        return subjects

# todo: add support for configuration (all vowels, single vowel, ...)
    def load(self):
        positive = self.load_subjects(os.path.join(self.base_path, 'p'))
        negative = self.load_subjects(os.path.join(self.base_path, 'h'))

        # merge, shuffle and split into test/val/train
        negative = [(n, 0) for n in negative]
        positive = [(p, 1) for p in positive]
        data = negative + positive
        random.shuffle(data)
        
        X = [t[0] for t in data]
        y = [t[1] for t in data]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)

        data = {
            "X_train": X_train,
            "y_train": y_train,
            # PC-Gita specific, there is no validation test set
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }
        return Data('svd-a-i-u', data)
