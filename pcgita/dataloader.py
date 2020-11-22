import os
import re
import random

from general.data import Data

class PcGitaDataLoader:
    MAX_WIDTH = 32
    base_path = '../../k_fold_by_vowels/'

    def __init__(self, spectrogram_loader):
        self.spectrogram_loader = spectrogram_loader

    # A/k_0/test/0/0004/
    def load_subject(self, path):
        # get 3 wavs and return them as RGB spectrogram
        files = os.listdir(path)
        
        regex = '.*{}.wav$'
        
        subject_data = [
            [file for file in files if re.match(regex.format(1), file)][0],
            [file for file in files if re.match(regex.format(2), file)][0],
            [file for file in files if re.match(regex.format(3), file)][0]
        ]
        
        subject_data = map(lambda f: os.path.join(path, f), subject_data)
        subject_data = map(lambda f: self.spectrogram_loader.load_spectrogram(f), subject_data)
        
        return list(subject_data)

    def should_skip_subject(self, subject, min_width=MAX_WIDTH):
        if any(s.shape[1] < min_width for s in subject):
            return True
        return False

    # A/k_0/test/0
    def load_subjects(self, path):
        subjects = []
        
        for subject_path in os.listdir(path):
            subject = self.load_subject(os.path.join(path, subject_path))
            if self.should_skip_subject(subject) == False:
                subjects.append(subject)
            
        return subjects 
        
    # A/k_0/test/
    def load_data(self, path):
        negative = self.load_subjects('{}/0'.format(path))
        positive = self.load_subjects('{}/1'.format(path))
        
        negative = [(n, 0) for n in negative]
        positive = [(p, 1) for p in positive]
        
        data = negative + positive
        random.shuffle(data)
        
        return [t[0] for t in data], [t[1] for t in data]
        
    def load_fold(self, vowel, n):
        path = '{}/{}/k_{}'.format(self.base_path, vowel, n)
        
        X_train, y_train = self.load_data('{}/train'.format(path))
        X_test, y_test = self.load_data('{}/test'.format(path))
        
        data = {
            "X_train": X_train,
            "y_train": y_train,
            # PC-Gita specific, there is no validation test set
            "X_val": X_test,
            "y_val": y_test,
            "X_test": X_test,
            "y_test": y_test
        }
        return Data('{}-{}'.format(vowel, n), data)