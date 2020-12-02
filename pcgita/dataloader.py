import os
import re
import random
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.image import imread

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from general.data import Data
from general.visualization import visualize_spectrogram

"""
Classes responsible for working with Pc-Gita dataset.
Initial approach was computing spectrograms ad-hoc, but this process was hard to verify.

_PcGitaDataConverter_ is responsible for processing original k_folded dataset into pre-computed
dataset of spectrograms ready to be used.
"""

class PcGitaDataConverter:
    base_path = 'D:/Users/lVavrek/research/data/pcgita/original/'
    output_path = 'D:/Users/lVavrek/research/data/pcgita/processed/'

    def __init__(self, spectrogram_loader):
        self.spectrogram_loader = spectrogram_loader

    def process(self):
        vowels = ['A']
        folds = 5
        sets = ['train', 'test']
        targets = [0, 1]
        for vowel in vowels:
            for fold in range(folds):
                for dataset in sets:
                    for target in targets:
                        self.load_subjects(vowel, str(fold), dataset, str(target))
    
    # A/k_0/test/0
    def load_subjects(self, vowel, fold, dataset, target):
        path = os.path.join(self.base_path, vowel, "k_"+fold, dataset, target)
        
        for subject_path in os.listdir(path):
            print(subject_path)
            self.load_subject(os.path.join(path, subject_path), vowel, fold, dataset, target)

    # A/k_0/test/0/0004/
    def load_subject(self, path, vowel, fold, dataset, target):
        # get 3 wavs and return them as RGB spectrogram
        files = os.listdir(path)
        
        regex = '.*{}.wav$'
        
        subject_data = [
            [file for file in files if re.match(regex.format(1), file)][0],
            [file for file in files if re.match(regex.format(2), file)][0],
            [file for file in files if re.match(regex.format(3), file)][0]
        ]
        
        for subject in subject_data:
            spectrogram = self.spectrogram_loader.load_spectrogram(os.path.join(path, subject))
            
            out_dir = os.path.join(self.output_path, vowel, fold, dataset, target)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            fig = plt.figure()
            visualize_spectrogram(spectrogram, sr=self.spectrogram_loader.config["sr"], hop_length=self.spectrogram_loader.config["hop_length"], show_colorbar=False)
            plt.axes().set_axis_off()
            fig.savefig(os.path.join(out_dir, subject)+".png", bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)

            #plt.imsave(os.path.join(out_dir, subject)+".png", spectrogram)
            #io.imsave(os.path.join(out_dir, subject)+".png", spectrogram)

class PcGitaDataLoader:
    base_path = '../data/pcgita/processed/'

    def __init__(self, spectrogram_loader):
        self.spectrogram_loader = spectrogram_loader

    # A/k_0/test/0/0004/
    def load_subject(self, path):
        return imread(path)[:, :, :3]
        
    # A/k_0/test/0
    def load_subjects(self, path):
        subjects = []
        
        for subject_path in os.listdir(path):
            subject = self.load_subject(os.path.join(path, subject_path))
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
        path = '{}/{}/{}'.format(self.base_path, vowel, n)
        
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

    def load_generators(self, vowel, n):
        path = '{}/{}/{}'.format(self.base_path, vowel, n)

        train_datagen = ImageDataGenerator(
            rescale=1./255, # ??
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            path + '/train',
            target_size=(369, 496),
            batch_size=20,
            class_mode='binary')

        validaton_generator = test_datagen.flow_from_directory(
            path + '/test',
            target_size=(369, 496),
            batch_size=20,
            class_mode='binary')

        data = {
            'train_generator': train_generator,
            'validation_generator': validaton_generator
        }

        return Data('{}-{}'.format(vowel, n), data)

