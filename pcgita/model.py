from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from general.keras import disable_training_layers

class PcGitaModel:
    MAX_WIDTH = 32

    def build_model(self):
        # should we introduce constants or obtain this as a parameter?
        n_mels = 128  * 2

        conv_base = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(n_mels, self.MAX_WIDTH, 3))

        disable_training_layers(conv_base)

        flattened = layers.Flatten()(conv_base.output)
        x = layers.Dense(16, activation='relu')(flattened)
        x = layers.Dropout(rate=0.4)(x)
        x = layers.Dense(8, activation='relu')(x)
        x = layers.Dropout(rate=0.2)(x)
        answer = layers.Dense(1, activation='sigmoid')(x) 

        model = models.Model(conv_base.input, answer)
        
        ### 1e-6 -> velmi zaujimave vysledky na 50 epochach

        model.compile(
            #optimizer=optimizers.Adam(lr=0.0001), #lr= 1e-6 0.001
            optimizer=optimizers.Adam(lr=0.00001), #lr= 1e-6 0.001
#           optimizer=optimizers.SGD(lr=0.1),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model