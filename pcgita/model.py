from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from general.keras import disable_training_layers

class PcGitaModel:
    MAX_WIDTH = 32
    # should we introduce constants or obtain this as a parameter?
    N_MELS = 128  * 2

    def __init__(self, model='vgg'):
        self.model = model

    def build_model(self):
        if self.model == 'cnn':
            model = self.build_cnn_model()
        elif self.model == 'vgg':
            model = self.build_vgg16_model()
        
        ### 1e-6 -> velmi zaujimave vysledky na 50 epochach
        model.compile(
            #optimizer=optimizers.Adam(lr=0.0001), #lr= 1e-6 0.001
            optimizer=optimizers.Adam(), #lr= 1e-6 0.001
#           optimizer=optimizers.SGD(lr=0.1),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model

    def build_cnn_model(self):
        #channels = [8, 16, 32]

        channels = [8, 16]

        input_layer = layers.Input(shape=(self.N_MELS, self.MAX_WIDTH, 3))
        x = input_layer

        for channel in channels:
            x = layers.Conv2D(channel, (3, 3), activation='relu')(x)
            x = layers.Conv2D(channel, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
       
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        answer = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(input_layer, answer)
        return model

    def build_vgg16_model(self):
        conv_base = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.N_MELS, self.MAX_WIDTH, 3))

        disable_training_layers(conv_base)

        flattened = layers.Flatten()(conv_base.output)
        x = layers.Dense(16, activation='relu')(flattened)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        answer = layers.Dense(1, activation='sigmoid')(x) 

        model = models.Model(conv_base.input, answer)
        return model