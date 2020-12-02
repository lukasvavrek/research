from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from general.keras import disable_training_layers

class PcGitaModel:
    INPUT_SHAPE = (369, 496, 3)

    def __init__(self, model='vgg'):
        self.model = model

    def build_model(self):
        if self.model == 'cnn':
            model = self.build_cnn_model()
        elif self.model == 'vgg':
            model = self.build_vgg16_model()
        elif self.model == 'xception':
            model = self.build_xception_model()
        
        ### 1e-6 -> velmi zaujimave vysledky na 50 epochach
        model.compile(
            #optimizer=optimizers.Adam(), #lr= 1e-6 0.001
            optimizer=optimizers.RMSprop(lr=2e-5), #lr= 1e-6 0.001
#           optimizer=optimizers.SGD(),
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
            input_shape=self.INPUT_SHAPE)

        disable_training_layers(conv_base)

        flattened = layers.Flatten()(conv_base.output)
        #flattened = layers.GlobalAveragePooling2D(conv_base.output) # how to use it?
        x = layers.Dense(256, activation='relu')(flattened)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer='l1')(x)
        answer = layers.Dense(1, activation='sigmoid')(x) 

        model = models.Model(conv_base.input, answer)
        return model

    def build_xception_model(self):
        conv_base = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=self.INPUT_SHAPE)

        disable_training_layers(conv_base)

        flattened = layers.Flatten()(conv_base.output)
        x = layers.Dense(128, activation='relu')(flattened)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        answer = layers.Dense(1, activation='sigmoid')(x) 

        model = models.Model(conv_base.input, answer)
        return model