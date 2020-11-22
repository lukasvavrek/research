from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from general.keras import disable_training_layers

class SVDModel:
    MAX_WIDTH = 32

    def __init__(self, weights=None):
        self.weights = weights

    def build_model(self):
        # should we introduce constants or obtain this as a parameter?
        n_mels = 128  * 2

        conv_base = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(n_mels, self.MAX_WIDTH, 3))

        disable_training_layers(conv_base)

        flattened = layers.Flatten()(conv_base.output)
        x = layers.Dense(32, activation='relu')(flattened)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        answer = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(conv_base.input, answer)
        optimizer = optimizers.Adam()

        # specifically for PC-GITA transfer learning
        if self.weights is not None:
            print('Transfer learning from pre-trained custom model')
            model.load_weights(self.weights)
            disable_training_layers(model)
            for layer in model.layers[-4:]:
                layer.trainable = True
            optimizer = optimizers.Adam(1e-4)

            x = layers.Dense(8, activation='relu')(model.layers[-2].output)
            x = layers.Dropout(rate=0.4)(x)
            answer = layers.Dense(1, activation='sigmoid')(x)

            model = models.Model(conv_base.input, answer)

        model.compile(
            optimizer=optimizer, #lr= 1e-6 0.001
#           optimizer=optimizers.SGD(lr=0.1),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model