import numpy as np

class Data:
    def __init__(self, identifier, data):
        self.identifier = identifier
        self.data = data

    def preprocess(self, preprocessor, normalize_data=True):
        self.data = preprocessor(self.data)

    def normalize(self):
        # Get min, max value aming all elements for each column
        X_train = self.data["X_train"]
        X_val = self.data["X_val"]
            
        x_min = np.min(X_train, axis=tuple(range(X_train.ndim-1)), keepdims=1)
        x_max = np.max(X_train, axis=tuple(range(X_train.ndim-1)), keepdims=1)

        # Normalize with those min, max values leveraging broadcasting
        X_train = (X_train - x_min)/ (x_max - x_min)
        X_val = (X_val - x_min) / (x_max - x_min)

        self.data["X_train"] = X_train
        self.data["X_val"] = X_val
