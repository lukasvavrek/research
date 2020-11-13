import tensorflow
from tensorflow.keras.backend import set_session

def prepare_session():
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tensorflow.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

# there has to be a better place for this method
def disable_training_layers(model):
    for layer in model.layers:
        layer.trainable = False

from tensorflow.keras import callbacks

class Trainer:
    EPOCHS = 100
    BATCH_SIZE = 20
    EARLY_STOP_PATIENCE = 10
    FILE_NAME_FORMAT = '../output/{}.mdl_wts.hdf5'

    # should we provide model_builder or model directly?
    def __init__(self, model_builder):
        self.model_builder = model_builder

    def fit_model(self, data, early_stop=False):
        self.model = self.model_builder.build_model()

        X_train = data.data["X_train"]
        X_val = data.data["X_val"]
        y_train = data.data["y_train"]
        y_val = data.data["y_val"]
        
        file_name = self.FILE_NAME_FORMAT.format(data.identifier)
    
        # usually doesn't help
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.EARLY_STOP_PATIENCE, 
            verbose=0, 
            mode='min')
        mcp_save = callbacks.ModelCheckpoint(
            file_name,
            save_best_only=True,
            monitor='val_loss',
            mode='min')

        cbs = [mcp_save]
        if early_stop:
            cbs.append(early_stop)

        return self.model.fit(
            X_train, 
            y_train, 
            epochs=self.EPOCHS, 
            batch_size=self.BATCH_SIZE, 
            validation_data=(X_val, y_val), 
            callbacks=cbs)

    # this is cheating a bit as model already saw val data
    # however, we want to have the ability to save the best model
    def evaluate(self, data):
        # TODO: normalize
        X_test = data.data["X_test"]
        y_test = data.data["y_test"]

        file_name = self.FILE_NAME_FORMAT.format(data.identifier)
        self.model.load_weights(file_name)

        return self.model.evaluate(X_test, y_test)