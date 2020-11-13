import numpy as np

class SVDPreprocessor:
    MAX_WIDTH = 60

    # copy of data would allow us to re-run preprocessing on the original data
    def preprocess(self, data):
        X_train = data.data["X_train"]
        X_val = data.data["X_val"]
        X_test = data.data["X_test"]
        y_train = data.data["y_train"]
        y_val = data.data["y_val"]
        y_test = data.data["y_test"]

        X_train = np.array([sample for subject in X_train for sample in self.process_subject(subject)])
        X_val = np.array([sample for subject in X_val for sample in self.process_subject(subject)])
        X_test = np.array([sample for subject in X_test for sample in self.process_subject(subject)])

        y_train = np.array([sample for subject in y_train for sample in [subject, subject, subject]])
        y_val = np.array([sample for subject in y_val for sample in [subject, subject, subject]])
        y_test = np.array([sample for subject in y_test for sample in [subject, subject, subject]])

        data.data["X_train"] = X_train
        data.data["X_val"] = X_val
        data.data["X_test"] = X_test
        data.data["y_train"] = y_train
        data.data["y_val"] = y_val
        data.data["y_test"] = y_test

    # reusing methods from pcgita preprocessor (TODO: extract them out)
    """alternative to cut_and_stack_samples"""
    def process_subject(self, subject, width=MAX_WIDTH):
        x = []
        
        # it _copies_ each sample three times into RGB
        for i in range(3):
            x.append(np.stack([
                subject[i][:, :width],
                subject[i][:, :width],
                subject[i][:, :width]
            ], axis=2))

        return x
        
    def cut_and_stack_samples(self, subject, width=MAX_WIDTH):   
        return np.stack([
            subject[0][:, :width],
            subject[1][:, :width],
            subject[2][:, :width]
        ], axis=2)

    # including old methods, just in case
    def all_samples_are_long_enough(self, sample):
        return all(s.shape[1] >= 32 for s in sample)

    def reshape_samples(self, sample):
        return [np.asarray(s[:512, :32]) for s in sample]

    def filter_short_data_samples(self, data):
        return [self.reshape_samples(s) for s in data if self.all_samples_are_long_enough(s)]
