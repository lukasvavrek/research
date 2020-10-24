import numpy as np

class PcGitaPreprocessor:
    MAX_WIDTH = 60

    def preprocess(self, data):
        X_train = data.data["X_train"]
        X_val = data.data["X_val"]
        y_train = data.data["y_train"]
        y_val = data.data["y_val"]

        X_train = np.array([sample for subject in X_train for sample in self.process_subject(subject)])
        X_val = np.array([sample for subject in X_val for sample in self.process_subject(subject)])
        
        y_train = np.array([sample for subject in y_train for sample in [subject, subject, subject]])
        y_val = np.array([sample for subject in y_val for sample in [subject, subject, subject]])

        data.data["X_train"] = X_train
        data.data["X_val"] = X_val
        data.data["y_train"] = y_train
        data.data["y_val"] = y_val

    """alternative to cut_and_stack_samples"""
    def process_subject(self, subject, width=MAX_WIDTH):
        x = []
        
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
