import numpy as np


class DatasetNormalizer(object):
    def __init__(self, dataset, epsilon=1e-4):
        self.mean = dataset.mean(axis=0)
        self.std = dataset.std(axis=0)
        # To handle cases where the standard deviation is too small.
        self.std[self.std < epsilon] = epsilon

    def normalize(self, x):    
        z = (x - self.mean) / self.std
        return np.clip(z, -5.0, 5.0)

    def unnormalize(self, x):
        return x * self.std + self.mean