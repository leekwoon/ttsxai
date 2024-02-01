import numpy as np


class DatasetNormalizer(object):
    def __init__(self, dataset):
        self.std = dataset.std(axis=0)
        self.mean = dataset.mean(axis=0)

    def normalize(self, x):    
        z = (x - self.mean) / self.std
        return np.clip(z, -5.0, 5.0)

    def unnormalize(self, x):
        return x * self.std + self.mean