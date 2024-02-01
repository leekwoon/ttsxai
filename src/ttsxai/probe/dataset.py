import numpy as np

import torch
from torch.utils.data import Dataset

from ttsxai.normalization import DatasetNormalizer


class ProbeDataset(Dataset):
    def __init__(self, dataframe, input_columns, label_column, neurons_to_keep=None):
        """
        Args:
            dataframe (pandas.DataFrame): The pandas DataFrame containing the data.
            input_columns (list of str): The names of the columns to be used as input features.
            label_column (str): The name of the column to be used as the label.
        """
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.label_column = label_column
        self.neurons_to_keep = neurons_to_keep

        self.X = np.stack(dataframe[input_columns].to_numpy())
        self.y = dataframe[label_column].to_numpy()
        if self.label_column in ['duration', 'pitch', 'energy']:
            # # ignore invalid data
            valid_idxs = self.y > 0
            self.X = self.X[valid_idxs]
            self.y = self.y[valid_idxs]

            # log scale
            self.y = np.log(self.y) 

            if neurons_to_keep is not None:
                self.X = self.X[:, neurons_to_keep]
                print(f'only keep neurons {neurons_to_keep}')
        else:
            raise NotImplementedError
        
    def make_normalizer(self, X):
        self.normalizer = DatasetNormalizer(X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        activation = self.normalizer.normalize(self.X[idx])
        label = self.y[idx] 

        # to tensor
        activation = torch.tensor(activation, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return activation, label

