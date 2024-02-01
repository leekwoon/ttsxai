import torch
from torch.utils.data import Dataset


class ProbeDataset(Dataset):
    def __init__(self, dataframe, input_columns, label_column):
        """
        Args:
            dataframe (pandas.DataFrame): 데이터를 포함하고 있는 pandas DataFrame.
            input_columns (list of str): 입력 특성으로 사용될 컬럼의 이름 목록.
            label_column (str): 레이블로 사용될 컬럼의 이름.
        """
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.label_column = label_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.dataframe.iloc[idx][self.input_columns]
        print(inputs)
        label = self.dataframe.iloc[idx][self.label_column]

        # 데이터를 Tensor로 변환
        inputs = torch.tensor(inputs, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return inputs, label