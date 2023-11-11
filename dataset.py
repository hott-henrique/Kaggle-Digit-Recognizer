import torch

from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class TestDataset(Dataset):

    def __init__(self, path: str, transform = None):
        self.X, self.y = TestDataset.get_input_and_targets(path=path)
        self.transform = transform

    def __getitem__(self, idx) -> np.ndarray:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.X[idx], self.y[idx])

        if self.transform:
            sample = (self.transform(sample[0]), self.transform(sample[1]))

        return sample

    def __len__(self) -> int:
        return len(self.X)

    @staticmethod
    def get_input_and_targets(path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)

        X = (df.iloc[:, 1:].to_numpy() / 255.0).astype(np.float32)

        y = df['label']

        return X, y

class TrainDataset(Dataset):

    def __init__(self, path: str, transform = None):
        self.X = TrainDataset.get_input(path=path)
        self.transform = transform

    def __getitem__(self, idx) -> np.ndarray:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.X)

    @staticmethod
    def get_input(path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)

        X = df.to_numpy(dtype=np.float32)

        return X
