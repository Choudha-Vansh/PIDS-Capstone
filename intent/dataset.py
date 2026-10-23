"""Torch Dataset over prepared sequence arrays."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path).astype(np.float32)   # (N, T, F)
        self.y = np.load(y_path).astype(np.float32)   # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

    @property
    def feature_dim(self) -> int:
        return self.X.shape[2]
