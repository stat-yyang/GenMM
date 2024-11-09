# dataset.py

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, N_list, M_list, Y):
        self.X = X
        self.N_list = N_list
        self.M_list = M_list
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'X': self.X[idx],
            'N_list': [N[idx] for N in self.N_list],
            'M_list': [M[idx] for M in self.M_list],
            'Y': self.Y[idx]
        }
        return sample
