import pandas as pd
import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    """User-item interaction dataset for training"""
    def __init__(self, interactions):
        self.data = interactions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, rating = self.data.iloc[idx]
        return torch.tensor(user), torch.tensor(item), torch.tensor(rating)
