import pandas as pd
import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    """Dataset for user–item interactions"""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user, item, rating = self.df.iloc[idx]
        return torch.tensor(user), torch.tensor(item), torch.tensor(rating)
