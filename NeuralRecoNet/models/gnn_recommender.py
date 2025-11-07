import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN

class GNNRecommender(nn.Module):
    """Graph Neural Network based recommender using LightGCN"""
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.model = LightGCN(num_users + num_items, embedding_dim, num_layers=num_layers)

    def forward(self, edge_index):
        return self.model(edge_index)
