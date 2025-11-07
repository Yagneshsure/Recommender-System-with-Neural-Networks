import torch
import torch.nn as nn

class HybridRecommender(nn.Module):
    """Combines RNN and GNN embeddings for hybrid recommendations"""
    def __init__(self, gnn_dim=64, rnn_dim=128, output_dim=64, num_items=1000):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(gnn_dim + rnn_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, num_items)
        )

    def forward(self, gnn_emb, rnn_emb):
        combined = torch.cat((gnn_emb, rnn_emb), dim=1)
        return self.fc(combined)
