import torch
import torch.nn as nn

class RNNRecommender(nn.Module):
    """Recurrent model (LSTM/GRU) for sequential recommendations"""
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128, rnn_type='gru'):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        rnn_class = nn.GRU if rnn_type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_class(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, sequences):
        x = self.embedding(sequences)
        out, _ = self.rnn(x)
        preds = self.fc(out[:, -1, :])
        return preds
