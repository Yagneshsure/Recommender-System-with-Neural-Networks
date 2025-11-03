import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4Rec(nn.Module):
    """GRU-based session recommendation model."""
    
    def __init__(self, n_items, embedding_dim=128, hidden_dim=256, 
                 num_layers=1, dropout=0.3):
        super(GRU4Rec, self).__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Item embedding layer
        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        
        # GRU layers
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, n_items)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, item_seq, hidden=None):
        """
        Args:
            item_seq: [batch_size, seq_len] - sequence of item IDs
            hidden: hidden state from previous step (optional)
        
        Returns:
            logits: [batch_size, n_items] - scores for all items
            hidden: updated hidden state
        """
        # Embed items
        embedded = self.item_embedding(item_seq)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Pass through GRU
        output, hidden = self.gru(embedded, hidden)  # output: [batch, seq_len, hidden]
        
        # Take last output
        last_output = output[:, -1, :]  # [batch, hidden]
        
        # Project to item space
        logits = self.fc(last_output)  # [batch, n_items]
        
        return logits, hidden
    
    def predict(self, item_seq, k=10):
        """Predict top-k items for a given sequence."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(item_seq)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_items = torch.topk(probs, k, dim=-1)
        
        return top_items, top_probs


class SessionDataset(torch.utils.data.Dataset):
    """PyTorch dataset for session data."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input': torch.LongTensor(item['input']),
            'target': torch.LongTensor([item['target']])
        }


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Find max sequence length in batch
    max_len = max([len(item['input']) for item in batch])
    
    # Pad sequences
    padded_inputs = []
    targets = []
    
    for item in batch:
        seq = item['input']
        # Pad with zeros
        padded = seq.tolist() + [0] * (max_len - len(seq))
        padded_inputs.append(padded)
        targets.append(item['target'].item())
    
    return {
        'input': torch.LongTensor(padded_inputs),
        'target': torch.LongTensor(targets)
    }


def create_dataloader(data, batch_size, shuffle=True, num_workers=4):
    """Create DataLoader for training/evaluation."""
    dataset = SessionDataset(data)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader