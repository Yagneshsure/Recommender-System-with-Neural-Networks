import os

# Define all files with their initial content
FILES = {
    # === Root files ===
    ".gitignore": """# Python
__pycache__/
*.py[cod]
*.pyo
*.so
*.ipynb_checkpoints
venv/
.env/
*.egg-info/
data/raw/
data/processed/
checkpoints/
logs/
results/
.DS_Store
""",

    "requirements.txt": """torch
torch-geometric
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
networkx
pyyaml
streamlit
fastapi
uvicorn
""",

    "setup.py": """from setuptools import setup, find_packages

setup(
    name='NeuralRecoNet',
    version='1.0',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author='Your Name',
    description='Hybrid RNN + GNN Recommender System',
)
""",

    "README.md": """# NeuralRecoNet
NeuralRecoNet — Hybrid Recommender System combining RNNs and GNNs.

## Overview
This project models both sequential and relational dependencies in user–item interactions
using a combination of RNNs (for sequence modeling) and GNNs (for structural reasoning).

## Features
- RNN-based sequence model
- GNN-based user–item graph model
- Hybrid fusion model
- Modular training/evaluation scripts
- Streamlit/FastAPI-ready for deployment
""",

    "LICENSE": "MIT License\n\nCopyright (c) 2025",

    # === Configs ===
    "configs/__init__.py": "",
    "configs/config.py": """# Configuration settings
DATA_PATH = 'data/processed/'
RAW_DATA_PATH = 'data/raw/'
CHECKPOINT_DIR = 'checkpoints/'
LOG_DIR = 'logs/'
RESULTS_DIR = 'results/'

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
DEVICE = 'cuda'  # or 'cpu'
""",

    "configs/model_configs.yaml": """rnn:
  embedding_dim: 64
  hidden_dim: 128
  rnn_type: 'gru'
gnn:
  embedding_dim: 64
  num_layers: 3
hybrid:
  fusion_dim: 64
  output_dim: 128
""",

    # === Models ===
    "models/__init__.py": "",
    "models/rnn_recommender.py": """import torch
import torch.nn as nn

class RNNRecommender(nn.Module):
    \"\"\"Recurrent model (LSTM/GRU) for sequential recommendations\"\"\"
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
""",

    "models/gnn_recommender.py": """import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN

class GNNRecommender(nn.Module):
    \"\"\"Graph-based recommender using LightGCN\"\"\"
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.model = LightGCN(num_users + num_items, embedding_dim, num_layers=num_layers)

    def forward(self, edge_index):
        return self.model(edge_index)
""",

    "models/hybrid_recommender.py": """import torch
import torch.nn as nn

class HybridRecommender(nn.Module):
    \"\"\"Combines RNN and GNN embeddings for hybrid recommendations\"\"\"
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
""",

    # === Utils ===
    "utils/__init__.py": "",
    "utils/data_loader.py": """import pandas as pd
import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    \"\"\"Dataset for user–item interactions\"\"\"
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user, item, rating = self.df.iloc[idx]
        return torch.tensor(user), torch.tensor(item), torch.tensor(rating)
""",

    "utils/metrics.py": """import numpy as np

def recall_at_k(actual, predicted, k):
    predicted = predicted[:k]
    return int(any(a in predicted for a in actual))

def ndcg_at_k(actual, predicted, k):
    predicted = predicted[:k]
    if not actual:
        return 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            return 1 / np.log2(i + 2)
    return 0.0
""",

    "utils/graph_utils.py": """import torch
from torch_geometric.data import Data

def build_user_item_graph(user_ids, item_ids):
    edge_index = torch.tensor([user_ids, item_ids], dtype=torch.long)
    return Data(edge_index=edge_index)
""",

    # === Scripts ===
    "scripts/train_rnn.py": """from models.rnn_recommender import RNNRecommender
print('Training RNN model (placeholder)')
""",

    "scripts/train_gnn.py": """from models.gnn_recommender import GNNRecommender
print('Training GNN model (placeholder)')
""",

    "scripts/train_hybrid.py": """from models.hybrid_recommender import HybridRecommender
print('Training Hybrid model (placeholder)')
""",

    "scripts/evaluate.py": """from utils.metrics import recall_at_k, ndcg_at_k
print('Evaluating model (placeholder)')
""",

    # === Tests ===
    "tests/__init__.py": "",
    "tests/test_models.py": """def test_imports():
    from models.rnn_recommender import RNNRecommender
    assert RNNRecommender is not None
""",

    "tests/test_data_loader.py": """def test_data_loader_import():
    from utils.data_loader import InteractionDataset
    assert InteractionDataset is not None
""",

    # === Notebooks placeholders ===
    "notebooks/01_data_exploration.ipynb": "",
    "notebooks/02_model_testing.ipynb": "",
    "notebooks/03_results_analysis.ipynb": "",

    # === Gitkeep files ===
    "data/.gitkeep": "",
    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "checkpoints/.gitkeep": "",
    "logs/.gitkeep": "",
    "results/.gitkeep": "",
}

# Create directories if needed and write files
for path, content in FILES.items():
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ All NeuralRecoNet files created successfully!")
