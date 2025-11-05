"""
NeuralRecoNet Configuration File
Centralized configuration for paths, hyperparameters, and model settings
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_ROOT = PROJECT_ROOT / r"C:\Users\yagne\OneDrive\Desktop\JOB\projects\Recommender-System-with-Neural-Networks\data"
RAW_DATA = DATA_ROOT / r"C:\Users\yagne\OneDrive\Desktop\JOB\projects\Recommender-System-with-Neural-Networks\data\raw_data"
PROCESSED_DATA = DATA_ROOT / "processed"
GRAPHS_DATA = DATA_ROOT / "graphs"

# Model checkpoints
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "experiments"

# Create directories if they don't exist
for directory in [RAW_DATA, PROCESSED_DATA, GRAPHS_DATA, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATA PATHS ====================
# MovieLens 1M Dataset Paths
MOVIELENS_RATINGS = RAW_DATA / "ratings.dat"
MOVIELENS_MOVIES = RAW_DATA / "movies.dat"
MOVIELENS_USERS = RAW_DATA / "users.dat"

# Processed data paths
TRAIN_DATA = PROCESSED_DATA / "train_sessions.pkl"
VAL_DATA = PROCESSED_DATA / "val_sessions.pkl"
TEST_DATA = PROCESSED_DATA / "test_sessions.pkl"
ITEM_MAPPING = PROCESSED_DATA / "item_mapping.pkl"
STATS_FILE = PROCESSED_DATA / "dataset_stats.json"

# Graph data paths
TRAIN_GRAPHS = GRAPHS_DATA / "train_graphs.pkl"
VAL_GRAPHS = GRAPHS_DATA / "val_graphs.pkl"
TEST_GRAPHS = GRAPHS_DATA / "test_graphs.pkl"


# ==================== MODEL HYPERPARAMETERS ====================
@dataclass
class GRUConfig:
    """Configuration for GRU-based Recommender"""
    name: str = "GRU"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs
        }


@dataclass
class GNNConfig:
    """Configuration for SR-GNN Recommender"""
    name: str = "SR-GNN"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    step: int = 1  # Number of message passing steps
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    num_epochs: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'step': self.step,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs
        }


@dataclass
class HybridConfig:
    """Configuration for Hybrid GRU-GNN Recommender"""
    name: str = "Hybrid"
    embedding_dim: int = 128
    gru_hidden_dim: int = 256
    gnn_hidden_dim: int = 256
    num_gru_layers: int = 2
    num_gnn_layers: int = 2
    dropout: float = 0.3
    fusion_method: str = "attention"  # 'concat', 'attention', 'weighted'
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    num_epochs: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'embedding_dim': self.embedding_dim,
            'gru_hidden_dim': self.gru_hidden_dim,
            'gnn_hidden_dim': self.gnn_hidden_dim,
            'num_gru_layers': self.num_gru_layers,
            'num_gnn_layers': self.num_gnn_layers,
            'dropout': self.dropout,
            'fusion_method': self.fusion_method,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs
        }


# ==================== DATA PROCESSING CONFIG ====================
@dataclass
class DataConfig:
    """Configuration for data processing"""
    min_session_length: int = 3
    max_session_length: int = 50
    session_timedelta_minutes: int = 30
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    min_item_support: int = 5  # Minimum number of interactions per item
    random_seed: int = 42


# ==================== TRAINING CONFIG ====================
@dataclass
class TrainingConfig:
    """Configuration for training"""
    device: str = "cuda"  # 'cuda' or 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 5
    gradient_clip: float = 5.0
    log_interval: int = 100
    save_best_only: bool = True
    
    # Top-K evaluation metrics
    top_k_list: list = None
    
    def __post_init__(self):
        if self.top_k_list is None:
            self.top_k_list = [5, 10, 20]


# ==================== MODEL SELECTION ====================
class ModelConfigs:
    """Centralized model configurations"""
    GRU = GRUConfig()
    GNN = GNNConfig()
    HYBRID = HybridConfig()
    
    @staticmethod
    def get_config(model_name: str):
        """Get configuration by model name"""
        configs = {
            'gru': ModelConfigs.GRU,
            'gnn': ModelConfigs.GNN,
            'srgnn': ModelConfigs.GNN,
            'hybrid': ModelConfigs.HYBRID
        }
        return configs.get(model_name.lower())


# ==================== GLOBAL CONFIGS ====================
DATA_CFG = DataConfig()
TRAIN_CFG = TrainingConfig()

# ==================== UTILITY FUNCTIONS ====================
def get_checkpoint_path(model_name: str, epoch: int = None) -> Path:
    """Get checkpoint path for a model"""
    if epoch is None:
        return CHECKPOINTS_DIR / f"{model_name}_best.pth"
    return CHECKPOINTS_DIR / f"{model_name}_epoch_{epoch}.pth"


def get_log_path(model_name: str) -> Path:
    """Get log file path for a model"""
    return LOGS_DIR / f"{model_name}_training.log"


def get_results_path(model_name: str) -> Path:
    """Get results file path for a model"""
    return RESULTS_DIR / f"{model_name}_results.json"


def print_config_summary():
    """Print configuration summary"""
    print("=" * 60)
    print("NeuralRecoNet Configuration Summary")
    print("=" * 60)
    print(f"\n📁 Data Paths:")
    print(f"  Raw Data: {RAW_DATA}")
    print(f"  Processed Data: {PROCESSED_DATA}")
    print(f"  Graphs Data: {GRAPHS_DATA}")
    
    print(f"\n💾 Model Paths:")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    print(f"\n🔧 Data Config:")
    print(f"  Min Session Length: {DATA_CFG.min_session_length}")
    print(f"  Max Session Length: {DATA_CFG.max_session_length}")
    print(f"  Train/Val/Test Split: {DATA_CFG.train_ratio}/{DATA_CFG.val_ratio}/{DATA_CFG.test_ratio}")
    
    print(f"\n🎯 Training Config:")
    print(f"  Device: {TRAIN_CFG.device}")
    print(f"  Top-K Metrics: {TRAIN_CFG.top_k_list}")
    print(f"  Early Stopping Patience: {TRAIN_CFG.early_stopping_patience}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()