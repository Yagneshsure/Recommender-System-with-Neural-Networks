"""
Global configuration file for NeuralRecoNet
"""

DATA_PATH = "data/processed/"
RAW_DATA_PATH = "data/raw/"
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
DEVICE = "cuda"  # or "cpu"
