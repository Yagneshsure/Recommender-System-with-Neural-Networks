import os
import torch
import random
import numpy as np

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
MOVIES_DATA_PATH = os.path.join(RAW_DATA_PATH, "ml-1m", "movies.dat")
RATINGS_DATA_PATH = os.path.join(RAW_DATA_PATH, "ml-1m", "ratings.dat")
USER_DATA_PATH = os.path.join(RAW_DATA_PATH, "ml-1m", "users.dat")

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if missing
for path in [PROCESSED_DATA_PATH, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# ---------- TRAINING PARAMETERS ----------
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
HIDDEN_DIM = 128

# ---------- ENVIRONMENT ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------- REPRODUCIBILITY ----------
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Auto set seed
set_seed()

print(f"[CONFIG] Using device: {DEVICE}")
print(f"[CONFIG] Base directory: {BASE_DIR}")
