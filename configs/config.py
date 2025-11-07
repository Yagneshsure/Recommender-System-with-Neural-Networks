# Configuration settings

### RAW Data path ###
RAW_DATA_PATH = r'data\raw'
MOVIES_DATA_PATH = r"data\raw\ml-1m\movies.dat"
RATINGS_DATA_PATH = r"data\raw\ml-1m\ratings.dat"
USER_DATA_PATH = r"data\raw\ml-1m\users.dat"

###### Processed data path #######
DATA_PATH = r'data\processed'


CHECKPOINT_DIR = 'checkpoints/'
LOG_DIR = 'logs/'
RESULTS_DIR = 'results/'

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
DEVICE = 'cuda'  # or 'cpu'
