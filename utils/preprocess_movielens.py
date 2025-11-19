"""
Data Preprocessing for MovieLens-1M Dataset
Converts raw .dat files into processed tensors ready for model training
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from collections import defaultdict

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import (
    RATINGS_DATA_PATH, 
    MOVIES_DATA_PATH, 
    USER_DATA_PATH,
    PROCESSED_DATA_PATH
)


class MovieLensPreprocessor:
    """
    Preprocesses MovieLens-1M dataset for recommendation models
    
    Tasks:
    1. Load raw .dat files
    2. Create user/item mappings (re-index to 0-based)
    3. Split train/validation/test sets
    4. Create sequences for RNN model
    5. Build interaction matrix for GNN
    """
    
    def __init__(self, test_ratio=0.2, val_ratio=0.1, min_interactions=5):
        """
        Args:
            test_ratio: Proportion of data for testing
            val_ratio: Proportion of training data for validation
            min_interactions: Minimum interactions per user to keep
        """
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.min_interactions = min_interactions
        
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        
        self.n_users = 0
        self.n_items = 0
        
    def load_data(self):
        """Load raw MovieLens data files"""
        print("[PREPROCESSING] Loading raw data...")
        
        # Load ratings: UserID::MovieID::Rating::Timestamp
        self.ratings_df = pd.read_csv(
            RATINGS_DATA_PATH,
            sep='::',
            engine='python',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Load movies: MovieID::Title::Genres
        self.movies_df = pd.read_csv(
            MOVIES_DATA_PATH,
            sep='::',
            engine='python',
            names=['item_id', 'title', 'genres'],
            encoding='latin-1'
        )
        
        # Load users: UserID::Gender::Age::Occupation::Zip-code
        self.users_df = pd.read_csv(
            USER_DATA_PATH,
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
            encoding='latin-1'
        )
        
        print(f"✓ Loaded {len(self.ratings_df)} ratings")
        print(f"✓ Loaded {len(self.movies_df)} movies")
        print(f"✓ Loaded {len(self.users_df)} users")
        
    def filter_data(self):
        """Remove users/items with too few interactions"""
        print(f"[PREPROCESSING] Filtering users with < {self.min_interactions} interactions...")
        
        # Count interactions per user
        user_counts = self.ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        
        # Filter ratings
        original_size = len(self.ratings_df)
        self.ratings_df = self.ratings_df[self.ratings_df['user_id'].isin(valid_users)]
        
        print(f"✓ Kept {len(self.ratings_df)}/{original_size} ratings")
        print(f"✓ Kept {self.ratings_df['user_id'].nunique()} users")
        print(f"✓ Kept {self.ratings_df['item_id'].nunique()} items")
        
    def create_mappings(self):
        """Create user and item ID mappings (0-indexed for embeddings)"""
        print("[PREPROCESSING] Creating ID mappings...")
        
        # Get unique users and items
        unique_users = sorted(self.ratings_df['user_id'].unique())
        unique_items = sorted(self.ratings_df['item_id'].unique())
        
        # Create mappings
        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}
        
        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)
        
        # Map IDs in dataframe
        self.ratings_df['user_idx'] = self.ratings_df['user_id'].map(self.user2idx)
        self.ratings_df['item_idx'] = self.ratings_df['item_id'].map(self.item2idx)
        
        print(f"✓ Created mappings: {self.n_users} users, {self.n_items} items")
        
    def temporal_split(self):
        """
        Split data temporally (more realistic for sequential models):
        - For each user, last 20% interactions → test
        - From remaining, last 10% → validation
        - Rest → train
        """
        print("[PREPROCESSING] Creating temporal train/val/test split...")
        
        train_data = []
        val_data = []
        test_data = []
        
        # Sort by timestamp for each user
        self.ratings_df = self.ratings_df.sort_values(['user_idx', 'timestamp'])
        
        for user_idx in range(self.n_users):
            user_ratings = self.ratings_df[self.ratings_df['user_idx'] == user_idx]
            
            n_ratings = len(user_ratings)
            n_test = max(1, int(n_ratings * self.test_ratio))
            n_val = max(1, int((n_ratings - n_test) * self.val_ratio))
            
            # Split
            test_data.append(user_ratings.iloc[-n_test:])
            val_data.append(user_ratings.iloc[-(n_test + n_val):-n_test])
            train_data.append(user_ratings.iloc[:-(n_test + n_val)])
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        print(f"✓ Train: {len(train_df)} interactions")
        print(f"✓ Val: {len(val_df)} interactions")
        print(f"✓ Test: {len(test_df)} interactions")
        
        return train_df, val_df, test_df
    
    def create_sequences(self, df, seq_length=10):
        """
        Create sequential data for RNN model
        Returns: list of (sequence, target) tuples
        """
        sequences = []
        
        for user_idx in range(self.n_users):
            user_items = df[df['user_idx'] == user_idx]['item_idx'].tolist()
            
            # Create sliding window sequences
            for i in range(len(user_items) - seq_length):
                seq = user_items[i:i + seq_length]
                target = user_items[i + seq_length]
                sequences.append((user_idx, seq, target))
        
        return sequences
    
    def create_interaction_matrix(self, df):
        """
        Create user-item interaction matrix (for GNN)
        Returns: sparse matrix of shape (n_users, n_items)
        """
        from scipy.sparse import csr_matrix
        
        users = df['user_idx'].values
        items = df['item_idx'].values
        ratings = df['rating'].values
        
        # Binary interactions (1 if interacted, 0 otherwise)
        # Can also use actual ratings if needed
        interaction_matrix = csr_matrix(
            (np.ones(len(users)), (users, items)),
            shape=(self.n_users, self.n_items)
        )
        
        return interaction_matrix
    
    def save_processed_data(self, train_df, val_df, test_df):
        """Save all processed data to disk"""
        print("[PREPROCESSING] Saving processed data...")
        
        # Create sequences for RNN (sequence length = 10)
        train_sequences = self.create_sequences(train_df, seq_length=10)
        val_sequences = self.create_sequences(val_df, seq_length=10)
        test_sequences = self.create_sequences(test_df, seq_length=10)
        
        # Create interaction matrices for GNN
        train_matrix = self.create_interaction_matrix(train_df)
        val_matrix = self.create_interaction_matrix(val_df)
        test_matrix = self.create_interaction_matrix(test_df)
        
        # Save everything
        data_dict = {
            # Mappings
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'item2idx': self.item2idx,
            'idx2item': self.idx2item,
            'n_users': self.n_users,
            'n_items': self.n_items,
            
            # DataFrames
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            
            # Sequences (for RNN)
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            
            # Matrices (for GNN)
            'train_matrix': train_matrix,
            'val_matrix': val_matrix,
            'test_matrix': test_matrix,
        }
        
        save_path = os.path.join(PROCESSED_DATA_PATH, 'movielens_processed.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"✓ Saved to: {save_path}")
        print(f"✓ Train sequences: {len(train_sequences)}")
        print(f"✓ Val sequences: {len(val_sequences)}")
        print(f"✓ Test sequences: {len(test_sequences)}")
        
    def run(self):
        """Execute full preprocessing pipeline"""
        print("="*50)
        print("MovieLens-1M Preprocessing Pipeline")
        print("="*50)
        
        self.load_data()
        self.filter_data()
        self.create_mappings()
        train_df, val_df, test_df = self.temporal_split()
        self.save_processed_data(train_df, val_df, test_df)
        
        print("="*50)
        print("✓ Preprocessing Complete!")
        print("="*50)
        print(f"Dataset Statistics:")
        print(f"  Users: {self.n_users}")
        print(f"  Items: {self.n_items}")
        print(f"  Density: {len(train_df) / (self.n_users * self.n_items) * 100:.4f}%")
        

def load_processed_data():
    """
    Utility function to load preprocessed data
    Returns: dictionary with all processed data
    """
    save_path = os.path.join(PROCESSED_DATA_PATH, 'movielens_processed.pkl')
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Processed data not found at {save_path}. "
            "Run preprocessing first!"
        )
    
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded processed data from {save_path}")
    return data


if __name__ == "__main__":
    # Run preprocessing
    preprocessor = MovieLensPreprocessor(
        test_ratio=0.2,
        val_ratio=0.1,
        min_interactions=5
    )
    preprocessor.run()