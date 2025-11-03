import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

class SessionDataProcessor:
    """Preprocess MovieLens data into sessions for sequential recommendation."""
    
    def __init__(self, config):
        self.config = config
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def load_movielens(self, filepath):
        """Load MovieLens 1M dataset."""
        print(f"Loading data from {filepath}...")
        
        # MovieLens format: UserID::MovieID::Rating::Timestamp
        df = pd.read_csv(
            filepath,
            sep='::',
            engine='python',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        print(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} users")
        print(f"and {df['item_id'].nunique()} items")
        
        return df
    
    def create_sessions(self, df):
        """Split user history into sessions based on time gaps."""
        print("\nCreating sessions...")
        gap = timedelta(minutes=self.config['data']['session_gap_minutes'])
        min_len = self.config['data']['min_session_length']
        max_len = self.config['data']['max_session_length']
        
        sessions = []
        
        for user_id, group in df.groupby('user_id'):
            group = group.sort_values('timestamp')
            
            session_items = []
            prev_time = None
            
            for _, row in group.iterrows():
                curr_time = row['timestamp']
                
                # Start new session if time gap is too large
                if prev_time and (curr_time - prev_time) > gap:
                    if len(session_items) >= min_len:
                        # Truncate long sessions
                        if len(session_items) > max_len:
                            session_items = session_items[:max_len]
                        sessions.append({
                            'user_id': user_id,
                            'items': session_items.copy(),
                            'timestamps': [prev_time] * len(session_items)
                        })
                    session_items = []
                
                session_items.append(row['item_id'])
                prev_time = curr_time
            
            # Add last session
            if len(session_items) >= min_len:
                if len(session_items) > max_len:
                    session_items = session_items[:max_len]
                sessions.append({
                    'user_id': user_id,
                    'items': session_items.copy(),
                    'timestamps': [prev_time] * len(session_items)
                })
        
        print(f"Created {len(sessions)} sessions")
        return sessions
    
    def encode_items(self, sessions):
        """Encode item IDs to continuous integers."""
        print("\nEncoding items...")
        
        # Collect all unique items
        all_items = []
        for session in sessions:
            all_items.extend(session['items'])
        
        # Fit encoder
        self.item_encoder.fit(all_items)
        n_items = len(self.item_encoder.classes_)
        
        # Encode sessions
        for session in sessions:
            session['items'] = self.item_encoder.transform(session['items']).tolist()
        
        print(f"Encoded {n_items} unique items")
        return sessions, n_items
    
    def create_sequences(self, sessions):
        """Create input-target pairs for training."""
        sequences = []
        
        for session in sessions:
            items = session['items']
            # For each position, predict the next item
            for i in range(len(items) - 1):
                sequences.append({
                    'input': items[:i+1],  # Items so far
                    'target': items[i+1],  # Next item to predict
                    'user_id': session['user_id']
                })
        
        return sequences
    
    def split_data(self, sequences):
        """Split into train/val/test sets."""
        print("\nSplitting data...")
        
        np.random.shuffle(sequences)
        n = len(sequences)
        
        train_size = int(n * self.config['data']['train_ratio'])
        val_size = int(n * self.config['data']['val_ratio'])
        
        train_data = sequences[:train_size]
        val_data = sequences[train_size:train_size + val_size]
        test_data = sequences[train_size + val_size:]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def process(self):
        """Main preprocessing pipeline."""
        # Load data
        df = self.load_movielens(self.config['data']['raw_path'])
        
        # Create sessions
        sessions = self.create_sessions(df)
        
        # Encode items
        sessions, n_items = self.encode_items(sessions)
        
        # Create sequences
        sequences = self.create_sequences(sessions)
        
        # Split data
        train_data, val_data, test_data = self.split_data(sequences)
        
        # Save processed data
        output_path = Path(self.config['data']['processed_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'train.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open(output_path / 'val.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        with open(output_path / 'test.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        # Save metadata
        metadata = {
            'n_items': n_items,
            'item_encoder': self.item_encoder
        }
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nData saved to {output_path}")
        print(f"Number of unique items: {n_items}")
        
        return n_items


def load_processed_data(config, split='train'):
    """Load preprocessed data."""
    path = Path(config['data']['processed_path']) / f'{split}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_metadata(config):
    """Load metadata (n_items, encoders)."""
    path = Path(config['data']['processed_path']) / 'metadata.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)