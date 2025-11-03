import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class PopularityRecommender:
    """Recommend most popular items."""
    
    def __init__(self):
        self.item_counts = Counter()
        self.top_items = []
    
    def fit(self, train_data):
        """Count item frequencies."""
        for seq in train_data:
            self.item_counts.update(seq['input'])
            self.item_counts[seq['target']] += 1
        
        # Sort by popularity
        self.top_items = [item for item, _ in self.item_counts.most_common()]
        
        print(f"Popularity model: {len(self.top_items)} unique items")
    
    def predict(self, item_seq, k=10):
        """Return top-k most popular items."""
        # Remove items already in sequence
        seen = set(item_seq)
        recommendations = [item for item in self.top_items if item not in seen]
        return recommendations[:k]
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'item_counts': self.item_counts, 'top_items': self.top_items}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.item_counts = data['item_counts']
            self.top_items = data['top_items']


class ItemKNNRecommender:
    """Item-to-item collaborative filtering using cosine similarity."""
    
    def __init__(self, k_neighbors=50):
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.item_ids = None
        self.user_item_matrix = None
    
    def fit(self, train_data, n_items):
        """Build item-item similarity matrix."""
        print("Building item-item similarity matrix...")
        
        # Create sparse user-item matrix
        user_items = defaultdict(set)
        all_items = set()
        
        for seq in train_data:
            user_id = seq['user_id']
            items = seq['input'] + [seq['target']]
            user_items[user_id].update(items)
            all_items.update(items)
        
        # Convert to matrix (items x users)
        self.item_ids = sorted(all_items)
        item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        
        n_users = len(user_items)
        matrix = np.zeros((len(self.item_ids), n_users))
        
        for user_idx, (user_id, items) in enumerate(user_items.items()):
            for item in items:
                if item in item_to_idx:
                    matrix[item_to_idx[item], user_idx] = 1
        
        # Compute cosine similarity between items
        self.item_similarity = cosine_similarity(matrix)
        
        print(f"Item-KNN: Built similarity matrix for {len(self.item_ids)} items")
    
    def predict(self, item_seq, k=10):
        """Recommend items similar to those in the sequence."""
        if len(item_seq) == 0:
            return []
        
        item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        
        # Get similarities for items in sequence
        scores = np.zeros(len(self.item_ids))
        
        for item in item_seq:
            if item in item_to_idx:
                item_idx = item_to_idx[item]
                scores += self.item_similarity[item_idx]
        
        # Average scores
        scores /= len(item_seq)
        
        # Remove items already in sequence
        seen_indices = [item_to_idx[item] for item in item_seq if item in item_to_idx]
        scores[seen_indices] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(scores)[-k:][::-1]
        recommendations = [self.item_ids[idx] for idx in top_indices]
        
        return recommendations
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'item_similarity': self.item_similarity,
                'item_ids': self.item_ids,
                'k_neighbors': self.k_neighbors
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.item_similarity = data['item_similarity']
            self.item_ids = data['item_ids']
            self.k_neighbors = data['k_neighbors']