import torch
from torch_geometric.data import Data

def build_user_item_graph(user_ids, item_ids):
    edge_index = torch.tensor([user_ids, item_ids], dtype=torch.long)
    return Data(edge_index=edge_index)
