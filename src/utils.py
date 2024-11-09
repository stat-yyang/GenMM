# utils.py

import torch
from torch_geometric.data import Data

def create_graph_data(modalities):
    """
    Creates a fully connected undirected graph from modality data.
    """
    num_nodes = modalities.shape[0]
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it undirected
    data = Data(x=modalities, edge_index=edge_index)
    return data

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized modalities in batches.
    """
    X_batch = torch.stack([item['X'] for item in batch])
    N_list_batch = [[item['N_list'][i] for item in batch] for i in range(len(batch[0]['N_list']))]
    M_list_batch = [[item['M_list'][i] for item in batch] for i in range(len(batch[0]['M_list']))]
    Y_batch = torch.stack([item['Y'] for item in batch])

    # Stack modality data
    N_list_batch = [torch.stack(N_batch) for N_batch in N_list_batch]
    M_list_batch = [torch.stack(M_batch) for M_batch in M_list_batch]

    return {'X': X_batch, 'N_list': N_list_batch, 'M_list': M_list_batch, 'Y': Y_batch}
