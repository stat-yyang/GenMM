# test_imports.py

import torch
import torch_geometric
from torch_geometric.nn import GCNConv

print("PyTorch version:", torch.__version__)
print("PyTorch Geometric version:", torch_geometric.__version__)
print("GCNConv imported successfully.")