# models.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.generator(z)

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)  # Output is a single value per node

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()
