import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        # first layer: transforms raw input into hidden space
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Second layer: allows atoms to learn about neighbors of neighbors
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Output layer: final regression head
        self.out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(0.2)
        )

    def forward(self, x, edge_index, batch):
        # run the GNN layers (Message Passing)
        # x: node features, edge_index: connection map
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # global mean pool (readout)
        # This collapses all atom vectors into 1 vector for the whole molecule
        x = global_mean_pool(x, batch) 

        # Step 3: final output layer (regression head)
        return self.out(x)

# --- Configuration ---
# QM9 has 11 node features (Atomic number, aromaticity, etc.)
# Let's say we want to predict 1 target property (e.g. Energy)
# model = GNNModel(in_channels=11, hidden_channels=64, out_channels=1)
# print(model)