"""
Script 2: Train a Graph Isomorphism Network (GIN) from structure alone
======================================================================
- Uses PyTorch Geometric for GNN modeling
- Inputs are dummy features (constant or one-hot)
- Displays weights, loss, and embeddings at each epoch
- Visualizes embeddings using PCA
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# --- Load your edge and node data ---
edges = pd.read_csv("edges.csv")  # source, target
nodes = pd.read_csv("nodes.csv")  # ID column

# --- Convert to PyG format ---
from torch_geometric.utils import from_networkx
import networkx as nx
G = nx.from_pandas_edgelist(edges, source='source', target='target')
data = from_networkx(G)

# Dummy input features (all 1s)
num_nodes = data.num_nodes
data.x = torch.ones((num_nodes, 5))  # 5-dimensional dummy input

# --- Define GIN Model ---
from torch_geometric.nn import Sequential
from torch.nn import Linear, ReLU

class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GINConv(Sequential(Linear(5, 16), ReLU(), Linear(16, 16)))
        self.conv2 = GINConv(Sequential(Linear(16, 16), ReLU(), Linear(16, 2)))  # 2D for easy plotting

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

model = GIN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Dummy target: nodes closer together in graph should be closer in embedding
target = torch.arange(num_nodes)

# --- Training Loop ---
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # Self-supervised loss: pull all embeddings together
    loss = torch.norm(out @ out.T - torch.eye(num_nodes))  # (not ideal but shows training working)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    # Visualize embeddings every 10 epochs
    if epoch % 10 == 0:
        emb = out.detach().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(emb[:, 0], emb[:, 1])
        for i, label in enumerate(nodes['ID']):
            plt.text(emb[i, 0] + 0.01, emb[i, 1] + 0.01, label, fontsize=8)
        plt.title(f"GIN Embeddings at Epoch {epoch}")
        plt.grid(True)
        plt.show()