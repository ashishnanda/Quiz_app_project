"""
GIN Graph Embedding with Contrastive Loss
=========================================
- Uses a GIN model with dummy input features (constant)
- Learns embeddings by maximizing similarity for connected nodes (positive pairs)
  and minimizing it for randomly chosen unconnected nodes (negative pairs)
- Tracks and prints loss and visualizes embeddings at each step
"""

import torch
import torch.nn.functional as F
import torch_geometric
import matplotlib.pyplot as plt
import pandas as pd
import random
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.utils import from_networkx, negative_sampling
import networkx as nx
from sklearn.decomposition import PCA

# --- Load and prepare graph ---
edges = pd.read_csv("edges.csv")  # must have 'source' and 'target'
nodes = pd.read_csv("nodes.csv")  # must have 'ID'

G = nx.from_pandas_edgelist(edges, source='source', target='target')
node_id_map = {node: i for i, node in enumerate(G.nodes())}
edges = [(node_id_map[u], node_id_map[v]) for u, v in G.edges()]
G = nx.Graph()
G.add_edges_from(edges)
data = from_networkx(G)
num_nodes = data.num_nodes

# --- Dummy input features (constant) ---
data.x = torch.ones((num_nodes, 5))  # could also try one-hot or degree

# --- GIN model ---
class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GINConv(Sequential(Linear(5, 16), ReLU(), Linear(16, 16)))
        self.conv2 = GINConv(Sequential(Linear(16, 16), ReLU(), Linear(16, 16)))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x  # final embeddings

model = GIN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Contrastive loss function ---
def contrastive_loss(embeddings, edge_index, num_neg_samples=5):
    """
    Compute contrastive loss with positive and negative node pairs
    - embeddings: Tensor [N, D]
    - edge_index: Tensor [2, E]
    """
    pos_u, pos_v = edge_index
    pos_sim = F.cosine_similarity(embeddings[pos_u], embeddings[pos_v])

    # Sample random negative pairs
    neg_edges = negative_sampling(edge_index, num_nodes=embeddings.shape[0], num_neg_samples=pos_u.size(0))
    neg_u, neg_v = neg_edges
    neg_sim = F.cosine_similarity(embeddings[neg_u], embeddings[neg_v])

    # InfoNCE-style loss
    loss = -torch.log(torch.sigmoid(pos_sim)).mean() - torch.log(1 - torch.sigmoid(neg_sim)).mean()
    return loss

# --- Training loop ---
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data)
    loss = contrastive_loss(embeddings, data.edge_index)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch:02d} | Contrastive Loss: {loss.item():.4f}")

    # --- Visualize every 10 epochs ---
    if epoch % 10 == 0:
        emb = embeddings.detach().numpy()
        emb_2d = PCA(n_components=2).fit_transform(emb)
        plt.figure(figsize=(8, 6))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c='skyblue', edgecolors='black')
        for i, node_label in enumerate(node_id_map.keys()):
            plt.text(emb_2d[i, 0]+0.01, emb_2d[i, 1]+0.01, str(node_label), fontsize=8)
        plt.title(f"GIN Embeddings (Epoch {epoch})")
        plt.grid(True)
        plt.show()