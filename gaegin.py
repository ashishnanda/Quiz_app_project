import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def build_graph_from_df(graph_df, feature_dim=4):
    G = nx.from_pandas_edgelist(graph_df, source='id', target='target')
    data = from_networkx(G)
    data.x = torch.ones((data.num_nodes, feature_dim))  # constant features
    return data


class GINGraphAutoEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False)
            )
            self.convs.append(GINConv(mlp))
            input_dim = hidden_dim

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        return F.normalize(x, p=2, dim=1)

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)


def visualize_embeddings(embeddings, epoch):
    emb_np = embeddings.detach().cpu().numpy()
    reduced = PCA(n_components=2).fit_transform(emb_np)
    plt.figure(figsize=(6, 6))
    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, str(i), fontsize=8)
    plt.title(f"GIN Embeddings at Epoch {epoch}")
    plt.grid(True)
    plt.show()


def train_autoencoder(
    data, model, epochs=100, lr=0.01,
    use_negative_sampling=True, visualize_every=25
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)

        pos_pred = model.decode(z, data.edge_index)
        pos_label = torch.ones(pos_pred.size(0))

        if use_negative_sampling:
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_pred.size(0)
            )
            neg_pred = model.decode(z, neg_edge_index)
            neg_label = torch.zeros(neg_pred.size(0))

            pred = torch.cat([pos_pred, neg_pred])
            label = torch.cat([pos_label, neg_label])
        else:
            pred = pos_pred
            label = pos_label

        loss = F.binary_cross_entropy_with_logits(pred, label)
        loss.backward()
        optimizer.step()

        if epoch % visualize_every == 0 or epoch == epochs:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
            visualize_embeddings(z, epoch)

    return z


# --- Example usage ---
# Load your graph_df here:
# graph_df = pd.read_csv("your_edges.csv")

# Example dummy graph for testing
graph_df = pd.DataFrame({'id': [0, 1, 2, 3, 4], 'target': [1, 2, 3, 4, 0]})

# Build PyG data
data = build_graph_from_df(graph_df, feature_dim=4)

# Initialize model
model = GINGraphAutoEncoder(input_dim=4, hidden_dim=8, num_layers=2)

# Train
embeddings = train_autoencoder(
    data, model,
    epochs=100,
    lr=0.01,
    use_negative_sampling=False,
    visualize_every=20
)