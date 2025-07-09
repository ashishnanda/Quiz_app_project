import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
import pandas as pd
import networkx as nx

# ---------------------------
# CONFIGURATION
# ---------------------------

INPUT_DIM = 16         # Size of input feature vector for each node
HIDDEN_DIM = 32        # Hidden layer size in GIN
EMBEDDING_DIM = 8      # Output embedding size
NUM_LAYERS = 4         # Number of GIN layers
EPOCHS = 50
LEARNING_RATE = 0.01
DECAY_RATE = 0.95
VISUALIZE_TSNE = False  # Set to True to use t-SNE instead of PCA

# ---------------------------
# STEP 1: Load Your DataFrame
# ---------------------------

# Sample for demo (replace this with your actual graph_df)
# graph_df = pd.DataFrame({'id': [...], 'target': [...]})

# Ensure all nodes are treated as strings
graph_df['id'] = graph_df['id'].astype(str)
graph_df['target'] = graph_df['target'].astype(str)

# ---------------------------
# STEP 2: Build Graph & PyG Data
# ---------------------------

# Create graph using NetworkX
G = nx.from_pandas_edgelist(graph_df, source='id', target='target')
nodes = sorted(G.nodes())
node_to_index = {node: i for i, node in enumerate(nodes)}

# Convert edges to indexed format
indexed_edges = [(node_to_index[u], node_to_index[v]) for u, v in G.edges()]
edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Undirected

NUM_NODES = len(nodes)

# Initialize input features: one-hot-like identity matrix (or make it learnable)
x = torch.eye(NUM_NODES, INPUT_DIM)

# Create PyG data object
data = Data(x=x, edge_index=edge_index)

# ---------------------------
# STEP 3: Define GIN Model
# ---------------------------

class GINModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            nn_func = nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_func))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return self.output_proj(x)

# ---------------------------
# STEP 4: Contrastive Loss
# ---------------------------

def contrastive_loss(emb, edge_index, num_nodes):
    pos_edges = edge_index.t()
    neg_edges = negative_sampling(edge_index=edge_index, num_nodes=num_nodes).t()

    # Filter negative edges to exclude existing positive edges
    existing = set((u.item(), v.item()) for u, v in pos_edges)
    neg_edges = torch.stack([
        pair for pair in neg_edges
        if (pair[0].item(), pair[1].item()) not in existing
    ])

    pos_sim = (emb[pos_edges[:, 0]] * emb[pos_edges[:, 1]]).sum(dim=1)
    neg_sim = (emb[neg_edges[:, 0]] * emb[neg_edges[:, 1]]).sum(dim=1)

    loss = -torch.log(torch.sigmoid(pos_sim)).mean() - torch.log(1 - torch.sigmoid(neg_sim)).mean()
    return loss

# ---------------------------
# STEP 5: Visualize Embeddings
# ---------------------------

def visualize(embeddings, labels, epoch, method='pca'):
    reducer = TSNE(n_components=2) if method == 'tsne' else PCA(n_components=2)
    emb_2d = reducer.fit_transform(embeddings.detach().cpu().numpy())

    plt.figure(figsize=(10, 8))
    texts = []
    for i, label in enumerate(labels):
        plt.scatter(emb_2d[i, 0], emb_2d[i, 1])
        texts.append(plt.text(emb_2d[i, 0], emb_2d[i, 1], str(label), fontsize=9))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
    plt.title(f"GIN Embeddings (Epoch {epoch})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

# ---------------------------
# STEP 6: Train the GIN Model
# ---------------------------

model = GINModel(INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM, NUM_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data.x, data.edge_index)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    loss = contrastive_loss(embeddings, data.edge_index, NUM_NODES)
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"Epoch {epoch:02d} | Contrastive Loss: {loss.item():.4f}")

    if epoch % 10 == 0:
        visualize(embeddings, nodes, epoch, method='tsne' if VISUALIZE_TSNE else 'pca')