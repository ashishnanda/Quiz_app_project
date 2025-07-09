import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import from_networkx
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Create graph: node 0 and 1 are structural duplicates
G = nx.Graph()
edges = [
    (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 5), (3, 5), (4, 5)
]
G.add_edges_from(edges)

# 2. Convert to PyG Data
data = from_networkx(G)
num_nodes = data.num_nodes
data.x = torch.ones((num_nodes, 4))  # all-ones input features

# 3. Define GIN with no bias, no dropout, no batchnorm
class MinimalGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
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
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings

# 4. Set weights to 1
def reset_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.ones_(module.weight)

torch.manual_seed(42)
model = MinimalGIN(input_dim=4, hidden_dim=4, num_layers=2)
model.apply(reset_weights)

# 5. Inference
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

# 6. Visualize via PCA
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings.numpy())
plt.figure(figsize=(6, 6))
for i in range(num_nodes):
    plt.scatter(emb_2d[i, 0], emb_2d[i, 1])
    plt.text(emb_2d[i, 0] + 0.01, emb_2d[i, 1] + 0.01, str(i), fontsize=9)
plt.title("GIN Embeddings (No Bias/Dropout/BatchNorm)")
plt.grid(True)
plt.show()

# 7. Print distance between node 0 and 1 (should be ~0)
diff = torch.norm(embeddings[0] - embeddings[1]).item()
print("Embedding difference between node 0 and 1:", diff)