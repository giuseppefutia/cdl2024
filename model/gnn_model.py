import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

# --------- #
# GCN Model #
# --------- #

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --------- #
# GAT Model #
# --------- #

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ---------- #
# Sage Model #
# ---------- #

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --------- #
# GIN Model #
# --------- #

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Linear(input_dim, hidden_dim))
        self.conv2 = GINConv(torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
