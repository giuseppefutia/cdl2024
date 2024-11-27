import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GraphConv

class BaseGraphModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, conv_layer, **conv_kwargs):
        """
        A generic graph model with two layers and ReLU activation.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden feature dimension.
            out_dim (int): Output feature dimension.
            conv_layer (class): Graph convolution layer class (e.g., GCNConv, SAGEConv, GATConv, GINConv).
            **conv_kwargs: Additional keyword arguments for the convolution layer.
        """
        super(BaseGraphModel, self).__init__()
        self.is_gat = conv_layer == GATConv  # Special handling for GAT

        # Initialize the first convolution layer
        self.conv1 = conv_layer(input_dim, hidden_dim, **conv_kwargs)
        
        # Adjust hidden_dim if GAT and concat=True
        if self.is_gat:
            hidden_dim *= conv_kwargs["heads"]
            conv_kwargs["heads"] = 1

        # Initialize the second convolution layer
        self.conv2 = conv_layer(hidden_dim, out_dim, **conv_kwargs) if self.is_gat else conv_layer(hidden_dim, out_dim, **conv_kwargs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --------- #
# GCN Model #
# --------- #

class GCN(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim, add_self_loops=True):
        super(GCN, self).__init__(
            input_dim, 
            hidden_dim, 
            out_dim, 
            GCNConv,
            add_self_loops=add_self_loops
        )

# --------- #
# GraphConv #
# --------- #

class GraphConvModel(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GraphConvModel, self).__init__(
            input_dim,
            hidden_dim,
            out_dim,
            GraphConv)

# --------- #
# GAT Model #
# --------- #

class GAT(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim, num_heads=8, add_self_loops=True, dropout=0.6):
        super(GAT, self).__init__(
            input_dim,
            hidden_dim,
            out_dim,
            GATConv,
            heads=num_heads,
            add_self_loops=add_self_loops,
            dropout=dropout
        )

# ---------- #
# SAGE Model #
# ---------- #

class SAGE(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(SAGE, self).__init__(
            input_dim,
            hidden_dim,
            out_dim,
            SAGEConv)

# --------- #
# GIN Model #
# --------- #

class GIN(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GIN, self).__init__(
            input_dim, 
            hidden_dim, 
            out_dim, 
            GINConv,
            torch.nn.Linear(input_dim, out_dim)
        )
