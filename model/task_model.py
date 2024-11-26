import torch
import torch.nn.functional as F

# --------------- #
# Node Classifers #
# --------------- #

class NodeClassifier(torch.nn.Module):
    def __init__(self, gnn_model):
        """
        Initializes the NodeClassificationModel with a given GNN model.

        Args:
            gnn_model (torch.nn.Module): A GNN model (e.g., GCN, SAGE, etc.).
        """
        super().__init__()
        self.gnn = gnn_model

    def forward(self, data):
        """
        Forward pass for node classification.

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            torch.Tensor: Log-softmax of class probabilities for each node.
        """
        x = self.gnn(data)
        return F.log_softmax(x, dim=1)