import torch
import torch.nn.functional as F

# ---------------- #
# Node Classifiers #
# ---------------- #

class NodeClassifier(torch.nn.Module):
    def __init__(self, gnn_model):
        """
        Initializes the NodeClassifier with a given GNN model.

        Args:
            gnn_model (torch.nn.Module): A GNN model (e.g., GCN, SAGE, GAT, etc.).
                                         This model computes node embeddings.
        """
        super().__init__()
        self.gnn = gnn_model  # Backbone GNN for learning node representations

    def forward(self, x, edge_index):
        """
        Forward pass for the node classification task.

        Args:
            x (torch.Tensor): Node feature matrix of shape (num_nodes, num_features).
                              Each row represents the feature vector for a node.
            edge_index (torch.Tensor): Graph connectivity in COO format,
                                        with shape (2, num_edges).
                                        Each column represents an edge (source, target).

        Returns:
            torch.Tensor: Log-softmax of class probabilities for each node,
                          with shape (num_nodes, num_classes).
        """
        # Compute node embeddings using the GNN model
        x = self.gnn(x, edge_index)
        
        # Apply log-softmax to output class probabilities for each node
        return F.log_softmax(x, dim=1)
