import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from cuml.manifold import TSNE as cuTSNE

def extract_embeddings_with_hook(model, data, layer_name, device='cuda'):
    """
    Extract embeddings from a specific layer of a GNN model using hooks.

    Args:
        model (torch.nn.Module): Trained GNN model.
        data (torch_geometric.data.Data): Graph data object.
        layer_name (str): Name of the layer to extract embeddings from.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Node embeddings.
    """
    data = data.to(device)
    model = model.to(device)
    model.eval()

    embeddings = None

    def hook(module, input, output):
        nonlocal embeddings
        embeddings = output

    # Register the hook
    layer = dict(model.named_modules())[layer_name]
    hook_handle = layer.register_forward_hook(hook)

    # Forward pass
    with torch.no_grad():
        model(data)

    # Remove the hook
    hook_handle.remove()

    return embeddings

def plot_embeddings(embeddings_list,
                    labels,
                    model_names,
                    title="GPU-accelerated t-SNE Visualization of Node Embeddings",
                    figsize=(10, 10),  # Adjusted figsize for better multi-row layout
                    perplexity=30,
                    random_state=42,
                    alpha=0.7):
    """
    Visualize 2D embeddings from multiple models, each on its own subplot, using GPU-accelerated t-SNE.

    Args:
        embeddings_list (list): List of embedding matrices (numpy arrays) from different models.
        labels (numpy array): Ground truth labels for coloring the points.
        model_names (list): List of model names corresponding to the embeddings.
        title (str): Title of the plot. Default is "GPU-accelerated t-SNE Visualization of Node Embeddings".
        figsize (tuple): Size of the figure. Default is (10, 10).
        perplexity (int): t-SNE perplexity parameter. Default is 30.
        random_state (int): Random seed for reproducibility. Default is 42.
        alpha (float): Transparency of the scatter points. Default is 0.7.
    """
    if len(embeddings_list) != len(model_names):
        raise ValueError("Number of embeddings must match the number of model names.")

    # Initialize t-SNE
    tsne = cuTSNE(n_components=2, random_state=random_state, perplexity=perplexity)

    # Define a custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#0091ea", "#ffbd59"])

    # Determine the layout of subplots
    num_models = len(embeddings_list)
    n_cols = 2  # Maximum of 2 subplots per row
    n_rows = (num_models + n_cols - 1) // n_cols  # Calculate the number of rows

    # Create subplots with dynamic layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False, sharey=False)
    axes = axes.flatten() if n_rows > 1 else [axes]

    # Loop through embeddings and plot each model's results
    for idx, (embeddings, ax) in enumerate(zip(embeddings_list, axes)):
        # Apply t-SNE transformation
        embeddings_2d = tsne.fit_transform(embeddings)

        # Scatter plot for the current model
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            s=10, c=labels, cmap=custom_cmap, alpha=alpha
        )
        ax.set_title(model_names[idx])

    # Hide unused axes
    for ax in axes[num_models:]:
        ax.axis("off")

    # Set the overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Simulate embeddings for four models
    np.random.seed(42)
    embeddings_gcn = np.random.rand(1000, 128)  # Example embedding from GCN
    embeddings_gat = np.random.rand(1000, 128)  # Example embedding from GAT
    embeddings_gin = np.random.rand(1000, 128)  # Example embedding from GIN
    embeddings_sgc = np.random.rand(1000, 128)  # Example embedding from SGC

    # Simulated labels
    labels = np.random.randint(0, 5, size=1000)

    # Model names
    model_names = ["GCN", "GAT", "GIN", "SGC"]

    # Call the function
    plot_embeddings(
        embeddings_list=[embeddings_gcn, embeddings_gat, embeddings_gin, embeddings_sgc],
        labels=labels,
        model_names=model_names,
        title="t-SNE Visualization of Node Embeddings from GNN Models"
    )

# Example Usage
if __name__ == "__main__":
    # Simulate embeddings for four models
    np.random.seed(42)
    embeddings_gcn = np.random.rand(1000, 128)  # Example embedding from GCN
    embeddings_gat = np.random.rand(1000, 128)  # Example embedding from GAT
    embeddings_gin = np.random.rand(1000, 128)  # Example embedding from GIN
    embeddings_sgc = np.random.rand(1000, 128)  # Example embedding from SGC

    # Simulated labels
    labels = np.random.randint(0, 5, size=1000)

    # Model names
    model_names = ["GCN", "GAT", "GIN", "SGC"]

    # Call the function
    plot_embeddings(
        embeddings_list=[embeddings_gcn, embeddings_gat, embeddings_gin, embeddings_sgc],
        labels=labels,
        model_names=model_names,
        title="t-SNE Visualization of Node Embeddings from GNN Models"
    )