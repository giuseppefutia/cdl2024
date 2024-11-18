import matplotlib.pyplot as plt
import numpy as np
from cuml.manifold import TSNE as cuTSNE

def plot_embeddings(embeddings_list, 
                               labels, 
                               model_names, 
                               title="GPU-accelerated t-SNE Visualization of Node Embeddings",
                               figsize=(15, 5), 
                               perplexity=30, 
                               random_state=42, 
                               alpha=0.7):
    # WARNING: This function can be run only on GPU (remember to activate if you use it in colab)
    """
    Visualize 2D embeddings from multiple models, each on its own subplot, using GPU-accelerated t-SNE.
    
    Args:
        embeddings_list (list): List of embedding matrices (numpy arrays) from different models.
        labels (numpy array): Ground truth labels for coloring the points.
        model_names (list): List of model names corresponding to the embeddings.
        title (str): Title of the plot. Default is "GPU-accelerated t-SNE Visualization of Node Embeddings".
        figsize (tuple): Size of the figure. Default is (15, 5).
        perplexity (int): t-SNE perplexity parameter. Default is 30.
        random_state (int): Random seed for reproducibility. Default is 42.
        alpha (float): Transparency of the scatter points. Default is 0.7.
    """
    if len(embeddings_list) != len(model_names):
        raise ValueError("Number of embeddings must match the number of model names.")
    
    # Initialize t-SNE
    tsne = cuTSNE(n_components=2, random_state=random_state, perplexity=perplexity)

    # Create subplots
    num_models = len(embeddings_list)
    fig, axes = plt.subplots(1, num_models, figsize=figsize, sharex=False, sharey=False)
    
    if num_models == 1:  # Handle single-model case
        axes = [axes]
    
    # Loop through embeddings and plot each model's results
    for idx, (embeddings, ax) in enumerate(zip(embeddings_list, axes)):
        # Apply t-SNE transformation
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Scatter plot for the current model
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], 
            s=10, c=labels, cmap="jet", alpha=alpha
        )
        ax.set_title(model_names[idx])
        #ax.set_xlabel("t-SNE Dimension 1")
        #ax.set_ylabel("t-SNE Dimension 2")
        #ax.grid(True)

    # Set the overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Simulate embeddings for three models
    np.random.seed(42)
    embeddings_gcn = np.random.rand(1000, 128)  # Example embedding from GCN
    embeddings_gat = np.random.rand(1000, 128)  # Example embedding from GAT
    embeddings_gin = np.random.rand(1000, 128)  # Example embedding from GIN

    # Simulated labels
    labels = np.random.randint(0, 5, size=1000)

    # Model names
    model_names = ["GCN", "GAT", "GIN"]

    # Call the function
    plot_embeddings(
        embeddings_list=[embeddings_gcn, embeddings_gat, embeddings_gin],
        labels=labels,
        model_names=model_names,
        title="t-SNE Visualization of Node Embeddings from GNN Models"
    )
