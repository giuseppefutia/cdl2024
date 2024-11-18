import matplotlib.pyplot as plt

def plot_metrics(metrics, 
                 metric_key, 
                 title, 
                 xlabel='Epoch', 
                 ylabel=None, 
                 styles=None, 
                 metric_types=None, 
                 figsize=(8, 4)):
    """
    Plots specified metrics for multiple models over epochs.

    Args:
        metrics (dict): A dictionary where keys are model names and values are dictionaries
                        containing dataset metrics (e.g., {'gcn': {'val': {'precisions': [...]}, 'train': {...}}}).
        metric_key (str): The key for the metric to plot (e.g., 'precisions').
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis. Default is 'Epoch'.
        ylabel (str): Label for the y-axis. Default is `metric_key.capitalize()`.
        styles (dict): Optional dictionary for styling. Keys are model names, and values are dictionaries with:
                       {'color': ..., 'linestyle': ..., 'linewidth': ..., 'label': ...}.
        metric_types (list): List of metric types to plot (e.g., ['train', 'val']). Default is `['val']`.
        figsize (tuple): Size of the figure. Default is (8, 4).
    """
    metric_types = metric_types or ['val']
    plt.figure(figsize=figsize)
    
    for model, data in metrics.items():
        for metric_type in metric_types:
            if metric_type not in data:
                continue

            epochs = range(1, len(data[metric_type][metric_key]) + 1)
            style = styles.get(model, {}) if styles else {}
            
            color = style.get('color', None)
            linestyle = style.get('linestyle', '-')
            linewidth = style.get('linewidth', 1.2)
            label = f"{style.get('label', model.upper())} ({metric_type})"
            
            plt.plot(epochs, data[metric_type][metric_key], 
                     color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else metric_key.capitalize())
    plt.title(title)
    plt.legend(fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    metrics = {
        'gcn': {
            'val': {'precisions': [0.7, 0.75, 0.78, 0.82]},
            'train': {'precisions': [0.65, 0.7, 0.73, 0.8]}
        },
        'gat': {
            'val': {'precisions': [0.68, 0.73, 0.77, 0.8]},
            'train': {'precisions': [0.6, 0.65, 0.7, 0.76]}
        },
        'gin': {
            'val': {'precisions': [0.65, 0.7, 0.74, 0.78]},
            'train': {'precisions': [0.62, 0.68, 0.72, 0.75]}
        }
    }

    styles = {
        'gcn': {'color': 'C0', 'linestyle': '--', 'linewidth': 1.2, 'label': 'GCN'},
        'gat': {'color': 'C1', 'linestyle': '-.', 'linewidth': 1.2, 'label': 'GAT'},
        'gin': {'color': 'C2', 'linestyle': ':', 'linewidth': 1.2, 'label': 'GIN'}
    }

    plot_metrics(metrics, 
                    metric_key='precisions', 
                    title='Training and Validation Precisions Across GNN Models',
                    ylabel='Precision', 
                    styles=styles, 
                    metric_types=['train', 'val'])
