import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import math

from cdl2024.eval.eval_funcs import predict_probabilities

def show_roc_curve(ax, model_name, data, probabilities, mapped_classes):
    """
    Plots the ROC curve for the test data.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        model_name (str): Name of the model.
        data (torch_geometric.data.Data): Graph data object containing masks.
        probabilities (torch.Tensor): Predicted probabilities for the test data.
        mapped_classes (list): List of class names mapped to target indices.
    """
    y_true = data.y[data.test_mask].cpu().numpy()
    y_prob = probabilities[data.test_mask].cpu().numpy()

    for i, class_name in enumerate(mapped_classes):
        # One-vs-rest ROC curve computation
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid()


def show_multiple_roc_curves(models, data, mapped_classes):
    """
    Plots ROC curves for multiple models on test data in a two-column layout.

    Args:
        models (dict): Dictionary of model names and PyTorch models.
        data (torch_geometric.data.Data): Graph data object.
        mapped_classes (list): List of class names mapped to target indices.
    """
    num_models = len(models)
    cols = 2
    rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten()  # Flatten to easily iterate

    for idx, (model_name, model) in enumerate(models.items()):
        probabilities = predict_probabilities(model, data)
        show_roc_curve(axes[idx], model_name, data, probabilities, mapped_classes)

    # Hide any unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
