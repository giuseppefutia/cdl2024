import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import math
import torch
from tqdm import tqdm

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

    _, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten()  # Flatten to easily iterate

    for idx, (model_name, model) in enumerate(models.items()):
        probabilities = predict_probabilities(model, data)
        show_roc_curve(axes[idx], model_name, data, probabilities, mapped_classes)

    # Hide any unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

def show_multiple_roc_curves_batched(models_dict, val_loader, mapped_classes, device):
    # TODO: Should move the probability computation outside
    """
    Plots ROC curves for multiple models in a two-column layout.

    Args:
        models_dict (dict): Dictionary of models where keys are model names and values are model objects.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        mapped_classes (list): List of class labels, e.g., ['non-existing links', 'existing links'].
        device (torch.device): Device to run the models on.
    """
    # Create a figure with two columns
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("ROC Curves for Multiple Models", fontsize=16)

    # Iterate through each model
    for model_name, model in models_dict.items():
        preds = []
        ground_truths = []

        # Collect predictions and ground truths
        for sampled_data in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(sampled_data))
                ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)

        # Concatenate predictions and ground truths
        pred = torch.cat(preds, dim=0).cpu().numpy()  # Predicted probabilities for class 1
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()  # Binary ground truth labels

        # Compute and plot ROC curve for each class
        for i, class_name in enumerate(mapped_classes):
            ground_truth_binary = (ground_truth == i).astype(int)
            fpr, tpr, _ = roc_curve(ground_truth_binary, pred)
            auc_score = roc_auc_score(ground_truth_binary, pred)

            # Plot ROC curve for the current class on the respective axis
            ax[i].plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})")
            ax[i].set_title(f"ROC Curve - {class_name}")
            ax[i].set_xlabel("False Positive Rate")
            ax[i].set_ylabel("True Positive Rate")
            ax[i].legend(loc="lower right")
            ax[i].grid()

    plt.tight_layout()
    plt.show()
