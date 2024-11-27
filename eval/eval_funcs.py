import torch

def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    return pred

def predict_batched(model, data_loader):
    """
    Predicts labels for batched data using a DataLoader.

    Args:
        model (torch.nn.Module): Trained model to use for predictions.
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of graph data.

    Returns:
        torch.Tensor: Predicted labels for all batches.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch_data in data_loader:
            out = model(batch_data)
            probs = torch.sigmoid(out)
            preds.append((probs >= 0.5).float())

    return torch.cat(preds, dim=0)


def predict_probabilities(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probabilities = torch.exp(out)
    return probabilities