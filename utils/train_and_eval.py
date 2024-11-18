import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

def initialize_metrics_storage():
    return {
        'losses': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': []
    }

def train_step(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_step(model, data):
    return calculate_metrics(model, data, 'val')

def train(num_epochs, data, model, optimizer, criterion):
    # Initialize metrics storage
    train_metrics = initialize_metrics_storage()
    val_metrics = initialize_metrics_storage()

    for epoch in range(1, num_epochs + 1):
        # Training Step
        train_loss = train_step(model, optimizer, criterion, data)
        train_metrics_epoch = calculate_metrics(model, data, 'train')
        update_metrics(train_metrics, train_metrics_epoch, train_loss)

        # Validation Step
        val_metrics_epoch = validate_step(model, data)
        update_metrics(val_metrics, val_metrics_epoch)

        # Logging
        if epoch % 100 == 0:
            log_epoch(epoch, train_loss, train_metrics_epoch, val_metrics_epoch)

    return {
        'train': train_metrics,
        'val': val_metrics
    }

def calculate_metrics(model, data, mask_type='train'):
    mask = getattr(data, f"{mask_type}_mask")
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[mask].argmax(dim=1)
        correct = (pred == data.y[mask]).sum()
        accuracy = int(correct) / int(mask.sum())

        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def update_metrics(metrics, metrics_epoch, loss=None):
    if loss is not None:
        metrics['losses'].append(loss)
    metrics['accuracies'].append(metrics_epoch['accuracy'])
    metrics['precisions'].append(metrics_epoch['precision'])
    metrics['recalls'].append(metrics_epoch['recall'])
    metrics['f1_scores'].append(metrics_epoch['f1_score'])

def log_epoch(epoch, train_loss, train_metrics_epoch, val_metrics_epoch):
    print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Train - '
          f'Acc: {train_metrics_epoch["accuracy"]:.4f} - '
          f'Prec: {train_metrics_epoch["precision"]:.4f} - '
          f'Rec: {train_metrics_epoch["recall"]:.4f} - '
          f'F1: {train_metrics_epoch["f1_score"]:.4f}')
    print(f'Val - Acc: {val_metrics_epoch["accuracy"]:.4f} - '
          f'Prec: {val_metrics_epoch["precision"]:.4f} - '
          f'Rec: {val_metrics_epoch["recall"]:.4f} - '
          f'F1: {val_metrics_epoch["f1_score"]:.4f}')