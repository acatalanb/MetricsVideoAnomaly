import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import get_model
from metrics_manager import MetricsManager
from dataset import VideoDataset
import argparse
import os
import numpy as np
from config import DEFAULT_DATASET_ROOT, DEFAULT_DATASET_NAME, BATCH_SIZE


def run_training(model_name, epochs=5, dataset_path=None, dataset_name=None):
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_ROOT
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET_NAME

    metrics_manager = MetricsManager(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"✅ Training {model_name} on {num_gpus} GPUs")
    else:
        print(f"✅ Training {model_name} on {device}")

    full_dataset = VideoDataset(root_dir=dataset_path, dataset_name=dataset_name)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    model = get_model(model_name).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    metrics_manager.start_training_timer()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

    total_time = metrics_manager.stop_training_timer()

    # Save model
    os.makedirs('cache', exist_ok=True)
    safe_name = model_name.replace(" ", "_").replace("-", "_")
    save_path = f"cache/model_{safe_name}.pth"
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_path)

    # Evaluation on validation set
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    stats = metrics_manager.compute_metrics(all_labels, all_preds, all_probs)
    stats['epochs'] = epochs
    metrics_manager.save_metrics(stats, total_time)
    metrics_manager.plot_confusion_matrix(np.array(stats['confusion_matrix']))
    metrics_manager.plot_roc_curve(stats['roc_data']['fpr'], stats['roc_data']['tpr'], stats['auc'])

    print(f"✅ Training complete! Model saved: {save_path}")
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video anomaly detection model.')
    parser.add_argument('--model', type=str, default='3D CNN',
                        choices=['CNN-LSTM', '3D CNN', 'Video Transformer'],
                        help='Model name to train')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the video dataset root directory.')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name of the dataset subfolder (e.g., "kvasir").')

    args = parser.parse_args()
    run_training(args.model, args.epochs, args.dataset_path, args.dataset_name)