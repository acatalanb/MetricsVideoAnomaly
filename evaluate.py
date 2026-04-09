import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import get_model
from metrics_manager import MetricsManager
from dataset import VideoDataset
import argparse
import numpy as np
from config import DEFAULT_DATASET_ROOT, DEFAULT_DATASET_NAME, BATCH_SIZE


def run_evaluation(model_name, model_path, dataset_path=None, dataset_name=None, epochs=None):
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_ROOT
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET_NAME

    eval_model_name = f"{model_name.replace(' ', '_')}_eval_{dataset_name}"
    metrics_manager = MetricsManager(eval_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Evaluating {model_name} on {dataset_name} using {device}")

    evaluation_dataset = VideoDataset(root_dir=dataset_path, dataset_name=dataset_name)
    eval_loader = DataLoader(evaluation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(model_name).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in eval_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    stats = metrics_manager.compute_metrics(all_labels, all_preds, all_probs)
    if epochs is not None:
        stats['epochs'] = epochs
    metrics_manager.save_metrics(stats, 0)  # no training time
    metrics_manager.plot_confusion_matrix(np.array(stats['confusion_matrix']))
    metrics_manager.plot_roc_curve(stats['roc_data']['fpr'], stats['roc_data']['tpr'], stats['auc'])

    print(f"✅ Evaluation complete for {model_name} on {dataset_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained video anomaly detection model.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['CNN-LSTM', '3D CNN', 'Video Transformer'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (optional, for metrics)')

    args = parser.parse_args()
    run_evaluation(args.model, args.model_path, args.dataset_path, args.dataset_name, args.epochs)