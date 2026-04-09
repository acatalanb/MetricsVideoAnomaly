import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
import argparse
from config import DEFAULT_DATASET_NAME


def load_metrics_for_comparison(model_identifier):
    cache_dir = 'cache'
    safe_name = model_identifier.replace(" ", "_").replace("-", "_")
    metrics_file = os.path.join(cache_dir, f"metrics_{safe_name}.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return None


def run_comparison(dataset_name):
    # Load metrics (using evaluation identifiers)
    cnn_lstm_metrics = load_metrics_for_comparison(f"CNN-LSTM_eval_{dataset_name}")
    cnn3d_metrics = load_metrics_for_comparison(f"3D CNN_eval_{dataset_name}")
    video_transformer_metrics = load_metrics_for_comparison(f"Video Transformer_eval_{dataset_name}")

    def get_max_epochs(metrics_list):
        epochs = [m.get('epochs', 0) for m in metrics_list if m]
        return max(epochs) if epochs else 5

    common_epochs = get_max_epochs([cnn_lstm_metrics, cnn3d_metrics, video_transformer_metrics])

    # ROC Curves
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'ROC Curves Comparison on {dataset_name} Dataset ({common_epochs} Epochs)', fontsize=16)
    axes = axes.flatten()

    models_data = [
        (cnn_lstm_metrics, 'CNN-LSTM', 'blue'),
        (cnn3d_metrics, '3D CNN', 'red'),
        (video_transformer_metrics, 'Video Transformer', 'green')
    ]

    for i, (metrics, name, color) in enumerate(models_data):
        if metrics and 'roc_data' in metrics:
            fpr = np.array(metrics['roc_data']['fpr'])
            tpr = np.array(metrics['roc_data']['tpr'])
            auc = metrics['auc']
            epochs = metrics.get('epochs', 'N/A')
            axes[i].plot(fpr, tpr, color=color, lw=2)
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{name} (AUC = {auc:.2f}, Epochs = {epochs})')
            axes[i].grid(True)
        else:
            print(f"{name} ROC data not found for {dataset_name}.")

    axes[3].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'Confusion Matrices Comparison on {dataset_name} Dataset ({common_epochs} Epochs)', fontsize=16)
    axes = axes.flatten()

    for i, (metrics, name, _) in enumerate(models_data):
        if metrics and 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Abnormal'],
                        yticklabels=['Normal', 'Abnormal'], ax=axes[i])
            axes[i].set_title(name)
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        else:
            print(f"{name} Confusion Matrix data not found for {dataset_name}.")

    axes[3].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare trained models.')
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME,
                        help='Name of the dataset for comparison.')

    args = parser.parse_args()
    run_comparison(args.dataset_name)