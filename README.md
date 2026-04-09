# Video Anomaly Detection – PyCharm Professional Project

Deep learning pipeline for video anomaly detection using PyTorch.

## Features
- CNN-LSTM, 3D CNN (R(2+1)D), Video Transformer (VideoMAE)
- Full training, inference, evaluation & comparison
- Clean modular structure for PyCharm

## Setup
1. Open the folder in PyCharm Professional
2. Create virtual environment (recommended)
3. Install dependencies: `pip install -r requirements.txt`
4. Put your videos in `VideoDataset/kvasir/normal/` and `/abnormal/`

## Usage Examples

```bash
# Train
python train.py --model "3D CNN" --epochs 5 --dataset_name kvasir

# Inference
python inference.py --model "3D CNN" --model_path "cache/model_3D_CNN.pth" --video_path "test.mp4"

# Evaluate
python evaluate.py --model "3D CNN" --model_path "cache/model_3D_CNN.pth" --dataset_name kvasir

# Compare all models
python compare_models.py