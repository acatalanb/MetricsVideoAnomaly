import torch
from torch.utils.data import Dataset
import cv2
import glob
import os
import numpy as np
from config import IMG_SIZE, SEQ_LEN, DEFAULT_DATASET_ROOT


class VideoDataset(Dataset):
    def __init__(self, root_dir=None, dataset_name=None):
        if root_dir is None:
            root_dir = DEFAULT_DATASET_ROOT
        if dataset_name is None:
            from config import DEFAULT_DATASET_NAME
            dataset_name = DEFAULT_DATASET_NAME

        self.video_paths = []
        self.labels = []

        norm_files = glob.glob(os.path.join(root_dir, dataset_name, 'normal', '*.mp4'))
        abnorm_files = glob.glob(os.path.join(root_dir, dataset_name, 'abnormal', '*.mp4'))

        self.video_paths.extend(norm_files + abnorm_files)
        self.labels.extend([0] * len(norm_files) + [1] * len(abnorm_files))

        print(f"📂 Loaded {len(self.video_paths)} videos "
              f"({len(norm_files)} normal, {len(abnorm_files)} abnormal) from {dataset_name}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < SEQ_LEN:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
            else:
                frames.append(np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
        cap.release()
        return torch.tensor(np.array(frames[:SEQ_LEN])), torch.tensor(label)