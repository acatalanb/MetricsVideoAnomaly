import torch
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from model import get_model
import argparse
from config import IMG_SIZE, SEQ_LEN


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // SEQ_LEN)
    for i in range(SEQ_LEN):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
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
    return torch.tensor(np.array(frames)).unsqueeze(0)


class VideoPredictor:
    def __init__(self, model_name, model_path):
        self.model = get_model(model_name)
        state_dict = torch.load(model_path, map_location="cpu")
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✅ Loaded {model_name} successfully")

    def predict(self, video_path):
        input_tensor = process_video(video_path).to(next(self.model.parameters()).device)

        t1 = time.time()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        inference_time = time.time() - t1

        result = {'Normal': probs[0][0].item(), 'Abnormal': probs[0][1].item()}
        label = "🚨 ANOMALY DETECTED" if pred.item() == 1 else "✅ NORMAL SCAN"

        print(f"\n{label} (Confidence: {conf.item():.1%})")
        print(f"Inference time: {inference_time*1000:.0f} ms")

        # Bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(result.keys(), result.values(), color=['#2ecc71', '#e74c3c'])
        plt.ylabel("Probability")
        plt.title("Prediction Confidence")
        plt.ylim(0, 1)
        plt.show()

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference for video anomaly detection.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['CNN-LSTM', '3D CNN', 'Video Transformer'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, required=True)

    args = parser.parse_args()
    predictor = VideoPredictor(args.model, args.model_path)
    predictor.predict(args.video_path)