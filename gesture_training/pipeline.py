#!/usr/bin/env python3
"""
Jester dataset pipeline (continuation of the previous program).

Step 1: extract_jester() - MediaPipe Hand Landmarker -> 21-landmark normalized
                             sequences -> .npz files in F:\\jesterdata_sort\\sorted
Step 2: train_dynamic()  - train DynamicLSTM on the extracted .npz files and save
                             weights to gesture_training/models/dynamic_lstm.pth

Run:  python gesture_training/pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from gesture_training.dynamic_model import (
    DynamicLSTM, JESTER_DYNAMIC_CLASSES, SEQ_LEN, NUM_LM, LM_DIM,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JESTER_RAW_DIR = r"F:\20bn-jester-v1-complete\20bn-jester-v1"
OUTPUT_DIR = r"F:\jesterdata_sort\sorted"
MODEL_ASSET_PATH = r"F:\MODELS\hand_landmarker.task"
CSV_MAPPINGS = {
    "train": r"F:\jesterdata_sort\jester-v1-train.csv",
    "val": r"F:\jesterdata_sort\jester-v1-validation.csv",
    "test": r"F:\jesterdata_sort\jester-v1-test.csv",
}
LANDMARK_COUNT = 21
FEATURES_PER_LM = 6  # x, y, z, dx, dy, dz

MODELS_DIR = os.path.join(ROOT, "gesture_training", "models")
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_lstm.pth")


# ---------------------------------------------------------------------------
# Extraction (adapted from F:\jesterdata_sort\main.py)
# ---------------------------------------------------------------------------
def _download_mediapipe_model():
    import urllib.request
    if not os.path.exists(MODEL_ASSET_PATH):
        print("Downloading MediaPipe Hand Landmarker model...")
        os.makedirs(os.path.dirname(MODEL_ASSET_PATH), exist_ok=True)
        url = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
               "hand_landmarker/float16/1/hand_landmarker.task")
        urllib.request.urlretrieve(url, MODEL_ASSET_PATH)
        print("Model downloaded.")


def _normalize_and_extract_features(landmarks_sequence):
    """Center by wrist, scale by palm size, append delta velocities."""
    seq_len = len(landmarks_sequence)
    final = np.zeros((seq_len, LANDMARK_COUNT, FEATURES_PER_LM), dtype=np.float32)
    for t in range(seq_len):
        frame_lms = landmarks_sequence[t]
        wrist = frame_lms[0]
        centered = frame_lms - wrist
        palm_size = np.linalg.norm(centered[0] - centered[9])
        if palm_size == 0:
            palm_size = 1.0
        normalized = centered / palm_size
        final[t, :, :3] = normalized
        if t > 0:
            final[t, :, 3:6] = final[t, :, :3] - final[t - 1, :, :3]
    return final.reshape(seq_len, -1)


def extract_jester():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from tqdm import tqdm

    _download_mediapipe_model()
    base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1, min_hand_detection_confidence=0.3)
    detector = vision.HandLandmarker.create_from_options(options)

    for subset_name, csv_path in CSV_MAPPINGS.items():
        if not os.path.exists(csv_path):
            print(f"Skip {subset_name}: {csv_path} not found")
            continue
        is_test = (subset_name == "test")
        df = pd.read_csv(csv_path, sep=';',
                         names=['folder_id', 'gesture_name'] if not is_test else ['folder_id'])
        if is_test:
            df['gesture_name'] = 'unlabeled'

        unique_classes = sorted(df['gesture_name'].unique())
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        X_data, y_data = [], []

        print(f"\nProcessing MediaPipe for [{subset_name.upper()}]...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {subset_name}"):
            folder_id = str(row['folder_id']).strip()
            label_idx = class_to_idx[row['gesture_name']]
            video_path = os.path.join(JESTER_RAW_DIR, folder_id)
            if not os.path.exists(video_path):
                continue
            frame_files = sorted([f for f in os.listdir(video_path)
                                  if f.lower().endswith(('.jpg', '.jpeg'))])
            if len(frame_files) < 5:
                continue
            indices = np.linspace(0, len(frame_files) - 1, SEQ_LEN, dtype=int)
            clip_landmarks = []
            valid = True
            for idx in indices:
                frame_path = os.path.join(video_path, frame_files[idx])
                try:
                    mp_image = mp.Image.create_from_file(frame_path)
                    results = detector.detect(mp_image)
                except Exception:
                    valid = False
                    break
                if results.hand_landmarks:
                    hand = results.hand_landmarks[0]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                    clip_landmarks.append(coords)
                else:
                    if clip_landmarks:
                        clip_landmarks.append(clip_landmarks[-1])
                    else:
                        clip_landmarks.append(np.zeros((LANDMARK_COUNT, 3), dtype=np.float32))
            if valid and clip_landmarks:
                X_data.append(_normalize_and_extract_features(clip_landmarks))
                y_data.append(label_idx)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(OUTPUT_DIR, f"jester_{subset_name}.npz")
        np.savez_compressed(out_file, X=np.array(X_data, dtype=np.float32),
                            y=np.array(y_data, dtype=np.int64),
                            classes=np.array(unique_classes))
        print(f"Saved -> {out_file}")

    detector.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _load_npz(subset):
    path = os.path.join(OUTPUT_DIR, f"jester_{subset}.npz")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True)
    return data['X'], data['y'], list(data['classes'])


def train_dynamic(epochs=20, batch_size=32):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X, y, classes = _load_npz("train")
    if X is None:
        print("No training .npz found. Run extract_jester() first.")
        return
    # Map jester labels -> dynamic vocabulary (best-effort)
    label_map = {c: c for c in JESTER_DYNAMIC_CLASSES}
    y_mapped = np.array([label_map.get(classes[i], 0) for i in y], dtype=np.int64)

    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y_mapped).long()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model = DynamicLSTM(num_classes=len(JESTER_DYNAMIC_CLASSES))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training dynamic LSTM on {len(X)} samples ({device})...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(X):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), DYNAMIC_MODEL_PATH)
    print(f"Dynamic model saved -> {DYNAMIC_MODEL_PATH}")


def main():
    extract_jester()
    train_dynamic()


if __name__ == '__main__':
    main()
