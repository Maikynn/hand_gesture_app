#!/usr/bin/env python3
"""
Dynamic gesture model (LSTM) for Jester 21-landmark sequences.

Produces a predictor object compatible with GestureFusion.set_dynamic_model().
The network consumes a fixed-length buffer of normalized 21-landmark frames
(63 values each: x, y, z) and outputs a probability distribution over the
dynamic gesture vocabulary.
"""

import os
import numpy as np
import torch
import torch.nn as nn

# Dynamic gesture vocabulary (subset of Jester relevant to the app)
JESTER_DYNAMIC_CLASSES = [
    'no_gesture', 'swiping_left', 'swiping_right', 'swiping_up', 'swiping_down',
    'push', 'pull', 'zoom_in', 'zoom_out', 'rotate_cw', 'rotate_ccw',
    'thumb_up', 'thumb_down', 'stop', 'drumming', 'shaking',
]

SEQ_LEN = 24
NUM_LM = 21
LM_DIM = 3  # x, y, z


class DynamicLSTM(nn.Module):
    """Two-layer LSTM over a (SEQ_LEN, NUM_LM*LM_DIM) feature sequence."""

    def __init__(self, input_size: int = NUM_LM * LM_DIM,
                 hidden: int = 64,
                 num_classes: int = len(JESTER_DYNAMIC_CLASSES)):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # last time step
        return self.head(x)


class DynamicPredictor:
    """Wraps a trained DynamicLSTM for use inside GestureFusion."""

    def __init__(self, model_path: str, device: str = 'cpu', classes: list = None):
        self.classes = list(classes) if classes else list(JESTER_DYNAMIC_CLASSES)
        self.device = device
        self.model = DynamicLSTM(num_classes=len(self.classes))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[DynamicPredictor] weights loaded: {model_path}")
        else:
            print(f"[DynamicPredictor] WARNING: weights not found: {model_path}")
        self.model.to(device).eval()

    def predict(self, sequence: np.ndarray):
        """
        Args:
            sequence: np.ndarray of shape (SEQ_LEN, NUM_LM*LM_DIM) or
                      (SEQ_LEN, NUM_LM, LM_DIM).
        Returns:
            (gesture_name, confidence)
        """
        if sequence is None:
            return 'no_gesture', 0.0
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[0] != SEQ_LEN:
            # pad / truncate to SEQ_LEN
            if arr.shape[0] < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = arr[:SEQ_LEN]
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out[0], dim=0).cpu().numpy()
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])
