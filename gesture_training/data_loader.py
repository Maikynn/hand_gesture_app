#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

class JesterDataLoader:
    def __init__(self, data_dir: str = 'F:\jesterdata_sort\sorted'):
        self.data_dir = Path(data_dir)
        self.class_map = {name: idx for idx, name in enumerate(GESTURE_CLASSES)}
        self._load_csvs()

    def _load_csvs(self):
        # Process jester-v1-train.csv in chunks to avoid memory issues
        self.frames = []
        for csv_file in self.data_dir.glob('jester-v1-*.csv'):
            df = pd.read_csv(csv_file, chunksize=1000)
            for chunk in df:
                # Extract 21st point (index 20) from landmarks
                if len(chunk.columns) > 21:  # Assuming columns 0-20 are landmarks
                    landmarks = chunk.iloc[:, 1:22].values  # Columns 1-21
                    # Normalize by wrist-centering + palm-size scaling
                    wrist = landmarks[:, [0, 5, 9]]  # Example wrist landmarks
                    scale = np.mean(np.linalg.norm(landmarks[:, :10], axis=1))
                    normalized = (landmarks - wrist) / scale
                    self.frames.append(normalized)

    def get_batch(self, batch_size: int = 32):
        # Return batches of normalized 21-point landmarks
        return np.array(self.frames)[np.random.permutation(len(self.frames))[:batch_size]]

# Example usage
if __name__ == '__main__':
    loader = JesterDataLoader()
    batch = loader.get_batch()
    print(f'Batch shape: {batch.shape}')  # Should be (batch_size, 24, 21)