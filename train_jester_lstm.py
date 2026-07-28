#!/usr/bin/env python3
"""
Train a two-layer LSTM with Dropout on the Jester dynamic-gesture dataset.

The dataset (F:\\jesterdata_sort\\sorted) is a set of .npz files:
    jester_train.npz / jester_val.npz / jester_test.npz
each containing:
    X       : (N, 24, 126) float32  -- N sequences of 24 frames x 126 features
    y       : (N,) int64            -- class index in [0, 26]
    classes : (27,) str             -- class names (20BN-Jester v1 vocabulary)

Feature layout (verified by inspection):
    126 = left_hand(21 landmarks x 3 xyz) + right_hand(21 landmarks x 3 xyz)
    Each hand is stored WRIST-RELATIVE (landmark_i - landmark_0), so the wrist
    itself is always (0, 0, 0). Values are in normalized [0,1] MediaPipe space.

The same layout is reproduced live in GestureFusion (dynamic_lstm backend), so
the trained model drops straight into the camera pipeline.

Outputs (to --out, default F:\\MODELS\\jester_lstm):
    jester_lstm.pt   : state_dict + metadata (classes, mean, std, arch params)
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
SEQ_LEN = 24
INPUT_DIM = 126
NUM_CLASSES = 27


class JesterLSTM(nn.Module):
    """Two-layer LSTM + Dropout classifier over a (T, F) landmark sequence."""

    def __init__(self, input_dim=INPUT_DIM, hidden=128, num_classes=NUM_CLASSES,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden, 64)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, hidden)
        last = out[:, -1, :]           # take the final timestep
        h = self.drop(self.act(self.fc1(last)))
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Dataset (standardizes on the fly using training statistics)
# ---------------------------------------------------------------------------
class JesterDataset(Dataset):
    def __init__(self, X, y, mean, std):
        # Keep numpy on host; convert per-sample to save RAM.
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.mean = np.ascontiguousarray(mean, dtype=np.float32)
        self.std = np.ascontiguousarray(std + 1e-8, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"F:\jesterdata_sort\sorted")
    ap.add_argument("--out", default=r"F:\MODELS\jester_lstm")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.txt")
    log = open(log_path, "w", encoding="utf-8", buffering=1)

    def logprint(*a):
        line = " ".join(str(s) for s in a)
        print(line, flush=True)
        log.write(line + "\n")

    logprint("Loading dataset from", args.data_dir)
    tr = np.load(os.path.join(args.data_dir, "jester_train.npz"), allow_pickle=True)
    va = np.load(os.path.join(args.data_dir, "jester_val.npz"), allow_pickle=True)
    te = np.load(os.path.join(args.data_dir, "jester_test.npz"), allow_pickle=True)

    Xtr, ytr = tr["X"], tr["y"]
    Xva, yva = va["X"], va["y"]
    Xte, yte = te["X"], te["y"]
    classes = [str(c) for c in tr["classes"]]
    logprint("train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)
    logprint("classes", classes)

    # Per-feature statistics from TRAIN only (applied to all splits).
    mean = Xtr.reshape(-1, INPUT_DIM).mean(axis=0).astype(np.float32)
    std = Xtr.reshape(-1, INPUT_DIM).std(axis=0).astype(np.float32)
    logprint("feature mean[0:3]", np.round(mean[:3], 4),
             "std[0:3]", np.round(std[:3], 4))

    train_ds = JesterDataset(Xtr, ytr, mean, std)
    val_ds = JesterDataset(Xva, yva, mean, std)
    test_ds = JesterDataset(Xte, yte, mean, std)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                        num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False,
                         num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logprint("device:", device)

    model = JesterLSTM(hidden=args.hidden, dropout=args.dropout).to(device)
    logprint(model)
    n_params = sum(p.numel() for p in model.parameters())
    logprint("parameters:", n_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    def evaluate(dl):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum())
                total += int(yb.shape[0])
        return correct / total if total else 0.0

    best_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        train_loss = running_loss / max(1, n_batches)

        val_acc = evaluate(val_dl)
        scheduler.step(val_acc)
        logprint(f"epoch {epoch:03d} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "classes": classes,
                "mean": mean,
                "std": std,
                "seq_len": SEQ_LEN,
                "input_dim": INPUT_DIM,
                "hidden": args.hidden,
                "num_layers": 2,
                "dropout": args.dropout,
                "num_classes": NUM_CLASSES,
            }
            torch.save(ckpt, os.path.join(args.out, "jester_lstm.pt"))
            logprint(f"  -> saved best model (val_acc {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logprint(f"Early stopping at epoch {epoch} (no val improvement "
                         f"for {args.patience} epochs).")
                break

    logprint(f"Best val_acc {best_acc:.4f} at epoch {best_epoch}")

    # Final test evaluation with the best checkpoint.
    ckpt = torch.load(os.path.join(args.out, "jester_lstm.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_acc = evaluate(test_dl)
    logprint(f"TEST acc {test_acc:.4f}")

    # Persist a small human-readable metadata sidecar too.
    meta = {
        "classes": classes,
        "seq_len": SEQ_LEN,
        "input_dim": INPUT_DIM,
        "hidden": args.hidden,
        "num_layers": 2,
        "dropout": args.dropout,
        "num_classes": NUM_CLASSES,
        "test_acc": round(float(test_acc), 4),
        "val_acc": round(float(best_acc), 4),
    }
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logprint("DONE")
    log.close()


if __name__ == "__main__":
    main()
