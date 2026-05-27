"""Phase 2B step 2: train a small MLP head on frozen ESM2 embeddings.

The MLP head consumes whatever embedding dim was produced by
``extract_esm_embeddings.py`` — auto-detected from the cached npz so
the script works for ESM2-35M (480), ESM2-150M (640), or ESM2-650M
(1280) without changes.

Saves: ml/checkpoints/esm_head_best.pt
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
CHECKPOINTS = PROJECT_ROOT / "ml" / "checkpoints"
ESM_HEAD_CHECKPOINT = CHECKPOINTS / "esm_head_best.pt"

SEED = 42
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 80
PATIENCE = 15
DROPOUT = 0.3


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class EsmHead(nn.Module):
    """Small MLP over a mean-pooled ESM2 embedding. Hidden dims scale
    with the input embedding dim so the same recipe works for ESM2-35M
    through ESM2-650M without manual tuning."""

    def __init__(self, in_dim, dropout=DROPOUT):
        super().__init__()
        h1 = max(128, in_dim // 2)
        h2 = max(64, h1 // 2)
        self.fc1 = nn.Linear(in_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(TF.relu(self.bn1(self.fc1(x))))
        x = self.dropout(TF.relu(self.bn2(self.fc2(x))))
        return self.fc3(x).squeeze(-1)


def compute_auc(model, loader, device):
    model.eval()
    all_s, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_s.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(yb.numpy())
    return float(roc_auc_score(np.concatenate(all_y).astype(int),
                                np.concatenate(all_s)))


def main():
    for name in ("train", "val", "test"):
        p = PROCESSED_DIR / f"esm_{name}.npz"
        if not p.exists():
            print(f"[esm-head] {p} missing. Run dev.bat esm-embed first.")
            return 1

    set_seed(SEED)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esm-head] device: {device}")

    train = np.load(PROCESSED_DIR / "esm_train.npz", allow_pickle=True)
    val = np.load(PROCESSED_DIR / "esm_val.npz", allow_pickle=True)
    in_dim = int(train["X"].shape[1])
    print(f"[esm-head] embedding dim: {in_dim}")
    X_train = torch.from_numpy(train["X"].astype(np.float32))
    y_train = torch.from_numpy(train["y"].astype(np.float32))
    X_val = torch.from_numpy(val["X"].astype(np.float32))
    y_val = torch.from_numpy(val["y"].astype(np.float32))

    pos = float((train["y"] == 1).sum())
    neg = float((train["y"] == 0).sum())
    pos_weight = torch.tensor(neg / max(1.0, pos), dtype=torch.float32, device=device)
    print(f"[esm-head] X_train={X_train.shape}, pos_weight={pos_weight.item():.3f}")

    model = EsmHead(in_dim=in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[esm-head] MLP params: {n_params:,}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)

    best_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    print(f"[esm-head] training up to {MAX_EPOCHS} epochs, patience {PATIENCE}")
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(1, n)
        val_auc = compute_auc(model, val_loader, device)
        elapsed = time.time() - t_start
        print(f"  epoch {epoch:3d}: train_loss={train_loss:.4f} "
              f"val_auc={val_auc:.4f}  ({elapsed:.0f}s)")
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "in_dim": in_dim,
                "config": {"dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
                           "batch_size": BATCH_SIZE, "seed": SEED},
            }, ESM_HEAD_CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[esm-head] early stop at epoch {epoch}")
                break

    print(f"[esm-head] best val AUC {best_auc:.4f} at epoch {best_epoch}; "
          f"saved -> {ESM_HEAD_CHECKPOINT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
