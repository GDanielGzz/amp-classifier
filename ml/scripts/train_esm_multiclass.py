"""Phase 2C step 2: train multi-label activity head on ESM2 embeddings.

Reuses the cached ESM2-650M embeddings from Phase 2B but filters to
AMP-positive entries only, then trains a 4-class multi-label MLP head
(antibacterial, antifungal, antiviral, antiparasitic). Each output is
independent sigmoid (multi-label, not softmax — peptides can be
positive in multiple activity classes).

Saves: ml/checkpoints/esm_multiclass_head.pt
"""
from __future__ import annotations

import csv
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
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
CLUSTERS_DIR = PROJECT_ROOT / "ml" / "data" / "clusters"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
CHECKPOINTS = PROJECT_ROOT / "ml" / "checkpoints"
ACTIVITIES_CSV = RAW_DIR / "amp_activities.csv"
ID_MAP_TSV = CLUSTERS_DIR / "id_map.tsv"
MULTICLASS_CKPT = CHECKPOINTS / "esm_multiclass_head.pt"

ACTIVITY_CLASSES = ("antibacterial", "antifungal", "antiviral", "antiparasitic")

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


class MultiClassHead(nn.Module):
    """Same architecture as the binary EsmHead but with N_CLASSES outputs."""

    def __init__(self, in_dim, n_classes=len(ACTIVITY_CLASSES), dropout=DROPOUT):
        super().__init__()
        h1 = max(128, in_dim // 2)
        h2 = max(64, h1 // 2)
        self.fc1 = nn.Linear(in_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(TF.relu(self.bn1(self.fc1(x))))
        x = self.dropout(TF.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


def load_id_to_dramp_map():
    out = {}
    with ID_MAP_TSV.open("r", encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            sid, orig_header, is_amp, length = parts[0], parts[1], parts[2], parts[3]
            if int(is_amp) == 1:
                dramp_id = orig_header.split()[0].strip()
                out[sid] = dramp_id
    return out


def load_activities():
    """dramp_id -> dict of class -> 0/1."""
    out = {}
    with ACTIVITIES_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rid = row["dramp_id"]
            out[rid] = {c: int(row[c]) for c in ACTIVITY_CLASSES}
    return out


def filter_amp_with_labels(npz_path, id_to_dramp, activities):
    """Load an ESM cache, return X, Y (multi-label), kept_ids for AMP rows."""
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"]
    ids_all = data["ids"]
    X_keep, Y_keep, kept_ids = [], [], []
    for i, sid in enumerate(ids_all):
        sid = str(sid)
        if not sid.startswith("amp_"):
            continue
        dramp_id = id_to_dramp.get(sid)
        if dramp_id is None:
            continue
        labels = activities.get(dramp_id)
        if labels is None:
            continue
        X_keep.append(X_all[i])
        Y_keep.append([labels[c] for c in ACTIVITY_CLASSES])
        kept_ids.append(sid)
    return (np.array(X_keep, dtype=np.float32),
            np.array(Y_keep, dtype=np.float32),
            np.array(kept_ids, dtype=object))


def compute_macro_auc(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(yb.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    aucs = []
    for c in range(scores.shape[1]):
        if labels[:, c].sum() < 1 or labels[:, c].sum() == labels.shape[0]:
            continue
        aucs.append(roc_auc_score(labels[:, c], scores[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    for name in ("train", "val"):
        p = PROCESSED_DIR / f"esm_{name}.npz"
        if not p.exists():
            print(f"[multiclass] {p} missing. Run dev.bat esm first.")
            return 1
    if not ACTIVITIES_CSV.exists():
        print(f"[multiclass] {ACTIVITIES_CSV} missing. Run dev.bat activities first.")
        return 1
    if not ID_MAP_TSV.exists():
        print(f"[multiclass] {ID_MAP_TSV} missing. Run dev.bat cluster first.")
        return 1

    set_seed(SEED)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[multiclass] device: {device}")

    id_to_dramp = load_id_to_dramp_map()
    activities = load_activities()
    print(f"[multiclass] {len(id_to_dramp)} AMP id_map entries, "
          f"{len(activities)} activity rows")

    X_train, Y_train, _ = filter_amp_with_labels(
        PROCESSED_DIR / "esm_train.npz", id_to_dramp, activities)
    X_val, Y_val, _ = filter_amp_with_labels(
        PROCESSED_DIR / "esm_val.npz", id_to_dramp, activities)
    in_dim = int(X_train.shape[1])
    print(f"[multiclass] X_train={X_train.shape}, X_val={X_val.shape}, "
          f"in_dim={in_dim}")
    print("[multiclass] per-class train positive counts:")
    for i, c in enumerate(ACTIVITY_CLASSES):
        n_pos = int(Y_train[:, i].sum())
        print(f"  {c}: {n_pos}/{len(Y_train)} ({100*n_pos/len(Y_train):.1f}%)")

    # Per-class pos_weight = neg_count / pos_count, clipped for sanity
    pos_counts = Y_train.sum(axis=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weight = torch.tensor(
        np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 50.0),
        dtype=torch.float32, device=device,
    )
    print(f"[multiclass] pos_weight: {pos_weight.cpu().numpy().round(2).tolist()}")

    model = MultiClassHead(in_dim=in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[multiclass] MLP params: {n_params:,}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=BATCH_SIZE, shuffle=False)

    best_macro = -1.0
    best_epoch = -1
    patience_counter = 0
    print(f"[multiclass] training up to {MAX_EPOCHS} epochs, patience {PATIENCE}")
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
        val_macro = compute_macro_auc(model, val_loader, device)
        elapsed = time.time() - t_start
        print(f"  epoch {epoch:3d}: train_loss={train_loss:.4f} "
              f"val_macro_auc={val_macro:.4f}  ({elapsed:.0f}s)")
        if val_macro > best_macro:
            best_macro = val_macro
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_auc": val_macro,
                "in_dim": in_dim,
                "classes": list(ACTIVITY_CLASSES),
                "config": {"dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
                           "batch_size": BATCH_SIZE, "seed": SEED},
            }, MULTICLASS_CKPT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[multiclass] early stop at epoch {epoch}")
                break

    print(f"[multiclass] best val macro-AUC {best_macro:.4f} at epoch {best_epoch}; "
          f"saved -> {MULTICLASS_CKPT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
