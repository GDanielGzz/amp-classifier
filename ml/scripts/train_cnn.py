"""Train the small 1D CNN over one-hot encoded sequences.

Architecture (Veltri 2018 / AMP-Scanner-inspired, CPU-friendly):

  Input: one-hot encoded sequence, padded/truncated to length 100,
         shape (batch, 21, 100). 21 channels = 20 canonical AA + 1 PAD.
  Conv1D(21 -> 64, kernel 3, padding 1) -> ReLU -> Dropout(0.2)
  Conv1D(64 -> 128, kernel 3, padding 1) -> ReLU -> Dropout(0.2)
  Conv1D(128 -> 128, kernel 3, padding 1) -> ReLU -> Dropout(0.2)
  GlobalMaxPool1D (over length axis)
  Linear(128 -> 64) -> ReLU -> Dropout(0.2)
  Linear(64 -> 1)
  (Logits; BCEWithLogitsLoss provides sigmoid.)

Training:
  - BCEWithLogitsLoss with pos_weight = neg / pos (class imbalance)
  - AdamW, lr=3e-4, weight_decay=1e-4
  - Batch size 64
  - Up to 50 epochs, early stopping on val AUC, patience 10
  - Seed=42 across torch, numpy, python random

Target: train in <15 min on CPU.

Saves: ml/checkpoints/cnn_best.pt
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import CANONICAL_ALPHABET  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
CLUSTERS_DIR = PROJECT_ROOT / "ml" / "data" / "clusters"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
CHECKPOINTS = PROJECT_ROOT / "ml" / "checkpoints"
COMBINED_FASTA = CLUSTERS_DIR / "combined.fasta"
CNN_CHECKPOINT = CHECKPOINTS / "cnn_best.pt"

SEED = 42
MAX_LEN = 100
ALPHABET = CANONICAL_ALPHABET  # 20 canonical AA
N_CHANNELS = len(ALPHABET) + 1  # +1 for PAD channel

BATCH_SIZE = 64
LR = 1e-4               # was 3e-4: higher LR caused val AUC to peak at epoch 2 then drift
WEIGHT_DECAY = 1e-3     # was 1e-4: stronger weight decay damps overfit on this corpus
MAX_EPOCHS = 80         # was 50: lower LR needs more epochs to converge
PATIENCE = 15           # was 10: matched to the slower learning curve
DROPOUT = 0.4           # was 0.2: doubled because network overfit in 2 epochs at 0.2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_fasta(path):
    header, chunks = None, []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip().split()[0]
                chunks = []
            elif line.strip():
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def load_sequence_db():
    return {sid: seq.upper() for sid, seq in parse_fasta(COMBINED_FASTA)}


def load_split_csv(path):
    import csv
    ids, labels = [], []
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ids.append(row["sequence_id"])
            labels.append(int(row["is_amp"]))
    return ids, np.array(labels, dtype=int)


def onehot_encode(seq):
    """Return shape (N_CHANNELS, MAX_LEN). Truncates or pads to MAX_LEN."""
    arr = np.zeros((N_CHANNELS, MAX_LEN), dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(ALPHABET)}
    pad_idx = len(ALPHABET)  # channel 20 is PAD
    for i in range(MAX_LEN):
        if i < len(seq):
            ch = aa_to_idx.get(seq[i], pad_idx)
        else:
            ch = pad_idx
        arr[ch, i] = 1.0
    return arr


def encode_split(seq_db, ids, labels):
    X = np.stack([onehot_encode(seq_db[sid]) for sid in ids])
    y = labels.astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


class CNN(nn.Module):
    """V2: BatchNorm after each conv/fc, wider channels (96/192/192/96).

    The V1 model (64/128/128, no BN, ~70k params) plateaued at val AUC
    ~0.81 in <10 epochs across multiple regularization recipes. Hypothesis
    for V2: more capacity + BN-normalized activations let the network
    keep finding signal past the V1 ceiling without immediately overfit.
    """

    def __init__(self, in_channels=N_CHANNELS, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(192)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(192, 96)
        self.bn_fc = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 1)

    def forward(self, x):
        x = self.dropout(TF.relu(self.bn1(self.conv1(x))))
        x = self.dropout(TF.relu(self.bn2(self.conv2(x))))
        x = self.dropout(TF.relu(self.bn3(self.conv3(x))))
        x = TF.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = self.dropout(TF.relu(self.bn_fc(self.fc1(x))))
        return self.fc2(x).squeeze(-1)


def compute_auc(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(yb.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(int)
    return float(roc_auc_score(labels, scores))


def main():
    if not COMBINED_FASTA.exists():
        print("[cnn] combined.fasta missing. Run dev.bat cluster first.")
        return 1
    if not (SPLITS_DIR / "train.csv").exists():
        print("[cnn] splits missing. Run dev.bat splits first.")
        return 1
    set_seed(SEED)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[cnn] loading sequences ...")
    seq_db = load_sequence_db()
    train_ids, y_train_np = load_split_csv(SPLITS_DIR / "train.csv")
    val_ids,   y_val_np   = load_split_csv(SPLITS_DIR / "val.csv")
    test_ids,  y_test_np  = load_split_csv(SPLITS_DIR / "test.csv")

    print("[cnn] one-hot encoding ...")
    t0 = time.time()
    X_train, y_train = encode_split(seq_db, train_ids, y_train_np)
    X_val,   y_val   = encode_split(seq_db, val_ids,   y_val_np)
    X_test,  y_test  = encode_split(seq_db, test_ids,  y_test_np)
    print(f"[cnn] encoded in {time.time()-t0:.1f}s; "
          f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    pos = float((y_train_np == 1).sum())
    neg = float((y_train_np == 0).sum())
    pos_weight = torch.tensor(neg / max(1.0, pos), dtype=torch.float32)
    print(f"[cnn] pos_weight = {pos_weight.item():.3f} ({int(neg)}/{int(pos)})")

    device = torch.device("cpu")
    model = CNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cnn] CNN params: {n_params:,}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)

    best_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    print(f"[cnn] training up to {MAX_EPOCHS} epochs, patience {PATIENCE}")
    train_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total_loss / max(1, n)

        val_auc = compute_auc(model, val_loader, device)
        elapsed = time.time() - train_start
        print(f"  epoch {epoch:2d}: train_loss={train_loss:.4f} "
              f"val_auc={val_auc:.4f}  ({elapsed:.0f}s elapsed)")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "config": {
                    "max_len": MAX_LEN, "n_channels": N_CHANNELS,
                    "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE, "seed": SEED,
                },
            }, CNN_CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[cnn] early stop at epoch {epoch} "
                      f"(no val improvement for {PATIENCE} epochs)")
                break

    print(f"[cnn] best val AUC {best_auc:.4f} at epoch {best_epoch}; "
          f"checkpoint -> {CNN_CHECKPOINT.name}")

    # Cache the test split's CNN-ready tensors for eval_cnn.py to reuse
    cache = PROCESSED_DIR / "cnn_test.npz"
    np.savez(
        cache,
        X_test=X_test.numpy(),
        y_test=y_test_np,
        lengths=np.array([len(seq_db[s]) for s in test_ids], dtype=np.int32),
        charges=np.zeros(len(test_ids), dtype=np.float32),
    )
    print(f"[cnn] cached test tensors -> {cache.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
