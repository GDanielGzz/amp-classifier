"""Train the three baseline classifiers: LogReg, RandomForest, XGBoost.

All three consume the same 428-dim engineered feature vector from
``ml/features.py``. Features are extracted once per split and cached to
``ml/data/processed/features_{train,val,test}.npz`` so retraining is
cheap.

Saved checkpoints:

  ml/checkpoints/baseline_logreg.joblib
  ml/checkpoints/baseline_rf.joblib
  ml/checkpoints/baseline_xgb.json

Class imbalance is handled with ``class_weight='balanced'`` for sklearn
models and ``scale_pos_weight`` for XGBoost. The XGBoost training uses
early stopping on the val split's AUC.

Seed=42 across the board. Re-run via ``dev.bat baseline`` or
``python ml/scripts/train_baseline.py``.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml import features as F  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
CLUSTERS_DIR = PROJECT_ROOT / "ml" / "data" / "clusters"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
CHECKPOINTS = PROJECT_ROOT / "ml" / "checkpoints"
COMBINED_FASTA = CLUSTERS_DIR / "combined.fasta"

SEED = 42


def parse_fasta(path):
    """Stream (id, sequence) from a FASTA file."""
    header = None
    seq_chunks = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip().split()[0]
                seq_chunks = []
            elif line.strip():
                seq_chunks.append(line.strip())
    if header is not None:
        yield header, "".join(seq_chunks)


def load_sequence_db():
    """Build id -> sequence map from combined.fasta."""
    return {sid: seq.upper() for sid, seq in parse_fasta(COMBINED_FASTA)}


def load_split(split_csv: Path, seq_db: "dict[str, str]"):
    """Load (sequences, labels, ids) for one split."""
    seqs, labels, ids = [], [], []
    with split_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row["sequence_id"]
            seqs.append(seq_db[sid])
            labels.append(int(row["is_amp"]))
            ids.append(sid)
    return seqs, np.array(labels, dtype=int), ids


def featurise_and_cache(split_name: str, seqs, labels, ids):
    """Extract features for a split; cache to .npz keyed by split."""
    cache = PROCESSED_DIR / f"features_{split_name}.npz"
    if cache.exists():
        print(f"[features] cache hit: {cache.name}")
        data = np.load(cache, allow_pickle=True)
        return data["X"], data["y"], data["lengths"], data["charges"], data["ids"]
    print(f"[features] computing {split_name} ({len(seqs)} sequences) ...")
    t0 = time.time()
    X = np.array([F.feature_vector(s) for s in seqs], dtype=np.float32)
    lengths = np.array([len(s) for s in seqs], dtype=np.int32)
    charges = np.array([F.net_charge_at_ph7(s) for s in seqs], dtype=np.float32)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=labels, lengths=lengths, charges=charges,
             ids=np.array(ids, dtype=object))
    print(f"[features] {split_name}: {X.shape}, {time.time()-t0:.1f}s; cached to {cache.name}")
    return X, labels, lengths, charges, np.array(ids, dtype=object)


def train_logreg(X_train, y_train):
    print("[train] LogReg (L2, class_weight=balanced, max_iter=2000)")
    t0 = time.time()
    model = LogisticRegression(
        C=1.0, penalty="l2", class_weight="balanced",
        max_iter=2000, random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"[train] LogReg done in {time.time()-t0:.1f}s")
    return model


def train_rf(X_train, y_train):
    print("[train] RandomForest (200 trees, class_weight=balanced)")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"[train] RandomForest done in {time.time()-t0:.1f}s")
    return model


def train_xgb(X_train, y_train, X_val, y_val):
    print("[train] XGBoost (500 trees max, early stop on val AUC, patience 20)")
    t0 = time.time()
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale = neg / max(1, pos)
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="auc",
        early_stopping_rounds=20,
        tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"[train] XGBoost done in {time.time()-t0:.1f}s "
          f"(best_iteration={model.best_iteration})")
    return model


def main():
    if not COMBINED_FASTA.exists():
        print("[train] combined.fasta missing. Run dev.bat cluster first.")
        return 1
    if not (SPLITS_DIR / "train.csv").exists():
        print("[train] splits missing. Run dev.bat splits first.")
        return 1

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    seq_db = load_sequence_db()
    print(f"[train] sequence DB: {len(seq_db)} sequences from combined.fasta")

    splits = {}
    for name in ("train", "val", "test"):
        seqs, labels, ids = load_split(SPLITS_DIR / f"{name}.csv", seq_db)
        X, y, lengths, charges, sid = featurise_and_cache(name, seqs, labels, ids)
        splits[name] = {"X": X, "y": y, "lengths": lengths,
                        "charges": charges, "ids": sid}

    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    print(f"[train] X_train shape: {X_train.shape}; "
          f"pos={int(y_train.sum())}, neg={int((y_train == 0).sum())}")

    # NaN-safe: instability_index can be NaN on degenerate inputs.
    # Replace NaN with the column median computed on train.
    col_median = np.nanmedian(X_train, axis=0)
    for split in splits.values():
        nan_mask = np.isnan(split["X"])
        if nan_mask.any():
            split["X"] = np.where(nan_mask, col_median, split["X"])
    X_train, X_val = splits["train"]["X"], splits["val"]["X"]

    logreg = train_logreg(X_train, y_train)
    joblib.dump(logreg, CHECKPOINTS / "baseline_logreg.joblib")
    print(f"[train] saved baseline_logreg.joblib")

    rf = train_rf(X_train, y_train)
    joblib.dump(rf, CHECKPOINTS / "baseline_rf.joblib")
    print(f"[train] saved baseline_rf.joblib")

    xgb = train_xgb(X_train, y_train, X_val, y_val)
    xgb.save_model(str(CHECKPOINTS / "baseline_xgb.json"))
    print(f"[train] saved baseline_xgb.json")

    print("[train] all three baselines trained and saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
