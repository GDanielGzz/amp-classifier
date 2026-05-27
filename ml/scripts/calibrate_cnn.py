"""Phase 2A: Platt calibration of the CNN.

The CNN trained with BCE pos_weight + BatchNorm produces a sharply
shifted score distribution where the val-MCC-optimal threshold is
~0.05, not 0.5. AUC is honest but raw P(AMP) is uninterpretable as a
calibrated probability.

Platt scaling fits LogisticRegression on (CNN logit, true label) using
the val split, learning a 2-parameter sigmoid that maps raw logits to
calibrated probabilities. After calibration the CNN's threshold-0.5
call should agree with the val-tuned threshold call, and the demo can
show a single intuitive P(AMP) instead of "raw + tuned threshold".

Reference:
  Platt JC (1999) "Probabilistic outputs for support vector machines
  and comparisons to regularized likelihood methods." In Smola et al.,
  Advances in Large Margin Classifiers.

Saves: ml/checkpoints/cnn_calibrator.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.scripts.train_cnn import (  # noqa: E402
    CNN, load_sequence_db, load_split_csv, encode_split,
)
from ml.eval_common import find_optimal_threshold  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
CHECKPOINTS = PROJECT_ROOT / "ml" / "checkpoints"
CNN_CHECKPOINT = CHECKPOINTS / "cnn_best.pt"
CALIBRATOR_PATH = CHECKPOINTS / "cnn_calibrator.joblib"

SEED = 42


def cnn_logits(model, X, batch_size=128):
    """Return the model's raw pre-sigmoid logits for X."""
    logits = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            logits.append(model(X[i:i + batch_size]).numpy())
    return np.concatenate(logits)


def main():
    if not CNN_CHECKPOINT.exists():
        print(f"[calibrate] {CNN_CHECKPOINT} missing. Run dev.bat cnn first.")
        return 1

    print("[calibrate] loading val sequences ...")
    seq_db = load_sequence_db()
    val_ids, y_val = load_split_csv(SPLITS_DIR / "val.csv")
    X_val, _ = encode_split(seq_db, val_ids, y_val)

    ckpt = torch.load(CNN_CHECKPOINT, map_location="cpu", weights_only=False)
    model = CNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[calibrate] loaded CNN epoch {ckpt.get('epoch', '?')}, "
          f"val_auc={ckpt.get('val_auc', '?')}")

    # Get CNN logits + raw probabilities on val
    logits_val = cnn_logits(model, X_val)
    raw_proba_val = 1.0 / (1.0 + np.exp(-logits_val))

    # Pre-calibration diagnostics
    t_pre, mcc_pre = find_optimal_threshold(y_val, raw_proba_val, metric="mcc")
    print(f"[calibrate] BEFORE Platt: optimal val-MCC threshold t={t_pre:.3f}, "
          f"val MCC={mcc_pre:.3f}")

    # Fit Platt scaler: LogisticRegression on logits (1-D feature) -> labels
    # Note: we fit on LOGITS not on raw probabilities — logits live in unbounded
    # space which is what Platt's sigmoid is designed to map.
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(logits_val.reshape(-1, 1), y_val)
    cal_proba_val = platt.predict_proba(logits_val.reshape(-1, 1))[:, 1]

    # Post-calibration diagnostics
    t_post, mcc_post = find_optimal_threshold(y_val, cal_proba_val, metric="mcc")
    mcc_at_half = float(_mcc(y_val, (cal_proba_val >= 0.5).astype(int)))
    print(f"[calibrate] AFTER  Platt: optimal val-MCC threshold t={t_post:.3f}, "
          f"val MCC={mcc_post:.3f}; MCC@0.5={mcc_at_half:.3f}")
    print(f"[calibrate] Platt params: intercept={platt.intercept_[0]:+.3f}, "
          f"coef={platt.coef_[0, 0]:+.3f}")

    joblib.dump(platt, CALIBRATOR_PATH)
    print(f"[calibrate] saved {CALIBRATOR_PATH}")
    return 0


def _mcc(y_true, y_pred):
    """Inline MCC to avoid importing sklearn.metrics here."""
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y_true, y_pred)


if __name__ == "__main__":
    sys.exit(main())
