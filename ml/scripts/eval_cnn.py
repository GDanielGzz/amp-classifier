"""Evaluate the trained CNN on the held-out test split.

Two evaluation modes:
  - Raw + tuned threshold (legacy path, kept for transparency)
  - Platt-calibrated (preferred when cnn_calibrator.joblib exists)

After Phase 2A's `calibrate_cnn.py` runs, the Platt scaler maps raw CNN
logits to calibrated P(AMP), and metrics at the standard t=0.5 threshold
become directly interpretable. AUC is threshold-independent so it's
unchanged.

Writes ``docs/cnn_results.md`` with both reports for honest comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.eval_common import (  # noqa: E402
    LENGTH_BINS, CHARGE_BINS,
    compute_all_metrics, stratify,
    find_optimal_threshold,
    format_metrics_table, format_stratified_table,
)
from ml.scripts.train_cnn import (  # noqa: E402
    CNN, MAX_LEN, N_CHANNELS,
    load_sequence_db, load_split_csv, encode_split,
)

SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
CHECKPOINTS = PROJECT_ROOT / "ml" / "checkpoints"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_PATH = DOCS_DIR / "cnn_results.md"
CNN_CHECKPOINT = CHECKPOINTS / "cnn_best.pt"
CALIBRATOR_PATH = CHECKPOINTS / "cnn_calibrator.joblib"

SEED = 42
N_BOOTSTRAP = 1000


def cnn_logits(model, X, batch_size=128):
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            out.append(model(X[i:i + batch_size]).numpy())
    return np.concatenate(out)


def main():
    if not CNN_CHECKPOINT.exists():
        print(f"[eval-cnn] {CNN_CHECKPOINT} missing. Run dev.bat cnn first.")
        return 1
    feat_cache = PROCESSED_DIR / "features_test.npz"
    if not feat_cache.exists():
        print(f"[eval-cnn] {feat_cache} missing. Run dev.bat baseline first.")
        return 1

    print("[eval-cnn] loading sequences ...")
    seq_db = load_sequence_db()
    val_ids,  y_val_np  = load_split_csv(SPLITS_DIR / "val.csv")
    test_ids, y_test_np = load_split_csv(SPLITS_DIR / "test.csv")
    X_val,  _ = encode_split(seq_db, val_ids,  y_val_np)
    X_test, _ = encode_split(seq_db, test_ids, y_test_np)

    feat = np.load(feat_cache, allow_pickle=True)
    lengths = feat["lengths"]
    charges = feat["charges"]

    checkpoint = torch.load(CNN_CHECKPOINT, map_location="cpu", weights_only=False)
    model = CNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[eval-cnn] loaded checkpoint epoch {checkpoint.get('epoch', '?')}, "
          f"val_auc={checkpoint.get('val_auc', '?')}")

    # Raw logits + raw sigmoid probabilities
    val_logits = cnn_logits(model, X_val)
    test_logits = cnn_logits(model, X_test)
    val_raw = 1.0 / (1.0 + np.exp(-val_logits))
    test_raw = 1.0 / (1.0 + np.exp(-test_logits))

    # Tuned threshold (raw scores)
    tuned_t, tuned_val_mcc = find_optimal_threshold(y_val_np, val_raw, metric="mcc")
    print(f"[eval-cnn] tuned MCC threshold (raw): t={tuned_t:.3f}  "
          f"val_MCC={tuned_val_mcc:.3f}")

    metrics_tuned = compute_all_metrics(
        y_test_np, test_raw, n_bootstrap=N_BOOTSTRAP, seed=SEED, threshold=tuned_t,
    )
    length_strata_tuned = stratify(
        y_test_np, test_raw, lengths, LENGTH_BINS,
        n_bootstrap=N_BOOTSTRAP, seed=SEED, threshold=tuned_t,
    )
    charge_strata_tuned = stratify(
        y_test_np, test_raw, charges, CHARGE_BINS,
        n_bootstrap=N_BOOTSTRAP, seed=SEED, threshold=tuned_t,
    )
    metrics_raw_at_half = compute_all_metrics(
        y_test_np, test_raw, n_bootstrap=N_BOOTSTRAP, seed=SEED, threshold=0.5,
    )
    print(f"[eval-cnn] RAW  AUC: {metrics_tuned['auc']}  "
          f"MCC@tuned: {metrics_tuned['mcc']}  MCC@0.5: {metrics_raw_at_half['mcc']}")

    # Calibrated (if Platt scaler exists)
    metrics_cal = None
    length_strata_cal = None
    charge_strata_cal = None
    if CALIBRATOR_PATH.exists():
        platt = joblib.load(CALIBRATOR_PATH)
        test_cal = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
        metrics_cal = compute_all_metrics(
            y_test_np, test_cal, n_bootstrap=N_BOOTSTRAP, seed=SEED, threshold=0.5,
        )
        length_strata_cal = stratify(
            y_test_np, test_cal, lengths, LENGTH_BINS,
            n_bootstrap=N_BOOTSTRAP, seed=SEED, threshold=0.5,
        )
        charge_strata_cal = stratify(
            y_test_np, test_cal, charges, CHARGE_BINS,
            n_bootstrap=N_BOOTSTRAP, seed=SEED, threshold=0.5,
        )
        print(f"[eval-cnn] CAL  AUC: {metrics_cal['auc']}  "
              f"MCC@0.5: {metrics_cal['mcc']}")
    else:
        print(f"[eval-cnn] no calibrator at {CALIBRATOR_PATH.name}; "
              "raw + tuned threshold only. Run `python ml/scripts/calibrate_cnn.py` "
              "to add Platt calibration.")

    sections = [
        "# CNN results",
        "",
        f"*Auto-generated by `ml/scripts/eval_cnn.py`. Bootstrap n="
        f"{N_BOOTSTRAP}, seed={SEED}.*",
        "",
        f"Architecture (V2): 3x Conv1D (96, 192, 192 channels) + BatchNorm + "
        f"ReLU + Dropout(0.4); global max pool; FC(192->96) + BN + FC(96->1). "
        f"One-hot, MAX_LEN={MAX_LEN}, {N_CHANNELS} channels.",
        "",
        f"Checkpoint: epoch {checkpoint.get('epoch', '?')}, "
        f"val AUC {checkpoint.get('val_auc', float('nan')):.4f}.",
        "",
    ]

    if metrics_cal is not None:
        sections.extend([
            "## Phase 2A — Platt-calibrated (preferred)",
            "",
            "Platt scaler fit on val logits. Calibrated P(AMP) is now "
            "interpretable at the standard t=0.5 threshold.",
            "",
            format_metrics_table(metrics_cal),
            "",
            "### Stratified by length (calibrated, t=0.5)",
            "",
            format_stratified_table(length_strata_cal, "length"),
            "",
            "### Stratified by net charge at pH 7 (calibrated, t=0.5)",
            "",
            format_stratified_table(charge_strata_cal, "charge"),
            "",
        ])

    sections.extend([
        f"## Raw + val-tuned threshold (t={tuned_t:.3f})",
        "",
        "Pre-calibration baseline. Raw CNN sigmoid output is concentrated "
        "near zero (BCE pos_weight + BatchNorm); the val-MCC-optimal "
        "threshold is shown above. AUC is threshold-independent.",
        "",
        format_metrics_table(metrics_tuned),
        "",
        "### Same metrics at default t=0.5 (uncalibrated, for transparency)",
        "",
        format_metrics_table(metrics_raw_at_half),
        "",
        "### Stratified by length (raw scores, tuned threshold)",
        "",
        format_stratified_table(length_strata_tuned, "length"),
        "",
        "### Stratified by net charge at pH 7 (raw scores, tuned threshold)",
        "",
        format_stratified_table(charge_strata_tuned, "charge"),
        "",
        "## Comparison with baselines",
        "",
        "See `docs/baseline_results.md` for the LogReg / RandomForest / "
        "XGBoost numbers. The unified comparison table lives in "
        "`docs/model_card.md`.",
        "",
    ])

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[eval-cnn] wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
