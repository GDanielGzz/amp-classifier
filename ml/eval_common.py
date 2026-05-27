"""Shared evaluation machinery for the baseline and CNN models."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, recall_score, f1_score,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5

LENGTH_BINS = [
    ("short_le20", lambda L: L <= 20),
    ("medium_21_50", lambda L: 20 < L <= 50),
    ("long_gt50", lambda L: L > 50),
]
CHARGE_BINS = [
    ("charge_le0", lambda c: c <= 0),
    ("charge_0_5", lambda c: 0 < c <= 5),
    ("charge_gt5", lambda c: c > 5),
]


@dataclass(frozen=True)
class MetricWithCI:
    point: float
    ci_low: float
    ci_high: float

    def __str__(self):
        return f"{self.point:.3f} [{self.ci_low:.3f}-{self.ci_high:.3f}]"


def _compute_metric(y_true, y_score, metric, threshold=DEFAULT_THRESHOLD):
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if metric == "auc":
        if n_pos == 0 or n_neg == 0:
            return None
        return float(roc_auc_score(y_true, y_score))
    y_pred = (y_score >= threshold).astype(int)
    if metric == "mcc":
        if n_pos == 0 or n_neg == 0:
            return None
        n_pred_pos = int(y_pred.sum())
        if n_pred_pos == 0 or n_pred_pos == len(y_pred):
            return 0.0
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "sensitivity":
        if n_pos == 0:
            return None
        return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    if metric == "specificity":
        if n_neg == 0:
            return None
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        return tn / max(1, tn + fp)
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"unknown metric: {metric}")


def bootstrap_metric(y_true, y_score, *, metric,
                     n_bootstrap=DEFAULT_BOOTSTRAP_N, seed=DEFAULT_SEED,
                     threshold=DEFAULT_THRESHOLD):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    point = _compute_metric(y_true, y_score, metric, threshold=threshold)
    point_val = float("nan") if point is None else float(point)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = _compute_metric(y_true[idx], y_score[idx], metric, threshold=threshold)
        if m is not None:
            samples.append(m)
    if not samples:
        return MetricWithCI(point=point_val, ci_low=float("nan"), ci_high=float("nan"))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[max(0, int(0.975 * len(samples)) - 1)]
    return MetricWithCI(point=point_val, ci_low=float(lo), ci_high=float(hi))


def compute_all_metrics(y_true, y_score, *,
                        n_bootstrap=DEFAULT_BOOTSTRAP_N, seed=DEFAULT_SEED,
                        threshold=DEFAULT_THRESHOLD):
    return {
        m: bootstrap_metric(y_true, y_score, metric=m,
                            n_bootstrap=n_bootstrap, seed=seed, threshold=threshold)
        for m in ("auc", "mcc", "sensitivity", "specificity", "f1")
    }


def find_optimal_threshold(y_true, y_score, metric="mcc", n_steps=181):
    """Sweep thresholds 0.05..0.95 and pick the one that maximizes ``metric``.
    Returns (threshold, metric_value)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    best_t, best_v = 0.5, -float("inf")
    for i in range(n_steps):
        t = 0.05 + (0.90 / (n_steps - 1)) * i
        v = _compute_metric(y_true, y_score, metric, threshold=t)
        if v is not None and v > best_v:
            best_v = v
            best_t = t
    return float(best_t), float(best_v)


def stratify(y_true, y_score, axis, bins, *,
             n_bootstrap=DEFAULT_BOOTSTRAP_N, seed=DEFAULT_SEED,
             threshold=DEFAULT_THRESHOLD):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    axis = np.asarray(axis, dtype=float)
    out = {}
    for name, pred in bins:
        mask = np.array([pred(v) for v in axis])
        n = int(mask.sum())
        if n < 20:
            out[name] = {"n": n, "note": "skipped (n<20)"}
            continue
        out[name] = {
            "n": n,
            "auc": bootstrap_metric(
                y_true[mask], y_score[mask], metric="auc",
                n_bootstrap=n_bootstrap, seed=seed, threshold=threshold,
            ),
            "mcc": bootstrap_metric(
                y_true[mask], y_score[mask], metric="mcc",
                n_bootstrap=n_bootstrap, seed=seed, threshold=threshold,
            ),
        }
    return out


def format_metrics_table(metrics):
    rows = ["| metric | value (95% CI) |", "|---|---|"]
    for name in ("auc", "mcc", "sensitivity", "specificity", "f1"):
        if name in metrics:
            rows.append(f"| {name} | {metrics[name]} |")
    return "\n".join(rows)


def format_stratified_table(strata, axis_name):
    rows = [f"| {axis_name} bin | n | auc | mcc |", "|---|---|---|---|"]
    for name, data in strata.items():
        if "note" in data:
            rows.append(f"| {name} | {data['n']} | {data['note']} |  |")
        else:
            rows.append(f"| {name} | {data['n']} | {data['auc']} | {data['mcc']} |")
    return "\n".join(rows)
