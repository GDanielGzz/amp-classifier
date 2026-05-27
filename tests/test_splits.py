"""Cluster-purity tests for the train/val/test splits.

The single most important property in this project: no mmseqs2 cluster
spans multiple splits. That property is what makes the held-out test
AUC honestly comparable across models. Random splits inflate AUC by
10-15 points; this test catches it if make_splits.py regresses.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"


def _read_split(path):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            r["cluster_id"] = int(r["cluster_id"])
            r["is_amp"] = int(r["is_amp"])
            rows.append(r)
    return rows


pytestmark = pytest.mark.skipif(
    not (TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists()),
    reason="splits not generated yet; run dev.bat splits",
)


@pytest.fixture(scope="module")
def split_rows():
    return {
        "train": _read_split(TRAIN_CSV),
        "val": _read_split(VAL_CSV),
        "test": _read_split(TEST_CSV),
    }


def test_no_cluster_spans_splits(split_rows):
    """The non-negotiable property: each cluster lives in exactly one split."""
    cluster_to_splits = defaultdict(set)
    for split_name, rows in split_rows.items():
        for r in rows:
            cluster_to_splits[r["cluster_id"]].add(split_name)
    leakers = {cid: splits for cid, splits in cluster_to_splits.items() if len(splits) > 1}
    assert not leakers, (
        f"{len(leakers)} clusters span multiple splits "
        f"(eval would be dishonest). First three: "
        f"{list(leakers.items())[:3]}"
    )


def test_no_sequence_in_multiple_splits(split_rows):
    """A sequence_id must appear in exactly one split."""
    seen = defaultdict(set)
    for split_name, rows in split_rows.items():
        for r in rows:
            seen[r["sequence_id"]].add(split_name)
    dupes = {sid: splits for sid, splits in seen.items() if len(splits) > 1}
    assert not dupes, f"{len(dupes)} sequence_ids appear in multiple splits"


def test_split_fractions_within_tolerance(split_rows):
    """Per-class split sizes are within +/- 5pp of 80/10/10 target."""
    counts_amp = {name: sum(1 for r in rows if r["is_amp"] == 1) for name, rows in split_rows.items()}
    counts_neg = {name: sum(1 for r in rows if r["is_amp"] == 0) for name, rows in split_rows.items()}
    total_amp = sum(counts_amp.values())
    total_neg = sum(counts_neg.values())
    targets = {"train": 0.80, "val": 0.10, "test": 0.10}
    for split_name, target in targets.items():
        amp_frac = counts_amp[split_name] / total_amp if total_amp else 0.0
        neg_frac = counts_neg[split_name] / total_neg if total_neg else 0.0
        assert abs(amp_frac - target) <= 0.05, (
            f"{split_name} AMP fraction {amp_frac:.3f} not within 0.05 of target {target}"
        )
        assert abs(neg_frac - target) <= 0.05, (
            f"{split_name} NEG fraction {neg_frac:.3f} not within 0.05 of target {target}"
        )


def test_splits_are_nonempty(split_rows):
    """Each split has at least 50 sequences of each class (for stable bootstrap CIs)."""
    for split_name, rows in split_rows.items():
        n_amp = sum(1 for r in rows if r["is_amp"] == 1)
        n_neg = sum(1 for r in rows if r["is_amp"] == 0)
        assert n_amp >= 50, f"{split_name} has only {n_amp} positives (need >=50)"
        assert n_neg >= 50, f"{split_name} has only {n_neg} negatives (need >=50)"


def test_split_files_have_consistent_schema(split_rows):
    """All three split files use the same column names."""
    expected_keys = {"sequence_id", "cluster_id", "is_amp"}
    for split_name, rows in split_rows.items():
        if rows:
            assert set(rows[0].keys()) == expected_keys, (
                f"{split_name} schema mismatch: {set(rows[0].keys())}"
            )
