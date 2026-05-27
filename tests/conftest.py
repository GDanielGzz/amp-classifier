"""Shared pytest fixtures for the AMP Classifier test suite.

Tests in ``tests/test_features.py`` use the known-AMP fixtures in
``tests/fixtures/known_amps.py`` for regression assertions on the runtime
scoring path (Step 6 acceptance). Tests in ``tests/test_splits.py`` exercise
the cluster-purity property — the property that makes the held-out
evaluation honest (Step 5 acceptance).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``ml.*`` importable from tests without forcing a `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
