"""Smoke tests for the Phase 1 scaffold.

These verify package imports, public surface, and the known-peptide
fixture. The heavy feature math lives in tests/test_features.py and
the cluster-purity property lives in tests/test_splits.py.
"""
from __future__ import annotations

import pytest

from ml import features
from tests.fixtures.known_amps import (
    KNOWN_AMPS,
    KnownPeptide,
    get_demo_examples,
    get_known_amps,
)


def test_features_module_importable() -> None:
    assert features.CANONICAL_ALPHABET == "ACDEFGHIKLMNPQRSTVWY"
    assert isinstance(features.FEATURE_NAMES, tuple)
    assert callable(features.extract_features)
    assert callable(features.validate_alphabet)


def test_validate_alphabet_accepts_canonical() -> None:
    assert features.validate_alphabet("MAGAININ")
    assert features.validate_alphabet("LL")
    assert features.validate_alphabet("")


def test_validate_alphabet_rejects_non_canonical() -> None:
    assert not features.validate_alphabet("MAGAXININ")
    assert not features.validate_alphabet("MABCDEF")
    assert not features.validate_alphabet("seq")


def test_extract_features_implemented() -> None:
    """Step 6 lands the real implementation; returns ~430 named features."""
    result = features.extract_features("MAGAININ")
    assert isinstance(result, dict)
    assert len(result) == len(features.FEATURE_NAMES) == 428


def test_known_amps_fixture_well_formed() -> None:
    assert len(KNOWN_AMPS) >= 8
    seen_names = set()
    for peptide in KNOWN_AMPS:
        assert isinstance(peptide, KnownPeptide)
        assert peptide.name
        assert peptide.name not in seen_names
        seen_names.add(peptide.name)
        assert peptide.sequence
        assert peptide.sequence.isupper()
        assert features.validate_alphabet(peptide.sequence)
        assert peptide.citation


def test_known_amps_has_both_classes() -> None:
    positives = [p for p in KNOWN_AMPS if p.is_amp]
    negatives = [p for p in KNOWN_AMPS if not p.is_amp]
    assert len(positives) >= 5
    assert len(negatives) >= 2


def test_demo_examples_are_one_of_each() -> None:
    examples = get_demo_examples()
    assert len(examples) == 3
    amp_count = sum(1 for p in examples if p.is_amp)
    assert amp_count == 2
    assert len({p.name for p in examples}) == 3


def test_get_known_amps_returns_immutable() -> None:
    assert isinstance(get_known_amps(), tuple)
