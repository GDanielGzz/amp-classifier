"""Engineered physicochemical and composition features for AMP classification.

Shared between training and runtime scoring. The Gradio demo (``app.py``)
and the baseline training script both import ``extract_features`` from
here. There is exactly one feature extractor in this project.

Feature set (428 total, matching the iAMPpred / AMP-Scanner stack):

  20   amino acid composition (fraction per canonical residue)
  400  dipeptide composition (fraction per ordered AA pair)
  4    physicochemical via Biopython.ProtParam (length, pI, instability,
       aromaticity)
  4    derived scalars (net charge at pH 7, Kyte-Doolittle mean,
       Boman index, Eisenberg hydrophobic moment max over a sliding
       11-residue window)

References
----------
Kyte J, Doolittle RF (1982) J. Mol. Biol. 157(1), 105-132.
  Hydrophobicity scale used for the mean and for the Eisenberg moment.
Eisenberg D, Weiss RM, Terwilliger TC (1984) PNAS 81(1), 140-144.
  Hydrophobic moment: amphipathicity detector for alpha-helical AMPs.
  Standard window 11 residues, periodicity 100 degrees.
Boman HG (2003) J. Internal Medicine 254(3), 197-215.
  Boman index: average binding free energy contribution per residue,
  signed so that positive values indicate high protein-binding (AMP-like)
  potential.
Meher PK et al. (2017) Scientific Reports 7, 42362.
  iAMPpred — engineered-feature reference architecture.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from Bio.SeqUtils.ProtParam import ProteinAnalysis

CANONICAL_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"
SELENOCYSTEINE_OK = "U"

# Kyte-Doolittle hydropathy index (Kyte & Doolittle 1982 Table 1).
KYTE_DOOLITTLE: dict[str, float] = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# Boman 2003 sidechain transfer free energy table (kcal/mol, water -> cyclohexane).
# The Boman INDEX is the average of the NEGATIVES of these values (so high
# index = high binding potential = AMP-like).
BOMAN_TRANSFER: dict[str, float] = {
    "L":  -4.92, "I":  -4.92, "V":  -4.04, "F":  -2.98, "M": -2.35,
    "W":  -2.33, "A":  -1.81, "C":  -1.28, "G":  -0.94, "Y":  0.14,
    "T":   2.57, "S":   3.40, "H":   4.66, "Q":   5.54, "K":  5.55,
    "N":   6.64, "E":   6.81, "D":   8.72, "R":  14.92, "P":  0.00,
}

EISENBERG_WINDOW = 11
EISENBERG_OMEGA_DEG = 100.0  # alpha-helix periodicity (100 deg per residue)


def _build_feature_names() -> tuple[str, ...]:
    names: list[str] = [f"aa_{aa}" for aa in CANONICAL_ALPHABET]
    for a in CANONICAL_ALPHABET:
        for b in CANONICAL_ALPHABET:
            names.append(f"dp_{a}{b}")
    names.extend([
        "length",
        "isoelectric_point",
        "instability_index",
        "aromaticity",
        "net_charge_ph7",
        "kyte_doolittle_mean",
        "boman_index",
        "eisenberg_moment_max",
    ])
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _build_feature_names()


# ---------------------------------------------------------------------------
# Alphabet validation
# ---------------------------------------------------------------------------


def validate_alphabet(sequence: str) -> bool:
    """True if ``sequence`` uses only the canonical 20-AA alphabet."""
    return all(aa in CANONICAL_ALPHABET for aa in sequence)


# ---------------------------------------------------------------------------
# Composition features
# ---------------------------------------------------------------------------


def aa_composition(sequence: str) -> dict[str, float]:
    """Fraction of each canonical residue. Sums to 1.0 for non-empty input."""
    n = len(sequence)
    if n == 0:
        return {f"aa_{aa}": 0.0 for aa in CANONICAL_ALPHABET}
    counts = Counter(sequence)
    return {f"aa_{aa}": counts.get(aa, 0) / n for aa in CANONICAL_ALPHABET}


def dipeptide_composition(sequence: str) -> dict[str, float]:
    """Fraction of each ordered AA pair (overlapping windows of 2)."""
    keys = [f"dp_{a}{b}" for a in CANONICAL_ALPHABET for b in CANONICAL_ALPHABET]
    if len(sequence) < 2:
        return {k: 0.0 for k in keys}
    pairs = Counter(sequence[i:i + 2] for i in range(len(sequence) - 1))
    total = len(sequence) - 1
    return {k: pairs.get(k[3:], 0) / total for k in keys}


# ---------------------------------------------------------------------------
# Biopython-backed physicochemical features
# ---------------------------------------------------------------------------


def physicochemical(sequence: str) -> dict[str, float]:
    """length, isoelectric_point, instability_index, aromaticity.

    Uses Biopython's ``ProteinAnalysis``. Returns NaN for instability /
    aromaticity on degenerate inputs (single residues, etc.) by catching
    the underlying ZeroDivisionError; downstream code should treat NaN
    as missing and impute or drop.
    """
    pa = ProteinAnalysis(sequence)
    try:
        pi = pa.isoelectric_point()
    except Exception:
        pi = 7.0
    try:
        inst = pa.instability_index()
    except Exception:
        inst = float("nan")
    try:
        arom = pa.aromaticity()
    except Exception:
        arom = 0.0
    return {
        "length": float(len(sequence)),
        "isoelectric_point": float(pi),
        "instability_index": float(inst),
        "aromaticity": float(arom),
    }


def net_charge_at_ph7(sequence: str) -> float:
    """Net charge at pH 7 via Biopython's Henderson-Hasselbalch implementation."""
    pa = ProteinAnalysis(sequence)
    try:
        return float(pa.charge_at_pH(7.0))
    except Exception:
        # Fallback: simple side-chain count
        charge = 0.0
        for aa in sequence:
            if aa in ("K", "R"):
                charge += 1.0
            elif aa in ("D", "E"):
                charge -= 1.0
            elif aa == "H":
                charge += 0.1
        return charge


# ---------------------------------------------------------------------------
# Hydrophobicity scalars
# ---------------------------------------------------------------------------


def kyte_doolittle_mean(sequence: str) -> float:
    """Mean Kyte-Doolittle hydropathy across the sequence."""
    if not sequence:
        return 0.0
    return sum(KYTE_DOOLITTLE[aa] for aa in sequence) / len(sequence)


def boman_index(sequence: str) -> float:
    """Boman 2003 binding-potential index (kcal/mol per residue).

    Sum of the NEGATIVES of the transfer free energies divided by length.
    Positive values indicate high protein-binding potential (AMP-like).
    Proline contribution treated as 0 per Boman's convention.
    """
    if not sequence:
        return 0.0
    return -sum(BOMAN_TRANSFER[aa] for aa in sequence) / len(sequence)


def _eisenberg_moment_window(window_seq: str, omega: float) -> float:
    """Hydrophobic moment for a single window of residues."""
    sin_sum = 0.0
    cos_sum = 0.0
    for i, aa in enumerate(window_seq):
        h = KYTE_DOOLITTLE[aa]
        angle = i * omega
        sin_sum += h * math.sin(angle)
        cos_sum += h * math.cos(angle)
    return math.sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / len(window_seq)


def eisenberg_moment_max(sequence: str) -> float:
    """Maximum Eisenberg hydrophobic moment over 11-residue windows.

    For sequences shorter than the window, compute the whole-sequence
    moment. Periodicity is 100 deg per residue (alpha-helix).
    """
    if not sequence:
        return 0.0
    omega = math.radians(EISENBERG_OMEGA_DEG)
    n = len(sequence)
    if n < EISENBERG_WINDOW:
        return _eisenberg_moment_window(sequence, omega)
    return max(
        _eisenberg_moment_window(sequence[i:i + EISENBERG_WINDOW], omega)
        for i in range(n - EISENBERG_WINDOW + 1)
    )


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------


def extract_features(sequence: str) -> dict[str, float]:
    """Compute the engineered feature vector for a peptide sequence.

    Args:
        sequence: Canonical 20-AA peptide, upper case. ``validate_data.py``
            (Step 3) ensures this property before training.

    Returns:
        Dict keyed by ``FEATURE_NAMES``, ~428 entries.
    """
    if not validate_alphabet(sequence):
        raise ValueError(
            f"sequence contains non-canonical characters: "
            f"{sorted(set(sequence) - set(CANONICAL_ALPHABET))}"
        )
    out: dict[str, float] = {}
    out.update(aa_composition(sequence))
    out.update(dipeptide_composition(sequence))
    out.update(physicochemical(sequence))
    out["net_charge_ph7"] = net_charge_at_ph7(sequence)
    out["kyte_doolittle_mean"] = kyte_doolittle_mean(sequence)
    out["boman_index"] = boman_index(sequence)
    out["eisenberg_moment_max"] = eisenberg_moment_max(sequence)
    return out


def feature_vector(sequence: str) -> list[float]:
    """Return features in ``FEATURE_NAMES`` order (for sklearn / torch input)."""
    f = extract_features(sequence)
    return [f[name] for name in FEATURE_NAMES]


def feature_matrix(sequences: Sequence[str]) -> list[list[float]]:
    """Vectorised wrapper for a batch of sequences."""
    return [feature_vector(s) for s in sequences]
