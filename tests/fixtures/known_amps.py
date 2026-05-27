"""Published peptides with literature-grade AMP labels.

Used by ``tests/test_features.py`` and ``tests/test_models.py`` as
regression fixtures on the runtime scoring path. Each entry carries the
canonical sequence plus a short provenance line so the test is its own
documentation.

The list is intentionally small — 10 to 20 entries is plenty for
regression. The full evaluation runs against the cluster-aware held-out
test set produced by ``ml/scripts/make_splits.py`` (Step 5), not this
fixture. These exist to catch silly regressions in feature extraction or
runtime preprocessing, not to be a benchmark.

Step 6 of ``HANDOFF.md`` will use a subset of these (one classical AMP,
one classical non-AMP fragment, one edge case) as the three pre-populated
examples in the Gradio demo.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownPeptide:
    """A published peptide with a literature-grade AMP / non-AMP label."""

    name: str
    sequence: str
    is_amp: bool
    citation: str
    notes: str = ""


# Classical antimicrobial peptides. Sequences from APD3 / DRAMP /
# the original isolation papers. All are public-domain biochemistry,
# included here as test fixtures only.
KNOWN_AMPS: tuple[KnownPeptide, ...] = (
    KnownPeptide(
        name="Magainin-2",
        sequence="GIGKFLHSAKKFGKAFVGEIMNS",
        is_amp=True,
        citation="Zasloff (1987) PNAS 84:5449",
        notes="Classical alpha-helical AMP from Xenopus laevis skin.",
    ),
    KnownPeptide(
        name="LL-37",
        sequence="LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        is_amp=True,
        citation="Gudmundsson et al. (1996) Eur J Biochem 238:325",
        notes="Human cathelicidin; canonical mammalian AMP.",
    ),
    KnownPeptide(
        name="Melittin",
        sequence="GIGAVLKVLTTGLPALISWIKRKRQQ",
        is_amp=True,
        citation="Habermann (1972) Science 177:314",
        notes="Major component of bee venom; also haemolytic.",
    ),
    KnownPeptide(
        name="Cecropin A",
        sequence="KWKLFKKIEKVGQNVRDGIIKAGPAVAVVGQATQIAK",
        is_amp=True,
        citation="Steiner et al. (1981) Nature 292:246",
        notes="Insect immune peptide; helical N-terminus.",
    ),
    KnownPeptide(
        name="Indolicidin",
        sequence="ILPWKWPWWPWRR",
        is_amp=True,
        citation="Selsted et al. (1992) J Biol Chem 267:4292",
        notes="Tryptophan-rich; bovine neutrophils.",
    ),
    KnownPeptide(
        name="Defensin HNP-1 (human alpha-defensin 1)",
        sequence="ACYCRIPACIAGERRYGTCIYQGRLWAFCC",
        is_amp=True,
        citation="Selsted et al. (1985) J Clin Invest 76:1436",
        notes="Disulfide-stabilised beta-sheet AMP.",
    ),
    # Non-AMP controls. Short fragments from cytoplasmic / structural
    # proteins whose AMP-keyword annotations in UniProt are negative.
    KnownPeptide(
        name="Cytochrome c N-terminal fragment",
        sequence="GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSY",
        is_amp=False,
        citation="UniProt P99999 (human cytochrome c), residues 1–48",
        notes="Cytoplasmic electron-transport protein; no antimicrobial annotation.",
    ),
    KnownPeptide(
        name="Insulin B-chain",
        sequence="FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        is_amp=False,
        citation="UniProt P01308 (human insulin), B-chain",
        notes="Endocrine hormone; non-AMP control with distinct length and charge profile.",
    ),
    KnownPeptide(
        name="Alpha-synuclein N-terminus",
        sequence="MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKE",
        is_amp=False,
        citation="UniProt P37840 residues 1–46",
        notes="Intrinsically disordered, lipid-binding; non-AMP control.",
    ),
    # Edge case — a sequence designed to sit on the model's decision
    # boundary. Used in the Gradio demo to illustrate uncertainty.
    KnownPeptide(
        name="Magainin-2 F12A mutant (designed edge case)",
        sequence="GIGKFLHSAKKAGKAFVGEIMNS",
        is_amp=True,
        citation="Designed for the demo — single F→A substitution at position 12",
        notes=(
            "Magainin-2 with one phenylalanine→alanine substitution; the "
            "literature suggests this reduces but does not abolish antimicrobial "
            "activity. Used in the Gradio demo as the boundary example."
        ),
    ),
)


def get_known_amps() -> tuple[KnownPeptide, ...]:
    """Return the immutable tuple of known peptides."""
    return KNOWN_AMPS


def get_demo_examples() -> tuple[KnownPeptide, KnownPeptide, KnownPeptide]:
    """The three Gradio demo example sequences.

    One classical AMP, one classical non-AMP, one boundary case. Chosen
    so the demo shows the model behaving sensibly on canonical inputs
    and reporting calibrated uncertainty on the edge case. Rationale
    repeated in ``docs/model_card.md`` so end users see the same story.
    """
    by_name = {p.name: p for p in KNOWN_AMPS}
    return (
        by_name["LL-37"],
        by_name["Cytochrome c N-terminal fragment"],
        by_name["Magainin-2 F12A mutant (designed edge case)"],
    )
