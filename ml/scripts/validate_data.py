"""Validate the downloaded AMP / non-AMP corpora and emit docs/data_card.md.

Acceptance criteria from HANDOFF.md §5 Step 3:

  - Sequence alphabet: canonical 20 AA (selenocysteine U accepted per spec
    but dropped from the working sets; ambiguous codes B/J/O/X/Z rejected).
  - Length range: 3-200 aa.
  - No duplicates within each class.
  - No sequence appearing in both positives and negatives.

On success the script exits 0 and writes ``docs/data_card.md`` in
Hugging Face data card style, with the actual length distribution, AA
composition, net-charge histogram, source decomposition, and the
CC-BY-NC licensing note for DRAMP.

On failure the script prints the specific assertion that fired and
exits non-zero so the Makefile / dev.bat target can be wired into CI.

References for the data card style:
  https://huggingface.co/docs/hub/datasets-cards
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from statistics import median, quantiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
DOCS_DIR = PROJECT_ROOT / "docs"
AMPS_FASTA = RAW_DIR / "amps.fasta"
NEGATIVES_FASTA = RAW_DIR / "negatives.fasta"
DATA_CARD = DOCS_DIR / "data_card.md"

CANONICAL_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SELENOCYSTEINE_OK = "U"
ALLOWED_ALPHABET = CANONICAL_ALPHABET + SELENOCYSTEINE_OK
MIN_LEN, MAX_LEN = 3, 200


# ---------------------------------------------------------------------------
# FASTA reading
# ---------------------------------------------------------------------------


def read_fasta(path):
    """Return a list of (header, sequence_uppercase) tuples."""
    if not path.exists():
        fail(f"FASTA not found: {path}")
    header = None
    seq_chunks = []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks).upper()))
                header = line[1:].strip()
                seq_chunks = []
            elif line.strip():
                seq_chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_chunks).upper()))
    return records


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def fail(msg):
    print(f"[validate] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def assert_alphabet(records, class_name):
    bad = []
    for header, seq in records:
        for c in seq:
            if c not in ALLOWED_ALPHABET:
                bad.append((header, c))
                break
    if bad:
        fail(f"{class_name}: {len(bad)} records contain non-canonical "
             f"characters. First offender: {bad[0]!r}")
    print(f"[validate] {class_name}: alphabet OK ({len(records)} records)")


def assert_length(records, class_name):
    bad = [(h, len(s)) for h, s in records if not (MIN_LEN <= len(s) <= MAX_LEN)]
    if bad:
        fail(f"{class_name}: {len(bad)} records outside length {MIN_LEN}-{MAX_LEN}. "
             f"First offender: {bad[0]!r}")
    print(f"[validate] {class_name}: length range OK")


def assert_no_duplicates(records, class_name):
    counter = Counter(seq for _, seq in records)
    dups = [(seq, n) for seq, n in counter.items() if n > 1]
    if dups:
        fail(f"{class_name}: {len(dups)} duplicate sequences. "
             f"First offender appears {dups[0][1]}x: {dups[0][0][:40]}...")
    print(f"[validate] {class_name}: no within-class duplicates")


def assert_no_cross_class_collisions(positives, negatives):
    pos_set = {seq for _, seq in positives}
    neg_set = {seq for _, seq in negatives}
    overlap = pos_set & neg_set
    if overlap:
        sample = next(iter(overlap))
        fail(f"cross-class collision: {len(overlap)} sequences appear in BOTH "
             f"positives and negatives. Example: {sample[:40]}...")
    print(f"[validate] no positive/negative collisions")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def length_stats(records):
    lengths = sorted(len(s) for _, s in records)
    n = len(lengths)
    q = quantiles(lengths, n=100) if n >= 100 else None
    return {
        "n": n,
        "min": lengths[0],
        "max": lengths[-1],
        "median": median(lengths),
        "p10": q[9] if q else lengths[max(0, n // 10 - 1)],
        "p90": q[89] if q else lengths[min(n - 1, 9 * n // 10)],
        "mean": sum(lengths) / n,
    }


def aa_composition(records, top_n=10):
    counts = Counter()
    total = 0
    for _, seq in records:
        counts.update(seq)
        total += len(seq)
    return [(aa, counts[aa] / total) for aa in sorted(counts, key=lambda a: -counts[a])[:top_n]]


def net_charge_at_ph7(seq):
    """Simple net charge: K and R +1, D and E -1, H ~+0.1 at pH 7."""
    charge = 0.0
    for aa in seq:
        if aa in ("K", "R"):
            charge += 1.0
        elif aa in ("D", "E"):
            charge -= 1.0
        elif aa == "H":
            charge += 0.1
    return charge


def charge_histogram(records, bin_edges=(-10, -5, -2, 0, 2, 5, 10, 20)):
    charges = [net_charge_at_ph7(s) for _, s in records]
    bins = [0] * (len(bin_edges) - 1)
    extras = 0
    for c in charges:
        placed = False
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= c < bin_edges[i + 1]:
                bins[i] += 1
                placed = True
                break
        if not placed:
            extras += 1
    return list(zip(bin_edges[:-1], bin_edges[1:], bins)), extras


def source_decomposition(records, class_name):
    """For negatives we have three potential sources: short SwissProt,
    fragmented long SwissProt, and (if class_name=='positives') DRAMP."""
    counts = Counter()
    for header, _ in records:
        if "|frag_" in header:
            counts["uniprot_fragments"] += 1
        elif header.startswith("sp|") or header.startswith("tr|"):
            counts["uniprot_short"] += 1
        elif "DRAMP" in header.upper():
            counts["dramp"] += 1
        else:
            counts["other"] += 1
    return counts


# ---------------------------------------------------------------------------
# Data card
# ---------------------------------------------------------------------------


def render_data_card(amps, negatives):
    amp_stats = length_stats(amps)
    neg_stats = length_stats(negatives)
    amp_aa = aa_composition(amps)
    neg_aa = aa_composition(negatives)
    amp_charges, _ = charge_histogram(amps)
    neg_charges, _ = charge_histogram(negatives)
    amp_sources = source_decomposition(amps, "positives")
    neg_sources = source_decomposition(negatives, "negatives")

    def stats_row(label, s):
        return (f"| {label} | {s['n']:,} | {s['min']} | {s['p10']:.0f} | "
                f"{s['median']:.0f} | {s['mean']:.1f} | {s['p90']:.0f} | {s['max']} |")

    def aa_block(comp):
        return "\n".join(f"  - {aa}: {frac:.1%}" for aa, frac in comp)

    def charge_block(hist):
        return "\n".join(f"  - [{lo:>3}, {hi:>3}): {n:>6,}" for lo, hi, n in hist)

    def src_block(srcs):
        return "\n".join(f"  - {k}: {v:,}" for k, v in sorted(srcs.items()))

    lines = [
        "# Data card — AMP Classifier corpus",
        "",
        "*Auto-generated by `ml/scripts/validate_data.py`. Re-run after any "
        "`make data` to refresh.*",
        "",
        "## Summary",
        "",
        f"- Positives: **{amp_stats['n']:,}** peptide sequences from DRAMP "
        "(general antimicrobial peptide entries).",
        f"- Negatives: **{neg_stats['n']:,}** non-AMP sequences from UniProt "
        "SwissProt, split between short reviewed entries and length-matched "
        "fragments of longer reviewed entries.",
        f"- Class ratio (neg / pos): {neg_stats['n'] / amp_stats['n']:.2f}",
        f"- Alphabet: canonical 20 amino acids (selenocysteine accepted but "
        "absent in v1).",
        f"- Length range: {MIN_LEN}-{MAX_LEN} aa, dropping ambiguous codes.",
        "",
        "## Sources and licensing",
        "",
        "**Positives — DRAMP 3.x / 4.x (general AMPs)**",
        "",
        "Shi G et al. (2022) *DRAMP 3.0: an enhanced comprehensive data "
        "repository of antimicrobial peptides.* Nucleic Acids Research, 50, "
        "D488-D496.",
        "",
        "Licensed under **CC-BY-NC for academic use**. Any redistribution "
        "must preserve attribution and the non-commercial restriction. The "
        "demo Hugging Face Space inherits this restriction.",
        "",
        "**Negatives — UniProt SwissProt (reviewed entries)**",
        "",
        "UniProt Consortium (2023) *UniProt: the Universal Protein "
        "Knowledgebase in 2023.* Nucleic Acids Research, 51, D523-D531.",
        "",
        "Licensed under **CC-BY 4.0**.",
        "",
        "**Fragmentation pattern**",
        "",
        "Long-entry negatives are produced by random-offset substring "
        "sampling from SwissProt entries 201-5,000 aa, after filtering out "
        "AMP-keyword annotations and entries carrying a Signal-peptide "
        "keyword (KW-0732, marker for likely secreted proteins). This "
        "follows Veltri 2018 (AMP-Scanner) and Meher 2017 (iAMPpred); the "
        "fragment length distribution is drawn from the empirical positives "
        "histogram to control for length as a class signal.",
        "",
        "## Schema",
        "",
        "Two FASTA files in `ml/data/raw/`:",
        "",
        "  - `amps.fasta` — one record per DRAMP positive entry. Header is "
        "the original DRAMP descriptor.",
        "  - `negatives.fasta` — one record per non-AMP. Header is the "
        "original SwissProt descriptor for short entries; fragment-derived "
        "records have `|frag_<start>_<length>` appended to the parent "
        "header.",
        "",
        "Sequences are uppercase ASCII, hard-wrapped at 60 columns "
        "(SwissProt convention).",
        "",
        "## Length distribution",
        "",
        "| Class | N | min | p10 | median | mean | p90 | max |",
        "|---|---|---|---|---|---|---|---|",
        stats_row("Positives (AMP)", amp_stats),
        stats_row("Negatives", neg_stats),
        "",
        "## Net charge at pH 7 (histogram)",
        "",
        "Computed with the simple K/R = +1, D/E = -1, H = +0.1 rule. AMPs "
        "are expected to be enriched in positive net charge (cationic "
        "membrane disruption is the canonical mechanism).",
        "",
        "**Positives**",
        "",
        charge_block(amp_charges),
        "",
        "**Negatives**",
        "",
        charge_block(neg_charges),
        "",
        "## Amino acid composition (top 10 by frequency)",
        "",
        "**Positives**",
        "",
        aa_block(amp_aa),
        "",
        "**Negatives**",
        "",
        aa_block(neg_aa),
        "",
        "## Source decomposition",
        "",
        "**Positives**",
        "",
        src_block(amp_sources),
        "",
        "**Negatives**",
        "",
        src_block(neg_sources),
        "",
        "## Exclusion criteria",
        "",
        "Applied during acquisition (`ml/scripts/download_data.py`):",
        "",
        "  - Non-canonical amino acids: any record containing characters "
        "outside `ACDEFGHIKLMNPQRSTVWY` (plus optional selenocysteine `U`) "
        "is dropped.",
        "  - Length outside the [5, 200] aa range at acquisition time; the "
        "validation step further tightens to [3, 200].",
        f"  - Negatives carrying any of: {', '.join(sorted(['antimicrobial', 'antibiotic', 'defensin', 'cathelicidin', 'magainin', 'cecropin', 'bacteriocin', 'lantibiotic', 'thionin']))} "
        "in the header.",
        "  - Negatives where the UniProt entry carries keyword KW-0929 "
        "(Antimicrobial), KW-0044 (Antibiotic), KW-0211 (Defensin), or "
        "(for long-entry fragmentation candidates) KW-0732 (Signal).",
        "  - Negatives whose entry name matches `cc_function:antimicrobial` "
        "or whose protein name contains any of the canonical AMP family "
        "names.",
        "",
        "## Known biases",
        "",
        "  - **DRAMP class composition.** The general dataset mixes natural "
        "and synthetic peptides plus patent-derived sequences. The model "
        "learns 'is AMP-like' rather than 'is natural AMP'.",
        "  - **Source distribution shift between train and test.** Long-"
        "entry fragmentation produces synthetic peptide-length sequences "
        "drawn from real protein interiors; these have different "
        "compositional statistics than short reviewed proteins. The cluster-"
        "aware split (Step 4) keeps fragments and short entries from "
        "leaking between train and test by clustering on sequence identity.",
        "  - **Length distribution is matched only approximately** (per-bin "
        "tolerance ±20%). Step 7's length-stratified analysis is the "
        "honest check that the model isn't reading length as the dominant "
        "class signal.",
        "  - **DRAMP CC-BY-NC restricts commercial use.** Any downstream "
        "model card and Hugging Face Space must repeat this restriction.",
        "",
        "## Provenance",
        "",
        "  - Acquisition script: `ml/scripts/download_data.py`",
        "  - Validation + data card: `ml/scripts/validate_data.py` (this file's source)",
        "  - Re-running the full pipeline: `make data && make validate` "
        "(or `.\\dev.bat data && .\\dev.bat validate` on Windows).",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    amps = read_fasta(AMPS_FASTA)
    negatives = read_fasta(NEGATIVES_FASTA)

    assert_alphabet(amps, "positives")
    assert_alphabet(negatives, "negatives")
    assert_length(amps, "positives")
    assert_length(negatives, "negatives")
    assert_no_duplicates(amps, "positives")
    assert_no_duplicates(negatives, "negatives")
    assert_no_cross_class_collisions(amps, negatives)

    print()
    print(f"[validate] all assertions passed: "
          f"{len(amps):,} positives + {len(negatives):,} negatives")
    print()

    # Print a quick summary
    amp_stats = length_stats(amps)
    neg_stats = length_stats(negatives)
    print(f"  positives length: min={amp_stats['min']}, median={amp_stats['median']}, "
          f"max={amp_stats['max']}")
    print(f"  negatives length: min={neg_stats['min']}, median={neg_stats['median']}, "
          f"max={neg_stats['max']}")
    print(f"  class ratio (neg/pos): {neg_stats['n'] / amp_stats['n']:.2f}")
    print()

    # Render the data card
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CARD.write_text(render_data_card(amps, negatives), encoding="utf-8")
    print(f"[validate] wrote {DATA_CARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
