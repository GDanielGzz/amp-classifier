"""Phase 2D: greedy beam search over single-residue mutations.

Given a parent peptide, expand by trying every single-residue substitution,
score the candidates with the trained ESM2-head AMP classifier, keep the
top ``beam_width`` per hop, repeat for ``n_mutations`` hops. Return the
final top-K mutants ranked by predicted P(AMP).

The discriminator-in-the-loop pattern: a strong classifier scores
candidates; a simple combinatorial generator proposes them. The
generator does not need to learn anything; the classifier does the
heavy lifting.

Public entry point:

  design_mutants(parent, tokenizer, backbone, head, device, *,
                 n_mutations=3, beam_width=5, top_k=10) -> list[Mutant]

Each Mutant carries the sequence, score, parent score delta, and the
list of (position, parent_aa, mutant_aa) changes from the parent.

Importable from app.py for the Design tab; also callable from the CLI
for batch design over multiple parents.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence as SeqType

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import CANONICAL_ALPHABET  # noqa: E402

# Well-known AMPs to pre-populate the demo dropdown.
PARENT_LIBRARY = {
    "LL-37 (human cathelicidin)": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin-2 (Xenopus skin)": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Cecropin A (insect)": "KWKLFKKIEKVGQNVRDGIIKAGPAVAVVGQATQIAK",
    "Indolicidin (bovine neutrophil)": "ILPWKWPWWPWRR",
    "Melittin (bee venom)": "GIGAVLKVLTTGLPALISWIKRKRQQ",
}

DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class Mutant:
    sequence: str
    score: float
    delta: float  # score - parent_score
    changes: tuple  # tuple of (position_1based, parent_aa, mutant_aa)

    def diff_string(self):
        """Human-readable mutation list like 'F4W,R12K'."""
        return ",".join(f"{pa}{p}{ma}" for p, pa, ma in self.changes)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def batch_score_esm(sequences, tokenizer, backbone, head, device,
                    batch_size=DEFAULT_BATCH_SIZE):
    """Return per-sequence P(AMP) under the ESM2-head model."""
    if not sequences:
        return np.zeros(0, dtype=np.float32)
    out = []
    head.eval()
    backbone.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = list(sequences[i:i + batch_size])
            encoded = tokenizer(batch, padding=True, return_tensors="pt",
                                truncation=True, max_length=200)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = backbone(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            logits = head(pooled)
            probs = torch.sigmoid(logits).cpu().numpy()
            out.append(probs)
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Mutation enumeration
# ---------------------------------------------------------------------------


def enumerate_single_mutations(seq):
    """Yield (mutant_seq, position_1based, parent_aa, mutant_aa) for every
    single-residue substitution of ``seq`` over the canonical alphabet."""
    for i, parent_aa in enumerate(seq):
        for mut_aa in CANONICAL_ALPHABET:
            if mut_aa == parent_aa:
                continue
            mutant = seq[:i] + mut_aa + seq[i + 1:]
            yield mutant, i + 1, parent_aa, mut_aa


def _diff_from_parent(child, parent):
    """Compute (pos_1based, parent_aa, mutant_aa) tuples where child differs."""
    return tuple(
        (i + 1, parent[i], child[i])
        for i in range(min(len(child), len(parent)))
        if child[i] != parent[i]
    )


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def design_mutants(parent, tokenizer, backbone, head, device, *,
                   n_mutations=3, beam_width=8, top_k=10,
                   batch_size=DEFAULT_BATCH_SIZE):
    """Greedy beam search for high-scoring mutants of ``parent``.

    At each hop, expand each current beam member by all single-residue
    mutations, score all expansions in one batched ESM forward pass, and
    keep the top ``beam_width``. After ``n_mutations`` hops, return the
    top ``top_k`` Mutant objects sorted by score (descending).
    """
    parent = parent.upper().strip()
    if not all(c in CANONICAL_ALPHABET for c in parent):
        bad = sorted(set(parent) - set(CANONICAL_ALPHABET))
        raise ValueError(f"parent contains non-canonical residues: {''.join(bad)}")
    parent_score = float(batch_score_esm([parent], tokenizer, backbone, head,
                                          device, batch_size)[0])

    # Beam state: list of (seq, score). Start from parent.
    beam = [(parent, parent_score)]
    # Track all unique mutants we've ever seen to dedupe and to surface the
    # global top-K (not just the final hop's beam).
    all_seen = {parent: parent_score}

    for hop in range(n_mutations):
        # Enumerate all single-mutation neighbours of every beam member
        proposals = set()
        for seq, _ in beam:
            for mutant, *_ in enumerate_single_mutations(seq):
                if mutant in all_seen:
                    continue
                proposals.add(mutant)
        if not proposals:
            break
        proposals = list(proposals)
        scores = batch_score_esm(proposals, tokenizer, backbone, head,
                                  device, batch_size)
        for s, sc in zip(proposals, scores):
            all_seen[s] = float(sc)
        # New beam: top ``beam_width`` of (current beam + new proposals)
        candidates = sorted(
            set([(s, sc) for s, sc in zip(proposals, scores)] + list(beam)),
            key=lambda x: -x[1],
        )[:beam_width]
        beam = candidates

    # Final ranking: surface the global top-K excluding the parent itself
    ranked = sorted(
        [(s, sc) for s, sc in all_seen.items() if s != parent],
        key=lambda x: -x[1],
    )[:top_k]
    return [
        Mutant(
            sequence=s,
            score=sc,
            delta=sc - parent_score,
            changes=_diff_from_parent(s, parent),
        )
        for s, sc in ranked
    ], parent_score


# ---------------------------------------------------------------------------
# CLI (batch design over multiple parents)
# ---------------------------------------------------------------------------


def _load_models():
    """Load ESM tokenizer + backbone + head. Mirrors app.py's loader."""
    from transformers import AutoTokenizer, AutoModel
    from ml.scripts.train_esm_head import EsmHead
    import os

    name = os.environ.get("ESM_MODEL_NAME", "facebook/esm2_t33_650M_UR50D")
    ckpt_path = PROJECT_ROOT / "ml" / "checkpoints" / "esm_head_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing {ckpt_path}; run dev.bat esm first")

    print(f"[design] loading {name} ...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    backbone = AutoModel.from_pretrained(name).eval()
    head_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_dim = head_ckpt.get("in_dim", backbone.config.hidden_size)
    head = EsmHead(in_dim=in_dim)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    head = head.to(device)
    print(f"[design] ready on {device}")
    return tokenizer, backbone, head, device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parent", help="Parent peptide sequence "
                                      "(or PARENT_LIBRARY key)")
    p.add_argument("--n-mutations", type=int, default=3)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--all-library", action="store_true",
                   help="Run on every parent in PARENT_LIBRARY")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    tokenizer, backbone, head, device = _load_models()

    if args.all_library:
        targets = list(PARENT_LIBRARY.items())
    elif args.parent in PARENT_LIBRARY:
        targets = [(args.parent, PARENT_LIBRARY[args.parent])]
    elif args.parent:
        targets = [("user", args.parent)]
    else:
        print("[design] specify --parent or --all-library")
        return 1

    for name, parent_seq in targets:
        print(f"\n=== {name} ({len(parent_seq)} aa) ===")
        mutants, parent_score = design_mutants(
            parent_seq, tokenizer, backbone, head, device,
            n_mutations=args.n_mutations,
            beam_width=args.beam_width,
            top_k=args.top_k,
        )
        print(f"Parent score: {parent_score:.4f}")
        for i, m in enumerate(mutants, 1):
            print(f"  {i:2d}. {m.score:.4f} "
                  f"(Δ {m.delta:+.4f})  {m.diff_string():<30}  {m.sequence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
