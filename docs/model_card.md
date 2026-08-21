# Model card — AMP Classifier

*Hugging Face style model card for the v1 AMP Classifier. Covers the
binary discriminator, the Phase 2A-calibrated CNN, the Phase 2B
ESM2-650M champion, the Phase 2C multi-label activity head, and the
Phase 2D discriminator-in-the-loop design tool.*

## Model details

**Task**: binary classification, antimicrobial peptide vs. non-AMP,
plus a multi-label activity-class head and a discriminator-driven
mutation design tool.

**Inputs**: a single peptide sequence over the canonical 20 amino
acids, length 5–100 residues.

**Outputs**: AMP probability from each binary model; per-class
activity probabilities (antibacterial / antifungal / antiviral /
antiparasitic) from the multi-label head; top-K mutant proposals
from the design tool.

**Models trained**

| ID | Algorithm | Inputs | Parameters |
|---|---|---|---|
| `baseline_logreg` | L2 logistic regression, balanced class weight | 428 engineered features | 429 |
| `baseline_rf` | Random forest, 200 trees, balanced class weight | 428 engineered features | ~20 M tree splits |
| `baseline_xgb` | XGBoost, 500 trees max, early stop on val AUC | 428 engineered features | ~3 M tree splits |
| `cnn_best` + `cnn_calibrator` | 3× Conv1D + BN + 2 FC, Platt-calibrated | one-hot 21ch × 100 len | 192k + 2 |
| **`esm_head_best`** | **3-layer MLP over frozen ESM2-650M embeddings** | **1280-dim mean-pooled** | **~1 M** |
| `esm_multiclass_head` | 3-layer MLP, 4 sigmoid outputs | 1280-dim ESM2 embeddings | ~1 M |

## Evaluation

Cluster-aware 80/10/10 split via **mmseqs2 easy-cluster at 40%
sequence identity, 80% bidirectional coverage**. Whole clusters live
in a single split — the property tested by `tests/test_splits.py`.
Random splits on AMP data inflate AUC by 10–15 points because of
sequence homology; this discipline is what makes the held-out numbers
honestly comparable.

Bootstrap n=1000, seed=42 for all 95% CIs.

### Head-to-head on the binary held-out test set (1,969 sequences, 882 pos / 1,087 neg)

| Model | AUC | MCC | Threshold | Notes |
|---|---|---|---|---|
| LogReg | 0.825 [0.805–0.844] | 0.542 [0.504–0.579] | 0.5 | 428 engineered features |
| RandomForest | 0.863 [0.845–0.879] | 0.636 [0.602–0.668] | 0.5 | 428 engineered features |
| XGBoost | 0.864 [0.845–0.880] | 0.621 [0.587–0.652] | 0.5 | 428 engineered features (Phase 1 winner) |
| CNN (raw + tuned) | 0.837 [0.818–0.854] | 0.477 [0.449–0.506] | tuned t=0.050 | Phase 1 CNN, original report |
| CNN (Phase 2A Platt) | 0.837 [0.818–0.854] | 0.570 [0.534–0.606] | 0.5 | Platt scaler on val logits — preferred CNN |
| **ESM2-650M + MLP head** | **0.919 [0.906–0.931]** | **0.697 [0.667–0.730]** | 0.5 | **Phase 2B champion** |

The ESM2-head wins by **+5.5 AUC and +0.076 MCC over XGBoost with
non-overlapping 95% bootstrap CIs on both metrics** — a statistically
clean Phase 2B win.

### Phase 2A calibration note

The Phase-1 CNN's score distribution was shifted near zero (BCE
pos_weight + BatchNorm) so MCC at the standard t=0.5 was 0.138. Platt
scaling preserved AUC (monotonic) and lifted MCC to 0.570 at t=0.5 —
better than the raw + val-tuned-threshold report of 0.477. The CNN's
demo output is now interpretable as a single calibrated probability.

The ESM head did not need Platt calibration; its output distribution
was well-calibrated by default (MCC@0.5 0.697 > MCC@tuned 0.670).

## Phase 2C — multi-label activity head (DRAMP four-class)

Once a sequence is called AMP, a downstream consumer wants to know
*what kind*. A 4-output sigmoid head trained on the AMP-positive
subset of the cluster-aware splits, with per-class `pos_weight`
clipped at 50 (raw antiparasitic weight was ~110× and destabilised
training).

| Activity | n test positives | Test AUC (95% CI) | Read |
|---|---|---|---|
| antiviral | 218 | **0.872 [0.844–0.898]** | Tightest CI; antiviral peptides have distinctive motifs |
| antibacterial | 854 | 0.809 [0.746–0.864] | Hard *within-AMP* signal (only 28 non-antibacterial AMPs in test) |
| antiparasitic | 5 | 0.811 [0.547–0.979] | Point estimate fine, CI uninformative (too few positives) |
| antifungal | 66 | 0.730 [0.668–0.787] | Weakest; mechanism overlaps with antibacterial |
| **macro-AUC** | | **0.806** | **Phase 2C win condition (≥ 0.80) met** |

### Multi-class caveats

- **antiparasitic has only 5 positives in the held-out test set**.
  CI [0.547–0.979] is uninformative. Future v2 should pool with
  antiprotozoal or drop the class explicitly.
- **antibacterial AUC ≈ 0.81** is low because the *within-AMP*
  problem is harder than the binary AMP-vs-non-AMP problem when
  96% of AMPs are antibacterial.
- **antifungal and antibacterial are biologically conflated** — both
  use membrane-disrupting mechanisms.

The multi-class head is for *ranking* candidate activities, not for
clinical commitment. Read the four probabilities as "these are
plausible activity classes given DRAMP-like training data."

## Phase 2D — discriminator-in-the-loop mutation design

A Design tab in the Gradio app wraps the ESM head in a **greedy beam
search over single-residue mutations**. For a parent peptide and a
mutation budget *n*, the search:

1. Expands every beam member by all 19 × len single-residue substitutions
2. Scores all candidates with the ESM head in one batched forward pass
3. Keeps the top beam-width per hop
4. After *n* hops, surfaces the global top-K mutants ranked by P(AMP)

On a 4060 Laptop GPU, a ~25-aa parent with `n=2` and `beam_width=10`
completes in ~10–15 seconds. Five pre-populated parents are in the
library (LL-37, magainin-2, cecropin A, indolicidin, melittin), plus
arbitrary custom sequences via a textbox.

### Magainin-2 case study — model rediscovers Trp-anchor heuristic

Applied to **magainin-2** (`GIGKFLHSAKKFGKAFVGEIMNS`, parent
P(AMP) = 0.9954, near ceiling), all ten top-ranked mutants after two
hops with beam-width 10 introduce **Tryptophan substitutions**. The
substituted positions cluster at residues 6, 10, 13, 15, 17, 19, 21
— which lie on the same face of magainin-2's amphipathic α-helix
(positions 10, 13, 17, 21 are within one helical turn-width of each
other given 3.6 residues per turn).

This is the **canonical "Trp-anchor amphipathic enhancement"** design
heuristic from the AMP literature: aromatic residues at the
hydrophobic face of an α-helical AMP improve membrane partitioning.
Indolicidin (Trp-rich by design) is the textbook example. The
classifier rediscovered this design recipe from DRAMP training data
alone, without any hand-engineered helical-wheel intermediate.
This is a useful piece of model introspection: the ESM head's local
gradient near magainin-2 points in a literature-validated direction,
giving face-validity to the AUC-only generalization claim.

### Design caveats

- **In-silico only.** Top-ranked mutants have not been wet-lab
  validated. Any application claim requires synthesis + assay against
  indicator strains.
- **Hemolytic / cytotoxic side-effects not modeled.** Trp-rich AMPs
  trend toward higher mammalian membrane disruption (the inherent
  risk of the design recipe the model recommends). Wet-lab follow-up
  should include a mammalian-cell hemolysis control alongside MIC
  measurement on indicator strains.
- **Local search only.** Beam search over single mutations explores
  the immediate neighbourhood of the parent; it doesn't propose
  sequences far from training distribution. It's a refinement tool
  for known scaffolds, not a de-novo generator.
- **Diminishing returns near ceiling.** A parent already at
  P(AMP) ≈ 0.99 has ~0.005 of headroom; the design tab is most useful
  on marginal parents in the 0.5–0.9 range.

## Intended use

Research and portfolio demo. Useful for ranking candidate peptides
by "AMP-like-ness", screening designs from a generator,
discriminator-in-the-loop refinement of known AMP scaffolds, or as a
teaching artifact for the cluster-aware split discipline. **Not
validated for any clinical or regulatory purpose.**

## Limitations

- **Binary scope is primary.** Multi-class activity is preliminary;
  antiparasitic is undersampled.
- **Length cap.** Training and inference bounded to 5–200 aa;
  demo further caps at 100. Longer peptides need re-training with a
  larger max-length.
- **No structure information.** Sequence-only model; helical packing,
  disulfide bonds, oligomerisation are invisible. AMPs with
  structure-dependent mechanisms (β-sheet defensins, polycyclic
  bacteriocins) may be under-predicted.
- **No fine-tuning of ESM.** Frozen embeddings + small head only.
  Full or partial ESM fine-tuning could add 1–3 AUC points but
  requires careful learning-rate scheduling and probably a larger
  corpus.
- **Domain shift.** Designed cyclic peptides, peptidomimetics, and
  D-amino-acid variants are out-of-distribution. Predictions on those
  should be treated as OOD.
- **DRAMP CC-BY-NC.** Commercial deployment would require license
  renegotiation. The HF Space deploy is academic-only.
- **Design tab is local search only.** Won't surface peptides far
  from training distribution; complements but doesn't replace a
  proper generative model.

## Training data

- **Positives**: DRAMP 3.x / 4.x general AMP entries
  ([Shi et al. 2022](https://doi.org/10.1093/nar/gkab651),
  CC-BY-NC academic use).
- **Negatives**: UniProt SwissProt entries without antimicrobial
  keywords, length-matched to positives. Long entries (>200 aa) were
  fragmented via random offsets to fill short-length bins
  ([Veltri 2018](https://doi.org/10.1093/bioinformatics/bty179),
  [Meher 2017](https://doi.org/10.1038/srep42362) pattern).
- **Activity labels** (Phase 2C): DRAMP general-AMP XLSX metadata,
  free-text "Activity" column substring-matched to four canonical
  classes.

Final corpus after dedup + cross-class collision removal: 9,224
positives + 11,252 negatives (1.22:1 ratio). Length range 5–~100 aa,
median 20, mean ~25. Full provenance in `docs/data_card.md`.

## Reproduction

```bash
git clone https://github.com/SleepyKomodo/amp-classifier
cd amp-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make data && make validate && make cluster && make splits
make baseline        # train + eval engineered baselines
make cnn             # train + eval the small CNN
make calibrate       # Phase 2A: Platt scaler on val logits
make esm             # Phase 2B: ESM2-650M embeddings + binary head
make activities      # Phase 2C step 1: download DRAMP activity metadata
make multiclass      # Phase 2C: train + eval multi-label head
make app             # local Gradio demo (Predict + Design tabs)
```

All deterministic at seed=42. Cluster step needs `mmseqs2`; on
Windows download the precompiled binary from
[MMseqs2 releases](https://github.com/soedinglab/MMseqs2/releases).
Phase 2B and 2D need GPU torch for usable wall-clock; CPU torch works
but is ~100× slower for ESM2-650M inference.

## Citation

If you use this work, please cite the underlying papers:

- Shi G et al. (2022) DRAMP 3.0. *Nucleic Acids Research* 50(D1), D488–D496.
- Lin Z et al. (2023) Evolutionary-scale prediction of atomic-level
  protein structure with a language model. *Science* 379(6637), 1123–1130.
  (ESM2)
- Steinegger M, Söding J (2017) MMseqs2. *Nature Biotechnology* 35, 1026–1028.
- Veltri D, Kamath U, Shehu A (2018) Deep learning improves
  antimicrobial peptide recognition. *Bioinformatics* 34(16), 2740–2747.
- Meher PK et al. (2017) Predicting antimicrobial peptides with
  improved accuracy. *Scientific Reports* 7, 42362.
- Boman HG (2003) Antibacterial peptides. *J Internal Medicine* 254(3), 197–215.
- Eisenberg D, Weiss RM, Terwilliger TC (1984) Hydrophobic moment.
  *PNAS* 81(1), 140–144.
- Kyte J, Doolittle RF (1982) Hydropathic character. *J Mol Biol* 157(1), 105–132.
- Platt JC (1999) Probabilistic outputs for support vector machines.
  In Smola et al., *Advances in Large Margin Classifiers*.
