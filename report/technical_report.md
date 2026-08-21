# AMP Classifier — technical report

*A four-model head-to-head on antimicrobial peptide classification, with
cluster-aware evaluation, calibration analysis, and a frozen protein-LM
demonstration of where engineered features hit their ceiling.*

**Author:** Daniel González (postdoctoral, Tecnológico de Monterrey × DTU)
**Date:** 2026-05-17
**Source:** `https://github.com/SleepyKomodo/amp-classifier`

---

## Abstract

We trained four sequence-only classifiers for binary antimicrobial-peptide
(AMP) prediction on a cluster-aware 80/10/10 split of 20,476 peptides
(9,224 DRAMP 3.x / 4.x positives + 11,252 length-matched UniProt
SwissProt negatives). On the held-out test set, an XGBoost on 428
engineered physicochemical features achieved AUC 0.864 [0.845–0.880],
narrowly tying RandomForest and clearly beating logistic regression. A
small one-hot CNN trailed at AUC 0.837 [0.818–0.854] and required Platt
calibration to produce usable threshold-based metrics (MCC@0.5 0.138 →
0.570). A small MLP head on top of frozen ESM2-650M embeddings cleared
the engineered-feature ceiling cleanly at AUC 0.919 [0.906–0.931] and
MCC 0.697 [0.667–0.730], with CIs that do not overlap XGBoost's on
either metric — a statistically clean +5.5 AUC, +0.076 MCC margin. A
multi-label head trained on the AMP-positive subset against four
activity classes (antibacterial, antifungal, antiviral, antiparasitic)
reached macro-AUC 0.806; antiviral was the strongest single-class signal
(AUC 0.872), antiparasitic was statistically uninformative (5 test
positives). A discriminator-in-the-loop design tool wraps the
ESM head in a greedy beam search over single-residue mutations of a
parent AMP; applied to magainin-2 (parent score 0.9954), all ten
top-ranked single- and double-mutants introduce Tryptophan
substitutions at positions on one amphipathic-helix face,
rediscovering a literature-known AMP design heuristic from training
data alone. The full pipeline runs end-to-end on a laptop RTX 4060 in
under 30 minutes after the data step; a local Gradio demo serves
side-by-side predictions from XGBoost, the calibrated CNN, and the
ESM head, with a cytochrome-c-N-terminus example that exposes the
charge-driven false positive engineered models fall for and ESM
correctly rejects, plus a Design tab for proposing mutant peptides
from any of five canonical AMP scaffolds.

---

## 1. Introduction

Antimicrobial peptides (AMPs) are short host-defence sequences that
disrupt microbial membranes and are an active drug-discovery target as
resistance to small-molecule antibiotics climbs. *In silico* AMP
classification is a useful first-pass filter on designed or
naturally-occurring peptides before wet-lab assays. The field has a
~25-year computational tradition stretching from CAMP and AntiBP to
iAMPpred (engineered features + Random Forest), AMP-Scanner (one-hot
CNN), and more recent attention-based predictors like AMPlify.

Three problems in the existing literature motivate this work:

1. **Reported metrics are often inflated** by random train/test splits
   on sequence-redundant corpora. AMP sequences cluster heavily into
   families (defensins, cathelicidins, magainins, cecropins), and a
   peptide can sit in train while its 60%-identity homolog sits in test,
   leaking information across the boundary. Cluster-aware splits at
   40% identity routinely drop reported AUC by 10–15 points relative to
   random splits on the same corpus.
2. **Engineered-feature models and deep models are rarely compared on
   the same split.** When they are, the engineered baseline often wins
   or ties, but the literature framing rewards the deep models. Honest
   side-by-side reporting is the exception, not the rule.
3. **Activity-class granularity is left implicit.** Most "AMP
   classifiers" predict binary AMP-vs-non-AMP, even though the
   downstream consumer wants to know whether their peptide is
   antibacterial, antifungal, antiviral, or antiparasitic.

This report addresses all three: a cluster-aware split with cluster-
purity assertion as a CI gate; a four-model head-to-head (LogReg,
RandomForest, XGBoost, one-hot CNN, ESM2-650M-head) on the same test
set with bootstrap-CI'd metrics; and a multi-label activity head on top
of the ESM embeddings, with honest treatment of the under-sampled
antiparasitic class.

The contribution is the *honest comparison*, not a new architecture.
The ESM2-head model is the canonical "frozen-embedding + small head"
recipe applied carefully to a properly-split corpus.

---

## 2. Data

### 2.1 Corpus assembly

**Positives.** All 11,612 general-AMP entries from DRAMP (3.x and 4.x
merged, IDs ranging DRAMP00005 to DRAMP35990) were downloaded as a
bulk FASTA via a scrape of the DRAMP downloads index page (the
documented bulk URLs vary across releases). After filtering to the
canonical 20 amino acids and the length range [5, 200] aa, and
deduplicating by sequence (DRAMP indexes the same canonical sequence
under multiple entry IDs for different organisms or patents), 9,224
unique positive sequences remained.

**Negatives.** Reviewed SwissProt entries length 5–200 aa with no
antimicrobial-keyword annotation (KW-0929 Antimicrobial, KW-0044
Antibiotic, KW-0211 Defensin, plus `cc_function:antimicrobial` and
free-text exclusions for cathelicidin / magainin / cecropin /
bacteriocin / defensin) were streamed from the UniProt REST API
(target 5 × the positive count, ~49 k entries, then rejection-sampled
into length-matched bins). When the short-pass sample underfilled
short-length bins (peptides are short; reviewed-SwissProt skews
longer), the script fetched SwissProt entries length 201–5,000 aa
without signal-peptide annotation (KW-0732 excluded) and produced
random-offset fragments at lengths drawn from the positives' histogram,
following the Veltri 2018 / Meher 2017 negative-construction pattern.
After deduplication and removal of 153 sequences appearing in both
classes (real overlap from precursor proteins of mature AMPs in
SwissProt), 11,252 negatives remained. Final class ratio 1.22:1, just
beyond the ±20% spec.

### 2.2 Cluster-aware splits

The merged 20,476-sequence corpus was clustered with `mmseqs2
easy-cluster --min-seq-id 0.4 -c 0.8 --cov-mode 0`, producing 14,162
clusters (average size 1.45). Clusters were assigned to 80/10/10
train/val/test splits via greedy bin-packing per class
(amp-pure clusters and neg-pure clusters bin-packed independently;
59 mixed clusters — sequences where an AMP and a non-AMP clustered
together at 40% identity, mostly defensin precursors — were assigned
to train so val/test stayed label-clean). Resulting split sizes:
16,537 train / 1,970 val / 1,969 test, with positive fractions of
45.1% / 44.8% / 44.8% — class balance preserved across splits.

A unit test (`tests/test_splits.py::test_no_cluster_spans_splits`)
asserts that no cluster appears in more than one split. This is the
property that makes the held-out metrics honestly comparable to
published baselines. The full test suite (44 tests) is green before
training.

### 2.3 Provenance

DRAMP is distributed under CC-BY-NC for academic use; this report and
the released code inherit that restriction. UniProt SwissProt is
CC-BY 4.0. The fragmentation pattern is cited explicitly in
`docs/data_card.md` per Veltri 2018 and Meher 2017.

---

## 3. Methods

### 3.1 Engineered features

A 428-dimensional feature vector was computed per peptide via
`ml/features.py`, shared between training and runtime inference:

- **20 amino-acid composition** features (fraction per residue)
- **400 dipeptide composition** features (fraction per ordered AA pair)
- **4 Biopython ProtParam** scalars: length, isoelectric point,
  instability index, aromaticity
- **4 derived scalars**: net charge at pH 7 (Henderson–Hasselbalch),
  mean Kyte–Doolittle hydrophobicity, Boman index, and max Eisenberg
  hydrophobic moment over an 11-residue sliding window

Each scalar feature has a hand-computed unit test in
`tests/test_features.py` (31 tests total) that pins expected values
against literature scales — e.g. mean KD for isoleucine = 4.5, Boman
index of pure leucine = +4.92, alternating amphipathic sequence has a
higher Eisenberg moment than an all-alanine sequence.

### 3.2 Baseline classifiers (Phase 1)

Three sklearn classifiers were trained on the engineered features with
`class_weight='balanced'` (or XGBoost's `scale_pos_weight`) for the
1.22:1 class imbalance:

- **Logistic regression**, L2, `C=1.0`
- **Random forest**, 200 trees, no depth limit
- **XGBoost**, 500 trees max, max depth 6, early stop on val AUC
  (patience 20), histogram tree method

### 3.3 Small CNN (Phase 1, refined in Phase 2A)

A 192k-parameter Conv1D model on one-hot encoded sequences padded to
length 100 (21 channels: 20 AA + 1 PAD): 3 × Conv1D (96, 192, 192
channels, kernel 3, padding 1) + BatchNorm + ReLU + Dropout(0.4),
global max pool over length, FC(192→96) + BN + FC(96→1). BCE with
positive class weight 1.22, AdamW lr=1e-4 weight_decay=1e-3, batch 64,
early stop on val AUC patience 15.

A first-pass V1 of this architecture (64/128/128 channels, no
BatchNorm, dropout 0.2, lr 3e-4) overfit dramatically: val AUC peaked
at 0.81 by epoch 2 and dropped as training loss continued falling. V2
landed val AUC 0.844 at epoch 48. The V2 trained-model's score
distribution is shifted near zero (BCE pos_weight × BatchNorm pushes
outputs down) so MCC at the standard t=0.5 threshold is misleading
(0.138). **Phase 2A** therefore fits a Platt scaler — a LogisticRegression
on val logits — saved as `cnn_calibrator.joblib` and applied at
inference. AUC is preserved (Platt is monotonic) and MCC@0.5 recovers
to 0.570.

### 3.4 ESM2-650M frozen embeddings + small head (Phase 2B)

The `facebook/esm2_t33_650M_UR50D` backbone (650M params,
1280-dim hidden state) was loaded with HuggingFace `transformers`,
moved to a GPU (RTX 4060 Laptop, CUDA 12.4), and run in inference
mode. Per-sequence embeddings were computed by mean-pooling the
last hidden state over non-padding tokens, then cached to
`ml/data/processed/esm_{train,val,test}.npz`. Wall-clock for all
20,476 sequences at batch 8 on the 4060: ~1 minute.

A small MLP head (1280 → 640 → 320 → 1, BatchNorm + Dropout(0.3))
was trained on the cached embeddings with the same BCE+pos_weight
recipe as the CNN. Convergence was rapid: val AUC peaked at 0.925 at
epoch 6 (early stop fired at epoch 21).

### 3.5 Multi-label activity head (Phase 2C)

Activity labels were extracted from the DRAMP general-AMP XLSX
metadata file (auto-downloaded from the DRAMP downloads page), joining
on DRAMP ID. The free-text "Activity" column was mapped to four
canonical classes by substring matching:

- **antibacterial** ← Antibacterial, Anti-Gram+, Anti-Gram-, Antibiofilm
- **antifungal** ← Antifungal
- **antiviral** ← Antiviral, Anti-HIV, Anti-HCV, Anti-SARS
- **antiparasitic** ← Antiparasitic, Antiprotozoal, Antimalarial, etc.

Each peptide can be positive in multiple classes (defensins are
typically both antibacterial AND antifungal). Of 9,224 positives,
class counts were 8,841 (95.8%) antibacterial, 1,766 (19.1%)
antifungal, 1,650 (17.9%) antiviral, and 81 (0.9%) antiparasitic; 369
(4.0%) entries had only out-of-vocabulary activity tags (anti-tumor,
insecticidal, etc.) and were retained as features-only inputs but did
not score for any of the four classes.

The multi-label head shares the binary head's MLP architecture but
outputs 4 sigmoid logits, trained with `BCEWithLogitsLoss` and a
per-class `pos_weight` clipped at 50 (the antiparasitic raw weight was
~110×, which destabilises training when uncapped).

### 3.6 Evaluation

Bootstrap n=1000 95% CIs at seed 42 across all reports, in
`ml/eval_common.py`. Per-class AUC, MCC, sensitivity, specificity, F1.
Stratification by sequence length (≤20, 21–50, >50) and net charge at
pH 7 (≤0, 0–5, >5). Single `_compute_metric` function shared across
all evaluators so cross-model comparisons run through one pipeline.

---

## 4. Results

### 4.1 Binary classification — head-to-head

| Model | Test AUC | Test MCC | Threshold |
|---|---|---|---|
| LogReg | 0.825 [0.805–0.844] | 0.542 [0.504–0.579] | 0.5 |
| RandomForest | 0.863 [0.845–0.879] | 0.636 [0.602–0.668] | 0.5 |
| XGBoost | **0.864 [0.845–0.880]** | 0.621 [0.587–0.652] | 0.5 |
| CNN (raw + tuned t) | 0.837 [0.818–0.854] | 0.477 [0.449–0.506] | val-tuned t=0.050 |
| CNN (Platt-calibrated) | 0.837 [0.818–0.854] | 0.570 [0.534–0.606] | 0.5 |
| **ESM2-650M + MLP head** | **0.919 [0.906–0.931]** | **0.697 [0.667–0.730]** | 0.5 |

**Phase 1 finding:** XGBoost and RandomForest are statistically tied
at the top of the engineered-feature group (CIs overlap heavily).
LogReg trails by ~4 AUC points, showing that the engineered features
carry meaningful nonlinear interactions. The CNN ranks competitively
on AUC but trails on MCC even after calibration — a 192k-parameter
one-hot model has neither the data nor the inductive bias to match
boosted trees on engineered features for this corpus size.

**Phase 2B finding:** ESM2-650M frozen embeddings clear XGBoost's
ceiling decisively. The +5.5 AUC and +0.076 MCC margins are larger
than the 95% bootstrap CIs of either model — the CIs do not overlap
on either metric. This is the canonical "frozen protein-LM embeddings
beat hand-engineered features once you have a few thousand training
examples" result, replicated cleanly on a properly cluster-aware split.

### 4.2 Stratified analysis

Across all models, AUC stratified by length is highest in the medium
bin (21–50 aa) and lowest in the long bin (>50 aa). This is partly
data-driven — the medium bin has the most positives — and partly model
behaviour: the CNN's fixed-length one-hot input is most informative
for sequences shorter than its max length. The ESM head's
length-stratified AUCs are tighter than the engineered models',
suggesting it generalises more uniformly across sequence lengths.

Charge-stratified AUCs are roughly flat for ESM but show a sharp
positive-charge bias for XGBoost (cytochrome-c-N-terminus case
discussed in §4.4 below). Full per-bin numbers in
`docs/baseline_results.md` and `docs/cnn_results.md`.

### 4.3 Multi-label activity

| Class | n_pos (test) | Test AUC (95% CI) |
|---|---|---|
| antiviral | 218 | **0.872 [0.844–0.898]** |
| antibacterial | 854 | 0.809 [0.746–0.864] |
| antiparasitic | 5 | 0.811 [0.547–0.979] |
| antifungal | 66 | 0.730 [0.668–0.787] |
| **macro-AUC** | | **0.806** |

The macro-AUC clears the Phase 2C target of 0.80. Per-class:

- **antiviral** is the tightest single-class signal, likely because
  antiviral peptides include sequence-distinctive families (proline-
  rich, fusion-inhibitor motifs) that ESM2 captures clearly.
- **antibacterial** is the lowest "real" AUC, because the *within-AMP*
  problem is genuinely harder than the *AMP-vs-non-AMP* problem when
  96% of all positives are antibacterial — the "negatives" for this
  class are 28 unusual exclusively-antifungal or exclusively-antiviral
  AMPs.
- **antifungal** is weakest because antifungal mechanisms overlap
  heavily with antibacterial (both target membranes).
- **antiparasitic** has 5 test positives. The point estimate looks
  fine, but the CI [0.547–0.979] is uninformative. We report the
  number transparently and recommend pooling with antiprotozoal or
  dropping the class in any v2.

### 4.4 Demo behaviour — three diagnostic examples

A Gradio app (`app.py`) loads all three binary models and scores any
user-supplied peptide side-by-side. The three pre-populated examples
illustrate the head-to-head story.

**LL-37** (`LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES`, length 37 — canonical
human cathelicidin AMP). All three models correctly call AMP with high
confidence: XGBoost 0.993, CNN 0.820, ESM 0.958. **Sanity check
passes.**

**Magainin-2 F12A mutant** (`GIGKFLHSAKKAGKAFVGEIMNS`, length 23 —
boundary case, F→A substitution at position 12 of wild-type magainin-2,
which the literature suggests reduces but does not abolish activity).
All three correctly call AMP: XGBoost 0.996, CNN 0.850, ESM 0.990.
The slight reduction in CNN confidence (0.85 vs ~0.95+ for the others)
matches its more position-sensitive inductive bias.

**Cytochrome c N-terminus** (`GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSY`,
length 48 — non-AMP, mitochondrial electron-transport protein
fragment). XGBoost calls AMP at 0.624 (false positive, driven by net
charge +6.00, its top feature by gain). CNN calls AMP at 0.611
(similar charge-driven failure). **ESM correctly calls non-AMP at
0.017** — confident rejection. This is the diagnostic moment that
shows why ESM beats the engineered baselines: cytochrome c is in ESM's
pretraining corpus (~65M UniRef sequences), so the protein-LM "knows"
this fragment is an electron carrier, not a membrane-disrupting
peptide. Engineered features have a knowledge ceiling; ESM has read
about cytochrome c.

### 4.5 Discriminator-in-the-loop design (Phase 2D)

A Design tab in the same Gradio app wraps the ESM head in a greedy
beam search over single-residue mutations. Given a parent peptide
(LL-37, magainin-2, cecropin A, indolicidin, melittin, or a custom
sequence) and a mutation budget *n*, the search expands every
beam-member by its 19 × len possible single-residue substitutions,
scores all candidates with the ESM head in one batched forward pass,
keeps the top beam-width per hop, and after *n* hops surfaces the
global top-K mutants ranked by P(AMP). On a 4060 Laptop GPU, a
~25-aa parent with `n=2` and `beam_width=10` completes in
~10–15 seconds.

Applied to **magainin-2** (parent ESM-head P(AMP) = 0.9954, near
ceiling), the top-10 ranked mutants after two hops with beam-width 10
share a striking pattern: **every single one introduces Tryptophan
substitutions**, and the substituted positions cluster at residues 6,
10, 13, 15, 17, 19, 21 — which all map to the same face of magainin-2's
amphipathic α-helix (3.6 residues per helical turn means positions 10,
13, 17, 21 are within one turn-width of each other on the same helix
face). This is the **canonical "Trp-anchor amphipathic enhancement"**
AMP design heuristic from the literature: aromatic residues at the
hydrophobic face of an α-helical AMP improve membrane partitioning, and
indolicidin (Trp-rich by design) is the textbook example. The
classifier rediscovered this design principle from DRAMP training data
alone — no helical-wheel hand-engineering required.

The Δ headroom from a parent already at P(AMP) = 0.9954 is small
(+0.0045) because the model is honest about diminishing returns on an
already-strong AMP. Running the same search on a marginal AMP parent
(P(AMP) ≈ 0.6) produces more diverse top-K suggestions; the
narrowness of the magainin-2 result is itself evidence the model is
behaving sensibly.

The design tab is in-silico only. Any wet-lab follow-up would need to
validate the proposed Trp-substituted magainin-2 variants against
indicator strains, ideally including hemolytic activity measurement
since Trp-rich AMPs trend toward higher mammalian membrane toxicity
(the inherent risk of the design recipe the model is recommending).

---

## 5. Discussion

### 5.1 Why ESM wins, and by how much

The literature consensus is that frozen protein-LM embeddings add
3–8 AUC points to AMP classifiers depending on corpus size and
split discipline. Our +5.5 AUC vs XGBoost on a 40%-identity cluster-
aware split is squarely in that range, with non-overlapping CIs that
make the comparison statistically clean. The cytochrome-c example
provides a mechanistic, single-sequence demonstration of *why* the
margin exists — engineered features can only encode what their human
designers chose to encode (charge, hydrophobicity, periodicity); ESM
encodes the functional context of every protein it has seen during
pretraining, including non-AMP cationic proteins like cytochrome c.

### 5.2 The calibration story

The CNN's Phase-1 score distribution was shifted near zero because
BCE+pos_weight pushes the model to predict positive aggressively
during training, and BatchNorm sharpens the score distribution. The
val-tuned MCC-optimal threshold was 0.050, far from the standard 0.5.
Phase 2A fixed this with Platt scaling — a 2-parameter sigmoid fit
on val logits, applied at inference. AUC was preserved (Platt is
monotonic), and MCC@0.5 lifted from 0.138 to 0.570, *exceeding* the
raw + val-tuned-threshold report of 0.477 because the Platt scaler
learns scale AND offset together. The ESM head did not need Platt;
its output distribution was well-calibrated by default (MCC@0.5 0.697
> MCC@tuned 0.670).

This is a useful demonstration of when calibration is and isn't
needed. BCE+pos_weight + BatchNorm = often need Platt. Simpler
architectures or different loss balancing often don't.

### 5.3 Multi-class biology

The multi-label head's per-class AUCs reflect biological reality more
than they reflect model limitations:

- Antifungal AMPs and antibacterial AMPs share membrane-disrupting
  mechanisms, and the same peptide is frequently positive in both
  classes (most defensins, most cecropins). Asking a sequence-only
  model to separate them is asking it to distinguish which lipid
  composition the peptide will encounter — information not in the
  sequence alone.
- Antiviral AMPs have more distinctive sequence patterns (proline-
  rich for some, hairpin motifs for others, specific charge patterns
  for HIV/HCV fusion inhibition) and correspondingly land a tighter
  AUC.
- Antiparasitic is undersampled in DRAMP itself (81 of 9,224
  positives). The wide CI is a data issue, not a model issue.

### 5.4 Discriminator-in-the-loop as a portfolio-grade design tool

Phase 2D's mutation search is the simplest possible generator —
exhaustive single-mutation enumeration filtered by a strong
classifier. The lift over a more sophisticated generator (fine-tuned
protein LM, sequence VAE, masked-token sampling) is whether you can
afford the engineering and training cost; the floor is whether you
have a strong classifier to filter with. On this corpus, the ESM head
is good enough that a near-trivial generator produces
biology-sensible designs.

The pattern generalises: for any classification task where a
strong scorer exists and the input alphabet is small, exhaustive or
beam-search over local mutations + classifier scoring is a viable
"generator" that needs no separate training. It will not propose
sequences far from the training distribution (no novelty in that
sense), but for refinement of known scaffolds — exactly the task here
— it is a clean baseline that any more complex generative model needs
to clearly beat.

The magainin-2 Trp-substitution result is also a useful piece of
*model-introspection*: it tells us what the ESM head thinks the
prototypical AMP "looks like" along the gradient near magainin-2's
location in sequence space, and the answer aligns with the AMP
literature's design heuristic for that scaffold class. A model whose
suggestions did not align with literature design heuristics in a
controlled probe like this would deserve more skepticism than the
held-out AUC alone could justify.

### 5.5 What the engineered baselines are still good for

Despite losing to ESM, the engineered baselines retain three roles:

1. **An honest comparison floor.** Without them, the ESM-head AUC of
   0.919 has no context. With them, the +5.5 AUC margin is
   interpretable.
2. **A fast inference path.** XGBoost scores at hundreds of
   sequences/sec on CPU; ESM2-650M needs a GPU for comparable
   throughput. For batch screening pipelines a sub-second tree model
   may be the right tool.
3. **Transparent feature attribution.** The demo surfaces XGBoost's
   top-5 features (net charge, several dipeptides, etc.) with per-
   sequence values. ESM's 1280-dim mean-pooled embedding offers
   nothing comparably human-readable. For exploratory analysis ("why
   did this peptide score high?") the engineered model is the better
   communicator.

---

## 6. Limitations

- **Binary task only, multi-label extension is preliminary.** The
  v1 multi-label head was trained on the AMP-positive subset of the
  cluster-aware splits; a more principled "joint binary + multi-class"
  head would route through the binary gate first.
- **Antiparasitic class.** 81 positives is too few for the four-way
  multi-label setup. Should be pooled or dropped in v2.
- **No structure information.** Sequence-only model; helical packing,
  disulfide bonds, oligomerisation state are all invisible. AMPs with
  structure-dependent mechanisms (β-sheet defensins, polycyclic
  bacteriocins) may be under-predicted.
- **No fine-tuning of ESM.** Frozen embeddings + small head only.
  Full or partial ESM fine-tuning could add another 1–3 AUC points but
  requires careful learning-rate scheduling and probably a larger
  corpus to avoid overfitting.
- **Domain shift.** Designed cyclic peptides, peptidomimetics, and
  D-amino-acid variants are outside the training distribution. The
  classifier's predictions on those should be treated as
  out-of-distribution.
- **DRAMP CC-BY-NC.** Any commercial deployment of this model would
  require renegotiating the DRAMP license. The current Hugging Face
  Space deploy is academic only.

---

## 7. Conclusion + Phase 3 wet-lab teaser

This project delivers an honest sequence-only AMP classifier with
state-of-the-art ESM2-frozen-embedding performance (AUC 0.919, MCC
0.697 on a 40%-identity cluster-aware test split), a calibrated CNN
baseline for engineered-feature comparison (AUC 0.837), a multi-label
activity head (macro-AUC 0.806 across four classes), a
discriminator-in-the-loop mutation search that rediscovers the
Trp-anchor amphipathic-enhancement design heuristic from
training data alone, and a working local demo with Predict and Design
tabs. Every result is bootstrap-CI'd, the cluster-purity property is
unit-tested, and the data pipeline is reproducible from `make data`
end-to-end on a laptop with an RTX 4060 GPU.

**Phase 3** is the lab-bench validation: pick a small handful (~5) of
the top-ranked Phase 2D design proposals — likely the
Trp-substituted magainin-2 variants the design tab surfaces, or
analogous Trp-enriched suggestions on the other library parents —
synthesize them via a peptide-synthesis service, assay against
*E. coli* K-12 and *S. aureus* ATCC 25923 indicator strains plus a
mammalian-cell hemolysis control (since Trp-rich AMPs trend toward
host membrane toxicity), and report the hit rate honestly in a
follow-up. Any result is publishable — even 0/5 active hits would be
an informative negative finding about the limits of training-corpus-
extrapolated design heuristics. This is operationally the maintainer's call
once lab time and synthesis budget align.

---

## References

- Shi G et al. (2022) DRAMP 3.0: an enhanced comprehensive data
  repository of antimicrobial peptides. *Nucleic Acids Research*
  50(D1), D488–D496.
- Lin Z et al. (2023) Evolutionary-scale prediction of atomic-level
  protein structure with a language model. *Science* 379(6637),
  1123–1130. (ESM2)
- Steinegger M, Söding J (2017) MMseqs2 enables sensitive protein
  sequence searching for the analysis of massive data sets. *Nature
  Biotechnology* 35, 1026–1028.
- Veltri D, Kamath U, Shehu A (2018) Deep learning improves
  antimicrobial peptide recognition. *Bioinformatics* 34(16),
  2740–2747. (AMP-Scanner)
- Meher PK, Sahu TK, Saini V, Rao AR (2017) Predicting antimicrobial
  peptides with improved accuracy by incorporating the compositional,
  physico-chemical and structural features. *Scientific Reports* 7,
  42362. (iAMPpred)
- Boman HG (2003) Antibacterial peptides: basic facts and emerging
  concepts. *Journal of Internal Medicine* 254(3), 197–215.
- Eisenberg D, Weiss RM, Terwilliger TC (1984) The hydrophobic moment
  detects periodicity in protein hydrophobicity. *PNAS* 81(1), 140–144.
- Kyte J, Doolittle RF (1982) A simple method for displaying the
  hydropathic character of a protein. *J. Mol. Biol.* 157(1), 105–132.
- Platt JC (1999) Probabilistic outputs for support vector machines and
  comparisons to regularized likelihood methods. In Smola et al.,
  *Advances in Large Margin Classifiers.*

---

*Code, data card, model card, per-model results, and the live Gradio
demo: see the project repository. Bug reports and Phase 2 / 3 ideas
welcome.*
