---
title: AMP Classifier
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Antimicrobial peptide classifier, ESM2-650M + MLP head
models:
  - facebook/esm2_t33_650M_UR50D
tags:
  - biology
  - protein
  - peptides
  - antimicrobial-peptides
  - esm2
---

# AMP Classifier

Paste a peptide sequence and get the probability that it is an
antimicrobial peptide (AMP). Research code and a portfolio demo — **not
for clinical, regulatory or production use.**

Source, training pipeline and full documentation:
<https://github.com/GDanielGzz/amp-classifier>

## What the demo does

- **Predict tab** — three binary AMP / non-AMP models score the sequence
  side by side:
  - **ESM2-650M + MLP head** (champion): a small MLP over frozen,
    mean-pooled [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
    embeddings.
  - **XGBoost** over 428 engineered physicochemical and composition
    features, with per-feature attribution.
  - **CNN** over one-hot sequences, Platt-calibrated.
- **Design tab** — greedy beam search over single-residue substitutions
  of a parent AMP (LL-37, magainin-2, cecropin A, indolicidin, melittin,
  or your own sequence), ranked by the ESM head. It is a ranker of local
  mutations, not a generator of novel peptides.

Input: canonical 20 amino acids; the demo caps input at 100 residues.
Models whose checkpoint is missing from the Space are skipped, and the
Design tab needs the ESM head. On the free CPU tier, ESM2-650M inference
takes roughly 5–15 s per sequence, and the first start downloads the
backbone from the Hub.

## Model and results

Positives come from DRAMP 3.0, negatives from length-matched UniProt
SwissProt entries without antimicrobial keywords (9,224 positives /
11,252 negatives after dedup). Splits are cluster-aware: mmseqs2 at 40%
sequence identity, so no cluster spans the train / test boundary.

Held-out test set: 1,969 sequences (882 AMP / 1,087 non-AMP). 95%
bootstrap CIs, n=1000.

| Model | AUC | MCC |
|---|---|---|
| **ESM2-650M + MLP head** | **0.919 [0.906–0.931]** | **0.697 [0.667–0.730]** |
| XGBoost (428 engineered features) | 0.864 [0.845–0.880] | 0.621 [0.587–0.652] |
| CNN, Platt-calibrated | 0.837 [0.818–0.854] | 0.570 [0.534–0.606] |

The ESM head beats XGBoost by +5.5 AUC and +0.076 MCC, with
non-overlapping 95% CIs on both metrics. Full tables, including the
LogReg and RandomForest baselines and the multi-label activity head, are
in the [model card](https://github.com/GDanielGzz/amp-classifier/blob/main/docs/model_card.md).

## Intended use

Ranking candidate peptides by "AMP-likeness", screening designs before
wet-lab assay, and teaching the cluster-aware split discipline. Hits are
starting points for experimental validation (e.g. MIC assays), not
finished candidates. Not a substitute for wet-lab validation or for any
clinical, therapeutic, regulatory or commercial decision.

## Key limitations

- Sequence-only: no structure, so structure-dependent AMPs (β-sheet
  defensins, polycyclic bacteriocins) may be under-predicted.
- Out of distribution: peptides longer than the training range, cyclic
  peptides, peptidomimetics and D-amino-acid variants.
- ESM2 is frozen, not fine-tuned. No external test corpus; all numbers
  come from the held-out cluster-aware DRAMP + UniProt split.
- Design-tab mutants are in-silico only. Hemolysis and cytotoxicity are
  not modelled, and the Trp-rich mutants the model tends to favour carry
  that risk.
- **Data license:** DRAMP is CC-BY-NC for academic use. This Space is for
  academic and portfolio use; commercial redistribution would need a
  separate agreement with the DRAMP authors.

Full list: [LIMITATIONS.md](https://github.com/GDanielGzz/amp-classifier/blob/main/LIMITATIONS.md)
· [model card](https://github.com/GDanielGzz/amp-classifier/blob/main/docs/model_card.md)
· [data card](https://github.com/GDanielGzz/amp-classifier/blob/main/docs/data_card.md)

## Citation

Cite the software via
[CITATION.cff](https://github.com/GDanielGzz/amp-classifier/blob/main/CITATION.cff)
(GitHub's "Cite this repository" button on the
[repo page](https://github.com/GDanielGzz/amp-classifier)). Please also
cite DRAMP 3.0 (Shi *et al.*, 2022, *Nucleic Acids Research*
50(D1):D488–D496) and ESM2 (Lin *et al.*, 2023, *Science*
379(6637):1123–1130).

Code: [MIT License](https://github.com/GDanielGzz/amp-classifier/blob/main/LICENSE).
Author: Daniel González Lozano.
