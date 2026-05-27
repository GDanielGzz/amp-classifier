# AMP Classifier

> **Status: research code, not production.**
> Results are reproducible against the corpus snapshot in
> `docs/data_card.md` with the random seeds pinned in
> `ml/scripts/make_splits.py` and `ml/eval/bootstrap.py`. The code is
> tested for the cases it was built against and is not hardened for
> production deployment. Known limitations are documented in
> [`LIMITATIONS.md`](LIMITATIONS.md). Issues and pull requests welcome.

A web-based antimicrobial peptide (AMP) discriminator. The binary
AMP / non-AMP scorer is a frozen ESM2-650M + MLP head trained on the
DRAMP 3.0 corpus and compared head-to-head against engineered-feature
baselines (LogReg, RandomForest, XGBoost, CNN) on a cluster-aware
held-out test set with bootstrap confidence intervals. Paste a peptide
sequence to get an AMP probability with a feature-level explanation.

## Headline results

Bootstrap 95% CIs from n=1000 resamples on the 1,969-sequence held-out
cluster-aware test split (882 positives + 1,087 negatives, no cluster
spans the train / test boundary at 40% identity).

| Model | AUC | MCC |
|---|---|---|
| LogReg (428 engineered features) | 0.825 [0.805–0.844] | 0.542 [0.504–0.579] |
| RandomForest | 0.863 [0.845–0.879] | 0.636 [0.602–0.668] |
| XGBoost | 0.864 [0.845–0.880] | 0.621 [0.587–0.652] |
| CNN (one-hot, 192k params) | 0.837 [0.818–0.854] | 0.477 [0.449–0.506] |
| CNN, Platt-calibrated | 0.837 [0.818–0.854] | 0.570 [0.534–0.606] |
| **ESM2-650M + MLP head** | **0.919 [0.906–0.931]** | **0.697 [0.667–0.730]** |

ESM2-650M frozen embeddings plus a tiny MLP head clear the engineered-
feature ceiling by **+5.5 AUC and +0.076 MCC vs XGBoost, with non-
overlapping 95% confidence intervals** on both metrics. Within the
engineered-feature group, XGBoost and RandomForest are statistically tied
(CIs overlap heavily); LogReg trails by ~4 AUC points.

Full stratified analysis (length and net-charge bins) in
`docs/baseline_results.md`, `docs/cnn_results.md`, `docs/esm_results.md`.
Unified comparison and caveats in `docs/model_card.md`.

## Why this project exists

AMP discrimination is a useful in-silico screen for narrowing candidate
peptide libraries before wet-lab assay. But many published benchmarks
overstate model performance by 10+ AUC points because naive train/test
splits leak sequence homology — clusters of near-identical peptides end up
on both sides of the boundary, so models effectively memorise rather than
generalise. This project addresses the leakage problem with three methodological
choices:

- **Cluster-aware evaluation at 40% identity.** mmseqs2 clustering (CD-HIT
  fallback) ensures no cluster spans the train/test boundary. Absolute
  AUC drops compared with naive splits, but the numbers generalise.
- **A trained model with a measured baseline on identical splits.**
  ESM2-650M frozen + MLP head head-to-head against an XGBoost over 428
  engineered features, evaluated on the same cluster-aware held-out set
  with bootstrap CIs on both metrics. The comparison reports which model
  wins by how much, with explicit uncertainty.
- **An interactive web demo.** A Gradio app lets biologists paste a
  peptide sequence and get a probability with feature-level explanation;
  runs locally now, with a Hugging Face Spaces deployment planned for
  follow-up.

## How to reproduce

Requires Python 3.11 through 3.13.

```bash
# Clone
git clone https://github.com/GDanielGzz/amp-classifier.git
cd amp-classifier

# Install
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Run the pipeline
make data       # DRAMP positives + UniProt non-AMP negatives
make validate   # alphabet + length + dedup
make cluster    # mmseqs2 at 40% identity (CD-HIT fallback)
make splits     # cluster-aware 80/10/10
make baseline   # LogReg + RandomForest + XGBoost with bootstrap CIs
make cnn        # small Conv1D over one-hot
make esm        # ESM2-650M frozen embeddings + MLP head
make app        # Gradio on http://127.0.0.1:7860
```

On Windows PowerShell, `dev.bat` mirrors the Makefile targets one-to-one.

Expected end-to-end runtime on CPU: ~2 hours (dominated by ESM2
embedding extraction). On a GPU: ~20 minutes.

## Repo layout

```
app.py             Gradio demo entry point
ml/                Data acquisition, scripts, features, eval, checkpoints
tests/             pytest unit + cluster-purity invariant tests
docs/              data_card.md, model_card.md, baseline_results.md,
                   cnn_results.md, esm_results.md
hf_space/          Hugging Face Space deploy artifacts
report/            Technical report PDF
LIMITATIONS.md     Known limitations and caveats
CITATION.cff       Machine-readable citation metadata
LICENSE            MIT
```

## In scope / out of scope

**In scope (v0.1):**

- Binary AMP / non-AMP classification on peptides ≤200 aa.
- DRAMP 3.0 as the positive corpus; UniProt SwissProt as the source of
  length-matched, antimicrobial-keyword-excluded negatives.
- mmseqs2 clustering at 40% identity for the test split (CD-HIT fallback).
- Engineered features (composition + Boman + Eisenberg + standard
  physicochemical) for the baseline; one-hot encoding for the CNN; frozen
  ESM2-650M embeddings for the deep model.
- Local Gradio demo + Hugging Face Space.

**Out of scope (v0.1):**

- Multi-class activity prediction at production quality (Phase 2 ships
  it, but the 96% antibacterial class imbalance limits the other three
  classes; see `docs/model_card.md`).
- Structure-based prediction or AlphaFold integration.
- Generative AMP design — this is a discriminator, not a generator. The
  Phase 2 discriminator-in-the-loop mutant designer is a *ranker* of
  proposed mutations, not a generator of novel peptides.
- ESM2-650M full fine-tuning (frozen head only).
- GPU-only inference paths. Free Hugging Face Spaces tier is CPU.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full caveats list.

## Live demo

A Hugging Face Spaces deployment is planned; the deploy bundle is
prepared in `hf_space/` and will go live in a follow-up release. In the
meantime, run the Gradio demo locally with `make app` (see *How to
reproduce* above) — paste a peptide and get the AMP probability with a
feature-level explanation.

## Citation

If you use this work, please cite the technical report:

```
González Lozano, D. (2026). AMP Classifier: A trained ESM2-650M head
for antimicrobial peptide discrimination on cluster-aware splits.
Zenodo. https://doi.org/10.5281/zenodo.[PENDING_DEPOSIT]
```

A machine-readable citation lives in [`CITATION.cff`](CITATION.cff);
GitHub renders a "Cite this repository" button in the sidebar.

## Acknowledgments

Training corpus and published baselines come from:

- Shi, G. *et al.* (2022). *DRAMP 3.0: an enhanced comprehensive data
  repository of antimicrobial peptides.* Nucleic Acids Research,
  50(D1), D488–D496. <https://doi.org/10.1093/nar/gkab651>
- Meher, P. K. *et al.* (2017). *Predicting antimicrobial peptides with
  improved accuracy by incorporating the compositional, physico-chemical
  and structural features into Chou's general PseAAC.* Scientific
  Reports, 7, 42362. <https://doi.org/10.1038/srep42362>
- Veltri, D. *et al.* (2018). *Deep learning improves antimicrobial
  peptide recognition.* Bioinformatics, 34(16), 2740–2747.
  <https://doi.org/10.1093/bioinformatics/bty179>

ESM2-650M from Lin, Z. *et al.* (2023). *Evolutionary-scale prediction
of atomic level protein structure with a language model.* Science,
379(6637), 1123–1130. <https://doi.org/10.1126/science.ade2574>

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Daniel González Lozano · `sleepy.komodo@protonmail.com` ·
[ORCID](https://orcid.org/0009-0002-1737-276X)

Postdoctoral researcher, Tecnológico de Monterrey × Technical University
of Denmark (Novo Nordisk Foundation Center for Biosustainability).
