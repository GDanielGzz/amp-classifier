# Known Limitations — AMP Classifier

This document is the deliberate "what would a careful reviewer catch?" pass
for the project. Listing limitations up front is a credibility move, not an
embarrassment. The discipline that makes this kind of model trustworthy in
the AMP literature is exactly the same discipline that surfaces these
caveats.

Last updated: 2026-05-23 (initial scaffolding by CodeNya)
Edit this file in place as the project evolves; resolved limitations get
struck through with a date and a brief note.

---

## Data limitations

- **Corpus snapshot date**: DRAMP 3.0 (positives) and UniProt SwissProt
  (negatives) as of the date in `docs/data_card.md`. Upstream additions
  to DRAMP since the snapshot are not reflected.
- **Positive class composition**: 9,224 DRAMP entries after dedup; class
  balance vs negatives is engineered (length-matched, antimicrobial-keyword-
  excluded UniProt subset) — this is a deliberate choice and is documented
  in the data card, but it does limit comparability to studies that use
  different negative-set construction.
- **96% antibacterial class imbalance** in the Phase 2 multi-label activity
  head. Antifungal / antiviral / antiparasitic predictions inherit this
  imbalance and are reported with that caveat in `docs/model_card.md`.
- **Peptide length cap**: 200 amino acids. Longer peptides are out of
  distribution and predictions are unreliable.

## Evaluation limitations

- **Cluster-aware splits at 40% identity** (mmseqs2; CD-HIT fallback). At
  stricter identity thresholds (25–30%) absolute AUC numbers drop; the
  *direction* of the ESM2-vs-XGBoost comparison is robust at the
  thresholds we tested but extreme thresholds were not exhaustively
  evaluated.
- **Bootstrap CIs** computed at n=1000 resamples on the held-out test
  set. Confidence intervals reported in the README and the model card
  are 95% percentile intervals. Single-split + bootstrap is a deliberate
  choice over k-fold CV; the trade-off is documented in
  `docs/baseline_results.md`.
- **No external test corpus.** All evaluation is on the held-out cluster-
  aware DRAMP+UniProt split. A truly independent corpus (e.g., a freshly-
  curated AMP set from a different lab) would strengthen the claim;
  not present in v0.1.
- **Threshold tuning was performed on the validation set, not the test
  set.** The Platt-calibrated CNN's standard-threshold MCC of 0.570 is
  reported alongside the val-tuned t=0.050 result for the uncalibrated
  CNN, with both labeled.

## Model limitations

- **ESM2-650M frozen, not fine-tuned.** The +5.5 AUC improvement over
  XGBoost is achieved with frozen embeddings + a small MLP head. ESM2
  fine-tuning was out of v1 scope; further gains are plausible but
  unverified.
- **Hyperparameter search budget was bounded.** The MLP head was tuned
  on the validation split with a small grid; the XGBoost baseline was
  tuned similarly. Both used the same compute envelope for a fair
  comparison, but neither saw an exhaustive search.
- **CNN architecture choices** (192k-param Conv1D + BatchNorm,
  val-tuned threshold, Platt calibration for the Phase 2A result) were
  set by iterative experimentation; not exhaustively ablated.
- **Inference cost**: ESM2-650M embedding extraction is the bottleneck;
  CPU-only inference on the free HF Spaces tier is 5–15 s per sequence
  depending on length.

## Scope limitations

- **In scope**: binary AMP / non-AMP classification on peptides ≤200 aa,
  trained on DRAMP 3.0 positives + length-matched UniProt negatives,
  cluster-aware evaluation at 40% identity.
- **Out of scope**: multi-class activity prediction at production
  quality (Phase 2 ships it with honest caveats but the 96% antibacterial
  imbalance limits the other three classes), structure-based prediction,
  AlphaFold integration, generative AMP design (the discriminator-in-
  the-loop mutant designer is a *ranker*, not a *generator*), ESM2
  full fine-tuning, GPU-only inference paths.
- **Not a substitute for**: wet-lab MIC validation, regulatory
  approval, or any clinical / therapeutic / commercial decision. AMP
  hits from this classifier are starting points for experimental
  validation, not finished candidates.

## Reproducibility caveats

- **Pinned dependencies**: see `requirements.txt`. The pins are
  compatible-range to permit Python 3.11–3.13 wheel selection.
- **Random seeds**: bootstrap RNG (`np.random.default_rng(42)`) and
  split RNG (separate seed in `ml/scripts/make_splits.py`) are
  independent.
- **mmseqs2 version**: results were computed against the version
  reported in `docs/data_card.md`. Different versions can produce
  slightly different cluster boundaries; the headline conclusion is
  robust across the versions we tested.

## Known bugs and open questions

- *No open bugs at v0.1.* The full test suite passes; the model card
  documents the model's behavior on the test set in detail. If bugs are
  discovered after publication, they'll be tracked as GitHub Issues and
  resolved in a versioned release.

## Things I'm uncertain about (honest section)

- Whether the +5.5 AUC ESM2 advantage transfers to a more recent AMP
  corpus snapshot. The headline is solid against the snapshot we used;
  re-evaluating on a 2027 DRAMP refresh is the natural next test.
- Whether 40% identity is the *right* threshold for the cluster-aware
  split in this domain. The AMP literature uses 30–50%; we picked 40%
  as a defensible middle. A sensitivity analysis across 25–50% would
  strengthen the methodology section.
- The Phase 2A Platt-calibrated CNN reaches MCC 0.570 at the standard
  0.5 threshold (vs 0.477 uncalibrated at the val-tuned threshold). The
  calibration helps, but Platt scaling on small validation sets can be
  unstable; this should be re-verified on a larger or external validation
  set.
- Multi-label activity head (Phase 2B) is honest about the class
  imbalance but I'd want a domain expert to spot-check whether the
  per-class predictions are mechanistically plausible before publishing
  any specific peptide as e.g. "predicted antifungal."

---

## Resolved limitations (kept for history)

*None yet — first publication.*
