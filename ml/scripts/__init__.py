"""CLI entry points for the AMP Classifier pipeline.

Each script is callable from the Makefile and exits with a meaningful
status code. The order in §5 of AMP_CLASSIFIER_HANDOFF.md is:

  1. download_data.py  — DRAMP + UniProt negatives → ml/data/raw/
  2. validate_data.py  — alphabet, length, dedup, no positive/negative collisions
  3. make_clusters.py  — mmseqs2 / CD-HIT clustering at 40% identity
  4. make_splits.py    — cluster-aware 80/10/10 train/val/test
  5. train_baseline.py — LogReg, RandomForest, XGBoost
  6. eval_baseline.py  — AUC + MCC + bootstrap CIs, stratified analysis
  7. train_cnn.py      — small Conv1D over one-hot
  8. eval_cnn.py       — same eval pipeline as baselines
"""
