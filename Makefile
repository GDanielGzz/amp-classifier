# Makefile for the AMP Classifier project.
#
# Targets assume the venv is already activated. The project folder contains
# a space — when invoking make from a parent directory, quote the path:
#
#   make -C "AMP Classifier" baseline
#
# Run `make help` for a list of targets.

PYTHON ?= python
PIP    ?= pip
PORT   ?= 7860

.PHONY: help install install-dev data validate cluster splits baseline cnn eval test app clean

help:
	@echo "Common targets:"
	@echo "  install      Install runtime deps (pip install -r requirements.txt)"
	@echo "  install-dev  Install dev deps (includes runtime)"
	@echo "  data         Download DRAMP positives + UniProt non-AMP negatives"
	@echo "  validate     Validate the downloaded corpus (alphabet + length + dedup)"
	@echo "  cluster      Cluster sequences at 40% identity (mmseqs2, CD-HIT fallback)"
	@echo "  splits       Build cluster-aware train/val/test splits"
	@echo "  baseline     Train + evaluate LogReg + RandomForest + XGBoost baselines"
	@echo "  cnn          Train + evaluate the small one-hot CNN"
	@echo "  test         Run pytest"
	@echo "  app          Launch the Gradio demo locally on port $(PORT)"
	@echo "  clean        Remove caches (keeps data, clusters, splits, checkpoints)"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

data:
	$(PYTHON) ml/scripts/download_data.py

validate:
	$(PYTHON) ml/scripts/validate_data.py

cluster:
	$(PYTHON) ml/scripts/make_clusters.py

splits: cluster
	$(PYTHON) ml/scripts/make_splits.py

baseline: splits
	$(PYTHON) ml/scripts/train_baseline.py
	$(PYTHON) ml/scripts/eval_baseline.py

cnn: splits
	$(PYTHON) ml/scripts/train_cnn.py
	$(PYTHON) ml/scripts/eval_cnn.py

calibrate:
	$(PYTHON) ml/scripts/calibrate_cnn.py
	$(PYTHON) ml/scripts/eval_cnn.py

esm-embed:
	$(PYTHON) ml/scripts/extract_esm_embeddings.py

esm-train:
	$(PYTHON) ml/scripts/train_esm_head.py

esm-eval:
	$(PYTHON) ml/scripts/eval_esm_head.py

esm: esm-embed esm-train esm-eval

eval:
	$(PYTHON) ml/scripts/eval_baseline.py
	$(PYTHON) ml/scripts/eval_cnn.py

test:
	$(PYTHON) -m pytest -v

app:
	GRADIO_SERVER_PORT=$(PORT) $(PYTHON) app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
	rm -rf gradio_cached_examples flagged
