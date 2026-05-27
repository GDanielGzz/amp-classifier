"""Phase 2B step 1: extract per-sequence ESM2 embeddings.

Loads an ESM2 model from Hugging Face Hub, mean-pools the last hidden
state over non-padding tokens, and caches the embedding per sequence to:

  ml/data/processed/esm_train.npz
  ml/data/processed/esm_val.npz
  ml/data/processed/esm_test.npz

Model is configurable via the ``ESM_MODEL_NAME`` env var:

  ESM2-35M  (480-dim, ~130MB, CPU-friendly)   ← default
  ESM2-150M (640-dim, ~600MB, ~3-5 min GPU)
  ESM2-650M (1280-dim, ~2.5GB, ~10-20 min GPU; canonical Phase 2B)

PowerShell:
  $env:ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
  python ml\scripts\extract_esm_embeddings.py

Device selection is automatic: uses CUDA if available, falls back to CPU.
For CUDA you need a torch build with CUDA wheels:
  pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

Time budget on RTX 4060:
  - ESM2-35M:  ~30 seconds total
  - ESM2-150M: ~3-5 minutes
  - ESM2-650M: ~10-20 minutes
On CPU these are ~100x slower; pick a smaller model if no GPU.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.scripts.train_cnn import load_sequence_db, load_split_csv  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
ESM_MODEL_NAME = os.environ.get("ESM_MODEL_NAME", DEFAULT_ESM_MODEL)

# Batch size — scales with VRAM. ESM2-650M needs smaller batches on a 4060.
BATCH_SIZE = int(os.environ.get("ESM_BATCH_SIZE",
                                "8" if "650M" in ESM_MODEL_NAME else "32"))


def select_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[esm] CUDA available: {name} ({mem_gb:.1f}GB)")
        return torch.device("cuda")
    print("[esm] no CUDA; using CPU (will be slow for larger ESM2 variants)")
    return torch.device("cpu")


def load_esm_model(device):
    from transformers import AutoTokenizer, AutoModel
    print(f"[esm] loading {ESM_MODEL_NAME} (first run downloads weights) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME).eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[esm] loaded in {time.time()-t0:.1f}s; {n_params:,} params; "
          f"hidden_size={model.config.hidden_size}")
    return tokenizer, model, model.config.hidden_size


def embed_sequences(seqs, tokenizer, model, device, embedding_dim,
                    batch_size=BATCH_SIZE):
    out = np.zeros((len(seqs), embedding_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, return_tensors="pt",
                            truncation=True, max_length=200)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        sums = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = (sums / counts).cpu().numpy().astype(np.float32)
        out[i:i + len(batch)] = pooled
        if (i // batch_size) % 25 == 0:
            done = min(i + batch_size, len(seqs))
            rate = done / max(1e-3, time.time() - t0)
            eta_min = (len(seqs) - done) / max(1e-3, rate) / 60
            print(f"  [esm] {done}/{len(seqs)} ({rate:.1f} seq/s, ETA {eta_min:.1f} min)")
    print(f"[esm] embedded {len(seqs)} in {(time.time()-t0)/60:.2f} min")
    return out


def extract_split(split_name, seq_db, tokenizer, model, device, embedding_dim):
    cache = PROCESSED_DIR / f"esm_{split_name}.npz"
    if cache.exists():
        existing = np.load(cache, allow_pickle=True)
        if existing["X"].shape[1] == embedding_dim:
            print(f"[esm] cache hit: {cache.name} (shape OK)")
            return
        else:
            print(f"[esm] cache exists but dim mismatch "
                  f"({existing['X'].shape[1]} vs {embedding_dim}); re-extracting")
    print(f"[esm] computing {split_name} ...")
    ids, labels = load_split_csv(SPLITS_DIR / f"{split_name}.csv")
    seqs = [seq_db[sid] for sid in ids]
    feat_cache = PROCESSED_DIR / f"features_{split_name}.npz"
    if feat_cache.exists():
        feat = np.load(feat_cache, allow_pickle=True)
        lengths = feat["lengths"]
        charges = feat["charges"]
    else:
        lengths = np.array([len(s) for s in seqs], dtype=np.int32)
        charges = np.zeros(len(seqs), dtype=np.float32)
    X = embed_sequences(seqs, tokenizer, model, device, embedding_dim)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=labels.astype(np.int32),
             lengths=lengths, charges=charges,
             ids=np.array(ids, dtype=object),
             model_name=ESM_MODEL_NAME)
    print(f"[esm] wrote {cache.name}: X={X.shape}")


def main():
    device = select_device()
    seq_db = load_sequence_db()
    tokenizer, model, embedding_dim = load_esm_model(device)
    for split in ("train", "val", "test"):
        extract_split(split, seq_db, tokenizer, model, device, embedding_dim)
    print(f"[esm] all splits embedded with {ESM_MODEL_NAME}; "
          f"ready for train_esm_head.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
