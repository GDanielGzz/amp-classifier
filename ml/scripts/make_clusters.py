"""Cluster amps.fasta + negatives.fasta at 40% identity for cluster-aware splits.

The cluster-aware split is the discipline that makes the held-out test
metrics honest. Random splits on AMP data leak homologous sequences
between train and test, inflating AUC by 10-15 points (Veltri 2018 found
this directly). This script clusters BOTH classes together so the
splitting step (`make_splits.py`) can assign whole clusters to one
split, never letting a sequence and its near-twin fall in different
splits.

Tool priority:
  1. mmseqs2 (preferred — faster, easier Windows install via conda)
  2. CD-HIT (fallback)

If neither binary is on PATH, the script writes a clear MISSING.md and
exits non-zero. Do NOT fall back to a random split.

Outputs land in ``ml/data/clusters/``:

  combined.fasta          Concatenated input. IDs renumbered to
                          ``amp_<i>`` / ``neg_<i>`` so class is recoverable
                          from the cluster output without re-reading
                          the source FASTAs.
  id_map.tsv              ``id\torig_header\tis_amp\tlength``. Preserves
                          provenance for the data card and downstream
                          scripts.
  clusters.csv            ``sequence_id,cluster_id,is_amp`` — the contract
                          that ``make_splits.py`` reads.

Install
-------

mmseqs2 — recommended:

  conda install -c bioconda -c conda-forge mmseqs2

  Or Windows binary: https://github.com/soedinglab/MMseqs2/releases

CD-HIT — fallback:

  conda install -c bioconda cd-hit

References:
  Steinegger M, Soeding J (2017) MMseqs2 enables sensitive protein
  sequence searching for the analysis of massive data sets. Nat
  Biotechnol 35, 1026-1028.
  Li W, Godzik A (2006) Cd-hit: a fast program for clustering and
  comparing large sets of protein or nucleotide sequences.
  Bioinformatics 22, 1658-1659.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
CLUSTERS_DIR = PROJECT_ROOT / "ml" / "data" / "clusters"
AMPS_FASTA = RAW_DIR / "amps.fasta"
NEGATIVES_FASTA = RAW_DIR / "negatives.fasta"
COMBINED_FASTA = CLUSTERS_DIR / "combined.fasta"
ID_MAP_TSV = CLUSTERS_DIR / "id_map.tsv"
CLUSTERS_CSV = CLUSTERS_DIR / "clusters.csv"
MISSING_PATH = CLUSTERS_DIR / "MISSING.md"

MMSEQS_MIN_SEQ_ID = 0.40
MMSEQS_COVERAGE = 0.80
CDHIT_WORD_SIZE = 2  # required for c < 0.5


# ---------------------------------------------------------------------------
# FASTA reading + combined input writer
# ---------------------------------------------------------------------------


def parse_fasta(path):
    header = None
    seq_chunks = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            elif line.strip():
                seq_chunks.append(line.strip())
    if header is not None:
        yield header, "".join(seq_chunks)


def build_combined_fasta():
    """Write the concatenated FASTA and the id_map.tsv. Returns id_to_class."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    id_to_class = {}
    with COMBINED_FASTA.open("w", encoding="utf-8", newline="\n") as fa, \
         ID_MAP_TSV.open("w", encoding="utf-8", newline="\n") as tsv:
        tsv.write("id\torig_header\tis_amp\tlength\n")
        for i, (header, seq) in enumerate(parse_fasta(AMPS_FASTA)):
            sid = f"amp_{i:06d}"
            seq_upper = seq.upper()
            fa.write(f">{sid}\n")
            for j in range(0, len(seq_upper), 60):
                fa.write(seq_upper[j:j + 60] + "\n")
            tsv.write(f"{sid}\t{header}\t1\t{len(seq_upper)}\n")
            id_to_class[sid] = 1
        for i, (header, seq) in enumerate(parse_fasta(NEGATIVES_FASTA)):
            sid = f"neg_{i:06d}"
            seq_upper = seq.upper()
            fa.write(f">{sid}\n")
            for j in range(0, len(seq_upper), 60):
                fa.write(seq_upper[j:j + 60] + "\n")
            tsv.write(f"{sid}\t{header}\t0\t{len(seq_upper)}\n")
            id_to_class[sid] = 0
    print(f"[clusters] wrote combined FASTA ({len(id_to_class)} sequences) and id_map.tsv")
    return id_to_class


# ---------------------------------------------------------------------------
# Tool runners
# ---------------------------------------------------------------------------


def have_binary(name):
    return shutil.which(name) is not None


def space_free_temp_root():
    """Return a temp directory whose absolute path contains no spaces.

    mmseqs2's Cygwin Windows build mangles argv when paths contain spaces
    (the cmd.exe `%*` re-expansion in mmseqs.bat doesn't preserve quotes
    consistently). The project folder name "AMP Classifier" has a space,
    so we stage all mmseqs I/O under a guaranteed-space-free root and
    copy the cluster TSV back when done.

    Tries the OS temp dir first (typically `%TEMP%` on Windows which is
    under `C:\\Users\\<name>\\AppData\\Local\\Temp` — fine unless the
    username has a space). Falls back to `C:\\amp_cluster_tmp` on Windows
    or `/tmp/amp_cluster_tmp` elsewhere.
    """
    candidates = [tempfile.gettempdir()]
    if os.name == "nt":
        candidates.append("C:\\amp_cluster_tmp")
    else:
        candidates.append("/tmp/amp_cluster_tmp")
    for c in candidates:
        if " " in c:
            continue
        root = Path(c)
        root.mkdir(parents=True, exist_ok=True)
        return root
    raise RuntimeError(
        "No space-free temp directory available. Set a TMPDIR env var that "
        "points to a path without spaces and retry."
    )


def run_mmseqs2(work_dir):
    """Run `mmseqs easy-cluster` and return the cluster.tsv path on success.

    Stages input + output + scratch in a space-free temp directory because
    mmseqs's Cygwin Windows wrapper cannot handle spaces in paths.
    """
    stage_root = space_free_temp_root()
    stage = Path(tempfile.mkdtemp(prefix="ampcluster_", dir=str(stage_root)))
    try:
        staged_input = stage / "input.fasta"
        shutil.copy2(COMBINED_FASTA, staged_input)
        out_prefix = stage / "out"
        tmp_subdir = stage / "tmp"
        tmp_subdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mmseqs", "easy-cluster",
            str(staged_input), str(out_prefix), str(tmp_subdir),
            "--min-seq-id", str(MMSEQS_MIN_SEQ_ID),
            "-c", str(MMSEQS_COVERAGE),
            "--cov-mode", "0",
        ]
        print(f"[mmseqs2] running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"[mmseqs2] failed to invoke: {exc!r}", file=sys.stderr)
            return None
        if result.returncode != 0:
            print(f"[mmseqs2] returned {result.returncode}", file=sys.stderr)
            print(result.stdout[-2000:] if result.stdout else "", file=sys.stderr)
            print(result.stderr[-2000:] if result.stderr else "", file=sys.stderr)
            return None
        cluster_tsv_src = Path(str(out_prefix) + "_cluster.tsv")
        if not cluster_tsv_src.exists():
            print(f"[mmseqs2] no {cluster_tsv_src.name} produced", file=sys.stderr)
            return None
        # Copy the result out of the temp staging area before the with-block
        # cleanup deletes it.
        cluster_tsv_dst = work_dir / "mmseqs_out_cluster.tsv"
        shutil.copy2(cluster_tsv_src, cluster_tsv_dst)
        return cluster_tsv_dst
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def run_cdhit(work_dir):
    """Run cd-hit and return the .clstr path on success. Stages in a
    space-free temp dir for the same reason as mmseqs2."""
    stage_root = space_free_temp_root()
    stage = Path(tempfile.mkdtemp(prefix="ampcdhit_", dir=str(stage_root)))
    try:
        staged_input = stage / "input.fasta"
        shutil.copy2(COMBINED_FASTA, staged_input)
        out_fa = stage / "cdhit_out.fa"
        cmd = [
            "cd-hit",
            "-i", str(staged_input),
            "-o", str(out_fa),
            "-c", str(MMSEQS_MIN_SEQ_ID),
            "-n", str(CDHIT_WORD_SIZE),
            "-d", "0",
            "-M", "0",
            "-T", "0",
        ]
        print(f"[cd-hit] running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"[cd-hit] failed to invoke: {exc!r}", file=sys.stderr)
            return None
        if result.returncode != 0:
            print(f"[cd-hit] returned {result.returncode}", file=sys.stderr)
            print(result.stderr[-2000:] if result.stderr else "", file=sys.stderr)
            return None
        clstr_src = Path(str(out_fa) + ".clstr")
        if not clstr_src.exists():
            print(f"[cd-hit] no {clstr_src.name} produced", file=sys.stderr)
            return None
        clstr_dst = work_dir / "cdhit_out.fa.clstr"
        shutil.copy2(clstr_src, clstr_dst)
        return clstr_dst
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cluster output parsers
# ---------------------------------------------------------------------------


def parse_mmseqs_cluster_tsv(path):
    """Return dict[member_id -> cluster_id]. cluster_id is the integer rank
    of the representative when sorted lexicographically (deterministic)."""
    rep_members = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            rep_members.setdefault(rep, []).append(member)
    rep_to_cluster = {rep: i for i, rep in enumerate(sorted(rep_members))}
    out = {}
    for rep, members in rep_members.items():
        cid = rep_to_cluster[rep]
        for m in members:
            out[m] = cid
    print(f"[mmseqs2] parsed {len(rep_members)} clusters covering {len(out)} sequences")
    return out


def parse_cdhit_clstr(path):
    """Parse CD-HIT .clstr into dict[member_id -> cluster_id].

    Format:
      >Cluster 0
      0       23aa, >seq_id_1... *
      1       22aa, >seq_id_2... at 87.50%
    """
    out = {}
    current_cluster = None
    cluster_count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
                cluster_count = max(cluster_count, current_cluster + 1)
                continue
            if ">" in line:
                gt = line.find(">")
                dot = line.find("...", gt)
                if dot == -1:
                    continue
                member = line[gt + 1 : dot].strip()
                if current_cluster is not None:
                    out[member] = current_cluster
    print(f"[cd-hit] parsed {cluster_count} clusters covering {len(out)} sequences")
    return out


# ---------------------------------------------------------------------------
# CSV writer + MISSING
# ---------------------------------------------------------------------------


def write_clusters_csv(member_to_cluster, id_to_class):
    """Join cluster assignment with class membership and write the CSV."""
    missing_class = [m for m in member_to_cluster if m not in id_to_class]
    missing_cluster = [s for s in id_to_class if s not in member_to_cluster]
    if missing_class:
        print(f"[clusters] WARNING: {len(missing_class)} clustered IDs not in id_map "
              f"(first: {missing_class[:3]})", file=sys.stderr)
    if missing_cluster:
        print(f"[clusters] WARNING: {len(missing_cluster)} input sequences not in "
              f"cluster output (first: {missing_cluster[:3]})", file=sys.stderr)
    with CLUSTERS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sequence_id", "cluster_id", "is_amp"])
        for sid in sorted(id_to_class):
            cid = member_to_cluster.get(sid, -1)
            writer.writerow([sid, cid, id_to_class[sid]])
    print(f"[clusters] wrote {CLUSTERS_CSV}")


def write_missing(reason):
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    MISSING_PATH.write_text(
        "# Clustering failed\n\n"
        f"{reason}\n\n"
        "Install one of:\n\n"
        "  conda install -c bioconda -c conda-forge mmseqs2   # preferred\n"
        "  conda install -c bioconda cd-hit                   # fallback\n\n"
        "Or grab the prebuilt mmseqs2 binary for Windows from\n"
        "https://github.com/soedinglab/MMseqs2/releases and add it to PATH.\n\n"
        "Per HANDOFF.md section 9: do NOT fall back to a random split.\n",
        encoding="utf-8",
    )
    print(f"[clusters] wrote {MISSING_PATH}")




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tool", choices=("auto", "mmseqs2", "cdhit"), default="auto",
                   help="Force a specific clustering tool. Default: auto.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not AMPS_FASTA.exists() or not NEGATIVES_FASTA.exists():
        print("[clusters] amps.fasta and negatives.fasta must exist. Run dev.bat data first.")
        return 1

    if MISSING_PATH.exists():
        MISSING_PATH.unlink()

    id_to_class = build_combined_fasta()

    with tempfile.TemporaryDirectory(prefix="amp_cluster_", dir=str(CLUSTERS_DIR)) as work:
        work_dir = Path(work)
        cluster_path = None
        used_tool = None
        member_to_cluster = None
        if args.tool in ("auto", "mmseqs2") and have_binary("mmseqs"):
            cluster_path = run_mmseqs2(work_dir)
            if cluster_path is not None:
                member_to_cluster = parse_mmseqs_cluster_tsv(cluster_path)
                used_tool = "mmseqs2"
        if cluster_path is None and args.tool in ("auto", "cdhit") and have_binary("cd-hit"):
            cluster_path = run_cdhit(work_dir)
            if cluster_path is not None:
                member_to_cluster = parse_cdhit_clstr(cluster_path)
                used_tool = "cd-hit"
        if cluster_path is None:
            tools_tried = []
            if args.tool in ("auto", "mmseqs2"):
                tools_tried.append("mmseqs (not found on PATH)" if not have_binary("mmseqs")
                                   else "mmseqs2 (ran but failed)")
            if args.tool in ("auto", "cdhit"):
                tools_tried.append("cd-hit (not found on PATH)" if not have_binary("cd-hit")
                                   else "cd-hit (ran but failed)")
            write_missing("Could not produce a cluster assignment.\n\n"
                          "Tools tried:\n" + "\n".join(f"  - {t}" for t in tools_tried))
            return 1

    write_clusters_csv(member_to_cluster, id_to_class)

    n_clusters = len(set(member_to_cluster.values()))
    n_seqs = len(member_to_cluster)
    avg_size = n_seqs / max(1, n_clusters)
    print(f"[clusters] {used_tool}: {n_clusters} clusters, "
          f"{n_seqs} sequences, average cluster size {avg_size:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
