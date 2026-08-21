"""Acquire DRAMP positives plus length-matched UniProt non-AMP negatives.

Writes ``ml/data/raw/amps.fasta`` and ``ml/data/raw/negatives.fasta``.
Within-class duplicates are dropped before write (DRAMP indexes the same
canonical sequence under multiple entry IDs, and the fragmentation pass
can yield collisions across distinct parent entries).

DRAMP acquisition strategy:
  1. Documented bulk FASTA URLs across DRAMP 3.x and 4.x.
  2. Scrape /downloads/ index for any .fasta or .zip link, prefer "general".

UniProt acquisition uses the REST streaming endpoint. Short negatives are
oversampled 5x then rejection-sampled to match the positives' length
distribution. Long negatives are fragmented into AMP-length pieces to
fill any per-bin shortfall, following Veltri 2018 / Meher 2017.

If DRAMP or UniProt is unreachable the script writes
``ml/data/raw/MISSING.md`` and exits non-zero. Do NOT substitute APD3 or
fall back to a random split.

References:
  Shi G et al. (2022) DRAMP 3.0. Nucleic Acids Research 50, D488-D496.
  Veltri D, Kamath U, Shehu A (2018) Bioinformatics 34(16), 2740-2747.
  Meher PK et al. (2017) Scientific Reports 7, 42362.

DRAMP is CC-BY-NC for academic use; ``docs/data_card.md`` records this.
"""
from __future__ import annotations

import argparse
import io
import random
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
MISSING_PATH = RAW_DIR / "MISSING.md"
AMPS_FASTA = RAW_DIR / "amps.fasta"
NEGATIVES_FASTA = RAW_DIR / "negatives.fasta"

USER_AGENT = "AMPClassifier/0.1 (research; portfolio project)"
TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 4

DRAMP_BULK_URLS = (
    "https://dramp.cpu-bioinfor.org/downloads/download.php?type=general&file_type=fasta",
    "https://dramp.cpu-bioinfor.org/static/downloads/general_amps.fasta",
    "https://dramp.cpu-bioinfor.org/static/downloads/general_amps.zip",
    "https://dramp.cpu-bioinfor.org/static/downloads/General_AMPs.fasta",
    "https://dramp.cpu-bioinfor.org/static/downloads/general_amps_v3.fasta",
    "https://dramp.cpu-bioinfor.org/static/downloads/general_amps_v4.fasta",
    "https://dramp.cpu-bioinfor.org/static/downloads/General_amps.fasta",
    "https://dramp.cpu-bioinfor.org/static/downloads/general.fasta",
    "https://dramp.cpu-bioinfor.org/static/downloads/General.fasta",
)

DRAMP_INDEX_URLS = (
    "https://dramp.cpu-bioinfor.org/downloads/",
    "http://dramp.cpu-bioinfor.org/downloads/",
    "https://dramp.cpu-bioinfor.org/downloads.php",
    "http://dramp.cpu-bioinfor.org/downloads.php",
)

UNIPROT_NEGATIVE_QUERY = (
    "(reviewed:true) AND (length:[5 TO 200]) "
    "NOT (keyword:KW-0929) NOT (keyword:KW-0044) NOT (keyword:KW-0211) "
    "NOT (cc_function:antimicrobial) "
    "NOT (protein_name:cathelicidin) NOT (protein_name:magainin) "
    "NOT (protein_name:cecropin) NOT (protein_name:bacteriocin) "
    "NOT (protein_name:defensin)"
)
UNIPROT_LONG_NEGATIVE_QUERY = (
    "(reviewed:true) AND (length:[201 TO 5000]) "
    "NOT (keyword:KW-0929) NOT (keyword:KW-0044) NOT (keyword:KW-0211) "
    "NOT (keyword:KW-0732) "
    "NOT (cc_function:antimicrobial) "
    "NOT (protein_name:cathelicidin) NOT (protein_name:magainin) "
    "NOT (protein_name:cecropin) NOT (protein_name:bacteriocin) "
    "NOT (protein_name:defensin)"
)
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_PAGE_SIZE = 500
UNIPROT_OVERSAMPLE_MULTIPLIER = 5
UNIPROT_LONG_OVERSAMPLE_MULTIPLIER = 3

AMP_KEYWORD_DENYLIST = (
    "antimicrobial", "antibiotic", "defensin", "cathelicidin",
    "magainin", "cecropin", "bacteriocin", "lantibiotic", "thionin",
)


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/plain"})
    return s


def _retry_get(session, url, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            print(f"  [http] attempt {attempt} for {url} raised {exc!r}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_BACKOFF * attempt)
            continue
        if response.status_code == 200:
            return response
        if response.status_code in (429, 500, 502, 503, 504):
            print(f"  [http] attempt {attempt} for {url} -> HTTP {response.status_code}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                return response
            time.sleep(RETRY_BACKOFF * attempt)
            continue
        return response
    return None


def _try_dramp_url(session, url):
    response = _retry_get(session, url)
    if response is None or response.status_code != 200:
        return None
    payload = response.content
    if url.lower().endswith(".zip") or payload[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                fasta_member = next(
                    (n for n in zf.namelist() if n.lower().endswith(".fasta")), None
                )
                if fasta_member is None:
                    print("  [dramp] zip had no .fasta member")
                    return None
                return zf.read(fasta_member).decode("utf-8", errors="replace")
        except zipfile.BadZipFile:
            print("  [dramp] payload looked like zip but failed to open")
            return None
    text = payload.decode("utf-8", errors="replace")
    if text.lstrip().startswith(">") and text.count(">") >= 100:
        return text
    print(f"  [dramp] {url} returned {len(text)} bytes but did not look like FASTA")
    return None


def fetch_dramp_index_links(session):
    href_pattern = re.compile(r'href=["\']([^"\']+\.(?:fasta|zip))["\']', re.IGNORECASE)
    seen = set()
    ordered = []
    for index_url in DRAMP_INDEX_URLS:
        print(f"[dramp] scraping index {index_url}")
        response = _retry_get(session, index_url)
        if response is None or response.status_code != 200:
            continue
        for raw_href in href_pattern.findall(response.text):
            absolute = urljoin(index_url, raw_href)
            if absolute in seen:
                continue
            seen.add(absolute)
            ordered.append(absolute)

    def sort_key(url):
        path_low = url.lower()
        general_rank = 0 if "general" in path_low else 1
        ext_rank = 0 if path_low.endswith(".fasta") else 1
        return general_rank, ext_rank, path_low

    ordered.sort(key=sort_key)
    return ordered


def fetch_dramp_bulk(session):
    for url in DRAMP_BULK_URLS:
        print(f"[dramp] trying {url}")
        fasta = _try_dramp_url(session, url)
        if fasta is not None:
            return fasta
    index_links = fetch_dramp_index_links(session)
    if not index_links:
        print("[dramp] could not load /downloads/ index either")
        return None
    print(f"[dramp] index scrape found {len(index_links)} candidate link(s)")
    for url in index_links:
        print(f"[dramp] trying (from index) {url}")
        fasta = _try_dramp_url(session, url)
        if fasta is not None:
            return fasta
    return None


def _stream_uniprot(session, query, page_size, want, log_prefix):
    params = {"query": query, "format": "fasta", "size": page_size}
    url = UNIPROT_STREAM_URL
    chunks = []
    fetched = 0
    while url and fetched < want:
        print(f"[{log_prefix}] page request; collected {fetched} so far")
        response = _retry_get(session, url, params=params if chunks == [] else None)
        if response is None or response.status_code != 200:
            return None
        chunks.append(response.text)
        fetched += response.text.count(">")
        link_header = response.headers.get("Link", "")
        next_link = None
        for part in link_header.split(","):
            if 'rel="next"' in part:
                start = part.find("<")
                end = part.find(">")
                if start != -1 and end != -1:
                    next_link = part[start + 1 : end]
                    break
        url = next_link
    return "".join(chunks)


def fetch_uniprot_negatives(session, target_count):
    want = target_count * UNIPROT_OVERSAMPLE_MULTIPLIER
    return _stream_uniprot(session, UNIPROT_NEGATIVE_QUERY, UNIPROT_PAGE_SIZE, want, "uniprot")


def fetch_uniprot_long_negatives(session, target_count):
    want = target_count * UNIPROT_LONG_OVERSAMPLE_MULTIPLIER
    return _stream_uniprot(session, UNIPROT_LONG_NEGATIVE_QUERY, UNIPROT_PAGE_SIZE, want, "uniprot-long")


def parse_fasta(text):
    header = None
    seq_chunks = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_chunks)
            header = line[1:].strip()
            seq_chunks = []
        elif line.strip():
            seq_chunks.append(line.strip())
    if header is not None:
        yield header, "".join(seq_chunks)


def is_canonical(sequence):
    return bool(sequence) and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence)


def header_mentions_amp(header):
    lower = header.lower()
    return any(term in lower for term in AMP_KEYWORD_DENYLIST)


def length_histogram(lengths, bin_width=5):
    hist = {}
    for length in lengths:
        bin_key = length // bin_width
        hist[bin_key] = hist.get(bin_key, 0) + 1
    return hist


def sample_length_matched(candidates, positive_lengths, *, seed=42, bin_width=5, tolerance=0.20):
    rng = random.Random(seed)
    target = length_histogram(positive_lengths, bin_width=bin_width)
    by_bin = {}
    for header, seq in candidates:
        by_bin.setdefault(len(seq) // bin_width, []).append((header, seq))
    sampled = []
    for bin_key, want in target.items():
        pool = by_bin.get(bin_key, [])
        rng.shuffle(pool)
        take = min(len(pool), int(round(want * (1.0 + tolerance))))
        sampled.extend(pool[:take])
    rng.shuffle(sampled)
    return sampled


def fragment_to_fill_shortfall(long_entries, current_negatives, positive_lengths,
                                *, seed=42, bin_width=5, tolerance=0.20):
    """Top up negatives via random fragments of longer SwissProt entries."""
    if not long_entries:
        return []
    rng = random.Random(seed + 1)
    target = length_histogram(positive_lengths, bin_width=bin_width)
    cur_hist = length_histogram([len(s) for _, s in current_negatives], bin_width=bin_width)
    fragments = []
    for bin_key in sorted(target):
        want = int(round(target[bin_key] * (1.0 + tolerance)))
        have = cur_hist.get(bin_key, 0)
        if have >= want:
            continue
        shortfall = want - have
        lo = max(bin_key * bin_width, 5)
        hi = (bin_key + 1) * bin_width - 1
        if hi < lo:
            hi = lo
        attempts = 0
        added = 0
        while added < shortfall and attempts < shortfall * 8:
            attempts += 1
            parent_header, parent_seq = rng.choice(long_entries)
            frag_len = rng.randint(lo, hi)
            if len(parent_seq) < frag_len:
                continue
            start = rng.randrange(0, len(parent_seq) - frag_len + 1)
            frag = parent_seq[start : start + frag_len]
            if not is_canonical(frag):
                continue
            tag = f"{parent_header}|frag_{start}_{frag_len}"
            fragments.append((tag, frag))
            added += 1
    return fragments


def parse_existing_amps_fasta():
    text = AMPS_FASTA.read_text(encoding="utf-8")
    out = []
    for header, seq in parse_fasta(text):
        seq_upper = seq.upper().replace(" ", "")
        if is_canonical(seq_upper) and 5 <= len(seq_upper) <= 200:
            out.append((header, seq_upper))
    return out


def dedupe_by_sequence(records, class_name):
    """Drop duplicate sequences (keep first occurrence). Logs drop count."""
    seen = set()
    out = []
    for header, seq in records:
        if seq in seen:
            continue
        seen.add(seq)
        out.append((header, seq))
    dropped = len(records) - len(out)
    if dropped:
        print(f"[dedupe] {class_name}: dropped {dropped} duplicates "
              f"({len(records)} -> {len(out)})")
    return out


def drop_cross_class_collisions(positives, negatives):
    """Remove any negative whose sequence also appears in positives.

    Real-biology source: some DRAMP entries (defensin precursors,
    cathelicidin fragments) are substrings of longer SwissProt proteins,
    so the fragmentation pass occasionally regenerates them. Keeping the
    positive label is correct since the sequence is known-AMP.
    """
    pos_seqs = {seq for _, seq in positives}
    kept = [(h, s) for h, s in negatives if s not in pos_seqs]
    dropped = len(negatives) - len(kept)
    if dropped:
        print(f"[dedupe] negatives: dropped {dropped} cross-class collisions "
              f"({len(negatives)} -> {len(kept)})")
    return kept


def write_fasta(records, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")
    return len(records)


def _missing_text(reason):
    return (
        "# Raw-data acquisition failed\n\n"
        "download_data.py could not produce amps.fasta and negatives.fasta.\n\n"
        "Failure mode\n\n"
        f"{reason}\n\n"
        "Do not fall back to APD3 or a random split. Handoff section 9 calls\n"
        "for stopping here and asking the maintainer.\n"
    )


def write_missing(reason):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MISSING_PATH.write_text(_missing_text(reason), encoding="utf-8")
    print(f"[download] wrote {MISSING_PATH}")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negatives-only", action="store_true",
                   help="Skip DRAMP fetch and rebuild negatives.fasta only.")
    p.add_argument("--dedupe-existing", action="store_true",
                   help="Read existing FASTAs, dedupe by sequence, re-write in place. "
                        "No network needed.")
    return p.parse_args(argv)


def already_present():
    return (
        AMPS_FASTA.exists()
        and NEGATIVES_FASTA.exists()
        and AMPS_FASTA.stat().st_size > 0
        and NEGATIVES_FASTA.stat().st_size > 0
    )


def _format_dramp_failure_reason():
    docs = "\n".join(f"  - {u}" for u in DRAMP_BULK_URLS)
    idx = "\n".join(f"  - {u}" for u in DRAMP_INDEX_URLS)
    return (
        "DRAMP acquisition failed via documented URLs and index scrape.\n\n"
        f"Documented URLs tried:\n\n{docs}\n\n"
        f"Index pages scraped:\n\n{idx}\n\n"
        "Open https://dramp.cpu-bioinfor.org/downloads/ in a browser, find\n"
        "the general-AMP FASTA link, and update DRAMP_BULK_URLS or save the\n"
        "FASTA manually to ml/data/raw/amps.fasta."
    )


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.dedupe_existing:
        if not (AMPS_FASTA.exists() and NEGATIVES_FASTA.exists()):
            print("[dedupe] --dedupe-existing requires both FASTAs to exist.")
            return 1
        positives = [(h, s.upper()) for h, s in
                     parse_fasta(AMPS_FASTA.read_text(encoding="utf-8"))]
        negatives = [(h, s.upper()) for h, s in
                     parse_fasta(NEGATIVES_FASTA.read_text(encoding="utf-8"))]
        positives = dedupe_by_sequence(positives, "positives")
        negatives = dedupe_by_sequence(negatives, "negatives")
        negatives = drop_cross_class_collisions(positives, negatives)
        write_fasta(positives, AMPS_FASTA)
        write_fasta(negatives, NEGATIVES_FASTA)
        ratio = len(negatives) / max(1, len(positives))
        print(f"[dedupe] final: {len(positives)} positives, {len(negatives)} "
              f"negatives (ratio {ratio:.2f})")
        return 0

    if already_present() and not args.force and not args.negatives_only:
        print("[download] amps.fasta and negatives.fasta already present -- "
              "pass --force to refetch.")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if MISSING_PATH.exists():
        MISSING_PATH.unlink()

    session = _session()

    if args.negatives_only:
        if not AMPS_FASTA.exists() or AMPS_FASTA.stat().st_size == 0:
            print("[download] --negatives-only requires amps.fasta to exist.")
            return 1
        positives = parse_existing_amps_fasta()
        print(f"[dramp] reused {len(positives)} positives from existing amps.fasta")
    else:
        dramp_fasta = fetch_dramp_bulk(session)
        if dramp_fasta is None:
            write_missing(_format_dramp_failure_reason())
            return 1
        positives = []
        for header, seq in parse_fasta(dramp_fasta):
            seq_upper = seq.upper().replace(" ", "")
            if is_canonical(seq_upper) and 5 <= len(seq_upper) <= 200:
                positives.append((header, seq_upper))
        if len(positives) < 1000:
            write_missing(f"DRAMP fetch returned only {len(positives)} canonical entries.")
            return 1
        print(f"[dramp] kept {len(positives)} canonical positives after filter")

    target_neg_count = len(positives)

    uniprot_fasta = fetch_uniprot_negatives(session, target_neg_count)
    if uniprot_fasta is None:
        write_missing("UniProt REST returned no usable response (short pass).")
        return 1
    candidates = []
    for header, seq in parse_fasta(uniprot_fasta):
        seq_upper = seq.upper().replace(" ", "")
        if not is_canonical(seq_upper):
            continue
        if not (5 <= len(seq_upper) <= 200):
            continue
        if header_mentions_amp(header):
            continue
        candidates.append((header, seq_upper))

    positive_lengths = [len(s) for _, s in positives]
    negatives = sample_length_matched(candidates, positive_lengths, seed=args.seed)
    print(f"[uniprot] length-matched (short pass): {len(negatives)} negatives")

    if len(negatives) < int(round(target_neg_count * 0.80)):
        shortfall = target_neg_count - len(negatives)
        print(f"[uniprot-long] short by {shortfall}; fetching long entries")
        long_fasta = fetch_uniprot_long_negatives(session, shortfall)
        if long_fasta is None:
            write_missing(f"UniProt long fetch returned no response (short by {shortfall}).")
            return 1
        long_entries = []
        for header, seq in parse_fasta(long_fasta):
            seq_upper = seq.upper().replace(" ", "")
            if not is_canonical(seq_upper):
                continue
            if len(seq_upper) < 201:
                continue
            if header_mentions_amp(header):
                continue
            long_entries.append((header, seq_upper))
        fragments = fragment_to_fill_shortfall(long_entries, negatives,
                                                positive_lengths, seed=args.seed)
        print(f"[uniprot-long] minted {len(fragments)} fragments")
        negatives.extend(fragments)

    print(f"[uniprot] final negatives count: {len(negatives)}")

    if not args.negatives_only:
        positives = dedupe_by_sequence(positives, "positives")
        amps_count = write_fasta(positives, AMPS_FASTA)
    else:
        amps_count = len(positives)
    negatives = dedupe_by_sequence(negatives, "negatives")
    negatives = drop_cross_class_collisions(positives, negatives)
    negs_count = write_fasta(negatives, NEGATIVES_FASTA)
    print(f"[download] wrote {amps_count} positives to {AMPS_FASTA.name} and "
          f"{negs_count} negatives to {NEGATIVES_FASTA.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
