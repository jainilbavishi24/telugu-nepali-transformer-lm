#!/usr/bin/env python3
"""
scrape_nepali_cc.py  (parallel version)

Scrapes Nepali-language text from Common Crawl WET files and stores
matching documents as JSONL. Processes multiple WET files concurrently
using a process pool, since the work is a mix of network I/O (download)
and CPU (parsing + language ID), and each WET file is independent.

Pipeline (per WET file, run in a worker process):
  1. Download the WET file from https://data.commoncrawl.org/
  2. Parse records with warcio
  3. Filter by:
       - Devanagari-script character ratio >= threshold (cheap pre-filter)
       - fastText language ID == 'ne' with confidence >= threshold
       - minimum document length
  4. Return matching docs to the main process, which writes them to JSONL
     and updates the checkpoint file

Checkpointing: still file-level. The main process only writes a WET path
to the checkpoint file after that file's docs have been written to disk,
so re-running is safe and resumable, including mid-crawl interruptions.

Install deps:
    pip install warcio requests fasttext tqdm numpy

Download the fastText language ID model once:
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    # or, smaller/faster (recommended for this use case):
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

Usage:
    python scrape_nepali_cc.py \
        --crawl CC-MAIN-2024-30 \
        --lid-model lid.176.bin \
        --out nepali_corpus.jsonl \
        --max-files 200 \
        --workers 8

Run again with a different --crawl to accumulate more data across
multiple monthly snapshots (recommended for low-resource languages).
"""

import argparse
import gzip
import io
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import requests
from warcio.archiveiterator import ArchiveIterator

# --- Compatibility shim for fasttext + NumPy 2.x ---
_original_np_array = np.array
def _patched_np_array(*args, **kwargs):
    if kwargs.get("copy", None) is False:
        kwargs.pop("copy")
        return np.asarray(*args, **kwargs)
    return _original_np_array(*args, **kwargs)
np.array = _patched_np_array
# --- end shim ---

try:
    import fasttext
except ImportError:
    fasttext = None

CC_DATA_ROOT = "https://data.commoncrawl.org/"
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Zero-width joiner/non-joiner and other invisible formatting chars that
# frequently leak into scraped Devanagari text from font/CMS quirks.
# These aren't meaningful content and just add noise to dedup/tokenization.
ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")

# Collapse 3+ repeated whitespace/newlines down (boilerplate scraping often
# leaves long runs of blank lines between nav/footer fragments).
MULTI_BLANK_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_text(text: str) -> str:
    """
    Normalize scraped text before filtering/storage:
      - Unicode NFC normalization (canonical composition) so visually
        identical Devanagari sequences collapse to the same codepoints
      - Strip zero-width joiners/non-joiners and BOM stragglers
      - Collapse excess whitespace left over from HTML boilerplate

    NOTE: this does NOT fix legacy non-Unicode font encodings (e.g. Preeti/
    Kantipur-encoded Nepali text mislabeled as UTF-8). Those show up as
    garbled Latin-range characters, not valid Devanagari, and are filtered
    out downstream by the Devanagari-ratio check rather than "fixed" here.
    """
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()

# Each worker process loads its own copy of the fastText model once,
# on first use, instead of once per document. Avoided passing the loaded
# model object between processes since fastText models aren't reliably
# picklable/shareable across process boundaries.
_worker_model = None
_worker_model_path = None


def _get_worker_model(model_path: str):
    global _worker_model, _worker_model_path
    if _worker_model is None or _worker_model_path != model_path:
        _worker_model = fasttext.load_model(model_path)
        _worker_model_path = model_path
    return _worker_model


def get_wet_paths(crawl_id: str) -> list[str]:
    """Download and decompress the wet.paths.gz index for a crawl."""
    url = f"{CC_DATA_ROOT}crawl-data/{crawl_id}/wet.paths.gz"
    print(f"[+] Fetching WET path list: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8")
    paths = [line.strip() for line in text.splitlines() if line.strip()]
    print(f"[+] Found {len(paths)} WET files in {crawl_id}")
    return paths


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for c in letters if DEVANAGARI_RE.match(c))
    return dev / len(letters)


def load_checkpoint(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_checkpoint(path: str, wet_path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(wet_path + "\n")
        f.flush()


def download_wet(wet_url: str, session: requests.Session, retries: int = 3) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(wet_url, timeout=180)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            print(f"    [!] attempt {attempt}/{retries} failed for {wet_url}: {e}")
            time.sleep(2 * attempt)
    return None


def process_one_wet_file(args) -> dict:
    """
    Worker function: download + parse + filter a single WET file.
    Runs in a separate process. Returns a dict with results and stats
    instead of writing to disk directly, so the main process can own
    all file writes (avoids concurrent write issues).
    """
    (wet_path, lid_model_path, min_chars, lang_conf_threshold,
     devanagari_threshold, crawl_id) = args

    if fasttext is None:
        return {"wet_path": wet_path, "docs": [], "error": "fasttext not installed"}

    model = _get_worker_model(lid_model_path)
    wet_url = CC_DATA_ROOT + wet_path

    session = requests.Session()
    content = download_wet(wet_url, session)
    if content is None:
        return {"wet_path": wet_path, "docs": [], "error": "download failed"}

    docs = []
    try:
        stream = io.BytesIO(content)
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            try:
                text = record.content_stream().read().decode("utf-8", errors="ignore")
            except Exception:
                continue

            text = normalize_text(text)
            if len(text) < min_chars:
                continue

            dev_ratio = devanagari_ratio(text)
            if dev_ratio < devanagari_threshold:
                continue

            lid_input = " ".join(text.split())[:2000]
            pred_labels, pred_probs = model.predict(lid_input, k=1)
            label = pred_labels[0].replace("__label__", "")
            confidence = float(pred_probs[0])

            if label != "ne" or confidence < lang_conf_threshold:
                continue

            url = record.rec_headers.get_header("WARC-Target-URI")
            date = record.rec_headers.get_header("WARC-Date")

            docs.append({
                "url": url,
                "date": date,
                "crawl": crawl_id,
                "source_wet": wet_path,
                "lang": label,
                "lang_confidence": round(confidence, 4),
                "devanagari_ratio": round(dev_ratio, 4),
                "char_count": len(text),
                "text": text,
            })
    except Exception as e:
        return {"wet_path": wet_path, "docs": docs, "error": f"parse error: {e}"}

    return {"wet_path": wet_path, "docs": docs, "error": None}


def process_crawl(
    crawl_id: str,
    lid_model_path: str,
    out_path: str,
    checkpoint_path: str,
    max_files: int,
    min_chars: int,
    lang_conf_threshold: float,
    devanagari_threshold: float,
    workers: int,
):
    if fasttext is None:
        sys.exit("fasttext not installed. Run: pip install fasttext")
    if not os.path.exists(lid_model_path):
        sys.exit(f"LID model not found at {lid_model_path}")

    wet_paths = get_wet_paths(crawl_id)
    done = load_checkpoint(checkpoint_path)
    remaining = [p for p in wet_paths if p not in done]
    print(f"[+] {len(done)} files already processed, {len(remaining)} remaining")

    if max_files:
        remaining = remaining[:max_files]
    if not remaining:
        print("[+] Nothing to do.")
        return

    total_docs = 0
    total_chars = 0
    completed = 0

    tasks = [
        (wet_path, lid_model_path, min_chars, lang_conf_threshold, devanagari_threshold, crawl_id)
        for wet_path in remaining
    ]

    print(f"[+] Processing {len(tasks)} WET files with {workers} parallel workers")

    with open(out_path, "a", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one_wet_file, t): t[0] for t in tasks}

            for future in as_completed(futures):
                wet_path = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[{completed}/{len(tasks)}] {wet_path} -> worker crashed: {e}")
                    continue

                if result["error"]:
                    print(f"[{completed}/{len(tasks)}] {wet_path} -> ERROR: {result['error']} "
                          f"({len(result['docs'])} docs recovered before error)")
                else:
                    print(f"[{completed}/{len(tasks)}] {wet_path} -> {len(result['docs'])} Nepali docs")

                for doc in result["docs"]:
                    out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total_docs += 1
                    total_chars += doc["char_count"]
                out_f.flush()

                # Only checkpoint on clean success so a failed file gets retried next run
                if not result["error"]:
                    append_checkpoint(checkpoint_path, wet_path)

                if completed % 5 == 0 or completed == len(tasks):
                    print(f"    -- progress: {completed}/{len(tasks)} files, "
                          f"{total_docs} docs, ~{total_chars // 5:,} est. tokens so far --")

    print(f"\n[+] Done. {total_docs} documents written to {out_path}")
    print(f"[+] Approx tokens (chars/5 heuristic): {total_chars // 5:,}")
    print("    NOTE: run a real tokenizer on a sample to get an accurate count.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crawl", required=True, help="Crawl ID, e.g. CC-MAIN-2024-30")
    ap.add_argument("--lid-model", required=True, help="Path to fastText lid.176.bin or lid.176.ftz")
    ap.add_argument("--out", default="nepali_corpus.jsonl", help="Output JSONL path")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint file (default: <crawl>.checkpoint.txt)")
    ap.add_argument("--max-files", type=int, default=50, help="Max WET files to process this run (0 = all)")
    ap.add_argument("--min-chars", type=int, default=200, help="Minimum document length in characters")
    ap.add_argument("--lang-conf", type=float, default=0.7, help="Min fastText confidence for 'ne'")
    ap.add_argument("--devanagari-ratio", type=float, default=0.5, help="Min Devanagari char ratio")
    ap.add_argument("--workers", type=int, default=8,
                     help="Number of parallel worker processes (default: 8). "
                          "This workload is mostly network-bound, so you can often "
                          "safely set this higher than your CPU core count.")
    args = ap.parse_args()

    checkpoint_path = args.checkpoint or f"{args.crawl}.checkpoint.txt"

    process_crawl(
        crawl_id=args.crawl,
        lid_model_path=args.lid_model,
        out_path=args.out,
        checkpoint_path=checkpoint_path,
        max_files=args.max_files,
        min_chars=args.min_chars,
        lang_conf_threshold=args.lang_conf,
        devanagari_threshold=args.devanagari_ratio,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()