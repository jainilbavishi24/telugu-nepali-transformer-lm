#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data.nosync"

INPUT_FILES = [
    {"name": "ekantipur", "path": DATA_DIR / "ekantipur.jsonl"},
    {"name": "nepalese_cc", "path": DATA_DIR / "nepalese_cc_corpus.jsonl"},
    {"name": "nepalese_wikipedia", "path": DATA_DIR / "nepalese_wikipedia_clean.jsonl"},
    {"name": "onlinekhabar", "path": DATA_DIR / "onlinekhabar.jsonl"},
    {"name": "ratopati", "path": DATA_DIR / "ratopati.jsonl"},
]

OUTPUT_FILE = DATA_DIR / "nepalese_combined_corpus.jsonl"
STATS_FILE = DATA_DIR / "nepalese_merge_stats.json"


def normalize_source(corpus_name, record):
    existing = (record.get("source") or "").strip()
    if existing:
        return existing

    fallback = {
        "ekantipur": "ekantipur.com",
        "nepalese_cc": "nepalese_cc_corpus",
        "nepalese_wikipedia": "nepalese_wikipedia",
        "onlinekhabar": "onlinekhabar.com",
        "ratopati": "ratopati.com",
    }

    return fallback.get(corpus_name, corpus_name)


def normalize_record(corpus_name, line_number, record):
    text = (record.get("text") or "").strip()
    if not text:
        return None

    original_id = record.get("id", line_number)
    merged = {
        "id": f"{corpus_name}_{original_id}",
        "title": record.get("title", ""),
        "text": text,
        "source": normalize_source(corpus_name, record),
    }

    for field in [
        "author",
        "date",
        "language",
        "num_chars",
        "devanagari_ratio",
        "namespace",
        "nepali_word_ratio",
        "nepali_pattern_score",
        "dataset",
        "domain",
        "type",
    ]:
        if field in record and field != "source":
            merged[field] = record[field]

    return merged


def merge_files():
    stats = Counter()

    with OUTPUT_FILE.open("w", encoding="utf-8") as outfile:
        for config in INPUT_FILES:
            corpus_name = config["name"]
            input_path = config["path"]

            print(f"\nProcessing {corpus_name}: {input_path.name}")
            stats[f"{corpus_name}_files_seen"] += 1

            with input_path.open("r", encoding="utf-8") as infile:
                for line_number, raw_line in enumerate(infile, start=1):
                    stats[f"{corpus_name}_lines_total"] += 1
                    raw_line = raw_line.strip()
                    if not raw_line:
                        stats[f"{corpus_name}_empty_lines"] += 1
                        continue

                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        stats[f"{corpus_name}_invalid_json"] += 1
                        continue

                    normalized = normalize_record(corpus_name, line_number, record)
                    if normalized is None:
                        stats[f"{corpus_name}_empty_text"] += 1
                        continue

                    outfile.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    stats[f"{corpus_name}_kept"] += 1
                    stats["total_kept"] += 1

    stats["output_file"] = str(OUTPUT_FILE)
    STATS_FILE.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nMerge complete.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Stats: {STATS_FILE}")
    print(f"Total kept records: {stats['total_kept']}")


if __name__ == "__main__":
    merge_files()
