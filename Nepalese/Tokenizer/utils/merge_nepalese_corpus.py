#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data.nosync"

APPEND_SOURCE_FILES = [
    {"name": "gorkhapatra", "path": DATA_DIR / "gorkhapatra.jsonl"},
    {"name": "sangraha", "path": DATA_DIR / "nepalese_sangraha_raw.jsonl"},
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
        "gorkhapatra": "gorkhapatraonline.com",
        "sangraha": "ai4bharat_sangraha_verified_tel",
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


def append_new_sources():
    stats = Counter()
    existing_ids = set()

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as existing_file:
            for line in existing_file:
                if not line.strip():
                    continue
                try:
                    existing_ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue

    with OUTPUT_FILE.open("a", encoding="utf-8") as outfile:
        for config in APPEND_SOURCE_FILES:
            corpus_name = config["name"]
            input_path = config["path"]

            print(f"\nAppending {corpus_name}: {input_path.name}")
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

                    if normalized["id"] in existing_ids:
                        stats[f"{corpus_name}_duplicates_skipped"] += 1
                        continue

                    outfile.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    existing_ids.add(normalized["id"])
                    stats[f"{corpus_name}_kept"] += 1
                    stats["total_kept"] += 1

    stats["output_file"] = str(OUTPUT_FILE)
    STATS_FILE.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nAppend complete.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Stats: {STATS_FILE}")

    for config in APPEND_SOURCE_FILES:
        corpus_name = config["name"]
        added = stats.get(f"{corpus_name}_kept", 0)
        skipped = stats.get(f"{corpus_name}_duplicates_skipped", 0)
        print(f"{corpus_name}: appended={added}, skipped={skipped}")

    print(f"New rows added in this run: {stats['total_kept']} (not total corpus size)")


if __name__ == "__main__":
    append_new_sources()
