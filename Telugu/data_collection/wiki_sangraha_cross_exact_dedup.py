import json
import hashlib
import re
import unicodedata
from collections import Counter


# ============================================================
# FILES
# ============================================================

# Existing corpus:
# Wikipedia + Sangraha
COMBINED_CORPUS_FILE = "telugu_combined_corpus_v2.jsonl"

# New corpus to compare against the existing corpus
WIKISOURCE_FILE = "telugu_wikitionary_quality_filtered.jsonl"

# Wikisource after removing exact overlaps
OUTPUT_FILE = "telugu_wikitionary_cross_dedup.jsonl"

STATS_FILE = "wikitionary_cross_dataset_exact_dedup_stats.json"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_for_dedup(text):
    """
    Normalize text before hashing.

    Documents differing only in Unicode representation or
    whitespace are treated as exact duplicates.
    """

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize all whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# HASH
# ============================================================

def get_document_hash(text):

    normalized = normalize_for_dedup(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# LOAD EXISTING CORPUS HASHES
# ============================================================

def load_existing_hashes():

    print("=" * 60)
    print("LOADING EXISTING CORPUS HASHES")
    print("=" * 60)

    existing_hashes = set()

    counts = Counter()

    with open(
        COMBINED_CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for line in infile:

            counts["total"] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                counts["invalid_json"] += 1
                continue

            text = record.get("text", "")

            if not text or not text.strip():

                counts["empty"] += 1
                continue

            doc_hash = get_document_hash(text)

            existing_hashes.add(doc_hash)

            counts["loaded"] += 1

            if counts["total"] % 10_000 == 0:

                print(
                    f"Processed: {counts['total']:,} | "
                    f"Unique hashes: {len(existing_hashes):,}"
                )

    print()

    print(
        f"Existing corpus documents processed: "
        f"{counts['total']:,}"
    )

    print(
        f"Unique document hashes: "
        f"{len(existing_hashes):,}"
    )

    return existing_hashes, counts


# ============================================================
# DEDUPLICATE WIKISOURCE AGAINST EXISTING CORPUS
# ============================================================

def deduplicate_wikisource(existing_hashes):

    print("\n" + "=" * 60)
    print("CROSS-DATASET EXACT DEDUPLICATION")
    print("=" * 60)

    counts = Counter()

    input_characters = 0
    output_characters = 0
    overlap_characters = 0

    with open(
        WIKISOURCE_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            counts["total"] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                counts["invalid_json"] += 1
                continue

            text = record.get("text", "")

            if not text or not text.strip():

                counts["empty"] += 1
                continue

            input_characters += len(text)

            doc_hash = get_document_hash(text)

            # ------------------------------------------------
            # Check overlap with existing corpus
            # ------------------------------------------------

            if doc_hash in existing_hashes:

                counts["overlap"] += 1
                overlap_characters += len(text)

                continue

            # ------------------------------------------------
            # Keep non-overlapping document
            # ------------------------------------------------

            counts["kept"] += 1
            output_characters += len(text)

            outfile.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if counts["total"] % 10_000 == 0:

                overlap_percentage = (
                    counts["overlap"]
                    / counts["total"]
                    * 100
                )

                print(
                    f"Processed: {counts['total']:,} | "
                    f"Kept: {counts['kept']:,} | "
                    f"Overlaps: {counts['overlap']:,} | "
                    f"Overlap: {overlap_percentage:.4f}%"
                )

    return (
        counts,
        input_characters,
        output_characters,
        overlap_characters
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # Load hashes from existing corpus
    # --------------------------------------------------------

    existing_hashes, existing_counts = (
        load_existing_hashes()
    )

    # --------------------------------------------------------
    # STEP 2
    # Deduplicate Wikisource
    # --------------------------------------------------------

    (
        wikisource_counts,
        input_characters,
        output_characters,
        overlap_characters
    ) = deduplicate_wikisource(
        existing_hashes
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total = wikisource_counts["total"]
    overlap = wikisource_counts["overlap"]
    kept = wikisource_counts["kept"]

    overlap_percentage = (
        overlap / total * 100
        if total > 0
        else 0
    )

    stats = {

        "existing_corpus_file": COMBINED_CORPUS_FILE,

        "wikisource_file": WIKISOURCE_FILE,

        "output_file": OUTPUT_FILE,

        "existing_corpus": {

            "documents_processed":
                existing_counts["total"],

            "unique_hashes":
                len(existing_hashes),

            "invalid_json":
                existing_counts["invalid_json"],

            "empty":
                existing_counts["empty"]
        },

        "wikisource": {

            "documents_processed":
                total,

            "documents_kept":
                kept,

            "exact_overlaps_removed":
                overlap,

            "invalid_json":
                wikisource_counts["invalid_json"],

            "empty":
                wikisource_counts["empty"],

            "overlap_percentage":
                overlap_percentage
        },

        "characters": {

            "input":
                input_characters,

            "output":
                output_characters,

            "overlap_characters_removed":
                overlap_characters
        }
    }


    # --------------------------------------------------------
    # SAVE STATISTICS
    # --------------------------------------------------------

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=4
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-DATASET EXACT DEDUPLICATION COMPLETE")
    print("=" * 60)

    print("\nExisting corpus:")

    print(
        f"Documents processed : "
        f"{existing_counts['total']:,}"
    )

    print(
        f"Unique hashes       : "
        f"{len(existing_hashes):,}"
    )


    print("\nWikisource:")

    print(
        f"Documents processed : "
        f"{total:,}"
    )

    print(
        f"Documents kept      : "
        f"{kept:,}"
    )

    print(
        f"Exact overlaps      : "
        f"{overlap:,}"
    )

    print(
        f"Overlap percentage  : "
        f"{overlap_percentage:.4f}%"
    )


    print("\nCharacter statistics:")

    print(
        f"Input characters    : "
        f"{input_characters:,}"
    )

    print(
        f"Output characters   : "
        f"{output_characters:,}"
    )

    print(
        f"Overlap characters removed: "
        f"{overlap_characters:,}"
    )


    print(
        f"\nOutput saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Statistics saved to: "
        f"{STATS_FILE}"
    )


if __name__ == "__main__":
    main()