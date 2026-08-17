import json
import hashlib
import re
import unicodedata
from collections import Counter


# ============================================================
# FILES
# ============================================================

WIKIPEDIA_FILE = "telugu_wikipedia_quality_filtered.jsonl"

SANGRAHA_FILE = (
    "sangraha_verified_tel_sample_quality_filtered.jsonl"
)

OUTPUT_FILE = (
    "sangraha_verified_tel_cross_dedup.jsonl"
)

STATS_FILE = "cross_dataset_exact_dedup_stats.json"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_for_dedup(text):
    """
    Normalize text before hashing.

    Documents differing only in Unicode representation or
    whitespace are treated as exact duplicates.
    """

    text = unicodedata.normalize("NFC", text)

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
# LOAD WIKIPEDIA HASHES
# ============================================================

def load_wikipedia_hashes():

    print("=" * 60)
    print("LOADING WIKIPEDIA HASHES")
    print("=" * 60)

    wikipedia_hashes = set()

    counts = Counter()

    with open(
        WIKIPEDIA_FILE,
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

            wikipedia_hashes.add(doc_hash)

            counts["loaded"] += 1

            if counts["total"] % 10_000 == 0:

                print(
                    f"Processed: {counts['total']:,} | "
                    f"Hashes: {len(wikipedia_hashes):,}"
                )

    print()

    print(
        f"Wikipedia documents processed: "
        f"{counts['total']:,}"
    )

    print(
        f"Unique Wikipedia hashes: "
        f"{len(wikipedia_hashes):,}"
    )

    return wikipedia_hashes, counts


# ============================================================
# DEDUPLICATE SANGRAHA AGAINST WIKIPEDIA
# ============================================================

def deduplicate_sangraha(wikipedia_hashes):

    print("\n" + "=" * 60)
    print("CROSS-DATASET EXACT DEDUPLICATION")
    print("=" * 60)

    counts = Counter()

    input_characters = 0
    output_characters = 0
    overlap_characters = 0

    with open(
        SANGRAHA_FILE,
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

            input_characters += len(text)

            if not text or not text.strip():

                counts["empty"] += 1
                continue

            doc_hash = get_document_hash(text)

            # ------------------------------------------------
            # Check overlap with Wikipedia
            # ------------------------------------------------

            if doc_hash in wikipedia_hashes:

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
    # --------------------------------------------------------

    wikipedia_hashes, wiki_counts = (
        load_wikipedia_hashes()
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    (
        sangraha_counts,
        input_characters,
        output_characters,
        overlap_characters
    ) = deduplicate_sangraha(
        wikipedia_hashes
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total = sangraha_counts["total"]
    overlap = sangraha_counts["overlap"]
    kept = sangraha_counts["kept"]

    overlap_percentage = (
        overlap / total * 100
        if total > 0
        else 0
    )

    stats = {
        "wikipedia_file": WIKIPEDIA_FILE,
        "sangraha_file": SANGRAHA_FILE,
        "output_file": OUTPUT_FILE,

        "wikipedia": {
            "documents_processed": wiki_counts["total"],
            "unique_hashes": len(wikipedia_hashes),
            "invalid_json": wiki_counts["invalid_json"],
            "empty": wiki_counts["empty"]
        },

        "sangraha": {
            "documents_processed": total,
            "documents_kept": kept,
            "exact_overlaps_removed": overlap,
            "invalid_json": sangraha_counts["invalid_json"],
            "empty": sangraha_counts["empty"],
            "overlap_percentage": overlap_percentage
        },

        "characters": {
            "input": input_characters,
            "output": output_characters,
            "overlap_characters_removed": overlap_characters
        }
    }

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

    print("\nWikipedia:")

    print(
        f"Documents processed : "
        f"{wiki_counts['total']:,}"
    )

    print(
        f"Unique hashes       : "
        f"{len(wikipedia_hashes):,}"
    )

    print("\nSangraha:")

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