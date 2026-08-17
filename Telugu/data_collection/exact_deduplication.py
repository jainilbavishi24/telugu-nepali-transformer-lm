import json
import hashlib
import re
import unicodedata
from collections import Counter

INPUT_FILE = "telugu_wikisource_basic_filtered.jsonl"
OUTPUT_FILE = "telugu_wikisource_exact_dedup.jsonl"
STATS_FILE = "telugu_wikisource_exact_dedup_stats.json"


# ============================================================
# NORMALIZATION FOR DEDUPLICATION
# ============================================================

def normalize_for_dedup(text):
    """
    Normalize text before hashing.

    This allows documents that differ only in whitespace or
    Unicode representation to be treated as duplicates.
    """

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Convert all whitespace sequences into a single space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# HASH FUNCTION
# ============================================================

def get_document_hash(text):
    """
    Generate a SHA-256 hash for normalized document text.
    """

    normalized = normalize_for_dedup(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# MAIN DEDUPLICATION
# ============================================================

def deduplicate():

    seen_hashes = set()

    counts = Counter()

    input_characters = 0
    output_characters = 0
    duplicate_characters = 0


    with open(
        INPUT_FILE,
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

            # Generate hash
            doc_hash = get_document_hash(text)

            # ------------------------------------------------
            # Check duplicate
            # ------------------------------------------------

            if doc_hash in seen_hashes:

                counts["duplicates"] += 1
                duplicate_characters += len(text)

                continue

            # ------------------------------------------------
            # New document
            # ------------------------------------------------

            seen_hashes.add(doc_hash)

            counts["kept"] += 1
            output_characters += len(text)

            outfile.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if counts["total"] % 1000 == 0:

                print(
                    f"Processed: {counts['total']:,} | "
                    f"Unique: {counts['kept']:,} | "
                    f"Duplicates: {counts['duplicates']:,}"
                )


    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    stats = {
        "input_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,

        "document_counts": {
            "total": counts["total"],
            "unique": counts["kept"],
            "duplicates_removed": counts["duplicates"],
            "invalid_json": counts["invalid_json"]
        },

        "characters": {
            "input": input_characters,
            "output": output_characters,
            "duplicate_characters_removed": duplicate_characters
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


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("EXACT DEDUPLICATION COMPLETE")
    print("=" * 60)

    print(f"Total documents      : {counts['total']:,}")
    print(f"Unique documents     : {counts['kept']:,}")
    print(f"Duplicates removed   : {counts['duplicates']:,}")

    if counts["total"] > 0:

        duplicate_percentage = (
            counts["duplicates"]
            / counts["total"]
            * 100
        )

        print(
            f"Duplicate percentage : "
            f"{duplicate_percentage:.2f}%"
        )

    print("\nCharacter statistics:")

    print(f"Input characters     : {input_characters:,}")
    print(f"Output characters    : {output_characters:,}")

    print(
        f"Duplicate characters removed: "
        f"{duplicate_characters:,}"
    )

    print(f"\nStats saved to: {STATS_FILE}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    deduplicate()