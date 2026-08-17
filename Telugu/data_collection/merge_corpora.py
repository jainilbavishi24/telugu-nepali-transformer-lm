import json
from collections import Counter


# ============================================================
# INPUT / OUTPUT FILES
# ============================================================

WIKIPEDIA_FILE = "telugu_combined_corpus_v3.jsonl"

SANGRAHA_FILE = (
    "sangraha_verified_tel_500k_cross_dedup.jsonl"
)

OUTPUT_FILE = "telugu_combined_corpus_v4.jsonl"

STATS_FILE = "merge_corpora_stats_v4.json"


# ============================================================
# COPY WIKIPEDIA
# ============================================================

def process_wikipedia(outfile, stats):

    print("\nProcessing Wikipedia...")

    with open(
        WIKIPEDIA_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for line_number, line in enumerate(infile, start=1):

            stats["wikipedia_total"] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                stats["wikipedia_invalid_json"] += 1
                continue

            text = record.get("text", "").strip()

            if not text:
                stats["wikipedia_empty"] += 1
                continue

            # ------------------------------------------------
            # Standardized output record
            # ------------------------------------------------

            merged_record = {
                "id": record.get(
                    "id",
                    f"wiki_{line_number}"
                ),

                "title": record.get(
                    "title",
                    ""
                ),

                "text": text,

                "source": record.get(
                    "source",
                    "unknown"
                )
            }

            outfile.write(
                json.dumps(
                    merged_record,
                    ensure_ascii=False
                ) + "\n"
            )

            stats["wikipedia_kept"] += 1
            stats["wikipedia_characters"] += len(text)

            if stats["wikipedia_total"] % 10_000 == 0:

                print(
                    f"Wikipedia processed: "
                    f"{stats['wikipedia_total']:,}"
                )


# ============================================================
# COPY SANGRAHA
# ============================================================

def process_sangraha(outfile, stats):

    print("\nProcessing Sangraha...")

    with open(
        SANGRAHA_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for line_number, line in enumerate(infile, start=1):

            stats["sangraha_total"] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                stats["sangraha_invalid_json"] += 1
                continue

            text = record.get("text", "").strip()

            if not text:
                stats["sangraha_empty"] += 1
                continue

            # ------------------------------------------------
            # Preserve useful original metadata if present.
            # ------------------------------------------------

            merged_record = {
                "id": record.get(
                    "id",
                    f"sangraha_{line_number}"
                ),

                "title": record.get(
                    "title",
                    ""
                ),

                "text": text,

                "source": "sangraha"
            }

            # Optional metadata
            #
            # Keep these if they exist in the original dataset.

            if "type" in record:

                merged_record["type"] = record["type"]

            if "domain" in record:

                merged_record["domain"] = record["domain"]

            if "dataset" in record:

                merged_record["dataset"] = record["dataset"]

            outfile.write(
                json.dumps(
                    merged_record,
                    ensure_ascii=False
                ) + "\n"
            )

            stats["sangraha_kept"] += 1
            stats["sangraha_characters"] += len(text)

            if stats["sangraha_total"] % 10_000 == 0:

                print(
                    f"Sangraha processed: "
                    f"{stats['sangraha_total']:,}"
                )


# ============================================================
# MAIN
# ============================================================

def merge_corpora():

    stats = Counter()

    print("=" * 60)
    print("MERGING TELUGU CORPORA")
    print("=" * 60)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        # ----------------------------------------------------
        # Wikipedia
        # ----------------------------------------------------

        process_wikipedia(
            outfile,
            stats
        )

        # ----------------------------------------------------
        # Sangraha
        # ----------------------------------------------------

        process_sangraha(
            outfile,
            stats
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    total_documents = (
        stats["wikipedia_kept"]
        + stats["sangraha_kept"]
    )

    total_characters = (
        stats["wikipedia_characters"]
        + stats["sangraha_characters"]
    )

    output_stats = {

        "input_files": {
            "wikipedia": WIKIPEDIA_FILE,
            "sangraha": SANGRAHA_FILE
        },

        "output_file": OUTPUT_FILE,

        "wikipedia": {
            "processed": stats["wikipedia_total"],
            "kept": stats["wikipedia_kept"],
            "invalid_json": (
                stats["wikipedia_invalid_json"]
            ),
            "empty": stats["wikipedia_empty"],
            "characters": (
                stats["wikipedia_characters"]
            )
        },

        "sangraha": {
            "processed": stats["sangraha_total"],
            "kept": stats["sangraha_kept"],
            "invalid_json": (
                stats["sangraha_invalid_json"]
            ),
            "empty": stats["sangraha_empty"],
            "characters": (
                stats["sangraha_characters"]
            )
        },

        "combined": {
            "documents": total_documents,
            "characters": total_characters
        }
    }

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_stats,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    print("\nWikipedia:")

    print(
        f"Documents : "
        f"{stats['wikipedia_kept']:,}"
    )

    print(
        f"Characters: "
        f"{stats['wikipedia_characters']:,}"
    )

    print("\nSangraha:")

    print(
        f"Documents : "
        f"{stats['sangraha_kept']:,}"
    )

    print(
        f"Characters: "
        f"{stats['sangraha_characters']:,}"
    )

    print("\nCombined:")

    print(
        f"Documents : "
        f"{total_documents:,}"
    )

    print(
        f"Characters: "
        f"{total_characters:,}"
    )

    print(
        f"\nMerged corpus saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Statistics saved to: "
        f"{STATS_FILE}"
    )


if __name__ == "__main__":
    merge_corpora()