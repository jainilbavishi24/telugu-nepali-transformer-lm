import json
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILES = [
    {
        "name": "wikipedia",
        "path": "Telugu/data/telugu_wikipedia_cleaned.jsonl"
    },
    {
        "name": "sangraha",
        "path": "Telugu/data/telugu_sangraha_cleaned.jsonl"
    },
    {
        "name": "wikisource",
        "path": "Telugu/data/telugu_wikisource_cleaned.jsonl"
    },
    {
        "name": "wiktionary",
        "path": "Telugu/data/telugu_wiktionary_cleaned.jsonl"
    },
    {
        "name": "wikibooks",
        "path": "Telugu/data/telugu_wikibooks_cleaned.jsonl"
    }
]

OUTPUT_FILE = "Telugu/data/telugu_combined_corpus.jsonl"

STATS_FILE = "Telugu/data/merge_corpora_stats.json"

PRINT_EVERY = 10_000


# ============================================================
# PROCESS ONE CORPUS
# ============================================================

def process_corpus(
    corpus_name,
    input_path,
    outfile,
    stats
):

    print("\n" + "=" * 60)
    print(
        f"PROCESSING {corpus_name.upper()}"
    )
    print("=" * 60)

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as infile:

        for line_number, line in enumerate(
            infile,
            start=1
        ):

            stats[
                f"{corpus_name}_total"
            ] += 1

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                stats[
                    f"{corpus_name}_invalid_json"
                ] += 1

                continue

            # ------------------------------------------------
            # Get text
            # ------------------------------------------------

            text = record.get(
                "text",
                ""
            ).strip()

            if not text:

                stats[
                    f"{corpus_name}_empty"
                ] += 1

                continue

            # ------------------------------------------------
            # Standardized ID
            #
            # Prefixing prevents possible ID collisions between
            # datasets.
            # ------------------------------------------------

            original_id = record.get(
                "id",
                line_number
            )

            document_id = (
                f"{corpus_name}_{original_id}"
            )

            # ------------------------------------------------
            # Standardized output record
            # ------------------------------------------------

            merged_record = {

                "id": document_id,

                "title": record.get(
                    "title",
                    ""
                ),

                "text": text,

                "source": corpus_name
            }

            # ------------------------------------------------
            # Preserve optional metadata
            # ------------------------------------------------

            optional_fields = [
                "type",
                "domain",
                "dataset",
                "namespace"
            ]

            for field in optional_fields:

                if field in record:

                    merged_record[field] = (
                        record[field]
                    )

            # ------------------------------------------------
            # Write
            # ------------------------------------------------

            outfile.write(

                json.dumps(
                    merged_record,
                    ensure_ascii=False
                )

                + "\n"
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            stats[
                f"{corpus_name}_kept"
            ] += 1

            stats[
                f"{corpus_name}_characters"
            ] += len(text)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                stats[
                    f"{corpus_name}_total"
                ]
                % PRINT_EVERY
                == 0
            ):

                print(

                    f"Processed: "
                    f"{stats[f'{corpus_name}_total']:,} | "

                    f"Kept: "
                    f"{stats[f'{corpus_name}_kept']:,} | "

                    f"Characters: "
                    f"{stats[f'{corpus_name}_characters']:,}"
                )


# ============================================================
# MAIN MERGE
# ============================================================

def merge_corpora():

    stats = Counter()

    print("=" * 65)
    print("MERGING TELUGU CORPORA")
    print("=" * 65)

    # --------------------------------------------------------
    # Open output
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for corpus in INPUT_FILES:

            process_corpus(

                corpus_name=corpus["name"],

                input_path=corpus["path"],

                outfile=outfile,

                stats=stats
            )

    # ========================================================
    # BUILD FINAL STATISTICS
    # ========================================================

    total_documents = 0
    total_characters = 0

    corpus_stats = {}

    for corpus in INPUT_FILES:

        name = corpus["name"]

        processed = stats[
            f"{name}_total"
        ]

        kept = stats[
            f"{name}_kept"
        ]

        invalid_json = stats[
            f"{name}_invalid_json"
        ]

        empty = stats[
            f"{name}_empty"
        ]

        characters = stats[
            f"{name}_characters"
        ]

        corpus_stats[name] = {

            "input_file": corpus["path"],

            "processed": processed,

            "kept": kept,

            "invalid_json": invalid_json,

            "empty": empty,

            "characters": characters
        }

        total_documents += kept

        total_characters += characters

    output_stats = {

        "input_files": {
            corpus["name"]: corpus["path"]
            for corpus in INPUT_FILES
        },

        "output_file": OUTPUT_FILE,

        "sources": corpus_stats,

        "combined": {

            "documents": total_documents,

            "characters": total_characters
        }
    }

    # ========================================================
    # SAVE STATISTICS
    # ========================================================

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

    print("\n" + "=" * 65)
    print("MERGE COMPLETE")
    print("=" * 65)

    for corpus in INPUT_FILES:

        name = corpus["name"]

        print(f"\n{name.upper()}:")

        print(

            f"Documents : "
            f"{stats[f'{name}_kept']:,}"
        )

        print(

            f"Characters: "
            f"{stats[f'{name}_characters']:,}"
        )

    print("\nCOMBINED:")

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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    merge_corpora()