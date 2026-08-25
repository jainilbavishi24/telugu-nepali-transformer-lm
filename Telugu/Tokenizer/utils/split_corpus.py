import json
import hashlib
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Telugu/data/telugu_combined_corpus_clean.jsonl"

TRAIN_FILE = "Telugu/data/telugu_train.jsonl"
VALID_FILE = "Telugu/data/telugu_validation.jsonl"
TEST_FILE = "Telugu/data/telugu_test.jsonl"

STATS_FILE = "corpus_split_stats.json"


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.98
VALID_RATIO = 0.01
TEST_RATIO = 0.01


# ============================================================
# RANDOM SEED
# ============================================================

SEED = "telugu_corpus_split_v1"


# ============================================================
# VALIDATE RATIOS
# ============================================================

if abs(
    TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0
) > 1e-9:

    raise ValueError(
        "Train, validation, and test ratios must sum to 1.0"
    )


# ============================================================
# DETERMINISTIC HASH
# ============================================================

def get_split_value(record, line_number):

    doc_id = record.get("id")
    source = record.get(
        "source",
        "unknown"
    )

    if doc_id is None:

        text = record.get("text", "")

        key = (
            f"{SEED}|"
            f"{source}|"
            f"{line_number}|"
            f"{text}"
        )

    else:

        key = (
            f"{SEED}|"
            f"{source}|"
            f"{doc_id}"
        )

    hash_value = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()

    integer_value = int(
        hash_value[:16],
        16
    )

    return integer_value / (16 ** 16)


# ============================================================
# MAIN SPLITTING FUNCTION
# ============================================================

def split_corpus():

    print("=" * 65)
    print("SPLITTING TELUGU CORPUS")
    print("=" * 65)

    print(f"\nInput file : {INPUT_FILE}")

    print("\nSplit ratios:")

    print(
        f"Train      : {TRAIN_RATIO * 100:.2f}%"
    )

    print(
        f"Validation : {VALID_RATIO * 100:.2f}%"
    )

    print(
        f"Test       : {TEST_RATIO * 100:.2f}%"
    )

    stats = Counter()

    character_stats = Counter()

    # --------------------------------------------------------
    # Open files
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        TRAIN_FILE,
        "w",
        encoding="utf-8"
    ) as train_file, open(
        VALID_FILE,
        "w",
        encoding="utf-8"
    ) as valid_file, open(
        TEST_FILE,
        "w",
        encoding="utf-8"
    ) as test_file:

        # ----------------------------------------------------
        # Process documents
        # ----------------------------------------------------

        for line_number, line in enumerate(
            infile,
            start=1
        ):

            stats["total"] += 1

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                stats["invalid_json"] += 1
                continue

            text = record.get("text", "")

            if not text or not text.strip():

                stats["empty"] += 1
                continue

            # ------------------------------------------------
            # Get deterministic random value
            # ------------------------------------------------

            split_value = get_split_value(
                record,
                line_number
            )

            # ------------------------------------------------
            # Assign split
            # ------------------------------------------------

            if split_value < TRAIN_RATIO:

                train_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                stats["train"] += 1

                character_stats["train"] += len(text)

            elif split_value < (
                TRAIN_RATIO + VALID_RATIO
            ):

                valid_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                stats["validation"] += 1

                character_stats[
                    "validation"
                ] += len(text)

            else:

                test_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                stats["test"] += 1

                character_stats["test"] += len(text)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if stats["total"] % 10_000 == 0:

                print(
                    f"Processed: {stats['total']:,} | "
                    f"Train: {stats['train']:,} | "
                    f"Validation: "
                    f"{stats['validation']:,} | "
                    f"Test: {stats['test']:,}"
                )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    valid_documents = (
        stats["train"]
        + stats["validation"]
        + stats["test"]
    )

    train_percentage = (
        stats["train"]
        / valid_documents
        * 100
        if valid_documents > 0
        else 0
    )

    validation_percentage = (
        stats["validation"]
        / valid_documents
        * 100
        if valid_documents > 0
        else 0
    )

    test_percentage = (
        stats["test"]
        / valid_documents
        * 100
        if valid_documents > 0
        else 0
    )

    output_stats = {

        "input_file": INPUT_FILE,

        "split_ratios": {

            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO
        },

        "seed": SEED,

        "documents": {

            "total_processed": stats["total"],

            "invalid_json": (
                stats["invalid_json"]
            ),

            "empty": stats["empty"],

            "valid_documents": (
                valid_documents
            ),

            "train": stats["train"],

            "validation": (
                stats["validation"]
            ),

            "test": stats["test"]
        },

        "actual_percentages": {

            "train": train_percentage,

            "validation": (
                validation_percentage
            ),

            "test": test_percentage
        },

        "characters": {

            "train": (
                character_stats["train"]
            ),

            "validation": (
                character_stats["validation"]
            ),

            "test": (
                character_stats["test"]
            ),

            "total": (
                character_stats["train"]
                + character_stats["validation"]
                + character_stats["test"]
            )
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

    print("\n" + "=" * 65)
    print("CORPUS SPLITTING COMPLETE")
    print("=" * 65)

    print("\nDocuments:")

    print(
        f"Total processed : "
        f"{stats['total']:,}"
    )

    print(
        f"Invalid JSON    : "
        f"{stats['invalid_json']:,}"
    )

    print(
        f"Empty documents : "
        f"{stats['empty']:,}"
    )

    print(
        f"Valid documents : "
        f"{valid_documents:,}"
    )

    print("\nTrain:")

    print(
        f"Documents : {stats['train']:,}"
    )

    print(
        f"Percentage: {train_percentage:.2f}%"
    )

    print(
        f"Characters: "
        f"{character_stats['train']:,}"
    )

    print("\nValidation:")

    print(
        f"Documents : "
        f"{stats['validation']:,}"
    )

    print(
        f"Percentage: "
        f"{validation_percentage:.2f}%"
    )

    print(
        f"Characters: "
        f"{character_stats['validation']:,}"
    )

    print("\nTest:")

    print(
        f"Documents : "
        f"{stats['test']:,}"
    )

    print(
        f"Percentage: {test_percentage:.2f}%"
    )

    print(
        f"Characters: "
        f"{character_stats['test']:,}"
    )

    print("\nOutput files:")

    print(f"Train      : {TRAIN_FILE}")
    print(f"Validation : {VALID_FILE}")
    print(f"Test       : {TEST_FILE}")

    print(
        f"\nStatistics saved to: "
        f"{STATS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    split_corpus()