import json
import random
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "near_duplicate_pairs.jsonl"

OUTPUT_FILE = "near_duplicate_audit.txt"

# Number of examples to sample from each bucket
SAMPLES_PER_BUCKET = 25

# For reproducibility
RANDOM_SEED = 42


# ============================================================
# SIMILARITY BUCKETS
# ============================================================

def get_bucket(similarity):

    if 0.90 <= similarity < 0.92:
        return "0.90 - 0.92"

    elif 0.92 <= similarity < 0.95:
        return "0.92 - 0.95"

    elif 0.95 <= similarity < 0.98:
        return "0.95 - 0.98"

    elif 0.98 <= similarity <= 1.00:
        return "0.98 - 1.00"

    else:
        return "other"


# ============================================================
# LOAD RECORDS
# ============================================================

def load_pairs():

    pairs = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            try:
                record = json.loads(line)

                pairs.append(record)

            except json.JSONDecodeError:

                print(
                    f"Skipping invalid JSON "
                    f"at line {line_number}"
                )

    return pairs


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    print("Loading near-duplicate pairs...")

    pairs = load_pairs()

    print(f"Loaded {len(pairs):,} pairs")

    # --------------------------------------------------------
    # Group by similarity bucket
    # --------------------------------------------------------

    buckets = defaultdict(list)

    for pair in pairs:

        similarity = pair.get(
            "jaccard_similarity",
            0
        )

        bucket = get_bucket(similarity)

        buckets[bucket].append(pair)

    # --------------------------------------------------------
    # Print bucket statistics
    # --------------------------------------------------------

    print("\nSimilarity distribution:")

    bucket_order = [
        "0.90 - 0.92",
        "0.92 - 0.95",
        "0.95 - 0.98",
        "0.98 - 1.00",
        "other"
    ]

    for bucket in bucket_order:

        print(
            f"{bucket:15}: "
            f"{len(buckets[bucket]):,}"
        )

    # --------------------------------------------------------
    # Write audit file
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        outfile.write(
            "NEAR DUPLICATE AUDIT\n"
        )

        outfile.write(
            "=" * 80 + "\n\n"
        )

        for bucket in bucket_order:

            records = buckets[bucket]

            if not records:
                continue

            # Random sample
            sample_size = min(
                SAMPLES_PER_BUCKET,
                len(records)
            )

            samples = random.sample(
                records,
                sample_size
            )

            # ------------------------------------------------
            # Bucket heading
            # ------------------------------------------------

            outfile.write(
                "\n"
                + "=" * 80
                + "\n"
            )

            outfile.write(
                f"SIMILARITY BUCKET: {bucket}\n"
            )

            outfile.write(
                f"TOTAL PAIRS IN BUCKET: "
                f"{len(records)}\n"
            )

            outfile.write(
                f"SAMPLED PAIRS: "
                f"{sample_size}\n"
            )

            outfile.write(
                "=" * 80
                + "\n\n"
            )

            # ------------------------------------------------
            # Write sampled records
            # ------------------------------------------------

            for sample_number, pair in enumerate(
                samples,
                start=1
            ):

                similarity = pair.get(
                    "jaccard_similarity"
                )

                outfile.write(
                    "#" * 80 + "\n"
                )

                outfile.write(
                    f"SAMPLE {sample_number}\n"
                )

                outfile.write(
                    f"JACCARD SIMILARITY: "
                    f"{similarity}\n"
                )

                outfile.write(
                    "#" * 80 + "\n\n"
                )


                # ============================================
                # KEPT DOCUMENT
                # ============================================

                outfile.write(
                    "KEPT / MATCHED DOCUMENT\n"
                )

                outfile.write(
                    "-" * 80 + "\n"
                )

                outfile.write(
                    f"Index: "
                    f"{pair.get('matched_index')}\n"
                )

                outfile.write(
                    f"ID: "
                    f"{pair.get('matched_id')}\n"
                )

                outfile.write(
                    f"Title: "
                    f"{pair.get('matched_title')}\n"
                )

                outfile.write(
                    f"Characters: "
                    f"{pair.get('matched_characters')}\n\n"
                )

                outfile.write(
                    "Preview:\n"
                )

                outfile.write(
                    pair.get(
                        "matched_preview",
                        ""
                    )
                    + "\n\n"
                )


                # ============================================
                # REMOVED DOCUMENT
                # ============================================

                outfile.write(
                    "REMOVED DOCUMENT\n"
                )

                outfile.write(
                    "-" * 80 + "\n"
                )

                outfile.write(
                    f"Index: "
                    f"{pair.get('removed_index')}\n"
                )

                outfile.write(
                    f"ID: "
                    f"{pair.get('removed_id')}\n"
                )

                outfile.write(
                    f"Title: "
                    f"{pair.get('removed_title')}\n"
                )

                outfile.write(
                    f"Characters: "
                    f"{pair.get('removed_characters')}\n\n"
                )

                outfile.write(
                    "Preview:\n"
                )

                outfile.write(
                    pair.get(
                        "removed_preview",
                        ""
                    )
                    + "\n\n"
                )


                # ============================================
                # MANUAL LABEL
                # ============================================

                outfile.write(
                    "MANUAL AUDIT\n"
                )

                outfile.write(
                    "-" * 80 + "\n"
                )

                outfile.write(
                    "Label "
                    "[TRUE_DUPLICATE / "
                    "NEAR_DUPLICATE / "
                    "LEGITIMATE_DIFFERENT / "
                    "UNCERTAIN]: \n\n"
                )

                outfile.write(
                    "Notes: \n\n\n"
                )

    print(
        f"\nAudit file created: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()