import json
from datasets import load_dataset


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_NAME = "ai4bharat/sangraha"
DATA_DIR = "verified/tel"

# New output file
OUTPUT_FILE = "sangraha_verified_tel_300k.jsonl"


# ============================================================
# SAMPLING CONFIGURATION
# ============================================================

# Number of valid documents already collected previously
SKIP_DOCUMENTS = 600_000

# Number of NEW documents to collect
NUM_DOCUMENTS = 300_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DOWNLOADING NEXT SANGRAHA VERIFIED TELUGU SAMPLE")
    print("=" * 60)

    print("\nLoading dataset in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        streaming=True
    )

    print(
        f"Available splits: "
        f"{list(dataset.keys())}"
    )

    split_name = (
        "train"
        if "train" in dataset
        else list(dataset.keys())[0]
    )

    stream = dataset[split_name]

    print(f"\nUsing split: {split_name}")

    print(
        f"Skipping first valid documents: "
        f"{SKIP_DOCUMENTS:,}"
    )

    print(
        f"New documents to collect: "
        f"{NUM_DOCUMENTS:,}\n"
    )


    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    valid_documents_seen = 0
    documents_saved = 0

    total_characters = 0


    # ========================================================
    # STREAM DATASET
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for example in stream:

            # ------------------------------------------------
            # Extract text
            # ------------------------------------------------

            text = example.get("text", "")

            # Match the logic of your previous script
            if not text or not text.strip():
                continue


            # ------------------------------------------------
            # Skip previously collected valid documents
            # ------------------------------------------------

            if valid_documents_seen < SKIP_DOCUMENTS:

                valid_documents_seen += 1

                if valid_documents_seen % 10_000 == 0:

                    print(
                        f"Skipped: "
                        f"{valid_documents_seen:,} / "
                        f"{SKIP_DOCUMENTS:,}"
                    )

                continue


            # ------------------------------------------------
            # Stop after collecting requested amount
            # ------------------------------------------------

            if documents_saved >= NUM_DOCUMENTS:
                break


            # ------------------------------------------------
            # Save document
            # ------------------------------------------------

            record = {
                "id": example.get("doc_id"),
                "text": text,
                "source": (
                    "ai4bharat_sangraha_verified_tel"
                ),
                "type": example.get("type"),
            }


            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


            documents_saved += 1

            total_characters += len(text)


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if documents_saved % 10_000 == 0:

                print(
                    f"Downloaded: "
                    f"{documents_saved:,} / "
                    f"{NUM_DOCUMENTS:,} | "
                    f"Characters: "
                    f"{total_characters:,}"
                )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        f"\nPreviously skipped documents : "
        f"{SKIP_DOCUMENTS:,}"
    )

    print(
        f"New documents saved           : "
        f"{documents_saved:,}"
    )

    print(
        f"Total characters              : "
        f"{total_characters:,}"
    )

    print(
        f"Output file                   : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()