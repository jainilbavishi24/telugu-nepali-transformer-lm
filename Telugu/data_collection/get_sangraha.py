import json
from datasets import load_dataset



DATASET_NAME = "ai4bharat/sangraha"
DATA_DIR = "verified/tel"
OUTPUT_FILE = "Telugu/data/telugu_sangraha_raw.jsonl"

NUM_DOCUMENTS = 900_000
PRINT_EVERY = 10_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DOWNLOADING FIXED NUMBER OF SANGRAHA DOCUMENTS")
    print("=" * 60)

    print("\nDataset      :", DATASET_NAME)
    print("Subset       :", DATA_DIR)
    print(f"Target docs  : {NUM_DOCUMENTS:,}")
    print("Streaming mode: enabled")

    # --------------------------------------------------------
    # Load dataset in streaming mode
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        streaming=True
    )

    print(
        f"Available splits: "
        f"{list(dataset.keys())}"
    )

    # --------------------------------------------------------
    # Select split
    # --------------------------------------------------------

    if "train" in dataset:
        split_name = "train"
    else:
        split_name = list(dataset.keys())[0]

    stream = dataset[split_name]

    print(f"Using split: {split_name}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    documents_seen = 0
    documents_saved = 0
    documents_empty = 0

    total_characters = 0

    # --------------------------------------------------------
    # Download documents
    # --------------------------------------------------------

    print("\nDownloading...\n")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for example in stream:

            # Stop after collecting the requested number
            # of valid documents.
            if documents_saved >= NUM_DOCUMENTS:
                break

            documents_seen += 1

            text = example.get("text", "")

            # Skip empty documents
            if not text or not text.strip():

                documents_empty += 1
                continue

            # ------------------------------------------------
            # Standardized record
            # ------------------------------------------------

            record = {
                "id": example.get("doc_id"),
                "text": text,
                "source": "ai4bharat_sangraha_verified_tel",
                "type": example.get("type")
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

            if documents_saved % PRINT_EVERY == 0:

                print(
                    f"Saved: {documents_saved:,} | "
                    f"Seen: {documents_seen:,} | "
                    f"Characters: {total_characters:,}"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        f"\nDocuments seen    : "
        f"{documents_seen:,}"
    )

    print(
        f"Documents saved   : "
        f"{documents_saved:,}"
    )

    print(
        f"Empty documents   : "
        f"{documents_empty:,}"
    )

    print(
        f"Total characters  : "
        f"{total_characters:,}"
    )

    print(
        f"\nOutput file: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()