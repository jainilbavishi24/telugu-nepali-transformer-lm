# download_sangraha_sample.py

import json
from datasets import load_dataset


DATASET_NAME = "ai4bharat/sangraha"
DATA_DIR = "verified/tel"

OUTPUT_FILE = "sangraha_verified_tel_sample.jsonl"

NUM_DOCUMENTS = 100_000


def main():

    print("=" * 60)
    print("DOWNLOADING SANGRAHA VERIFIED TELUGU SAMPLE")
    print("=" * 60)

    print("\nLoading dataset in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        streaming=True
    )

    print(f"Available splits: {list(dataset.keys())}")

    split_name = (
        "train"
        if "train" in dataset
        else list(dataset.keys())[0]
    )

    stream = dataset[split_name]

    documents_saved = 0
    total_characters = 0

    print(f"\nUsing split: {split_name}")
    print(f"Target documents: {NUM_DOCUMENTS:,}\n")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for example in stream:

            if documents_saved >= NUM_DOCUMENTS:
                break

            text = example.get("text", "")

            if not text or not text.strip():
                continue

            record = {
                "id": example.get("doc_id"),
                "text": text,
                "source": "ai4bharat_sangraha_verified_tel",
                "type": example.get("type"),
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            documents_saved += 1
            total_characters += len(text)

            if documents_saved % 10_000 == 0:

                print(
                    f"Downloaded: {documents_saved:,} documents | "
                    f"Characters: {total_characters:,}"
                )

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"\nDocuments saved : {documents_saved:,}")
    print(f"Total characters: {total_characters:,}")
    print(f"Output file     : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()