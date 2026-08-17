import json
import re
import unicodedata


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_wikipedia_quality_filtered.jsonl"

OUTPUT_FILE = "telugu_tokenizer_corpus.txt"


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text):
    """
    Basic normalization before tokenizer training.

    SentencePiece also performs its own normalization, but
    we normalize Unicode and clean whitespace here so the
    training corpus is consistent.
    """

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    total_documents = 0
    total_characters = 0

    print("=" * 60)
    print("PREPARING TELUGU TOKENIZER CORPUS")
    print("=" * 60)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as input_file, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for line in input_file:

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get(
                "text",
                ""
            )

            text = normalize_text(text)

            if not text:
                continue

            # One Wikipedia article per line.
            output_file.write(text + "\n")

            total_documents += 1
            total_characters += len(text)

            if total_documents % 10000 == 0:

                print(
                    f"Processed "
                    f"{total_documents:,} documents"
                )

    print("\n" + "=" * 60)
    print("CORPUS PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Documents  : {total_documents:,}")
    print(f"Characters : {total_characters:,}")
    print(f"Output     : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()