import json
import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "Telugu/data/telugu_train.jsonl"
TEXT_FILE = "Telugu/data/telugu_train_text.txt"

MODEL_PREFIX = "Telugu/models/telugu_sentencepiece_bpe_24k/telugu_bpe_24k"

VOCAB_SIZE = 24_000


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_training_text():

    print("=" * 65)
    print("EXTRACTING TRAINING TEXT")
    print("=" * 65)

    documents = 0
    characters = 0

    with open(
        TRAIN_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        TEXT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                continue

            text = record.get(
                "text",
                ""
            ).strip()

            if not text:

                continue

            outfile.write(text + "\n")

            documents += 1
            characters += len(text)

            if documents % 10_000 == 0:

                print(
                    f"Processed: {documents:,} documents | "
                    f"Characters: {characters:,}"
                )

    print("\nExtraction complete")

    print(
        f"Documents: {documents:,}"
    )

    print(
        f"Characters: {characters:,}"
    )

    return documents, characters


# ============================================================
# TRAIN TOKENIZER
# ============================================================

def train_tokenizer():

    print("\n" + "=" * 65)
    print("TRAINING SENTENCEPIECE BPE TOKENIZER")
    print("=" * 65)

    spm.SentencePieceTrainer.train(
        input=TEXT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        normalization_rule_name="nmt_nfkc",
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        num_threads=16
    )

    print("\nTokenizer training complete.")

    print(
        f"\nModel saved to:\n"
        f"{MODEL_PREFIX}.model"
    )

    print(
        f"\nVocabulary saved to:\n"
        f"{MODEL_PREFIX}.vocab"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    extract_training_text()

    train_tokenizer()


if __name__ == "__main__":
    main()