import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from tiktoken._educational import bpe_train


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_wikipedia_quality_filtered.jsonl"

OUTPUT_DIR = Path("telugu_tiktoken_wiki_v1")

VOCAB_SIZE = 8_000 

# Number of special tokens we reserve
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
]

OUTPUT_MERGES = OUTPUT_DIR / "mergeable_ranks.json"
OUTPUT_CONFIG = OUTPUT_DIR / "tokenizer_config.json"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Apply Unicode NFC normalization and normalize whitespace.
    """

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus():

    texts = []

    total_documents = 0
    total_characters = 0

    print("Reading corpus...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = normalize_text(
                record.get("text", "")
            )

            if not text:
                continue

            texts.append(text)

            total_documents += 1
            total_characters += len(text)

            if total_documents % 10000 == 0:

                print(
                    f"Processed {total_documents:,} documents"
                )

    corpus = "\n".join(texts)

    print("\nCorpus statistics:")

    print(f"Documents  : {total_documents:,}")
    print(f"Characters : {total_characters:,}")

    return corpus


# ============================================================
# TRAIN BPE
# ============================================================

def train_bpe(corpus):

    print("\nTraining BPE tokenizer...")
    print(f"Target vocabulary size: {VOCAB_SIZE:,}")

    # bpe_train expects the total vocabulary size, including
    # the initial byte vocabulary.
    #
    # Therefore the requested vocab size must be > 256.
    mergeable_ranks = bpe_train(
        data=corpus,
        vocab_size=VOCAB_SIZE,
        pat_str=r"\S+",
    )

    return mergeable_ranks


# ============================================================
# SAVE TOKENIZER
# ============================================================

def save_tokenizer(mergeable_ranks):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # JSON cannot directly store bytes as dictionary keys.
    # Convert byte tokens to hexadecimal strings.
    serializable_ranks = {
        token.hex(): rank
        for token, rank in mergeable_ranks.items()
    }

    with open(
        OUTPUT_MERGES,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            serializable_ranks,
            f,
            indent=2
        )

    special_tokens = {
        token: VOCAB_SIZE + i
        for i, token in enumerate(SPECIAL_TOKENS)
    }

    config = {
        "name": "telugu_tiktoken_wiki_v1",
        "type": "Byte-level BPE",
        "vocab_size": len(mergeable_ranks),
        "target_vocab_size": VOCAB_SIZE,
        "special_tokens": special_tokens,
        "normalization": "Unicode NFC + whitespace normalization",
        "training_corpus": INPUT_FILE,
        "pre_tokenizer_pattern": r"\S+",
    }

    with open(
        OUTPUT_CONFIG,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nTokenizer saved:")

    print(f"  {OUTPUT_MERGES}")
    print(f"  {OUTPUT_CONFIG}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRAINING TELUGU TIKTOKEN BPE TOKENIZER")
    print("=" * 60)

    corpus = load_corpus()

    mergeable_ranks = train_bpe(
        corpus
    )

    save_tokenizer(
        mergeable_ranks
    )

    print("\n" + "=" * 60)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nFinal vocabulary size: "
        f"{len(mergeable_ranks):,}"
    )

    print(
        f"Target vocabulary size: "
        f"{VOCAB_SIZE:,}"
    )


if __name__ == "__main__":
    main()