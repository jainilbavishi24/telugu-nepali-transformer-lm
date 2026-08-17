import json
from pathlib import Path

import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_combined_corpus.jsonl"

MODEL_FILE = (
    "Telugu/telugu_sentencepiece_bpe_8k/"
    "telugu_bpe_8k.model"
)

OUTPUT_FILE = "token_count_stats.json"

TARGET_TOKENS = 500_000_000

PRINT_EVERY = 10_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COUNTING TOKENS IN TELUGU WIKIPEDIA CORPUS")
    print("=" * 65)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    sp = spm.SentencePieceProcessor(
        model_file=MODEL_FILE
    )

    print(f"\nTokenizer vocabulary size: {sp.get_piece_size():,}")
    print(f"Input corpus: {INPUT_FILE}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_documents = 0
    total_characters = 0
    total_tokens = 0

    documents_with_unk = 0
    total_unk_tokens = 0

    min_tokens = None
    max_tokens = 0

    # --------------------------------------------------------
    # Process corpus
    # --------------------------------------------------------

    print("\nCounting tokens...\n")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get("text", "")

            if not text.strip():
                continue

            # Encode directly to token IDs
            token_ids = sp.encode(
                text,
                out_type=int
            )

            num_tokens = len(token_ids)

            # Count unknown tokens
            unk_count = token_ids.count(
                sp.unk_id()
            )

            if unk_count > 0:
                documents_with_unk += 1
                total_unk_tokens += unk_count

            # Update statistics
            total_documents += 1
            total_characters += len(text)
            total_tokens += num_tokens

            if min_tokens is None:
                min_tokens = num_tokens
            else:
                min_tokens = min(
                    min_tokens,
                    num_tokens
                )

            max_tokens = max(
                max_tokens,
                num_tokens
            )

            # Progress
            if total_documents % PRINT_EVERY == 0:

                print(
                    f"Processed: {total_documents:,} documents | "
                    f"Tokens: {total_tokens:,} | "
                    f"Characters: {total_characters:,}"
                )

    # --------------------------------------------------------
    # Derived statistics
    # --------------------------------------------------------

    avg_tokens_per_document = (
        total_tokens / total_documents
        if total_documents > 0
        else 0
    )

    chars_per_token = (
        total_characters / total_tokens
        if total_tokens > 0
        else 0
    )

    tokens_per_character = (
        total_tokens / total_characters
        if total_characters > 0
        else 0
    )

    progress_percentage = (
        total_tokens / TARGET_TOKENS
    ) * 100

    remaining_tokens = max(
        0,
        TARGET_TOKENS - total_tokens
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TOKEN COUNT COMPLETE")
    print("=" * 65)

    print(f"\nDocuments processed      : {total_documents:,}")
    print(f"Total characters         : {total_characters:,}")
    print(f"Total tokens             : {total_tokens:,}")

    print("\nTokenizer efficiency:")

    print(
        f"Average tokens/document  : "
        f"{avg_tokens_per_document:,.2f}"
    )

    print(
        f"Characters/token         : "
        f"{chars_per_token:.4f}"
    )

    print(
        f"Tokens/character         : "
        f"{tokens_per_character:.4f}"
    )

    print(
        f"Shortest document        : "
        f"{min_tokens:,} tokens"
    )

    print(
        f"Longest document         : "
        f"{max_tokens:,} tokens"
    )

    print("\nUnknown token statistics:")

    print(
        f"Documents containing UNK : "
        f"{documents_with_unk:,}"
    )

    print(
        f"Total UNK tokens         : "
        f"{total_unk_tokens:,}"
    )

    print("\nProgress toward target:")

    print(
        f"Target                   : "
        f"{TARGET_TOKENS:,} tokens"
    )

    print(
        f"Current                  : "
        f"{total_tokens:,} tokens"
    )

    print(
        f"Progress                 : "
        f"{progress_percentage:.2f}%"
    )

    print(
        f"Remaining                : "
        f"{remaining_tokens:,} tokens"
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    stats = {
        "input_file": INPUT_FILE,
        "tokenizer_model": MODEL_FILE,
        "tokenizer_vocab_size": sp.get_piece_size(),

        "documents": total_documents,
        "characters": total_characters,
        "tokens": total_tokens,

        "average_tokens_per_document": avg_tokens_per_document,
        "characters_per_token": chars_per_token,
        "tokens_per_character": tokens_per_character,

        "min_tokens_per_document": min_tokens,
        "max_tokens_per_document": max_tokens,

        "documents_with_unk": documents_with_unk,
        "total_unk_tokens": total_unk_tokens,

        "target_tokens": TARGET_TOKENS,
        "progress_percentage": progress_percentage,
        "remaining_tokens": remaining_tokens,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nStatistics saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()