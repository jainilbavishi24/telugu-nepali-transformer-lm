import json
from collections import defaultdict

import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_combined_corpus_v4.jsonl"

MODEL_FILE = (
    "Telugu/telugu_sentencepiece_bpe_24k/"
    "telugu_bpe_24k.model"
)

OUTPUT_FILE = "token_count_stats.json"

TARGET_TOKENS = 500_000_000

PRINT_EVERY = 10_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COUNTING TOKENS IN TELUGU CORPUS")
    print("=" * 65)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    sp = spm.SentencePieceProcessor(
        model_file=MODEL_FILE
    )

    print(
        f"\nTokenizer vocabulary size: "
        f"{sp.get_piece_size():,}"
    )

    print(f"Input corpus: {INPUT_FILE}")

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    total_documents = 0
    total_characters = 0
    total_tokens = 0

    documents_with_unk = 0
    total_unk_tokens = 0

    min_tokens = None
    max_tokens = 0

    # --------------------------------------------------------
    # Per-source statistics
    #
    # defaultdict automatically creates a dictionary for every
    # new source encountered in the corpus.
    # --------------------------------------------------------

    source_stats = defaultdict(
        lambda: {
            "documents": 0,
            "characters": 0,
            "tokens": 0,
            "documents_with_unk": 0,
            "total_unk_tokens": 0,
        }
    )

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

            # ------------------------------------------------
            # Get source
            #
            # Use "unknown" if the source field is missing.
            # ------------------------------------------------

            source = record.get(
                "source",
                "unknown"
            )

            # ------------------------------------------------
            # Encode text
            # ------------------------------------------------

            token_ids = sp.encode(
                text,
                out_type=int
            )

            num_tokens = len(token_ids)

            # ------------------------------------------------
            # Count UNK tokens
            # ------------------------------------------------

            unk_count = token_ids.count(
                sp.unk_id()
            )

            # ------------------------------------------------
            # Update global statistics
            # ------------------------------------------------

            total_documents += 1
            total_characters += len(text)
            total_tokens += num_tokens

            if unk_count > 0:

                documents_with_unk += 1
                total_unk_tokens += unk_count

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

            # ------------------------------------------------
            # Update per-source statistics
            # ------------------------------------------------

            source_stats[source]["documents"] += 1

            source_stats[source]["characters"] += len(
                text
            )

            source_stats[source]["tokens"] += num_tokens

            if unk_count > 0:

                source_stats[source][
                    "documents_with_unk"
                ] += 1

                source_stats[source][
                    "total_unk_tokens"
                ] += unk_count

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if total_documents % PRINT_EVERY == 0:

                print(
                    f"Processed: {total_documents:,} documents | "
                    f"Tokens: {total_tokens:,} | "
                    f"Characters: {total_characters:,}"
                )

    # --------------------------------------------------------
    # Global derived statistics
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
        total_tokens / TARGET_TOKENS * 100
        if TARGET_TOKENS > 0
        else 0
    )

    remaining_tokens = max(
        0,
        TARGET_TOKENS - total_tokens
    )

    # --------------------------------------------------------
    # Calculate derived statistics for each source
    # --------------------------------------------------------

    for source in source_stats:

        stats = source_stats[source]

        documents = stats["documents"]
        characters = stats["characters"]
        tokens = stats["tokens"]

        stats["average_tokens_per_document"] = (
            tokens / documents
            if documents > 0
            else 0
        )

        stats["characters_per_token"] = (
            characters / tokens
            if tokens > 0
            else 0
        )

        stats["tokens_per_character"] = (
            tokens / characters
            if characters > 0
            else 0
        )

        stats["token_percentage"] = (
            tokens / total_tokens * 100
            if total_tokens > 0
            else 0
        )

        stats["document_percentage"] = (
            documents / total_documents * 100
            if total_documents > 0
            else 0
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

    # --------------------------------------------------------
    # Per-source results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TOKEN DISTRIBUTION BY SOURCE")
    print("=" * 65)

    for source, stats in sorted(
        source_stats.items(),
        key=lambda item: item[1]["tokens"],
        reverse=True
    ):

        print(f"\nSource: {source}")

        print(
            f"  Documents             : "
            f"{stats['documents']:,}"
        )

        print(
            f"  Document share        : "
            f"{stats['document_percentage']:.2f}%"
        )

        print(
            f"  Characters            : "
            f"{stats['characters']:,}"
        )

        print(
            f"  Tokens                : "
            f"{stats['tokens']:,}"
        )

        print(
            f"  Token share           : "
            f"{stats['token_percentage']:.2f}%"
        )

        print(
            f"  Avg tokens/document   : "
            f"{stats['average_tokens_per_document']:,.2f}"
        )

        print(
            f"  Characters/token      : "
            f"{stats['characters_per_token']:.4f}"
        )

        print(
            f"  Documents with UNK    : "
            f"{stats['documents_with_unk']:,}"
        )

        print(
            f"  Total UNK tokens      : "
            f"{stats['total_unk_tokens']:,}"
        )

    # --------------------------------------------------------
    # Global UNK statistics
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("UNKNOWN TOKEN STATISTICS")
    print("=" * 65)

    print(
        f"\nDocuments containing UNK : "
        f"{documents_with_unk:,}"
    )

    print(
        f"Total UNK tokens         : "
        f"{total_unk_tokens:,}"
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PROGRESS TOWARD TARGET")
    print("=" * 65)

    print(
        f"\nTarget                   : "
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

        "tokenizer_vocab_size": (
            sp.get_piece_size()
        ),

        "global": {

            "documents": total_documents,

            "characters": total_characters,

            "tokens": total_tokens,

            "average_tokens_per_document": (
                avg_tokens_per_document
            ),

            "characters_per_token": (
                chars_per_token
            ),

            "tokens_per_character": (
                tokens_per_character
            ),

            "min_tokens_per_document": (
                min_tokens
            ),

            "max_tokens_per_document": (
                max_tokens
            ),

            "documents_with_unk": (
                documents_with_unk
            ),

            "total_unk_tokens": (
                total_unk_tokens
            )
        },

        "sources": dict(
            source_stats
        ),

        "target_tokens": TARGET_TOKENS,

        "progress_percentage": (
            progress_percentage
        ),

        "remaining_tokens": (
            remaining_tokens
        )
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

    print(
        f"\nStatistics saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()