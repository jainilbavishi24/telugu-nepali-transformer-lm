import json
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Telugu/data/telugu_combined_corpus_clean.jsonl"


TOKENIZERS = {
    "bpe_8k": (
        "Telugu/models/telugu_sentencepiece_bpe_8k/"
        "telugu_bpe_8k.model"
    ),

    "bpe_24k": (
        "Telugu/models/telugu_sentencepiece_bpe_24k/"
        "telugu_bpe_24k.model"
    ),
}


OUTPUT_FILE = "token_count_by_source.json"

PRINT_EVERY = 10_000


# ============================================================
# LOAD TOKENIZERS
# ============================================================

def load_tokenizers():

    tokenizers = {}

    print("=" * 70)
    print("LOADING TOKENIZERS")
    print("=" * 70)

    for name, model_path in TOKENIZERS.items():

        if not Path(model_path).exists():

            raise FileNotFoundError(
                f"Tokenizer model not found:\n"
                f"{model_path}"
            )

        sp = spm.SentencePieceProcessor(
            model_file=model_path
        )

        tokenizers[name] = sp

        print(
            f"\n{name}"
        )

        print(
            f"  Model: {model_path}"
        )

        print(
            f"  Vocabulary size: "
            f"{sp.get_piece_size():,}"
        )

    return tokenizers


# ============================================================
# INITIALIZE SOURCE STATISTICS
# ============================================================

def create_source_stats():

    return {

        "documents": 0,

        "characters": 0,

        "tokens": {

            tokenizer_name: 0

            for tokenizer_name
            in TOKENIZERS
        }
    }


# ============================================================
# PROCESS CORPUS
# ============================================================

def process_corpus(tokenizers):

    print("\n" + "=" * 70)
    print("COUNTING TOKENS BY SOURCE")
    print("=" * 70)

    total_documents = 0
    total_characters = 0

    total_tokens = Counter()

    source_stats = defaultdict(
        create_source_stats
    )

    invalid_json = 0
    empty_documents = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for line in infile:

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                invalid_json += 1
                continue

            text = record.get(
                "text",
                ""
            ).strip()

            if not text:

                empty_documents += 1
                continue

            source = record.get(
                "source",
                "unknown"
            )

            # ------------------------------------------------
            # DOCUMENT / CHARACTER STATISTICS
            # ------------------------------------------------

            total_documents += 1

            total_characters += len(text)

            source_stats[source][
                "documents"
            ] += 1

            source_stats[source][
                "characters"
            ] += len(text)

            # ------------------------------------------------
            # TOKENIZE WITH EACH TOKENIZER
            # ------------------------------------------------

            for tokenizer_name, sp in tokenizers.items():

                token_ids = sp.encode(
                    text,
                    out_type=int
                )

                num_tokens = len(
                    token_ids
                )

                total_tokens[
                    tokenizer_name
                ] += num_tokens

                source_stats[source][
                    "tokens"
                ][
                    tokenizer_name
                ] += num_tokens

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if total_documents % PRINT_EVERY == 0:

                progress = (
                    f"Processed: "
                    f"{total_documents:,} documents"
                )

                for tokenizer_name in tokenizers:

                    progress += (
                        f" | {tokenizer_name}: "
                        f"{total_tokens[tokenizer_name]:,}"
                    )

                print(progress)

    return (
        total_documents,
        total_characters,
        total_tokens,
        source_stats,
        invalid_json,
        empty_documents
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    total_documents,
    total_characters,
    total_tokens,
    source_stats,
    invalid_json,
    empty_documents
):

    print("\n" + "=" * 70)
    print("TOKEN COUNT RESULTS")
    print("=" * 70)

    print(
        f"\nDocuments processed : "
        f"{total_documents:,}"
    )

    print(
        f"Total characters    : "
        f"{total_characters:,}"
    )

    print(
        f"Invalid JSON        : "
        f"{invalid_json:,}"
    )

    print(
        f"Empty documents     : "
        f"{empty_documents:,}"
    )

    # ========================================================
    # OVERALL TOKENIZER STATISTICS
    # ========================================================

    print("\n" + "-" * 70)
    print("OVERALL TOKEN COUNTS")
    print("-" * 70)

    for tokenizer_name, token_count in total_tokens.items():

        avg_tokens_per_doc = (

            token_count
            / total_documents

            if total_documents > 0
            else 0
        )

        chars_per_token = (

            total_characters
            / token_count

            if token_count > 0
            else 0
        )

        print(f"\n{tokenizer_name}")

        print(
            f"  Vocabulary size       : "
            f"{TOKENIZERS[tokenizer_name]}"
        )

        print(
            f"  Total tokens          : "
            f"{token_count:,}"
        )

        print(
            f"  Average tokens/document: "
            f"{avg_tokens_per_doc:.2f}"
        )

        print(
            f"  Characters/token      : "
            f"{chars_per_token:.4f}"
        )

    # ========================================================
    # SOURCE DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("TOKEN DISTRIBUTION BY SOURCE")
    print("=" * 70)

    for source in sorted(source_stats.keys()):

        stats = source_stats[source]

        documents = stats["documents"]

        characters = stats["characters"]

        print(f"\nSource: {source}")

        print(
            f"  Documents : "
            f"{documents:,}"
        )

        print(
            f"  Characters: "
            f"{characters:,}"
        )

        for tokenizer_name in TOKENIZERS:

            tokens = stats["tokens"][
                tokenizer_name
            ]

            token_percentage = (

                tokens
                / total_tokens[tokenizer_name]
                * 100

                if total_tokens[tokenizer_name] > 0
                else 0
            )

            avg_tokens_per_document = (

                tokens
                / documents

                if documents > 0
                else 0
            )

            chars_per_token = (

                characters
                / tokens

                if tokens > 0
                else 0
            )

            print(
                f"\n  {tokenizer_name}"
            )

            print(
                f"    Tokens              : "
                f"{tokens:,}"
            )

            print(
                f"    Token percentage    : "
                f"{token_percentage:.2f}%"
            )

            print(
                f"    Avg tokens/document : "
                f"{avg_tokens_per_document:.2f}"
            )

            print(
                f"    Characters/token    : "
                f"{chars_per_token:.4f}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    total_documents,
    total_characters,
    total_tokens,
    source_stats,
    invalid_json,
    empty_documents
):

    output = {

        "input_file": INPUT_FILE,

        "documents_processed": (
            total_documents
        ),

        "characters_processed": (
            total_characters
        ),

        "invalid_json": invalid_json,

        "empty_documents": (
            empty_documents
        ),

        "tokenizers": {},

        "sources": {}
    }

    # ========================================================
    # TOKENIZER TOTALS
    # ========================================================

    for tokenizer_name, token_count in total_tokens.items():

        sp = spm.SentencePieceProcessor(
            model_file=TOKENIZERS[tokenizer_name]
        )

        avg_tokens_per_doc = (
            token_count / total_documents
            if total_documents > 0
            else 0
        )

        chars_per_token = (
            total_characters / token_count
            if token_count > 0
            else 0
        )

        print(f"\n{tokenizer_name}")

        print(
            f"  Vocabulary size       : "
            f"{sp.get_piece_size():,}"
        )

        print(
            f"  Total tokens          : "
            f"{token_count:,}"
        )

        print(
            f"  Average tokens/document: "
            f"{avg_tokens_per_doc:.2f}"
        )

        print(
            f"  Characters/token      : "
            f"{chars_per_token:.4f}"
        )

    # ========================================================
    # SOURCE STATISTICS
    # ========================================================

    for source, stats in source_stats.items():

        source_output = {

            "documents": stats[
                "documents"
            ],

            "characters": stats[
                "characters"
            ],

            "tokenizers": {}
        }

        for tokenizer_name in TOKENIZERS:

            tokens = stats["tokens"][
                tokenizer_name
            ]

            source_output[
                "tokenizers"
            ][
                tokenizer_name
            ] = {

                "tokens": tokens,

                "percentage_of_total_tokens": (

                    tokens
                    / total_tokens[
                        tokenizer_name
                    ]
                    * 100

                    if total_tokens[
                        tokenizer_name
                    ] > 0
                    else 0
                ),

                "average_tokens_per_document": (

                    tokens
                    / stats["documents"]

                    if stats["documents"] > 0
                    else 0
                ),

                "characters_per_token": (

                    stats["characters"]
                    / tokens

                    if tokens > 0
                    else 0
                )
            }

        output["sources"][
            source
        ] = source_output

    # ========================================================
    # WRITE JSON
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        json.dump(
            output,
            outfile,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\nStatistics saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD TOKENIZERS
    # --------------------------------------------------------

    tokenizers = load_tokenizers()

    # --------------------------------------------------------
    # PROCESS CORPUS
    # --------------------------------------------------------

    results = process_corpus(
        tokenizers
    )

    (
        total_documents,
        total_characters,
        total_tokens,
        source_stats,
        invalid_json,
        empty_documents
    ) = results

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_results(
        total_documents,
        total_characters,
        total_tokens,
        source_stats,
        invalid_json,
        empty_documents
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        total_documents,
        total_characters,
        total_tokens,
        source_stats,
        invalid_json,
        empty_documents
    )


if __name__ == "__main__":
    main()