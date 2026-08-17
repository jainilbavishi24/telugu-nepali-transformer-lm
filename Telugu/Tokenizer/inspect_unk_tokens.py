import json
import unicodedata
from collections import Counter

import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_wikipedia_quality_filtered.jsonl"

MODEL_FILE = (
    "telugu_sentencepiece_bpe_8k/"
    "telugu_bpe_8k.model"
)

OUTPUT_FILE = "unk_examples.jsonl"

PRINT_EXAMPLES = 20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def describe_character(char):
    """
    Return useful Unicode information about a character.
    """

    return {
        "character": char,
        "codepoint": f"U+{ord(char):04X}",
        "unicode_name": unicodedata.name(
            char,
            "UNKNOWN"
        ),
        "category": unicodedata.category(char),
    }


def find_unknown_characters(sp, text):
    """
    Identify characters that individually encode to <unk>.

    This is useful for diagnosing which Unicode characters
    are not covered by the tokenizer vocabulary.
    """

    unknown_chars = []

    for char in set(text):

        # Ignore normal whitespace.
        if char.isspace():
            continue

        ids = sp.encode(
            char,
            out_type=int
        )

        if sp.unk_id() in ids:

            unknown_chars.append(
                describe_character(char)
            )

    return unknown_chars


def make_context(text, position, radius=80):

    start = max(
        0,
        position - radius
    )

    end = min(
        len(text),
        position + radius
    )

    return text[start:end]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INSPECTING SENTENCEPIECE <UNK> TOKENS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    sp = spm.SentencePieceProcessor(
        model_file=MODEL_FILE
    )

    unk_id = sp.unk_id()

    print(f"\nModel: {MODEL_FILE}")
    print(f"Vocabulary size: {sp.get_piece_size():,}")
    print(f"UNK token ID: {unk_id}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_documents = 0
    documents_with_unk = 0
    total_unk_tokens = 0

    unknown_character_counter = Counter()
    unicode_name_counter = Counter()

    examples_printed = 0

    # --------------------------------------------------------
    # Scan corpus
    # --------------------------------------------------------

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

            if not text.strip():
                continue

            total_documents += 1

            # Encode the complete document.
            token_ids = sp.encode(
                text,
                out_type=int
            )

            unk_count = token_ids.count(
                unk_id
            )

            if unk_count == 0:
                continue

            documents_with_unk += 1
            total_unk_tokens += unk_count

            # ------------------------------------------------
            # Identify unknown characters
            # ------------------------------------------------

            unknown_chars = find_unknown_characters(
                sp,
                text
            )

            for info in unknown_chars:

                char = info["character"]

                unknown_character_counter[
                    char
                ] += text.count(char)

                unicode_name_counter[
                    info["unicode_name"]
                ] += text.count(char)

            # ------------------------------------------------
            # Find approximate locations of suspicious chars
            # ------------------------------------------------

            contexts = []

            for info in unknown_chars:

                char = info["character"]

                start = 0

                while True:

                    position = text.find(
                        char,
                        start
                    )

                    if position == -1:
                        break

                    contexts.append({
                        "character": char,
                        "unicode": info,
                        "context": make_context(
                            text,
                            position
                        )
                    })

                    start = position + 1

                    # Avoid generating huge output
                    if len(contexts) >= 10:
                        break

            # ------------------------------------------------
            # Save example
            # ------------------------------------------------

            example = {
                "id": record.get("id"),
                "title": record.get("title"),
                "unk_token_count": unk_count,
                "unknown_characters": unknown_chars,
                "contexts": contexts,
            }

            output_file.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

            # ------------------------------------------------
            # Print first examples
            # ------------------------------------------------

            if examples_printed < PRINT_EXAMPLES:

                examples_printed += 1

                print("\n" + "=" * 70)

                print(
                    f"EXAMPLE "
                    f"{examples_printed}"
                )

                print("=" * 70)

                print(
                    f"ID: "
                    f"{record.get('id')}"
                )

                print(
                    f"Title: "
                    f"{record.get('title')}"
                )

                print(
                    f"UNK tokens: "
                    f"{unk_count}"
                )

                print(
                    "\nUnknown characters:"
                )

                for info in unknown_chars:

                    print(
                        f"  {repr(info['character'])} | "
                        f"{info['codepoint']} | "
                        f"{info['unicode_name']} | "
                        f"{info['category']}"
                    )

                print(
                    "\nText preview:"
                )

                print(
                    text[:500]
                )

            # Progress
            if total_documents % 10000 == 0:

                print(
                    f"\nProcessed "
                    f"{total_documents:,} documents..."
                )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("UNK INSPECTION COMPLETE")
    print("=" * 70)

    print(
        f"\nDocuments processed: "
        f"{total_documents:,}"
    )

    print(
        f"Documents with UNK: "
        f"{documents_with_unk:,}"
    )

    print(
        f"Total UNK tokens: "
        f"{total_unk_tokens:,}"
    )

    print("\nMost common unknown characters:")

    for char, count in unknown_character_counter.most_common(30):

        print(
            f"{repr(char):15} "
            f"{count:8,} | "
            f"U+{ord(char):04X} | "
            f"{unicodedata.name(char, 'UNKNOWN')}"
        )

    print("\nMost common Unicode categories/names:")

    for name, count in unicode_name_counter.most_common(30):

        print(
            f"{count:8,} | "
            f"{name}"
        )

    print(
        f"\nDetailed examples saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()