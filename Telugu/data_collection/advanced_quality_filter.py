import json
import re
import unicodedata
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "sangraha_verified_tel_sample_exact_dedup.jsonl"

OUTPUT_FILE = "sangraha_verified_tel_sample_quality_filtered.jsonl"

STATS_FILE = "sangraha_advanced_quality_filter_stats.json"


# ------------------------------------------------------------
# Sentence repetition
# ------------------------------------------------------------

MIN_SENTENCES_FOR_CHECK = 5

MAX_REPEATED_SENTENCE_RATIO = 0.30

MAX_DOMINANT_SENTENCE_RATIO = 0.20


# ------------------------------------------------------------
# Word repetition / lexical diversity
# ------------------------------------------------------------

MIN_WORDS_FOR_LEXICAL_CHECK = 100

MIN_LEXICAL_DIVERSITY = 0.15


# ------------------------------------------------------------
# Repeated n-grams
# ------------------------------------------------------------

NGRAM_SIZE = 5

MIN_WORDS_FOR_NGRAM_CHECK = 100

MAX_DOMINANT_NGRAM_RATIO = 0.08


# ------------------------------------------------------------
# Digits
# ------------------------------------------------------------

MIN_CHARS_FOR_DIGIT_CHECK = 500

MAX_DIGIT_RATIO = 0.25


# ------------------------------------------------------------
# Symbols / punctuation
# ------------------------------------------------------------

MIN_CHARS_FOR_SYMBOL_CHECK = 500

MAX_SYMBOL_RATIO = 0.30


# ------------------------------------------------------------
# Abnormally long tokens
# ------------------------------------------------------------

MIN_WORDS_FOR_LONG_TOKEN_CHECK = 50

MAX_LONG_TOKEN_RATIO = 0.05

LONG_TOKEN_LENGTH = 50


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def get_words(text):

    # Telugu Unicode block:
    # \u0C00 - \u0C7F
    #
    # Also retain Latin words and numbers.

    return re.findall(
        r"[\u0C00-\u0C7FA-Za-z0-9]+",
        text
    )


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def get_sentences(text):

    sentences = re.split(
        r"[.!?।]+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = normalize_text(sentence)

        if len(sentence) >= 10:
            cleaned.append(sentence)

    return cleaned


# ============================================================
# REPEATED SENTENCE CHECK
# ============================================================

def has_excessive_sentence_repetition(text):

    sentences = get_sentences(text)

    if len(sentences) < MIN_SENTENCES_FOR_CHECK:
        return False

    counts = Counter(sentences)

    total = len(sentences)

    unique = len(counts)

    repeated_ratio = (
        (total - unique) / total
    )

    dominant_ratio = (
        max(counts.values()) / total
    )

    if repeated_ratio > MAX_REPEATED_SENTENCE_RATIO:
        return True

    if dominant_ratio > MAX_DOMINANT_SENTENCE_RATIO:
        return True

    return False


# ============================================================
# LEXICAL DIVERSITY
# ============================================================

def has_low_lexical_diversity(text):

    words = get_words(text)

    if len(words) < MIN_WORDS_FOR_LEXICAL_CHECK:
        return False

    unique_words = len(set(words))

    diversity = (
        unique_words / len(words)
    )

    return diversity < MIN_LEXICAL_DIVERSITY


# ============================================================
# REPEATED N-GRAM CHECK
# ============================================================

def has_excessive_ngram_repetition(text):

    words = get_words(text)

    if len(words) < MIN_WORDS_FOR_NGRAM_CHECK:
        return False

    ngrams = []

    for i in range(
        len(words) - NGRAM_SIZE + 1
    ):

        ngram = tuple(
            words[i:i + NGRAM_SIZE]
        )

        ngrams.append(ngram)

    if not ngrams:
        return False

    counts = Counter(ngrams)

    dominant_ngram_count = max(
        counts.values()
    )

    dominant_ratio = (
        dominant_ngram_count
        / len(ngrams)
    )

    return (
        dominant_ratio
        > MAX_DOMINANT_NGRAM_RATIO
    )


# ============================================================
# DIGIT RATIO
# ============================================================

def has_high_digit_ratio(text):

    if len(text) < MIN_CHARS_FOR_DIGIT_CHECK:
        return False

    digit_count = sum(
        char.isdigit()
        for char in text
    )

    ratio = digit_count / len(text)

    return ratio > MAX_DIGIT_RATIO


# ============================================================
# SYMBOL / PUNCTUATION RATIO
# ============================================================

def has_high_symbol_ratio(text):

    if len(text) < MIN_CHARS_FOR_SYMBOL_CHECK:
        return False

    symbol_count = 0

    for char in text:

        if char.isspace():
            continue

        category = unicodedata.category(char)

        # P = punctuation
        # S = symbol
        if (
            category.startswith("P")
            or category.startswith("S")
        ):
            symbol_count += 1

    ratio = symbol_count / len(text)

    return ratio > MAX_SYMBOL_RATIO


# ============================================================
# ABNORMALLY LONG TOKENS
# ============================================================

def has_too_many_long_tokens(text):

    words = get_words(text)

    if len(words) < MIN_WORDS_FOR_LONG_TOKEN_CHECK:
        return False

    long_tokens = sum(
        len(word) >= LONG_TOKEN_LENGTH
        for word in words
    )

    ratio = (
        long_tokens / len(words)
    )

    return ratio > MAX_LONG_TOKEN_RATIO


# ============================================================
# FILTER ONE DOCUMENT
# ============================================================

def filter_document(text):

    text = normalize_text(text)

    if has_excessive_sentence_repetition(text):
        return False, "sentence_repetition"

    if has_low_lexical_diversity(text):
        return False, "low_lexical_diversity"

    if has_excessive_ngram_repetition(text):
        return False, "ngram_repetition"

    if has_high_digit_ratio(text):
        return False, "high_digit_ratio"

    if has_high_symbol_ratio(text):
        return False, "high_symbol_ratio"

    if has_too_many_long_tokens(text):
        return False, "long_token_ratio"

    return True, None


# ============================================================
# MAIN
# ============================================================

def main():

    removal_counts = Counter()

    total_documents = 0

    kept_documents = 0

    input_characters = 0

    output_characters = 0

    print("Starting advanced quality filtering...\n")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line_number, line in enumerate(
            infile,
            start=1
        ):

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                removal_counts[
                    "invalid_json"
                ] += 1

                continue

            text = record.get("text", "")

            text = normalize_text(text)

            total_documents += 1

            input_characters += len(text)

            keep, reason = filter_document(text)

            if not keep:

                removal_counts[reason] += 1

                continue

            # Save normalized text back
            record["text"] = text

            outfile.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            kept_documents += 1

            output_characters += len(text)

            # Progress
            if total_documents % 5000 == 0:

                print(
                    f"Processed: {total_documents:,} | "
                    f"Kept: {kept_documents:,} | "
                    f"Removed: "
                    f"{total_documents - kept_documents:,}"
                )

    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    removed_documents = (
        total_documents - kept_documents
    )

    stats = {

        "input_file": INPUT_FILE,

        "output_file": OUTPUT_FILE,

        "document_counts": {

            "input": total_documents,

            "kept": kept_documents,

            "removed": removed_documents
        },

        "removal_reasons": dict(
            removal_counts
        ),

        "character_statistics": {

            "input_characters":
                input_characters,

            "output_characters":
                output_characters,

            "removed_characters":
                input_characters
                - output_characters
        },

        "configuration": {

            "min_sentences_for_check":
                MIN_SENTENCES_FOR_CHECK,

            "max_repeated_sentence_ratio":
                MAX_REPEATED_SENTENCE_RATIO,

            "max_dominant_sentence_ratio":
                MAX_DOMINANT_SENTENCE_RATIO,

            "min_words_for_lexical_check":
                MIN_WORDS_FOR_LEXICAL_CHECK,

            "min_lexical_diversity":
                MIN_LEXICAL_DIVERSITY,

            "ngram_size":
                NGRAM_SIZE,

            "min_words_for_ngram_check":
                MIN_WORDS_FOR_NGRAM_CHECK,

            "max_dominant_ngram_ratio":
                MAX_DOMINANT_NGRAM_RATIO,

            "max_digit_ratio":
                MAX_DIGIT_RATIO,

            "max_symbol_ratio":
                MAX_SYMBOL_RATIO,

            "long_token_length":
                LONG_TOKEN_LENGTH,

            "max_long_token_ratio":
                MAX_LONG_TOKEN_RATIO
        }
    }

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as stats_file:

        json.dump(
            stats,
            stats_file,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "ADVANCED QUALITY FILTERING COMPLETE"
    )

    print("=" * 60)

    print(
        f"Total documents : "
        f"{total_documents:,}"
    )

    print(
        f"Kept            : "
        f"{kept_documents:,}"
    )

    print(
        f"Removed         : "
        f"{removed_documents:,}"
    )

    print("\nRemoval reasons:")

    for reason, count in sorted(
        removal_counts.items()
    ):

        print(
            f"{reason:30} "
            f"{count:,}"
        )

    print("\nCharacter statistics:")

    print(
        f"Input characters : "
        f"{input_characters:,}"
    )

    print(
        f"Output characters: "
        f"{output_characters:,}"
    )

    print(
        f"Removed characters: "
        f"{input_characters - output_characters:,}"
    )

    print(
        f"\nStats saved to: "
        f"{STATS_FILE}"
    )

    print(
        f"Filtered corpus saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()