import json
import re
import unicodedata
from collections import Counter


INPUT_FILE = "sangraha_verified_tel_500k.jsonl"
OUTPUT_FILE = "sangraha_verified_tel_500k_basic_filtered.jsonl"
STATS_FILE = "sangraha_verified_tel_500k_basic_filter_stats.json"

# ============================================================
# CONFIGURATION
# ============================================================

# Minimum number of characters in a document.
MIN_CHARS = 200

# Minimum fraction of alphabetic/script characters that should
# be Telugu Unicode characters.
MIN_TELUGU_RATIO = 0.70

# Maximum fraction of non-space characters that can be symbols /
# punctuation before the document is considered suspicious.
MAX_SYMBOL_RATIO = 0.35

# Maximum fraction of repeated lines.
MAX_REPEATED_LINE_RATIO = 0.30

# Maximum allowed consecutive repetition of the same character.
MAX_CONSECUTIVE_CHAR_REPEAT = 20


# ============================================================
# CHARACTER CLASSIFICATION
# ============================================================

def is_telugu(char):
    """
    Return True if the character belongs to the Telugu Unicode block.
    """
    return "\u0C00" <= char <= "\u0C7F"


def get_character_stats(text):
    """
    Compute character-level statistics for a document.

    We ignore whitespace when computing most ratios.
    """

    telugu = 0
    latin = 0
    digits = 0
    punctuation = 0
    other = 0

    for char in text:

        if char.isspace():
            continue

        if is_telugu(char):
            telugu += 1

        elif char.isascii() and char.isalpha():
            latin += 1

        elif char.isdigit():
            digits += 1

        elif unicodedata.category(char).startswith("P"):
            punctuation += 1

        elif unicodedata.category(char).startswith("S"):
            punctuation += 1

        else:
            other += 1

    total = telugu + latin + digits + punctuation + other

    if total == 0:
        return {
            "telugu": 0,
            "latin": 0,
            "digits": 0,
            "punctuation": 0,
            "other": 0,
            "total": 0,
            "telugu_ratio": 0.0,
            "symbol_ratio": 0.0
        }

    # Telugu ratio among meaningful script characters.
    # We compare Telugu against Telugu + Latin + Other.
    script_total = telugu + latin + other

    if script_total == 0:
        telugu_ratio = 0.0
    else:
        telugu_ratio = telugu / script_total

    symbol_ratio = punctuation / total

    return {
        "telugu": telugu,
        "latin": latin,
        "digits": digits,
        "punctuation": punctuation,
        "other": other,
        "total": total,
        "telugu_ratio": telugu_ratio,
        "symbol_ratio": symbol_ratio
    }


# ============================================================
# REPETITION DETECTION
# ============================================================

def repeated_line_ratio(text):
    """
    Calculate the fraction of lines that are duplicates.

    Example:

        Line A
        Line B
        Line A
        Line A

    Here, repeated lines are counted as duplicates.
    """

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if len(lines) <= 1:
        return 0.0

    counts = Counter(lines)

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / len(lines)


def has_excessive_character_repetition(
    text,
    max_repeat=MAX_CONSECUTIVE_CHAR_REPEAT
):
    """
    Detect sequences such as:

    హహహహహహహహహహహహహహహహ...
    !!!!!!!!!!!!!!!!!!!!
    """

    pattern = rf"(.)\1{{{max_repeat},}}"

    return re.search(pattern, text) is not None


# ============================================================
# OPTIONAL TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Apply Unicode NFC normalization and normalize whitespace.
    """

    text = unicodedata.normalize("NFC", text)

    # Normalize different newline formats.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces/tabs but preserve paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Limit excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# MAIN DOCUMENT FILTER
# ============================================================

def filter_document(text):
    """
    Returns:

        (keep, reason, statistics)

    Only the first failed filter is reported as the reason.
    """

    # Normalize first.
    text = normalize_text(text)

    # --------------------------------------------------------
    # 1. Empty document
    # --------------------------------------------------------

    if not text:
        return False, "empty", None

    # --------------------------------------------------------
    # 2. Minimum length
    # --------------------------------------------------------

    if len(text) < MIN_CHARS:
        return False, "too_short", None

    # --------------------------------------------------------
    # 3. Character statistics
    # --------------------------------------------------------

    char_stats = get_character_stats(text)

    # --------------------------------------------------------
    # 4. Telugu script ratio
    # --------------------------------------------------------

    if char_stats["telugu_ratio"] < MIN_TELUGU_RATIO:
        return False, "low_telugu_ratio", char_stats

    # --------------------------------------------------------
    # 5. Excessive punctuation / symbols
    # --------------------------------------------------------

    if char_stats["symbol_ratio"] > MAX_SYMBOL_RATIO:
        return False, "high_symbol_ratio", char_stats

    # --------------------------------------------------------
    # 6. Repeated lines
    # --------------------------------------------------------

    line_repeat_ratio = repeated_line_ratio(text)

    if line_repeat_ratio > MAX_REPEATED_LINE_RATIO:
        return False, "high_line_repetition", char_stats

    # --------------------------------------------------------
    # 7. Excessive repeated characters
    # --------------------------------------------------------

    if has_excessive_character_repetition(text):
        return False, "excessive_char_repetition", char_stats

    return True, "kept", char_stats


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    counts = Counter()

    total_input_characters = 0
    total_output_characters = 0

    total_telugu_ratio = 0.0
    kept_documents = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line_number, line in enumerate(infile, start=1):

            counts["total"] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue

            text = record.get("text", "")

            total_input_characters += len(text)

            keep, reason, char_stats = filter_document(text)

            counts[reason] += 1

            if not keep:
                continue

            # Normalize the text before saving.
            cleaned_text = normalize_text(text)

            record["text"] = cleaned_text

            outfile.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            kept_documents += 1
            total_output_characters += len(cleaned_text)

            total_telugu_ratio += char_stats["telugu_ratio"]

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if counts["total"] % 1000 == 0:

                print(
                    f"Processed: {counts['total']:,} | "
                    f"Kept: {counts['kept']:,} | "
                    f"Removed: {counts['total'] - counts['kept']:,}"
                )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    removed = counts["total"] - counts["kept"]

    stats = {
        "input_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,

        "configuration": {
            "min_chars": MIN_CHARS,
            "min_telugu_ratio": MIN_TELUGU_RATIO,
            "max_symbol_ratio": MAX_SYMBOL_RATIO,
            "max_repeated_line_ratio": MAX_REPEATED_LINE_RATIO,
            "max_consecutive_char_repeat":
                MAX_CONSECUTIVE_CHAR_REPEAT
        },

        "document_counts": {
            "total": counts["total"],
            "kept": counts["kept"],
            "removed": removed,

            "empty": counts["empty"],
            "too_short": counts["too_short"],
            "low_telugu_ratio":
                counts["low_telugu_ratio"],
            "high_symbol_ratio":
                counts["high_symbol_ratio"],
            "high_line_repetition":
                counts["high_line_repetition"],
            "excessive_char_repetition":
                counts["excessive_char_repetition"],
            "invalid_json":
                counts["invalid_json"]
        },

        "characters": {
            "input": total_input_characters,
            "output": total_output_characters
        },

        "average_telugu_ratio_kept": (
            total_telugu_ratio / kept_documents
            if kept_documents > 0
            else 0
        )
    }

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("BASIC FILTERING COMPLETE")
    print("=" * 60)

    print(f"Total documents : {counts['total']:,}")
    print(f"Kept            : {counts['kept']:,}")
    print(f"Removed         : {removed:,}")

    print("\nRemoval reasons:")

    for reason in [
        "empty",
        "too_short",
        "low_telugu_ratio",
        "high_symbol_ratio",
        "high_line_repetition",
        "excessive_char_repetition",
        "invalid_json"
    ]:

        print(
            f"{reason:30}: "
            f"{counts[reason]:,}"
        )

    print("\nCharacter statistics:")

    print(
        f"Input characters : "
        f"{total_input_characters:,}"
    )

    print(
        f"Output characters: "
        f"{total_output_characters:,}"
    )

    print(
        f"\nAverage Telugu ratio "
        f"(kept documents): "
        f"{stats['average_telugu_ratio_kept']:.4f}"
    )

    print(f"\nStatistics saved to: {STATS_FILE}")
    print(f"Filtered corpus saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_dataset()