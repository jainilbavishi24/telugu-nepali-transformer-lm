import json
import re
import unicodedata
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "Nepalese/data.nosync/nepalese_wikipedia_basic_filtered.jsonl"
)

OUTPUT_FILE = (
    "Nepalese/data.nosync/nepalese_wikipedia_clean.jsonl"
)

STATS_FILE = (
    "Nepalese/data.nosync/nepalese_wikipedia_cleaning_stats.json"
)

PRINT_EVERY = 10_000

# Minimum characters after cleaning.
MIN_CHARS = 50

# Minimum Devanagari ratio among alphabetic/script characters.
MIN_DEVANAGARI_RATIO = 0.50

# Minimum evidence of Nepali-specific vocabulary/patterns.
MIN_NEPALI_WORD_RATIO = 0.01


# ============================================================
# REMOVE BALANCED TEMPLATES
# ============================================================

def remove_templates(text):
    """
    Remove balanced {{ ... }} templates.

    Supports nested templates.

    If an unmatched '{{' is found, preserve the remainder
    rather than accidentally deleting the rest of the article.
    """

    result = []
    i = 0
    n = len(text)

    while i < n:

        if text.startswith("{{", i):

            start = i
            depth = 1
            i += 2

            while i < n and depth > 0:

                if text.startswith("{{", i):
                    depth += 1
                    i += 2

                elif text.startswith("}}", i):
                    depth -= 1
                    i += 2

                else:
                    i += 1

            # Balanced template.
            if depth == 0:
                continue

            # Unmatched '{{' -> preserve remainder.
            result.append(text[start:])
            break

        else:
            result.append(text[i])
            i += 1

    return "".join(result)


# ============================================================
# REMOVE REFERENCES
# ============================================================

def remove_references(text):

    # <ref> ... </ref>
    text = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # <ref ... />
    text = re.sub(
        r"<ref\b[^>]*/\s*>",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text


# ============================================================
# REMOVE HTML COMMENTS
# ============================================================

def remove_comments(text):

    return re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL
    )


# ============================================================
# REMOVE TABLES
# ============================================================

def remove_tables(text):
    """
    Remove balanced MediaWiki tables:

        {| ... |}

    If a table opening has no closing |}, preserve the
    remainder rather than destroying the document.
    """

    result = []
    i = 0
    n = len(text)

    while i < n:

        start = text.find("{|", i)

        if start == -1:
            result.append(text[i:])
            break

        result.append(text[i:start])

        end = text.find("|}", start + 2)

        if end == -1:
            result.append(text[start:])
            break

        i = end + 2

    return "".join(result)


# ============================================================
# REMOVE FILE / IMAGE LINKS
# ============================================================

def remove_files(text):

    pattern = (
        r"\[\["
        r"(?:File|Image|चित्र|फाइल)"
        r":[^\]]*"
        r"\]\]"
    )

    return re.sub(
        pattern,
        "",
        text,
        flags=re.IGNORECASE
    )


# ============================================================
# REMOVE CATEGORY LINKS
# ============================================================

def remove_categories(text):

    pattern = (
        r"\[\["
        r"(?:Category|श्रेणी|वर्ग)"
        r":[^\]]*"
        r"\]\]"
    )

    return re.sub(
        pattern,
        "",
        text,
        flags=re.IGNORECASE
    )


# ============================================================
# CLEAN INTERNAL WIKI LINKS
# ============================================================

def clean_wiki_links(text):
    """
    Convert:

        [[target]]
            -> target

        [[target|display text]]
            -> display text

        [[target#section]]
            -> target

        [[target#section|display]]
            -> display
    """

    def replace_link(match):

        content = match.group(1).strip()

        if "|" in content:

            parts = content.split("|")

            display = parts[-1].strip()

            return display

        # Remove fragment.
        if "#" in content:

            content = content.split(
                "#",
                1
            )[0].strip()

        return content

    return re.sub(
        r"\[\[([^\[\]]+)\]\]",
        replace_link,
        text
    )


# ============================================================
# CLEAN EXTERNAL LINKS
# ============================================================

def clean_external_links(text):

    # [https://example.com Display text]
    text = re.sub(
        r"\[https?://[^\s\]]+\s+([^\]]+)\]",
        r"\1",
        text
    )

    # [https://example.com]
    text = re.sub(
        r"\[https?://[^\]]+\]",
        "",
        text
    )

    return text


# ============================================================
# REMOVE HTML TAGS
# ============================================================

def remove_html(text):

    return re.sub(
        r"<[^>]+>",
        "",
        text
    )


# ============================================================
# REMOVE WIKI FORMATTING
# ============================================================

def remove_formatting(text):

    text = text.replace("'''''", "")
    text = text.replace("'''", "")
    text = text.replace("''", "")

    return text


# ============================================================
# CLEAN HEADINGS
# ============================================================

def clean_headings(text):

    return re.sub(
        r"(?m)^\s*=+\s*(.*?)\s*=+\s*$",
        r"\1",
        text
    )


# ============================================================
# CLEAN LIST MARKERS
# ============================================================

def clean_list_markers(text):
    """
    Convert MediaWiki list markers to ordinary text.

        * item
        # item
        : item
        ; item
    """

    return re.sub(
        r"(?m)^\s*[\*#:;]+\s*",
        "",
        text
    )


# ============================================================
# REMOVE LEFTOVER MEDIAWIKI MARKUP
# ============================================================

def remove_leftover_markup(text):

    # HTML entities.
    text = re.sub(
        r"&nbsp;",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"&amp;",
        "&",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"&lt;",
        "<",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"&gt;",
        ">",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"&quot;",
        '"',
        text,
        flags=re.IGNORECASE
    )

    # Remaining URLs.
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    return text


# ============================================================
# CLEAN STRAY WIKI CHARACTERS
# ============================================================

def clean_stray_markup(text):
    """
    Conservative cleanup.

    Do NOT remove normal punctuation.
    """

    text = text.replace(
        "{{",
        " "
    )

    text = text.replace(
        "}}",
        " "
    )

    text = text.replace(
        "[[",
        ""
    )

    text = text.replace(
        "]]",
        ""
    )

    text = text.replace(
        "{|",
        " "
    )

    text = text.replace(
        "|}",
        " "
    )

    return text


# ============================================================
# REMOVE CONTROL / INVISIBLE CHARACTERS
# ============================================================

def remove_control_characters(text):
    """
    IMPORTANT:
    Preserve:
        U+200C ZERO WIDTH NON-JOINER
        U+200D ZERO WIDTH JOINER

    Remove other accidental control/format characters.
    """

    cleaned = []

    for char in text:

        if char in "\n\t ":
            cleaned.append(char)
            continue

        # Preserve Indic ZWNJ/ZWJ.
        if char in ("\u200c", "\u200d"):
            cleaned.append(char)
            continue

        category = unicodedata.category(char)

        if category.startswith("C"):
            continue

        cleaned.append(char)

    return "".join(cleaned)


# ============================================================
# NORMALIZE UNICODE
# ============================================================

def normalize_unicode(text):

    return unicodedata.normalize(
        "NFC",
        text
    )


# ============================================================
# NORMALIZE WHITESPACE
# ============================================================

def normalize_whitespace(text):

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Normalize non-breaking and narrow spaces.
    text = (
        text
        .replace("\u00a0", " ")
        .replace("\u2009", " ")
        .replace("\u200a", " ")
        .replace("\u202f", " ")
    )

    # Collapse spaces/tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    # Remove spaces around newlines.
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# DEVANAGARI / NEPALI DETECTION
# ============================================================

def is_devanagari_character(char):

    code = ord(char)

    return (
        0x0900 <= code <= 0x097F
    )


NEPALI_WORDS = {
    "छ",
    "छन्",
    "थियो",
    "थिए",
    "हुन्छ",
    "भएको",
    "भएका",
    "गरेको",
    "गरेका",
    "गरेर",
    "गर्नु",
    "गर्ने",
    "गरिएको",
    "गरिएका",
    "गरिन्छ",
    "भएकोले",
    "त्यसैले",
    "यसअघि",
    "यसपछि",
    "अहिले",
    "यहाँ",
    "त्यहाँ",
    "किनभने",
    "नेपाल",
    "नेपालमा",
    "नेपाली",
    "सरकारले",
    "मन्त्रालयले",
    "व्यवस्था",
    "आवश्यकता",
    "सम्बन्धी",
    "सम्बन्धित",
    "मार्फत",
    "अनुसार",
    "लगायत",
    "तथा",
    "तर",
    "भने",
    "पनि",
    "नै",
}


NEPALI_PATTERNS = [
    r"\bछैन\b",
    r"\bछैनन्\b",
    r"\bहुनेछ\b",
    r"\bभनेर\b",
    r"\bगरेको\b",
    r"\bगरिरहेका\b",
    r"\bरहेको\b",
    r"\bरहेका\b",
    r"\bरहेकी\b",
    r"\bनेपालमा\b",
    r"\bनेपाली\b",
    r"\bसम्बन्धी\b",
    r"\bमार्फत\b",
    r"\bलगायत\b",
]

COMPILED_NEPALI_PATTERNS = [
    re.compile(pattern)
    for pattern in NEPALI_PATTERNS
]


def calculate_devanagari_ratio(text):

    relevant = 0
    devanagari = 0

    for char in text:

        if is_devanagari_character(char):

            devanagari += 1
            relevant += 1

        elif char.isalpha():

            relevant += 1

    if relevant == 0:
        return 0.0

    return (
        devanagari
        / relevant
    )


def extract_devanagari_words(text):

    return re.findall(
        r"[\u0900-\u097F]+",
        text
    )


def calculate_nepali_word_ratio(text):

    words = extract_devanagari_words(
        text
    )

    if not words:
        return 0.0

    matches = sum(
        1
        for word in words
        if word in NEPALI_WORDS
    )

    return matches / len(words)


def calculate_nepali_pattern_score(text):

    words = extract_devanagari_words(
        text
    )

    if not words:
        return 0.0

    matches = 0

    for pattern in COMPILED_NEPALI_PATTERNS:
        matches += len(
            pattern.findall(text)
        )

    return matches / len(words)


def is_probably_nepali(text):

    devanagari_ratio = (
        calculate_devanagari_ratio(
            text
        )
    )

    if (
        devanagari_ratio
        < MIN_DEVANAGARI_RATIO
    ):
        return False

    nepali_word_ratio = (
        calculate_nepali_word_ratio(
            text
        )
    )

    pattern_score = (
        calculate_nepali_pattern_score(
            text
        )
    )

    return (
        nepali_word_ratio
        >= MIN_NEPALI_WORD_RATIO
        or pattern_score >= 0.005
    )


# ============================================================
# COMPLETE CLEANING PIPELINE
# ============================================================

def clean_text(text):

    text = normalize_unicode(text)

    text = remove_comments(text)

    text = remove_templates(text)

    text = remove_references(text)

    text = remove_tables(text)

    text = remove_files(text)

    text = remove_categories(text)

    text = clean_wiki_links(text)

    text = clean_external_links(text)

    text = remove_html(text)

    text = remove_formatting(text)

    text = clean_headings(text)

    text = clean_list_markers(text)

    text = remove_leftover_markup(text)

    text = clean_stray_markup(text)

    text = remove_control_characters(text)

    text = normalize_whitespace(text)

    return text


# ============================================================
# PROCESS CORPUS
# ============================================================

def main():

    print("=" * 70)
    print("CLEANING COMBINED NEPALI CORPUS")
    print("=" * 70)

    total_documents = 0
    kept_documents = 0
    removed_documents = 0

    invalid_json = 0
    empty_after_cleaning = 0
    too_short = 0
    not_nepali = 0

    input_characters = 0
    output_characters = 0

    source_stats = {}

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            total_documents += 1

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                invalid_json += 1
                removed_documents += 1

                continue

            text = record.get(
                "text",
                ""
            )

            source = record.get(
                "source",
                "unknown"
            )

            if source not in source_stats:

                source_stats[source] = Counter()

            source_stats[source][
                "input_documents"
            ] += 1

            source_stats[source][
                "input_characters"
            ] += len(text)

            input_characters += len(text)

            # ------------------------------------------------
            # CLEAN
            # ------------------------------------------------

            cleaned = clean_text(text)

            # ------------------------------------------------
            # EMPTY
            # ------------------------------------------------

            if not cleaned:

                empty_after_cleaning += 1
                removed_documents += 1

                source_stats[source][
                    "removed"
                ] += 1

                continue

            # ------------------------------------------------
            # MINIMUM LENGTH
            # ------------------------------------------------

            if len(cleaned) < MIN_CHARS:

                too_short += 1
                removed_documents += 1

                source_stats[source][
                    "removed"
                ] += 1

                continue

            # ------------------------------------------------
            # NEPALI DETECTION
            # ------------------------------------------------

            if not is_probably_nepali(
                cleaned
            ):

                not_nepali += 1
                removed_documents += 1

                source_stats[source][
                    "not_nepali"
                ] += 1

                continue

            # ------------------------------------------------
            # KEEP
            # ------------------------------------------------

            output_record = record.copy()

            output_record["text"] = cleaned

            # Useful for later auditing.
            output_record[
                "devanagari_ratio"
            ] = round(
                calculate_devanagari_ratio(
                    cleaned
                ),
                4
            )

            output_record[
                "nepali_word_ratio"
            ] = round(
                calculate_nepali_word_ratio(
                    cleaned
                ),
                4
            )

            output_record[
                "nepali_pattern_score"
            ] = round(
                calculate_nepali_pattern_score(
                    cleaned
                ),
                4
            )

            outfile.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                )
                + "\n"
            )

            kept_documents += 1

            output_characters += len(
                cleaned
            )

            source_stats[source][
                "kept"
            ] += 1

            source_stats[source][
                "output_characters"
            ] += len(cleaned)

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                total_documents
                % PRINT_EVERY
                == 0
            ):

                print(
                    f"Processed: "
                    f"{total_documents:,} | "
                    f"Kept: "
                    f"{kept_documents:,} | "
                    f"Removed: "
                    f"{removed_documents:,} | "
                    f"Characters removed: "
                    f"{input_characters - output_characters:,}"
                )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    character_removal = (
        input_characters
        - output_characters
    )

    removal_percentage = (
        character_removal
        / input_characters
        * 100
        if input_characters > 0
        else 0
    )

    stats = {

        "input_file": INPUT_FILE,

        "output_file": OUTPUT_FILE,

        "documents": {

            "total": total_documents,

            "kept": kept_documents,

            "removed": removed_documents,

            "invalid_json": invalid_json,

            "empty_after_cleaning":
                empty_after_cleaning,

            "too_short":
                too_short,

            "not_nepali":
                not_nepali,
        },

        "characters": {

            "input": input_characters,

            "output": output_characters,

            "removed": character_removal,

            "removal_percentage":
                removal_percentage,
        },

        "sources": {

            source: dict(values)

            for source, values
            in source_stats.items()
        }
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

    print(
        "\n" + "=" * 70
    )

    print(
        "COMBINED NEPALI CORPUS CLEANING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDocuments processed : "
        f"{total_documents:,}"
    )

    print(
        f"Documents kept      : "
        f"{kept_documents:,}"
    )

    print(
        f"Documents removed   : "
        f"{removed_documents:,}"
    )

    print(
        f"\nInvalid JSON        : "
        f"{invalid_json:,}"
    )

    print(
        f"Empty after cleaning: "
        f"{empty_after_cleaning:,}"
    )

    print(
        f"Too short           : "
        f"{too_short:,}"
    )

    print(
        f"Not Nepali          : "
        f"{not_nepali:,}"
    )

    print(
        "\nCharacter statistics:"
    )

    print(
        f"Input characters    : "
        f"{input_characters:,}"
    )

    print(
        f"Output characters   : "
        f"{output_characters:,}"
    )

    print(
        f"Characters removed  : "
        f"{character_removal:,}"
    )

    print(
        f"Removal percentage  : "
        f"{removal_percentage:.2f}%"
    )

    print(
        "\nSource statistics:"
    )

    for source, values in source_stats.items():

        input_chars = values[
            "input_characters"
        ]

        output_chars = values.get(
            "output_characters",
            0
        )

        removed_chars = (
            input_chars
            - output_chars
        )

        percentage = (
            removed_chars
            / input_chars
            * 100
            if input_chars > 0
            else 0
        )

        print(
            f"\n{source}"
        )

        print(
            f"  Documents in: "
            f"{values['input_documents']:,}"
        )

        print(
            f"  Documents kept: "
            f"{values.get('kept', 0):,}"
        )

        print(
            f"  Documents removed: "
            f"{values.get('removed', 0):,}"
        )

        print(
            f"  Not Nepali: "
            f"{values.get('not_nepali', 0):,}"
        )

        print(
            f"  Characters in: "
            f"{input_chars:,}"
        )

        print(
            f"  Characters out: "
            f"{output_chars:,}"
        )

        print(
            f"  Characters removed: "
            f"{removed_chars:,} "
            f"({percentage:.2f}%)"
        )

    print(
        f"\nCleaned corpus saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Statistics saved to: "
        f"{STATS_FILE}"
    )


if __name__ == "__main__":
    main()