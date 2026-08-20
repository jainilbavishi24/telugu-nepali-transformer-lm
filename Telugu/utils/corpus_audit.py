import json
import re
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Telugu/data/telugu_combined_corpus.jsonl"

PRINT_EVERY = 10_000

MAX_EXAMPLES_PER_PATTERN = 5

# Number of most common characters to display
TOP_CHARACTERS = 100


# ============================================================
# SUSPICIOUS PATTERNS
# ============================================================

PATTERNS = {

    # --------------------------------------------------------
    # MediaWiki templates
    # --------------------------------------------------------

    "template_open": r"\{\{",
    "template_close": r"\}\}",

    # --------------------------------------------------------
    # Wiki links
    # --------------------------------------------------------

    "wiki_link_open": r"\[\[",
    "wiki_link_close": r"\]\]",

    # --------------------------------------------------------
    # Wiki tables
    # --------------------------------------------------------

    "table_open": r"\{\|",
    "table_close": r"\|\}",

    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    "ref_tag": r"<ref\b[^>]*>",
    "ref_close": r"</ref\s*>",

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    "html_tag": r"<[^>]+>",

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    "url": r"https?://",

    # --------------------------------------------------------
    # HTML entities
    # --------------------------------------------------------

    "html_entity": r"&[a-zA-Z]+;",

    # --------------------------------------------------------
    # Remaining formatting
    # --------------------------------------------------------

    "triple_apostrophe": r"'''",
    "double_apostrophe": r"''",

    # --------------------------------------------------------
    # Backslashes / escaped sequences
    # --------------------------------------------------------

    "backslash": r"\\",

    # --------------------------------------------------------
    # Other markup-like patterns
    # --------------------------------------------------------

    "multiple_equals": r"={2,}",
    "multiple_asterisk": r"\*{2,}",
}


# ============================================================
# INDIVIDUAL CHARACTERS TO AUDIT
# ============================================================

SUSPICIOUS_CHARACTERS = [
    "{",
    "}",
    "[",
    "]",
    "|",
    "<",
    ">",
    "\\",
    "`",
    "*",
    "#",
    "=",
    "_",
]


# ============================================================
# HELPERS
# ============================================================

def get_preview(text, start, window=250):
    """
    Return a small readable preview around a suspicious match.
    """

    left = max(0, start - 80)
    right = min(len(text), start + window)

    preview = text[left:right]

    preview = preview.replace("\n", "\\n")

    return preview


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_corpus():

    print("=" * 70)
    print("TELUGU CORPUS MARKUP / CHARACTER AUDIT")
    print("=" * 70)

    total_documents = 0
    total_characters = 0

    invalid_json = 0
    empty_documents = 0

    # --------------------------------------------------------
    # Global character frequencies
    # --------------------------------------------------------

    character_counter = Counter()

    # --------------------------------------------------------
    # Suspicious character counts
    # --------------------------------------------------------

    suspicious_character_counts = Counter()

    suspicious_character_documents = Counter()

    # --------------------------------------------------------
    # Pattern statistics
    # --------------------------------------------------------

    pattern_occurrences = Counter()

    pattern_documents = Counter()

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    examples = defaultdict(list)

    # --------------------------------------------------------
    # Source-level statistics
    # --------------------------------------------------------

    source_documents = Counter()

    source_characters = Counter()

    source_pattern_documents = defaultdict(Counter)

    source_pattern_occurrences = defaultdict(Counter)

    print("\nAuditing corpus...\n")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for line_number, line in enumerate(
            infile,
            start=1
        ):

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                invalid_json += 1
                continue

            text = record.get("text", "")

            if not isinstance(text, str):

                continue

            if not text.strip():

                empty_documents += 1
                continue

            source = record.get(
                "source",
                "unknown"
            )

            # ------------------------------------------------
            # Global statistics
            # ------------------------------------------------

            total_documents += 1
            total_characters += len(text)

            source_documents[source] += 1

            source_characters[source] += len(text)

            # ------------------------------------------------
            # Character frequencies
            # ------------------------------------------------

            character_counter.update(text)

            # ------------------------------------------------
            # Suspicious individual characters
            # ------------------------------------------------

            for char in SUSPICIOUS_CHARACTERS:

                count = text.count(char)

                if count > 0:

                    suspicious_character_counts[char] += count

                    suspicious_character_documents[
                        char
                    ] += 1

            # ------------------------------------------------
            # Pattern matching
            # ------------------------------------------------

            for pattern_name, pattern in PATTERNS.items():

                matches = list(
                    re.finditer(
                        pattern,
                        text,
                        flags=re.IGNORECASE
                    )
                )

                if not matches:
                    continue

                num_matches = len(matches)

                pattern_occurrences[
                    pattern_name
                ] += num_matches

                pattern_documents[
                    pattern_name
                ] += 1

                source_pattern_documents[
                    source
                ][pattern_name] += 1

                source_pattern_occurrences[
                    source
                ][pattern_name] += num_matches

                # --------------------------------------------
                # Store a few examples
                # --------------------------------------------

                if (
                    len(examples[pattern_name])
                    < MAX_EXAMPLES_PER_PATTERN
                ):

                    match = matches[0]

                    examples[pattern_name].append(
                        {
                            "line": line_number,
                            "id": record.get("id"),
                            "title": record.get(
                                "title",
                                ""
                            ),
                            "source": source,
                            "preview": get_preview(
                                text,
                                match.start()
                            )
                        }
                    )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if total_documents % PRINT_EVERY == 0:

                print(
                    f"Processed: "
                    f"{total_documents:,} documents | "
                    f"Characters: "
                    f"{total_characters:,}"
                )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("CORPUS SUMMARY")
    print("=" * 70)

    print(
        f"\nDocuments processed : "
        f"{total_documents:,}"
    )

    print(
        f"Characters processed: "
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
    # SOURCE DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("SOURCE DISTRIBUTION")
    print("=" * 70)

    for source, count in (
        source_documents.most_common()
    ):

        chars = source_characters[source]

        percentage = (
            count / total_documents * 100
            if total_documents > 0
            else 0
        )

        print(
            f"\n{source}"
        )

        print(
            f"  Documents : "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

        print(
            f"  Characters: "
            f"{chars:,}"
        )

    # ========================================================
    # SUSPICIOUS CHARACTER REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("SUSPICIOUS CHARACTER STATISTICS")
    print("=" * 70)

    print()

    print(
        f"{'Character':<15}"
        f"{'Occurrences':>15}"
        f"{'Documents':>15}"
        f"{'% Documents':>15}"
    )

    print("-" * 60)

    for char in SUSPICIOUS_CHARACTERS:

        occurrences = (
            suspicious_character_counts[char]
        )

        documents = (
            suspicious_character_documents[char]
        )

        percentage = (
            documents / total_documents * 100
            if total_documents > 0
            else 0
        )

        printable_char = repr(char)

        print(
            f"{printable_char:<15}"
            f"{occurrences:>15,}"
            f"{documents:>15,}"
            f"{percentage:>14.2f}%"
        )

    # ========================================================
    # PATTERN STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("LEFTOVER MARKUP / NOISE PATTERNS")
    print("=" * 70)

    print()

    print(
        f"{'Pattern':<30}"
        f"{'Occurrences':>15}"
        f"{'Documents':>15}"
        f"{'% Documents':>15}"
    )

    print("-" * 75)

    for pattern_name in PATTERNS:

        occurrences = (
            pattern_occurrences[pattern_name]
        )

        documents = (
            pattern_documents[pattern_name]
        )

        percentage = (
            documents / total_documents * 100
            if total_documents > 0
            else 0
        )

        print(
            f"{pattern_name:<30}"
            f"{occurrences:>15,}"
            f"{documents:>15,}"
            f"{percentage:>14.2f}%"
        )

    # ========================================================
    # TOP CHARACTERS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        f"TOP {TOP_CHARACTERS} MOST COMMON CHARACTERS"
    )
    print("=" * 70)

    for char, count in (
        character_counter.most_common(
            TOP_CHARACTERS
        )
    ):

        if char == "\n":
            printable = "\\n"

        elif char == "\t":
            printable = "\\t"

        elif char == " ":
            printable = "SPACE"

        else:
            printable = char

        print(
            f"{repr(printable):<20} "
            f"{count:,}"
        )

    # ========================================================
    # SOURCE × PATTERN DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("SUSPICIOUS PATTERNS BY SOURCE")
    print("=" * 70)

    for source in source_documents:

        print(f"\nSource: {source}")

        found_any = False

        for pattern_name in PATTERNS:

            documents = (
                source_pattern_documents[
                    source
                ][pattern_name]
            )

            occurrences = (
                source_pattern_occurrences[
                    source
                ][pattern_name]
            )

            if documents > 0:

                found_any = True

                print(
                    f"  {pattern_name:<30} "
                    f"Docs: {documents:>8,} | "
                    f"Occurrences: "
                    f"{occurrences:>10,}"
                )

        if not found_any:

            print(
                "  No suspicious patterns found."
            )

    # ========================================================
    # EXAMPLES
    # ========================================================

    print("\n" + "=" * 70)
    print("EXAMPLES OF SUSPICIOUS CONTENT")
    print("=" * 70)

    for pattern_name in PATTERNS:

        print(
            f"\n{'-' * 70}"
        )

        print(
            f"PATTERN: {pattern_name}"
        )

        print(
            f"{'-' * 70}"
        )

        if not examples[pattern_name]:

            print(
                "No examples found."
            )

            continue

        for i, example in enumerate(
            examples[pattern_name],
            start=1
        ):

            print(
                f"\nExample {i}"
            )

            print(
                f"Line   : "
                f"{example['line']}"
            )

            print(
                f"ID     : "
                f"{example['id']}"
            )

            print(
                f"Title  : "
                f"{example['title']}"
            )

            print(
                f"Source : "
                f"{example['source']}"
            )

            print(
                f"Preview:\n"
                f"{example['preview']}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_data = {

        "input_file": INPUT_FILE,

        "summary": {
            "documents": total_documents,
            "characters": total_characters,
            "invalid_json": invalid_json,
            "empty_documents": empty_documents
        },

        "source_distribution": {
            source: {
                "documents": source_documents[source],
                "characters": source_characters[source]
            }
            for source in source_documents
        },

        "suspicious_characters": {
            char: {
                "occurrences": (
                    suspicious_character_counts[char]
                ),
                "documents": (
                    suspicious_character_documents[char]
                )
            }
            for char in SUSPICIOUS_CHARACTERS
        },

        "patterns": {
            pattern_name: {
                "occurrences": (
                    pattern_occurrences[pattern_name]
                ),
                "documents": (
                    pattern_documents[pattern_name]
                )
            }
            for pattern_name in PATTERNS
        },

        "source_pattern_statistics": {
            source: {
                pattern_name: {
                    "documents": (
                        source_pattern_documents[
                            source
                        ][pattern_name]
                    ),
                    "occurrences": (
                        source_pattern_occurrences[
                            source
                        ][pattern_name]
                    )
                }
                for pattern_name in PATTERNS
            }
            for source in source_documents
        },

        "examples": dict(examples)
    }

    stats_file = "corpus_markup_audit.json"

    with open(
        stats_file,
        "w",
        encoding="utf-8"
    ) as outfile:

        json.dump(
            output_data,
            outfile,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to: "
        f"{stats_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    audit_corpus()