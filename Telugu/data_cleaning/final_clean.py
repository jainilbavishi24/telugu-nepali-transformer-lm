import json
import re
import unicodedata
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Telugu/data/telugu_combined_corpus_cleaned.jsonl"

OUTPUT_FILE = "Telugu/data/telugu_combined_corpus_clean.jsonl"

STATS_FILE = "Telugu/data/telugu_combined_corpus_cleaning_stats.json"

PRINT_EVERY = 10_000


# ============================================================
# REMOVE BALANCED TEMPLATES
# ============================================================

def remove_templates(text):
    """
    Remove balanced {{ ... }} templates.

    Supports nested templates.

    If an unmatched '{{' is found, leave the remainder
    unchanged rather than accidentally deleting text.
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

            # ------------------------------------------------
            # Balanced template found
            # ------------------------------------------------

            if depth == 0:
                continue

            # ------------------------------------------------
            # Unmatched {{
            #
            # Preserve it instead of deleting everything after it.
            # ------------------------------------------------

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

    If a table opening has no closing |}, preserve
    the remainder instead of destroying the document.
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

            # Preserve unmatched content
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
        r"(?:File|Image|చిత్రం)"
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
        r"(?:Category|వర్గం)"
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
            ->
        target

        [[target|display text]]
            ->
        display text

        [[target#section]]
            ->
        target

        [[target#section|display]]
            ->
        display
    """

    def replace_link(match):

        content = match.group(1).strip()

        if "|" in content:

            parts = content.split("|")

            # Last component is generally display text
            display = parts[-1].strip()

            return display

        # Remove fragment

        if "#" in content:

            content = content.split("#", 1)[0].strip()

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
    """
    Convert:

        == Heading ==

    into:

        Heading
    """

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
    Convert:

        * item
        # item
        : item
        ; item

    into:

        item
    """

    return re.sub(
        r"(?m)^\s*[*#:;]+\s*",
        "",
        text
    )


# ============================================================
# REMOVE LEFTOVER MEDIAWIKI MARKUP
# ============================================================

def remove_leftover_markup(text):

    # HTML entities

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

    # Remaining URLs

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

    Only remove obvious leftover MediaWiki markup.

    Do NOT remove normal punctuation such as:
        ( ) , . ! ? : ; - "
    """

    # Unmatched template delimiters

    text = text.replace("{{", " ")
    text = text.replace("}}", " ")

    # Unmatched wiki link delimiters

    text = text.replace("[[", "")
    text = text.replace("]]", "")

    # Table markers

    text = text.replace("{|", " ")
    text = text.replace("|}", " ")

    return text


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

    # Normalize line endings

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Collapse spaces/tabs

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Collapse excessive blank lines

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    # Remove spaces around newlines

    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    return text.strip()


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

    text = normalize_whitespace(text)

    return text


# ============================================================
# PROCESS CORPUS
# ============================================================

def main():

    print("=" * 70)
    print("CLEANING COMBINED TELUGU CORPUS")
    print("=" * 70)

    total_documents = 0
    kept_documents = 0
    removed_documents = 0

    invalid_json = 0
    empty_after_cleaning = 0

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

            source_stats[source]["input_documents"] += 1
            source_stats[source]["input_characters"] += len(text)

            input_characters += len(text)

            # ------------------------------------------------
            # CLEAN
            # ------------------------------------------------

            cleaned_text = clean_text(text)

            # ------------------------------------------------
            # EMPTY
            # ------------------------------------------------

            if not cleaned_text:

                empty_after_cleaning += 1
                removed_documents += 1

                source_stats[source]["removed"] += 1

                continue

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            output_record = record.copy()

            output_record["text"] = cleaned_text

            outfile.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                )
                + "\n"
            )

            kept_documents += 1

            output_characters += len(
                cleaned_text
            )

            source_stats[source]["kept"] += 1

            source_stats[source][
                "output_characters"
            ] += len(cleaned_text)

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if total_documents % PRINT_EVERY == 0:

                print(
                    f"Processed: {total_documents:,} | "
                    f"Kept: {kept_documents:,} | "
                    f"Removed: {removed_documents:,} | "
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
                empty_after_cleaning
        },

        "characters": {

            "input": input_characters,

            "output": output_characters,

            "removed": character_removal,

            "removal_percentage":
                removal_percentage
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

    print("\n" + "=" * 70)
    print("COMBINED CORPUS CLEANING COMPLETE")
    print("=" * 70)

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

    print("\nCharacter statistics:")

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

    # ========================================================
    # SOURCE STATISTICS
    # ========================================================

    print("\nSource statistics:")

    for source, values in source_stats.items():

        input_chars = (
            values["input_characters"]
        )

        output_chars = (
            values["output_characters"]
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

        print(f"\n{source}")

        print(
            f"  Documents in : "
            f"{values['input_documents']:,}"
        )

        print(
            f"  Documents kept: "
            f"{values['kept']:,}"
        )

        print(
            f"  Documents removed: "
            f"{values['removed']:,}"
        )

        print(
            f"  Characters in : "
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