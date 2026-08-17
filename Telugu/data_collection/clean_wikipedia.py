import json
import re
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_wikisource_raw.jsonl"

OUTPUT_FILE = "telugu_wikisource_clean.jsonl"

MIN_CHARS = 100


# ============================================================
# REMOVE BALANCED TEMPLATES
# ============================================================

def remove_templates(text):
    """
    Remove balanced {{ ... }} templates.

    Supports nested templates.

    If an unmatched '{{' is found, we stop instead of truncating
    the rest of the document.
    """

    while "{{" in text:

        start = text.find("{{")

        depth = 0
        end = None

        for i in range(start, len(text) - 1):

            if text[i:i + 2] == "{{":

                depth += 1

            elif text[i:i + 2] == "}}":

                depth -= 1

                if depth == 0:

                    end = i + 2
                    break

        # ----------------------------------------------------
        # Unmatched template
        # ----------------------------------------------------

        if end is None:
            break

        # Remove the balanced template
        text = text[:start] + text[end:]

    return text


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

    # <ref name="..." />
    text = re.sub(
        r"<ref\b[^>]*/>",
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
# CLEAN INTERNAL WIKI LINKS
# ============================================================

def clean_wiki_links(text):

    def replace_link(match):

        content = match.group(1)

        # [[target|display text]]
        if "|" in content:

            display = content.split("|")[-1]

        else:

            display = content

        # Remove section fragment
        if "#" in display:

            display = display.split("#")[-1]

        return display

    return re.sub(
        r"\[\[([^\[\]]+)\]\]",
        replace_link,
        text
    )


# ============================================================
# CLEAN EXTERNAL LINKS
# ============================================================

def clean_external_links(text):

    # [https://example.com Display Text]
    text = re.sub(
        r"\[https?://[^\s\]]+\s+([^\]]+)\]",
        r"\1",
        text
    )

    # [https://example.com]
    text = re.sub(
        r"\[https?://[^\s\]]+\]",
        "",
        text
    )

    return text


# ============================================================
# REMOVE FILE / IMAGE LINKS
# ============================================================

def remove_files(text):

    pattern = (
        r"\[\[(?:File|Image|చిత్రం)"
        r":[^\[\]]+\]\]"
    )

    return re.sub(
        pattern,
        "",
        text,
        flags=re.IGNORECASE
    )


# ============================================================
# REMOVE CATEGORIES
# ============================================================

def remove_categories(text):

    pattern = (
        r"\[\[(?:Category|వర్గం)"
        r":[^\[\]]+\]\]"
    )

    return re.sub(
        pattern,
        "",
        text,
        flags=re.IGNORECASE
    )


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
# REMOVE BOLD / ITALIC MARKUP
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
# REMOVE TABLES
# ============================================================

def remove_tables(text):

    while "{|" in text:

        start = text.find("{|")

        end = text.find("|}", start)

        # If unmatched, don't destroy the remainder
        if end == -1:
            break

        text = (
            text[:start]
            +
            text[end + 2:]
        )

    return text


# ============================================================
# CLEAN LIST MARKERS
# ============================================================

def clean_list_markers(text):

    return re.sub(
        r"(?m)^\s*[*#:;]+\s*",
        "",
        text
    )


# ============================================================
# REMOVE MISCELLANEOUS MARKUP
# ============================================================

def remove_misc_markup(text):

    # HTML entities
    text = re.sub(
        r"&nbsp;",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remaining URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # Remove remaining brackets
    text = text.replace("[", "")
    text = text.replace("]", "")

    return text


# ============================================================
# NORMALIZE WHITESPACE
# ============================================================

def normalize_whitespace(text):

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse spaces and tabs
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

    # Remove whitespace around newlines
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# MAIN CLEANING PIPELINE
# ============================================================

def clean_text(text):

    # 1. Comments
    text = remove_comments(text)

    # 2. Templates
    text = remove_templates(text)

    # 3. References
    text = remove_references(text)

    # 4. Tables
    text = remove_tables(text)

    # 5. Files/images
    text = remove_files(text)

    # 6. Categories
    text = remove_categories(text)

    # 7. Internal links
    text = clean_wiki_links(text)

    # 8. External links
    text = clean_external_links(text)

    # 9. HTML
    text = remove_html(text)

    # 10. Formatting
    text = remove_formatting(text)

    # 11. Headings
    text = clean_headings(text)

    # 12. Lists
    text = clean_list_markers(text)

    # 13. Miscellaneous markup
    text = remove_misc_markup(text)

    # 14. Whitespace
    text = normalize_whitespace(text)

    return text


# ============================================================
# PROCESS FILE
# ============================================================

def process_file():

    total = 0
    kept = 0
    removed = 0

    # Removal diagnostics
    removal_reasons = Counter()

    # Namespace diagnostics
    kept_by_namespace = Counter()
    removed_by_namespace = Counter()
    total_by_namespace = Counter()

    # Character statistics
    input_chars = 0
    output_chars = 0


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

            total += 1

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                removed += 1
                removal_reasons["invalid_json"] += 1

                continue


            title = record.get("title", "")
            text = record.get("text", "")
            namespace = record.get(
                "namespace",
                "unknown"
            )

            namespace = str(namespace)

            total_by_namespace[
                namespace
            ] += 1


            # ------------------------------------------------
            # Input characters
            # ------------------------------------------------

            input_chars += len(text)


            # ------------------------------------------------
            # Clean
            # ------------------------------------------------

            cleaned = clean_text(text)


            # ------------------------------------------------
            # Empty after cleaning
            # ------------------------------------------------

            if not cleaned.strip():

                removed += 1

                removal_reasons[
                    "empty_after_cleaning"
                ] += 1

                removed_by_namespace[
                    namespace
                ] += 1

                continue


            # ------------------------------------------------
            # Too short
            # ------------------------------------------------

            if len(cleaned) < MIN_CHARS:

                removed += 1

                removal_reasons[
                    "too_short"
                ] += 1

                removed_by_namespace[
                    namespace
                ] += 1

                continue


            # ------------------------------------------------
            # Save cleaned record
            # ------------------------------------------------

            output_record = {

                "id": record.get("id"),

                "title": title,

                "text": cleaned,

                "namespace": (
                    int(namespace)
                    if namespace.isdigit()
                    else namespace
                ),

                "source": "telugu_wikisource"
            }


            outfile.write(

                json.dumps(
                    output_record,
                    ensure_ascii=False
                )

                + "\n"
            )


            kept += 1

            kept_by_namespace[
                namespace
            ] += 1

            output_chars += len(cleaned)


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if total % 5000 == 0:

                print(

                    f"Processed: {total:,} | "
                    f"Kept: {kept:,} | "
                    f"Removed: {removed:,} | "
                    f"Main: {kept_by_namespace['0']:,} | "
                    f"Page: {kept_by_namespace['104']:,}"
                )


    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 65)
    print("WIKISOURCE CLEANING COMPLETE")
    print("=" * 65)


    print(f"\nTotal documents : {total:,}")
    print(f"Kept            : {kept:,}")
    print(f"Removed         : {removed:,}")


    # --------------------------------------------------------
    # REMOVAL REASONS
    # --------------------------------------------------------

    print("\nRemoval reasons:")

    for reason, count in (
        removal_reasons.most_common()
    ):

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(

            f"{reason:30} "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )


    # --------------------------------------------------------
    # TOTAL BY NAMESPACE
    # --------------------------------------------------------

    print("\nTotal by namespace:")

    for namespace, count in (
        total_by_namespace.most_common()
    ):

        print(

            f"Namespace {namespace:>7}: "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # KEPT BY NAMESPACE
    # --------------------------------------------------------

    print("\nKept by namespace:")

    for namespace, count in (
        kept_by_namespace.most_common()
    ):

        total_ns = total_by_namespace[
            namespace
        ]

        percentage = (
            count / total_ns * 100
            if total_ns > 0
            else 0
        )

        print(

            f"Namespace {namespace:>7}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )


    # --------------------------------------------------------
    # REMOVED BY NAMESPACE
    # --------------------------------------------------------

    print("\nRemoved by namespace:")

    for namespace, count in (
        removed_by_namespace.most_common()
    ):

        total_ns = total_by_namespace[
            namespace
        ]

        percentage = (
            count / total_ns * 100
            if total_ns > 0
            else 0
        )

        print(

            f"Namespace {namespace:>7}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )


    # --------------------------------------------------------
    # CHARACTER STATISTICS
    # --------------------------------------------------------

    print("\nCharacter statistics:")

    print(
        f"Input characters : "
        f"{input_chars:,}"
    )

    print(
        f"Output characters: "
        f"{output_chars:,}"
    )

    print(
        f"Characters removed: "
        f"{input_chars - output_chars:,}"
    )


    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print(
        f"\nOutput saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_file()