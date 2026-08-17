import json
import re
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_wikitionary_raw.jsonl"

PRINT_EVERY = 10_000


# ============================================================
# REMOVE BALANCED TEMPLATES
# ============================================================

def remove_templates(text):

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

        # If unmatched, leave the remaining text unchanged
        if end is None:
            break

        text = text[:start] + text[end:]

    return text


# ============================================================
# REMOVE REFERENCES
# ============================================================

def remove_references(text):

    text = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

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

        if "|" in content:
            display = content.split("|")[-1]
        else:
            display = content

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

    text = re.sub(
        r"\[https?://[^\s\]]+\s+([^\]]+)\]",
        r"\1",
        text
    )

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
# REMOVE HTML
# ============================================================

def remove_html(text):

    return re.sub(
        r"<[^>]+>",
        "",
        text
    )


# ============================================================
# REMOVE FORMATTING
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
        r"(?m)^[ \t]*={2,}[ \t]*(.*?)[ \t]*={2,}[ \t]*$",
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

        # If unmatched, leave remaining text
        if end == -1:
            break

        text = (
            text[:start]
            + text[end + 2:]
        )

    return text


# ============================================================
# CLEAN LIST MARKERS
# ============================================================

def clean_list_markers(text):
    return re.sub(
        r"(?m)^[ \t]*[*#:;]+[ \t]*",
        "",
        text
    )


# ============================================================
# REMOVE MISC MARKUP
# ============================================================

def remove_misc_markup(text):

    text = re.sub(
        r"&nbsp;",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    text = text.replace("[", "")
    text = text.replace("]", "")

    return text


# ============================================================
# NORMALIZE WHITESPACE
# ============================================================

def normalize_whitespace(text):

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n[ \t]*\n(?:[ \t]*\n)+",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]*\n[ \t]*",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# CLEANING STAGES
# ============================================================

CLEANING_STAGES = [
    ("remove_comments", remove_comments),
    ("remove_templates", remove_templates),
    ("remove_references", remove_references),
    ("remove_tables", remove_tables),
    ("remove_files", remove_files),
    ("remove_categories", remove_categories),
    ("clean_wiki_links", clean_wiki_links),
    ("clean_external_links", clean_external_links),
    ("remove_html", remove_html),
    ("remove_formatting", remove_formatting),
    ("clean_headings", clean_headings),
    ("clean_list_markers", clean_list_markers),
    ("remove_misc_markup", remove_misc_markup),
    ("normalize_whitespace", normalize_whitespace),
]


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("WIKTIONARY CLEANING AUDIT")
    print("=" * 70)

    total_documents = 0
    invalid_json = 0

    # Character counts after every stage
    stage_characters = Counter()

    # Number of documents that become empty at each stage
    stage_empty_documents = Counter()

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

            text = record.get("text", "")

            total_documents += 1

            # ------------------------------------------------
            # Raw characters
            # ------------------------------------------------

            stage_characters["raw"] += len(text)

            # Track whether the document was already empty
            was_empty = not text.strip()

            # ------------------------------------------------
            # Apply every cleaning stage
            # ------------------------------------------------

            for stage_name, cleaning_function in CLEANING_STAGES:

                before_length = len(text)

                text = cleaning_function(text)

                after_length = len(text)

                # Total characters remaining after this stage
                stage_characters[stage_name] += after_length

                # Did this stage make the document empty?
                if (
                    not was_empty
                    and not text.strip()
                ):
                    stage_empty_documents[
                        stage_name
                    ] += 1

                    was_empty = True

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if total_documents % PRINT_EVERY == 0:

                print(
                    f"Processed: {total_documents:,} documents"
                )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("CHARACTER LOSS BY CLEANING STAGE")
    print("=" * 70)

    raw_characters = stage_characters["raw"]

    previous_characters = raw_characters

    print(
        f"\n{'Stage':30}"
        f"{'Characters Remaining':>25}"
        f"{'Removed at Stage':>20}"
        f"{'Total Removed %':>18}"
    )

    print("-" * 95)

    print(
        f"{'raw':30}"
        f"{raw_characters:>25,}"
        f"{0:>20,}"
        f"{0.0:>17.2f}%"
    )

    for stage_name, _ in CLEANING_STAGES:

        current_characters = (
            stage_characters[stage_name]
        )

        removed_at_stage = (
            previous_characters
            - current_characters
        )

        total_removed = (
            raw_characters
            - current_characters
        )

        total_removed_percentage = (
            total_removed
            / raw_characters
            * 100
            if raw_characters > 0
            else 0
        )

        print(
            f"{stage_name:30}"
            f"{current_characters:>25,}"
            f"{removed_at_stage:>20,}"
            f"{total_removed_percentage:>17.2f}%"
        )

        previous_characters = current_characters

    # ========================================================
    # EMPTY DOCUMENT DIAGNOSTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("DOCUMENTS BECOMING EMPTY AT EACH STAGE")
    print("=" * 70)

    any_empty = False

    for stage_name, _ in CLEANING_STAGES:

        count = stage_empty_documents[stage_name]

        if count > 0:

            any_empty = True

            print(
                f"{stage_name:30}: "
                f"{count:,}"
            )

    if not any_empty:

        print("No documents became empty during cleaning.")

    # ========================================================
    # SUMMARY
    # ========================================================

    final_stage = CLEANING_STAGES[-1][0]

    final_characters = (
        stage_characters[final_stage]
    )

    total_removed = (
        raw_characters
        - final_characters
    )

    removal_percentage = (
        total_removed
        / raw_characters
        * 100
        if raw_characters > 0
        else 0
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\nDocuments processed : "
        f"{total_documents:,}"
    )

    print(
        f"Invalid JSON        : "
        f"{invalid_json:,}"
    )

    print(
        f"Raw characters      : "
        f"{raw_characters:,}"
    )

    print(
        f"Final characters    : "
        f"{final_characters:,}"
    )

    print(
        f"Characters removed  : "
        f"{total_removed:,}"
    )

    print(
        f"Removal percentage  : "
        f"{removal_percentage:.2f}%"
    )


if __name__ == "__main__":
    main()