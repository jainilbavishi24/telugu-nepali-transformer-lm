import json
import re


INPUT_FILE = "telugu_wikipedia_raw.jsonl"
OUTPUT_FILE = "telugu_wikipedia_clean.jsonl"


# ------------------------------------------------------------
# Remove balanced Wikipedia templates
# Example:
# {{Infobox person
# | name = ...
# | birth_date = ...
# }}
# ------------------------------------------------------------

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

        if end is None:
            # Unmatched template
            text = text[:start]
            break

        text = text[:start] + text[end:]

    return text


# ------------------------------------------------------------
# Remove <ref>...</ref>
# Also handles self-closing references:
# <ref name="something" />
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Remove HTML comments
# <!-- some comment -->
# ------------------------------------------------------------

def remove_comments(text):
    return re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL
    )


# ------------------------------------------------------------
# Convert Wikipedia links
#
# [[Hyderabad]]
#       -> Hyderabad
#
# [[Hyderabad|హైదరాబాద్]]
#       -> హైదరాబాద్
#
# [[Telugu language#History|చరిత్ర]]
#       -> చరిత్ర
# ------------------------------------------------------------

def clean_wiki_links(text):

    def replace_link(match):
        content = match.group(1)

        # [[target|display text]]
        if "|" in content:
            display = content.split("|")[-1]
        else:
            display = content

        # Remove section fragment
        display = display.split("#")[-1]

        return display

    text = re.sub(
        r"\[\[([^\[\]]+)\]\]",
        replace_link,
        text
    )

    return text


# ------------------------------------------------------------
# External links
#
# [https://example.com Example website]
#       -> Example website
#
# [https://example.com]
#       -> ""
# ------------------------------------------------------------

def clean_external_links(text):

    # [URL display text]
    text = re.sub(
        r"\[https?://[^\s\]]+\s+([^\]]+)\]",
        r"\1",
        text
    )

    # [URL]
    text = re.sub(
        r"\[https?://[^\s\]]+\]",
        "",
        text
    )

    return text


# ------------------------------------------------------------
# Remove file/image links
#
# [[File:Something.jpg|thumb|caption]]
# [[చిత్రం:Something.jpg]]
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Remove categories
#
# [[Category:Telugu people]]
# [[వర్గం:...]]
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Remove HTML tags
#
# <div>
# </div>
# <br>
# <small>
# etc.
# ------------------------------------------------------------

def remove_html(text):
    return re.sub(
        r"<[^>]+>",
        "",
        text
    )


# ------------------------------------------------------------
# Remove common Wikipedia formatting
#
# '''bold''' -> bold
# ''italic'' -> italic
# '''''bold italic''''' -> bold italic
# ------------------------------------------------------------

def remove_formatting(text):

    # Bold + italic
    text = text.replace("'''''", "")

    # Bold
    text = text.replace("'''", "")

    # Italic
    text = text.replace("''", "")

    return text


# ------------------------------------------------------------
# Clean headings
#
# == History ==
# === Early history ===
#
# We keep the heading text but remove the "=" characters.
# ------------------------------------------------------------

def clean_headings(text):
    return re.sub(
        r"(?m)^\s*=+\s*(.*?)\s*=+\s*$",
        r"\1",
        text
    )


# ------------------------------------------------------------
# Remove Wikipedia table markup
#
# {| class="wikitable"
# |-
# | something
# |}
#
# This is intentionally simple. Tables are usually noisy
# for language-model pretraining.
# ------------------------------------------------------------

def remove_tables(text):
    while "{|" in text:

        start = text.find("{|")
        end = text.find("|}", start)

        if end == -1:
            text = text[:start]
            break

        text = text[:start] + text[end + 2:]

    return text


# ------------------------------------------------------------
# Remove list markup
#
# * item
# ** sub-item
# # numbered item
# ; definition
# : description
#
# We retain the actual text.
# ------------------------------------------------------------

def clean_list_markers(text):
    return re.sub(
        r"(?m)^\s*[*#:;]+\s*",
        "",
        text
    )


# ------------------------------------------------------------
# Remove leftover Wikipedia-specific constructs
# ------------------------------------------------------------

def remove_misc_markup(text):

    # HTML entities
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)

    # Remaining external URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # Remaining square brackets
    text = text.replace("[", "")
    text = text.replace("]", "")

    return text


# ------------------------------------------------------------
# Normalize whitespace
# ------------------------------------------------------------

def normalize_whitespace(text):

    # Normalize different newline styles
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace
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


# ------------------------------------------------------------
# Main cleaning pipeline
# ------------------------------------------------------------

def clean_text(text):

    # 1. Remove comments
    text = remove_comments(text)

    # 2. Remove templates / infoboxes
    text = remove_templates(text)

    # 3. Remove references
    text = remove_references(text)

    # 4. Remove tables
    text = remove_tables(text)

    # 5. Remove images/files
    text = remove_files(text)

    # 6. Remove categories
    text = remove_categories(text)

    # 7. Convert internal Wikipedia links to plain text
    text = clean_wiki_links(text)

    # 8. Convert external links to display text
    text = clean_external_links(text)

    # 9. Remove HTML
    text = remove_html(text)

    # 10. Remove bold / italic markup
    text = remove_formatting(text)

    # 11. Clean headings
    text = clean_headings(text)

    # 12. Remove list markers
    text = clean_list_markers(text)

    # 13. Remove miscellaneous leftover markup
    text = remove_misc_markup(text)

    # 14. Normalize whitespace
    text = normalize_whitespace(text)

    return text


# ------------------------------------------------------------
# Process JSONL
# ------------------------------------------------------------

def process_file():

    total = 0
    kept = 0
    removed = 0

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

            record = json.loads(line)

            title = record["title"]
            text = record["text"]

            cleaned = clean_text(text)

            # Remove articles that become empty after cleaning
            if not cleaned.strip():
                removed += 1
                continue

            # Remove extremely short articles
            # You can change this threshold later.
            if len(cleaned) < 100:
                removed += 1
                continue

            output_record = {
                "id": record["id"],
                "title": title,
                "text": cleaned,
                "source": "telugu_wikipedia"
            }

            outfile.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                ) + "\n"
            )

            kept += 1

            if kept % 1000 == 0:
                print(
                    f"Processed: {total:,} | "
                    f"Kept: {kept:,} | "
                    f"Removed: {removed:,}"
                )

    print("\nFinished!")
    print(f"Total articles : {total:,}")
    print(f"Kept           : {kept:,}")
    print(f"Removed        : {removed:,}")


if __name__ == "__main__":
    process_file()