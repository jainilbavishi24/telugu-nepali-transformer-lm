import bz2
import json
import xml.etree.ElementTree as ET
from collections import Counter


DUMP_FILE = (
    "/Users/rithvik/Downloads/tewiktionary-latest-pages-articles.xml.bz2"
)

OUTPUT_FILE = "telugu_wikitionary_raw.jsonl"


def parse_wikisource():

    article_count = 0

    skip_reasons = Counter()
    namespace_counts = Counter()

    total_pages = 0

    with bz2.open(DUMP_FILE, "rb") as f, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as out:

        for event, elem in ET.iterparse(
            f,
            events=("end",)
        ):

            if not elem.tag.endswith("page"):
                continue

            total_pages += 1

            title = elem.findtext("./{*}title")
            page_id = elem.findtext("./{*}id")
            namespace = elem.findtext("./{*}ns")

            redirect = elem.find("./{*}redirect")

            text = elem.findtext(
                "./{*}revision/{*}text"
            )

            # --------------------------------------------
            # Track namespaces
            # --------------------------------------------

            namespace_counts[namespace] += 1

            # --------------------------------------------
            # Skip non-main namespace
            # --------------------------------------------

            if namespace != "0":

                skip_reasons["non_main_namespace"] += 1

                elem.clear()
                continue

            # --------------------------------------------
            # Skip redirects
            # --------------------------------------------

            if redirect is not None:

                skip_reasons["redirect"] += 1

                elem.clear()
                continue

            # --------------------------------------------
            # Skip missing text
            # --------------------------------------------

            if not text:

                skip_reasons["missing_text"] += 1

                elem.clear()
                continue

            # --------------------------------------------
            # Skip whitespace-only text
            # --------------------------------------------

            if not text.strip():

                skip_reasons["empty_text"] += 1

                elem.clear()
                continue

            # --------------------------------------------
            # Save page
            # --------------------------------------------

            record = {
                "id": page_id,
                "title": title,
                "text": text
            }

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            article_count += 1

            # --------------------------------------------
            # Progress
            # --------------------------------------------

            if total_pages % 10_000 == 0:

                print(
                    f"Pages processed: {total_pages:,} | "
                    f"Saved: {article_count:,}"
                )

            elem.clear()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    skipped_count = sum(
        skip_reasons.values()
    )

    print("\n" + "=" * 60)
    print("WIKISOURCE PARSING COMPLETE")
    print("=" * 60)

    print(f"\nTotal pages processed : {total_pages:,}")
    print(f"Pages saved           : {article_count:,}")
    print(f"Pages skipped         : {skipped_count:,}")

    print("\nSkip reasons:")

    for reason, count in skip_reasons.most_common():

        percentage = (
            count / total_pages * 100
        )

        print(
            f"{reason:25} "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    print("\nNamespace distribution:")

    for namespace, count in namespace_counts.most_common():

        percentage = (
            count / total_pages * 100
        )

        print(
            f"Namespace {namespace:>5}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    print("\nOutput:")
    print(f"Raw corpus saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parse_wikisource()