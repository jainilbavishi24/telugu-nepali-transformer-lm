import bz2
import json
import xml.etree.ElementTree as ET
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

DUMP_FILE = (
    "/Users/rithvik/Downloads/"
    "tewikisource-latest-pages-articles.xml.bz2"
)

OUTPUT_FILE = "telugu_wikisource_raw.jsonl"


# Keep:
# 0   -> Main namespace
# 104 -> Page namespace
KEEP_NAMESPACES = {"0", "104"}


# ============================================================
# PARSER
# ============================================================

def parse_wikisource():

    total_pages = 0
    saved_pages = 0

    namespace_counts = Counter()
    saved_by_namespace = Counter()
    skipped_by_reason = Counter()

    characters_by_namespace = Counter()


    with bz2.open(
        DUMP_FILE,
        "rb"
    ) as f, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as out:

        for event, elem in ET.iterparse(
            f,
            events=("end",)
        ):

            # We only process complete <page> elements.
            if not elem.tag.endswith("page"):
                continue

            total_pages += 1


            # ------------------------------------------------
            # EXTRACT PAGE DATA
            # ------------------------------------------------

            title = elem.findtext(
                "./{*}title"
            )

            page_id = elem.findtext(
                "./{*}id"
            )

            namespace = elem.findtext(
                "./{*}ns"
            )

            redirect = elem.find(
                "./{*}redirect"
            )

            text = elem.findtext(
                "./{*}revision/{*}text"
            )


            # ------------------------------------------------
            # NAMESPACE STATISTICS
            # ------------------------------------------------

            namespace_counts[namespace] += 1


            # ------------------------------------------------
            # SKIP UNWANTED NAMESPACES
            # ------------------------------------------------

            if namespace not in KEEP_NAMESPACES:

                skipped_by_reason[
                    "other_namespace"
                ] += 1

                elem.clear()
                continue


            # ------------------------------------------------
            # SKIP REDIRECTS
            # ------------------------------------------------

            if redirect is not None:

                skipped_by_reason[
                    "redirect"
                ] += 1

                elem.clear()
                continue


            # ------------------------------------------------
            # SKIP MISSING TEXT
            # ------------------------------------------------

            if not text:

                skipped_by_reason[
                    "missing_text"
                ] += 1

                elem.clear()
                continue


            # ------------------------------------------------
            # SKIP WHITESPACE-ONLY TEXT
            # ------------------------------------------------

            if not text.strip():

                skipped_by_reason[
                    "empty_text"
                ] += 1

                elem.clear()
                continue


            # ------------------------------------------------
            # SAVE RECORD
            # ------------------------------------------------

            record = {
                "id": page_id,
                "title": title,
                "text": text,
                "namespace": int(namespace),
                "source": "telugu_wikisource"
            }


            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


            saved_pages += 1

            saved_by_namespace[
                namespace
            ] += 1

            characters_by_namespace[
                namespace
            ] += len(text)


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if total_pages % 10_000 == 0:

                print(
                    f"Processed: {total_pages:,} | "
                    f"Saved: {saved_pages:,} | "
                    f"Main: {saved_by_namespace['0']:,} | "
                    f"Page: {saved_by_namespace['104']:,}"
                )


            # Free memory.
            elem.clear()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    skipped_pages = sum(
        skipped_by_reason.values()
    )


    print("\n" + "=" * 60)
    print("WIKISOURCE PARSING COMPLETE")
    print("=" * 60)


    print(
        f"\nTotal pages processed : "
        f"{total_pages:,}"
    )

    print(
        f"Pages saved           : "
        f"{saved_pages:,}"
    )

    print(
        f"Pages skipped         : "
        f"{skipped_pages:,}"
    )


    # --------------------------------------------------------
    # SAVED BY NAMESPACE
    # --------------------------------------------------------

    print("\nSaved by namespace:")

    print(
        f"Namespace 0 (Main)    : "
        f"{saved_by_namespace['0']:,}"
    )

    print(
        f"Namespace 104 (Page)  : "
        f"{saved_by_namespace['104']:,}"
    )


    # --------------------------------------------------------
    # CHARACTER STATISTICS
    # --------------------------------------------------------

    print("\nCharacter statistics:")

    print(
        f"Main namespace chars  : "
        f"{characters_by_namespace['0']:,}"
    )

    print(
        f"Page namespace chars  : "
        f"{characters_by_namespace['104']:,}"
    )

    total_characters = (
        characters_by_namespace["0"]
        + characters_by_namespace["104"]
    )

    print(
        f"Total characters      : "
        f"{total_characters:,}"
    )


    # --------------------------------------------------------
    # SKIP REASONS
    # --------------------------------------------------------

    print("\nSkip reasons:")

    for reason, count in (
        skipped_by_reason.most_common()
    ):

        print(
            f"{reason:25} "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # ORIGINAL NAMESPACE DISTRIBUTION
    # --------------------------------------------------------

    print("\nFull namespace distribution:")

    for namespace, count in (
        namespace_counts.most_common()
    ):

        print(
            f"Namespace {namespace:>5}: "
            f"{count:,}"
        )


    print(
        f"\nOutput saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parse_wikisource()