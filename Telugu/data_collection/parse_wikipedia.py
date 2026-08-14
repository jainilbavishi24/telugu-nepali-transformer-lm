import bz2
import json
import xml.etree.ElementTree as ET


DUMP_FILE = "/Users/rithvik/Downloads/tewiki-latest-pages-articles.xml.bz2"
OUTPUT_FILE = "telugu_wikipedia_raw.jsonl"


def parse_wikipedia():
    article_count = 0
    skipped_count = 0

    with bz2.open(DUMP_FILE, "rb") as f, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as out:

        for event, elem in ET.iterparse(f, events=("end",)):

            if not elem.tag.endswith("page"):
                continue

            title = elem.findtext("./{*}title")
            page_id = elem.findtext("./{*}id")

            namespace = elem.findtext("./{*}ns")

            redirect = elem.find("./{*}redirect")

            text = elem.findtext("./{*}revision/{*}text")

            if namespace != "0":
                skipped_count += 1
                elem.clear()
                continue

            if redirect is not None:
                skipped_count += 1
                elem.clear()
                continue

            if not text or not text.strip():
                skipped_count += 1
                elem.clear()
                continue


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


            if article_count % 1000 == 0:
                print(
                    f"Articles parsed: {article_count:,} | "
                    f"Skipped: {skipped_count:,}"
                )


            elem.clear()

    print("\nFinished!")
    print(f"Articles saved : {article_count:,}")
    print(f"Pages skipped  : {skipped_count:,}")


if __name__ == "__main__":
    parse_wikipedia()