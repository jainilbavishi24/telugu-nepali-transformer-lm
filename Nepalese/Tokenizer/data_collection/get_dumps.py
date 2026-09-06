import bz2
import json
import xml.etree.ElementTree as ET
from collections import Counter


class GetDumps:

    def __init__(self, dump_file_path, output_file_path, type="wikipedia"):
        self.dump_file_path = dump_file_path
        self.output_file_path = output_file_path
        self.type = type


    def extract_text_from_dump(self):

        print("=" * 60)
        print("PARSING WIKIPEDIA DUMP")
        print("=" * 60)

        total_pages = 0
        saved_pages = 0
        skipped_pages = 0

        skip_reasons = Counter()
        namespace_counts = Counter()

        print(f"\nInput dump : {self.dump_file_path}")
        print(f"Output file: {self.output_file_path}\n")

        # Open the compressed .bz2 dump
        with bz2.open(
            self.dump_file_path,
            "rb"
        ) as file, open(
            self.output_file_path,
            "w",
            encoding="utf-8"
        ) as output_file:

            # Stream through the XML instead of loading
            # the entire dump into memory.
            for event, elem in ET.iterparse(
                file,
                events=("end",)
            ):

                # Process only completed <page> elements.
                if not elem.tag.endswith("page"):
                    continue

                total_pages += 1

                # ------------------------------------------------
                # Extract metadata
                # ------------------------------------------------

                title = elem.findtext("./{*}title")

                page_id = elem.findtext("./{*}id")

                namespace = elem.findtext("./{*}ns")

                redirect = elem.find("./{*}redirect")

                text = elem.findtext(
                    "./{*}revision/{*}text"
                )

                # Track namespaces
                if (self.type == "wikisource"):
                    Allowed_namespaces = {"0", "104"}
                else:
                    Allowed_namespaces = {"0"}
                
                namespace_counts[namespace] += 1


                # ------------------------------------------------
                # Keep only main namespace articles
                #
                # Namespace 0 = normal Wikipedia articles
                # ------------------------------------------------

                if namespace not in Allowed_namespaces:

                    skipped_pages += 1
                    skip_reasons["non_main_namespace"] += 1

                    elem.clear()
                    continue


                # ------------------------------------------------
                # Skip redirects
                # ------------------------------------------------

                if redirect is not None:

                    skipped_pages += 1
                    skip_reasons["redirect"] += 1

                    elem.clear()
                    continue


                # ------------------------------------------------
                # Skip pages without text
                # ------------------------------------------------

                if not text or not text.strip():

                    skipped_pages += 1
                    skip_reasons["missing_text"] += 1

                    elem.clear()
                    continue


                # ------------------------------------------------
                # Save record
                # ------------------------------------------------

                record = {
                    "id": page_id,
                    "title": title,
                    "text": text,
                    "namespace": int(namespace),
                    "source": "telugu_wikipedia"
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                saved_pages += 1


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if total_pages % 10_000 == 0:

                    print(
                        f"Processed: {total_pages:,} | "
                        f"Saved: {saved_pages:,} | "
                        f"Skipped: {skipped_pages:,}"
                    )


                # ------------------------------------------------
                # Important:
                #
                # Clear the processed XML element so memory
                # does not continuously grow.
                # ------------------------------------------------

                elem.clear()


        # ========================================================
        # FINAL SUMMARY
        # ========================================================

        print("\n" + "=" * 60)
        print("WIKIPEDIA PARSING COMPLETE")
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
        # Skip reasons
        # --------------------------------------------------------

        print("\nSkip reasons:")

        for reason, count in skip_reasons.most_common():

            percentage = (
                count / total_pages * 100
                if total_pages > 0
                else 0
            )

            print(
                f"{reason:25} "
                f"{count:10,} "
                f"({percentage:.2f}%)"
            )


        # --------------------------------------------------------
        # Namespace distribution
        # --------------------------------------------------------

        print("\nNamespace distribution:")

        for namespace, count in (
            namespace_counts.most_common()
        ):

            percentage = (
                count / total_pages * 100
                if total_pages > 0
                else 0
            )

            print(
                f"Namespace {str(namespace):>5}: "
                f"{count:10,} "
                f"({percentage:.2f}%)"
            )


        print(
            f"\nOutput saved to: "
            f"{self.output_file_path}"
        )


if __name__ == "__main__":

    dump_file_path = (
        "/Users/rithvik/Downloads/"
        "newiki-latest-pages-articles.xml.bz2"
    )

    output_file_path = (
        "Nepalese/data.nosync/"
        "nepalese_wikipedia_raw.jsonl"
    )

    parser = GetDumps(
        dump_file_path,
        output_file_path,
        type="wikipedia"
    )

    parser.extract_text_from_dump()