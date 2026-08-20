import json
import hashlib
import re
import unicodedata
from collections import Counter


# ============================================================
# EXACT CORPUS DEDUPLICATION
# ============================================================

class ExactDeduplicator:

    def __init__(
        self,
        input_file,
        output_file,
        stats_file="exact_dedup_stats.json"
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.stats_file = stats_file

        self.stats = Counter()

        self.input_characters = 0
        self.output_characters = 0
        self.duplicate_characters = 0


    # ========================================================
    # NORMALIZE TEXT FOR DEDUPLICATION
    # ========================================================

    def normalize_for_dedup(self, text):
        """
        Normalize text before hashing.

        This makes documents differing only in:
        - Unicode representation
        - repeated spaces
        - tabs
        - newline formatting

        count as duplicates.
        """

        # Normalize Unicode
        text = unicodedata.normalize(
            "NFC",
            text
        )

        # Normalize all whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    # ========================================================
    # GENERATE DOCUMENT HASH
    # ========================================================

    def get_document_hash(self, text):
        """
        Generate a SHA-256 hash of normalized text.
        """

        normalized = self.normalize_for_dedup(
            text
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()


    # ========================================================
    # PROCESS FILE
    # ========================================================

    def process_file(self):

        print("=" * 65)
        print("EXACT CORPUS DEDUPLICATION")
        print("=" * 65)

        print(f"\nInput file : {self.input_file}")
        print(f"Output file: {self.output_file}")

        # ----------------------------------------------------
        # Store hashes of previously seen documents
        # ----------------------------------------------------

        seen_hashes = set()

        with open(
            self.input_file,
            "r",
            encoding="utf-8"
        ) as infile, open(
            self.output_file,
            "w",
            encoding="utf-8"
        ) as outfile:

            for line_number, line in enumerate(
                infile,
                start=1
            ):

                self.stats["total"] += 1


                # ------------------------------------------------
                # Parse JSON
                # ------------------------------------------------

                try:

                    record = json.loads(line)

                except json.JSONDecodeError:

                    self.stats["invalid_json"] += 1
                    continue


                # ------------------------------------------------
                # Get text
                # ------------------------------------------------

                text = record.get(
                    "text",
                    ""
                )


                # ------------------------------------------------
                # Empty document
                # ------------------------------------------------

                if not text or not text.strip():

                    self.stats["empty"] += 1
                    continue


                # ------------------------------------------------
                # Character statistics
                # ------------------------------------------------

                self.input_characters += len(text)


                # ------------------------------------------------
                # Generate normalized hash
                # ------------------------------------------------

                document_hash = (
                    self.get_document_hash(
                        text
                    )
                )


                # ------------------------------------------------
                # Duplicate
                # ------------------------------------------------

                if document_hash in seen_hashes:

                    self.stats["duplicates"] += 1

                    self.duplicate_characters += len(
                        text
                    )

                    continue


                # ------------------------------------------------
                # New unique document
                # ------------------------------------------------

                seen_hashes.add(
                    document_hash
                )

                outfile.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                self.stats["kept"] += 1

                self.output_characters += len(
                    text
                )


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if self.stats["total"] % 10_000 == 0:

                    duplicate_percentage = (
                        self.stats["duplicates"]
                        / self.stats["total"]
                        * 100
                    )

                    print(
                        f"Processed: "
                        f"{self.stats['total']:,} | "

                        f"Kept: "
                        f"{self.stats['kept']:,} | "

                        f"Duplicates: "
                        f"{self.stats['duplicates']:,} | "

                        f"Duplicate rate: "
                        f"{duplicate_percentage:.4f}%"
                    )


        # ----------------------------------------------------
        # Save statistics
        # ----------------------------------------------------

        self.save_statistics(
            len(seen_hashes)
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        self.print_summary(
            len(seen_hashes)
        )


    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    def save_statistics(
        self,
        unique_documents
    ):

        total = self.stats["total"]

        duplicate_percentage = (
            self.stats["duplicates"]
            / total
            * 100
            if total > 0
            else 0
        )

        stats = {

            "input_file": self.input_file,

            "output_file": self.output_file,

            "documents": {

                "total": total,

                "unique": unique_documents,

                "kept": self.stats["kept"],

                "duplicates_removed": (
                    self.stats["duplicates"]
                ),

                "invalid_json": (
                    self.stats["invalid_json"]
                ),

                "empty": (
                    self.stats["empty"]
                ),

                "duplicate_percentage": (
                    duplicate_percentage
                )
            },

            "characters": {

                "input": (
                    self.input_characters
                ),

                "output": (
                    self.output_characters
                ),

                "duplicate_characters_removed": (
                    self.duplicate_characters
                )
            }
        }


        with open(
            self.stats_file,
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

    def print_summary(
        self,
        unique_documents
    ):

        total = self.stats["total"]

        duplicate_percentage = (
            self.stats["duplicates"]
            / total
            * 100
            if total > 0
            else 0
        )

        print("\n" + "=" * 65)
        print("EXACT DEDUPLICATION COMPLETE")
        print("=" * 65)

        print(
            f"\nTotal documents      : "
            f"{total:,}"
        )

        print(
            f"Unique documents     : "
            f"{unique_documents:,}"
        )

        print(
            f"Duplicates removed   : "
            f"{self.stats['duplicates']:,}"
        )

        print(
            f"Duplicate percentage : "
            f"{duplicate_percentage:.4f}%"
        )


        # ----------------------------------------------------
        # Invalid / empty
        # ----------------------------------------------------

        if (
            self.stats["invalid_json"] > 0
            or self.stats["empty"] > 0
        ):

            print("\nOther skipped documents:")

            print(
                f"Invalid JSON         : "
                f"{self.stats['invalid_json']:,}"
            )

            print(
                f"Empty documents      : "
                f"{self.stats['empty']:,}"
            )


        # ----------------------------------------------------
        # Character statistics
        # ----------------------------------------------------

        print("\nCharacter statistics:")

        print(
            f"Input characters     : "
            f"{self.input_characters:,}"
        )

        print(
            f"Output characters    : "
            f"{self.output_characters:,}"
        )

        print(
            f"Duplicate characters removed: "
            f"{self.duplicate_characters:,}"
        )


        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        print(
            f"\nOutput saved to: "
            f"{self.output_file}"
        )

        print(
            f"Statistics saved to: "
            f"{self.stats_file}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    INPUT_FILE = (
        "telugu_wiktionary_basic_filtered.jsonl"
    )

    OUTPUT_FILE = (
        "telugu_wiktionary_exact_dedup.jsonl"
    )

    STATS_FILE = (
        "telugu_wiktionary_exact_dedup_stats.json"
    )


    deduplicator = ExactDeduplicator(

        input_file=INPUT_FILE,

        output_file=OUTPUT_FILE,

        stats_file=STATS_FILE
    )


    deduplicator.process_file()