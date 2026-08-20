import json
import re
import unicodedata
from collections import Counter


# ============================================================
# BASIC CORPUS FILTER
# ============================================================

class BasicCorpusFilter:

    def __init__(
        self,
        input_file,
        output_file,
        min_chars=50,
        min_telugu_ratio=0.50
    ):
        self.input_file = input_file
        self.output_file = output_file

        self.min_chars = min_chars
        self.min_telugu_ratio = min_telugu_ratio

        self.stats = Counter()

        self.input_characters = 0
        self.output_characters = 0


    # ========================================================
    # REMOVE HTML COMMENTS
    # ========================================================

    def remove_comments(self, text):

        return re.sub(
            r"<!--.*?-->",
            "",
            text,
            flags=re.DOTALL
        )


    # ========================================================
    # REMOVE HTML TAGS
    # ========================================================

    def remove_html(self, text):

        return re.sub(
            r"<[^>]+>",
            "",
            text
        )


    # ========================================================
    # REMOVE URLS
    # ========================================================

    def remove_urls(self, text):

        return re.sub(
            r"https?://\S+|www\.\S+",
            "",
            text,
            flags=re.IGNORECASE
        )


    # ========================================================
    # REMOVE CONTROL / INVISIBLE CHARACTERS
    # ========================================================

    def remove_control_characters(self, text):

        cleaned = []

        for char in text:

            category = unicodedata.category(char)

            # Keep normal whitespace characters
            if char in "\n\t ":
                cleaned.append(char)

            # Remove control characters
            elif category.startswith("C"):

                continue

            else:
                cleaned.append(char)

        return "".join(cleaned)


    # ========================================================
    # REMOVE OBVIOUS GARBAGE CHARACTERS
    # ========================================================

    def remove_noise(self, text):

        # Unicode replacement character
        text = text.replace("\ufffd", "")

        # Null characters
        text = text.replace("\x00", "")

        # Remove repeated weird separators
        text = re.sub(
            r"[•●◆■□▪▫★☆]+",
            " ",
            text
        )

        return text


    # ========================================================
    # NORMALIZE UNICODE
    # ========================================================

    def normalize_unicode(self, text):

        return unicodedata.normalize(
            "NFC",
            text
        )


    # ========================================================
    # NORMALIZE WHITESPACE
    # ========================================================

    def normalize_whitespace(self, text):

        # Normalize line endings
        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )

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

        # Remove spaces around newlines
        text = re.sub(
            r" *\n *",
            "\n",
            text
        )

        return text.strip()


    # ========================================================
    # TELUGU CHARACTER CHECK
    # ========================================================

    def is_telugu_character(self, char):

        code = ord(char)

        # Telugu Unicode block
        return 0x0C00 <= code <= 0x0C7F


    # ========================================================
    # CALCULATE TELUGU RATIO
    # ========================================================

    def calculate_telugu_ratio(self, text):

        # Only consider alphabetic/script characters.
        # We don't want punctuation and spaces to reduce
        # the Telugu ratio.
        relevant_characters = 0
        telugu_characters = 0

        for char in text:

            # Telugu
            if self.is_telugu_character(char):

                telugu_characters += 1
                relevant_characters += 1

            # Other alphabetic Unicode characters
            elif char.isalpha():

                relevant_characters += 1

        if relevant_characters == 0:

            return 0.0

        return (
            telugu_characters
            / relevant_characters
        )


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    def clean_text(self, text):

        # Unicode normalization first
        text = self.normalize_unicode(text)

        # Remove comments
        text = self.remove_comments(text)

        # Remove URLs
        text = self.remove_urls(text)

        # Remove HTML
        text = self.remove_html(text)

        # Remove control characters
        text = self.remove_control_characters(text)

        # Remove obvious noise
        text = self.remove_noise(text)

        # Normalize whitespace
        text = self.normalize_whitespace(text)

        return text


    # ========================================================
    # PROCESS FILE
    # ========================================================

    def process_file(self):

        print("=" * 65)
        print("BASIC TELUGU CORPUS FILTERING")
        print("=" * 65)

        print(f"\nInput file : {self.input_file}")
        print(f"Output file: {self.output_file}")

        print(
            f"\nMinimum characters: "
            f"{self.min_chars}"
        )

        print(
            f"Minimum Telugu ratio: "
            f"{self.min_telugu_ratio:.2f}"
        )

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


                text = record.get(
                    "text",
                    ""
                )

                # ------------------------------------------------
                # Empty input
                # ------------------------------------------------

                if not text or not text.strip():

                    self.stats["empty_input"] += 1
                    continue


                self.input_characters += len(text)


                # ------------------------------------------------
                # Clean
                # ------------------------------------------------

                cleaned = self.clean_text(text)


                # ------------------------------------------------
                # Empty after cleaning
                # ------------------------------------------------

                if not cleaned:

                    self.stats[
                        "empty_after_cleaning"
                    ] += 1

                    continue


                # ------------------------------------------------
                # Minimum length
                # ------------------------------------------------

                if len(cleaned) < self.min_chars:

                    self.stats[
                        "too_short"
                    ] += 1

                    continue


                # ------------------------------------------------
                # Telugu ratio
                # ------------------------------------------------

                telugu_ratio = (
                    self.calculate_telugu_ratio(
                        cleaned
                    )
                )

                if (
                    telugu_ratio
                    < self.min_telugu_ratio
                ):

                    self.stats[
                        "low_telugu_ratio"
                    ] += 1

                    continue


                # ------------------------------------------------
                # Keep document
                # ------------------------------------------------

                output_record = record.copy()

                output_record["text"] = cleaned

                # Optional: save ratio for auditing
                # output_record["telugu_ratio"] = round(
                #     telugu_ratio,
                #     4
                # )

                outfile.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                self.stats["kept"] += 1

                self.output_characters += len(
                    cleaned
                )


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if self.stats["total"] % 10_000 == 0:

                    print(
                        f"Processed: "
                        f"{self.stats['total']:,} | "

                        f"Kept: "
                        f"{self.stats['kept']:,} | "

                        f"Removed: "
                        f"{self.stats['total'] - self.stats['kept']:,}"
                    )


        self.print_summary()


    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(self):

        total = self.stats["total"]

        removed = (
            total
            - self.stats["kept"]
        )

        print("\n" + "=" * 65)
        print("BASIC FILTERING COMPLETE")
        print("=" * 65)

        print(
            f"\nTotal documents : "
            f"{total:,}"
        )

        print(
            f"Kept            : "
            f"{self.stats['kept']:,}"
        )

        print(
            f"Removed         : "
            f"{removed:,}"
        )

        print("\nRemoval reasons:")

        removal_reasons = [
            "invalid_json",
            "empty_input",
            "empty_after_cleaning",
            "too_short",
            "low_telugu_ratio"
        ]

        for reason in removal_reasons:

            count = self.stats[reason]

            if count == 0:
                continue

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


        print("\nCharacter statistics:")

        print(
            f"Input characters : "
            f"{self.input_characters:,}"
        )

        print(
            f"Output characters: "
            f"{self.output_characters:,}"
        )

        print(
            f"Characters removed: "
            f"{self.input_characters - self.output_characters:,}"
        )

        print(
            f"\nOutput saved to: "
            f"{self.output_file}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    INPUT_FILE = (
        "telugu_wikipedia_raw.jsonl"
    )

    OUTPUT_FILE = (
        "telugu_wikipedia_basic_filtered.jsonl"
    )

    filter_pipeline = BasicCorpusFilter(

        input_file=INPUT_FILE,

        output_file=OUTPUT_FILE,

        min_chars=50,

        min_telugu_ratio=0.50
    )

    filter_pipeline.process_file()