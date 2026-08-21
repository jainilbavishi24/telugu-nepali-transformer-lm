import json
import re
import unicodedata
from collections import Counter


# ============================================================
# BASIC NEPALI CORPUS FILTER
# ============================================================

class BasicCorpusFilter:

    def __init__(
        self,
        input_file,
        output_file,
        min_chars=50,
        min_nepali_ratio=0.70,
        min_nepali_word_ratio=0.02,
    ):
        self.input_file = input_file
        self.output_file = output_file

        self.min_chars = min_chars
        self.min_nepali_ratio = min_nepali_ratio
        self.min_nepali_word_ratio = min_nepali_word_ratio

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

            # Keep normal whitespace.
            if char in "\n\t ":
                cleaned.append(char)

            # IMPORTANT:
            # Preserve ZWJ and ZWNJ.
            elif char in ("\u200c", "\u200d"):
                cleaned.append(char)

            # Remove other control/format characters.
            elif category.startswith("C"):
                continue

            else:
                cleaned.append(char)

        return "".join(cleaned)

    # ========================================================
    # REMOVE OBVIOUS GARBAGE CHARACTERS
    # ========================================================

    def remove_noise(self, text):

        # Unicode replacement character.
        text = text.replace("\ufffd", "")

        # Null characters.
        text = text.replace("\x00", "")

        # Remove repeated decorative symbols.
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

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )

        # Collapse spaces and tabs.
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Collapse excessive blank lines.
        text = re.sub(
            r"\n\s*\n\s*\n+",
            "\n\n",
            text
        )

        # Remove spaces around newlines.
        text = re.sub(
            r" *\n *",
            "\n",
            text
        )

        return text.strip()

    # ========================================================
    # DEVANAGARI CHARACTER CHECK
    # ========================================================

    def is_devanagari_character(self, char):
        code = ord(char)

        return (
            0x0900 <= code <= 0x097F
        )

    # ========================================================
    # CALCULATE DEVANAGARI RATIO
    # ========================================================

    def calculate_devanagari_ratio(self, text):

        relevant_characters = 0
        devanagari_characters = 0

        for char in text:

            if self.is_devanagari_character(char):
                devanagari_characters += 1
                relevant_characters += 1

            elif char.isalpha():
                relevant_characters += 1

        if relevant_characters == 0:
            return 0.0

        return (
            devanagari_characters
            / relevant_characters
        )

    # ========================================================
    # NEPALI FUNCTION WORDS
    # ========================================================
    #
    # These are not proof that text is Nepali, but they provide
    # useful evidence when combined with Devanagari ratio.
    #
    # Keep this list conservative. Avoid common words such as
    # "नेपाल" alone being sufficient.
    # ========================================================

    NEPALI_WORDS = {
        "छ",
        "छन्",
        "थियो",
        "थिए",
        "हुन्छ",
        "हुन्छ",
        "भएको",
        "भएका",
        "गरेको",
        "गरेका",
        "गरेर",
        "गर्नु",
        "गर्ने",
        "गरिएको",
        "गरिएका",
        "गरिन्छ",
        "भएकोले",
        "त्यसैले",
        "यसअघि",
        "यसपछि",
        "अहिले",
        "यहाँ",
        "त्यहाँ",
        "किनभने",
        "नेपालमा",
        "नेपाली",
        "सरकारले",
        "मन्त्रालयले",
        "प्रयोगकर्ता",
        "व्यवस्था",
        "आवश्यकता",
        "सम्बन्धी",
        "सम्बन्धित",
        "मार्फत",
        "अनुसार",
        "लगायत",
        "तथा",
        "एवं",
        "तर",
        "भने",
        "पनि",
        "नै",
    }

    # ========================================================
    # NEPALI-SPECIFIC ORTHOGRAPHIC PATTERNS
    # ========================================================
    #
    # Nepali and Hindi overlap heavily, so these are signals,
    # not absolute rules.
    # ========================================================

    NEPALI_PATTERNS = [
        r"\bछैन\b",
        r"\bछैनन्\b",
        r"\bहुनेछ\b",
        r"\bभनेर\b",
        r"\bगरेको\b",
        r"\bगरिरहेका\b",
        r"\bगरिरहेक[ाो]\b",
        r"\bरहेको\b",
        r"\bरहेका\b",
        r"\bरहेकी\b",
        r"\bनेपालमा\b",
        r"\bनेपाली\b",
        r"\bसम्बन्धी\b",
        r"\bमाध्यमबाट\b",
        r"\bमार्फत\b",
        r"\bलगायत\b",
    ]

    COMPILED_NEPALI_PATTERNS = [
        re.compile(pattern)
        for pattern in NEPALI_PATTERNS
    ]

    # ========================================================
    # NEPALI WORD EXTRACTION
    # ========================================================

    def extract_devanagari_words(self, text):

        return re.findall(
            r"[\u0900-\u097F]+",
            text
        )

    # ========================================================
    # NEPALI WORD RATIO
    # ========================================================

    def calculate_nepali_word_ratio(self, text):

        words = self.extract_devanagari_words(
            text
        )

        if not words:
            return 0.0

        nepali_words = 0

        for word in words:

            if word in self.NEPALI_WORDS:
                nepali_words += 1

        return (
            nepali_words
            / len(words)
        )

    # ========================================================
    # NEPALI PATTERN SCORE
    # ========================================================

    def calculate_nepali_pattern_score(self, text):

        matches = 0

        for pattern in self.COMPILED_NEPALI_PATTERNS:
            matches += len(
                pattern.findall(text)
            )

        # Normalize so extremely long documents don't
        # automatically dominate.
        words = self.extract_devanagari_words(
            text
        )

        if not words:
            return 0.0

        return matches / len(words)

    # ========================================================
    # NEPALI TEXT DETECTOR
    # ========================================================

    def is_probably_nepali(self, text):

        devanagari_ratio = (
            self.calculate_devanagari_ratio(
                text
            )
        )

        if (
            devanagari_ratio
            < self.min_nepali_ratio
        ):
            return False, {
                "reason": "low_devanagari_ratio",
                "devanagari_ratio": devanagari_ratio,
                "nepali_word_ratio": 0.0,
                "pattern_score": 0.0,
            }

        nepali_word_ratio = (
            self.calculate_nepali_word_ratio(
                text
            )
        )

        pattern_score = (
            self.calculate_nepali_pattern_score(
                text
            )
        )

        # Strong evidence:
        #
        # 1. Some Nepali vocabulary OR
        # 2. Several Nepali-specific patterns.
        #
        # This is deliberately permissive.
        is_nepali = (
            nepali_word_ratio
            >= self.min_nepali_word_ratio
            or pattern_score
            >= 0.005
        )

        return is_nepali, {
            "reason": (
                "accepted"
                if is_nepali
                else "weak_nepali_evidence"
            ),
            "devanagari_ratio": devanagari_ratio,
            "nepali_word_ratio": nepali_word_ratio,
            "pattern_score": pattern_score,
        }

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    def clean_text(self, text):

        # Unicode normalization first.
        text = self.normalize_unicode(
            text
        )

        # Remove comments.
        text = self.remove_comments(
            text
        )

        # Remove URLs.
        text = self.remove_urls(
            text
        )

        # Remove HTML.
        text = self.remove_html(
            text
        )

        # Remove control characters.
        text = self.remove_control_characters(
            text
        )

        # Remove obvious noise.
        text = self.remove_noise(
            text
        )

        # Normalize whitespace.
        text = self.normalize_whitespace(
            text
        )

        return text

    # ========================================================
    # PROCESS FILE
    # ========================================================

    def process_file(self):

        print("=" * 70)
        print("BASIC NEPALI CORPUS FILTERING")
        print("=" * 70)

        print(
            f"\nInput file : "
            f"{self.input_file}"
        )

        print(
            f"Output file: "
            f"{self.output_file}"
        )

        print(
            f"\nMinimum characters: "
            f"{self.min_chars}"
        )

        print(
            f"Minimum Devanagari ratio: "
            f"{self.min_nepali_ratio:.2f}"
        )

        print(
            f"Minimum Nepali word ratio: "
            f"{self.min_nepali_word_ratio:.3f}"
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
                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    self.stats[
                        "invalid_json"
                    ] += 1

                    continue

                text = record.get(
                    "text",
                    ""
                )

                # ------------------------------------------------
                # Empty input
                # ------------------------------------------------

                if (
                    not text
                    or not text.strip()
                ):

                    self.stats[
                        "empty_input"
                    ] += 1

                    continue

                self.input_characters += len(
                    text
                )

                # ------------------------------------------------
                # Clean
                # ------------------------------------------------

                cleaned = self.clean_text(
                    text
                )

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

                if (
                    len(cleaned)
                    < self.min_chars
                ):

                    self.stats[
                        "too_short"
                    ] += 1

                    continue

                # ------------------------------------------------
                # Nepali detection
                # ------------------------------------------------

                is_nepali, scores = (
                    self.is_probably_nepali(
                        cleaned
                    )
                )

                # Save diagnostic statistics.
                self.stats[
                    "checked_nepali"
                ] += 1

                if not is_nepali:

                    self.stats[
                        "not_nepali"
                    ] += 1

                    reason = scores[
                        "reason"
                    ]

                    self.stats[
                        f"not_nepali_{reason}"
                    ] += 1

                    continue

                # ------------------------------------------------
                # Keep document
                # ------------------------------------------------

                output_record = record.copy()

                output_record[
                    "text"
                ] = cleaned

                # Useful for auditing the corpus.
                output_record[
                    "devanagari_ratio"
                ] = round(
                    scores[
                        "devanagari_ratio"
                    ],
                    4
                )

                output_record[
                    "nepali_word_ratio"
                ] = round(
                    scores[
                        "nepali_word_ratio"
                    ],
                    4
                )

                output_record[
                    "nepali_pattern_score"
                ] = round(
                    scores[
                        "pattern_score"
                    ],
                    4
                )

                outfile.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                self.stats[
                    "kept"
                ] += 1

                self.output_characters += len(
                    cleaned
                )

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    self.stats[
                        "total"
                    ] % 10_000 == 0
                ):

                    removed = (
                        self.stats["total"]
                        - self.stats["kept"]
                    )

                    print(
                        f"Processed: "
                        f"{self.stats['total']:,} | "
                        f"Kept: "
                        f"{self.stats['kept']:,} | "
                        f"Removed: "
                        f"{removed:,}"
                    )

        self.print_summary()

    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(self):

        total = self.stats[
            "total"
        ]

        removed = (
            total
            - self.stats["kept"]
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "NEPALI FILTERING COMPLETE"
        )

        print(
            "=" * 70
        )

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

        print(
            "\nRemoval reasons:"
        )

        removal_reasons = [
            "invalid_json",
            "empty_input",
            "empty_after_cleaning",
            "too_short",
            "not_nepali",
            "not_nepali_low_devanagari_ratio",
            "not_nepali_weak_nepali_evidence",
        ]

        for reason in removal_reasons:

            count = self.stats[
                reason
            ]

            if count == 0:
                continue

            percentage = (
                count / total * 100
                if total > 0
                else 0
            )

            print(
                f"{reason:40} "
                f"{count:10,} "
                f"({percentage:.2f}%)"
            )

        print(
            "\nCharacter statistics:"
        )

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
        "Nepalese/data.nosync/"
        "nepalese_wikipedia_raw.jsonl"
    )

    OUTPUT_FILE = (
        "Nepalese/data.nosync/"
        "nepalese_wikipedia_basic_filtered.jsonl"
    )

    filter_pipeline = BasicCorpusFilter(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,

        min_chars=50,

        # Fraction of alphabetic characters that must be
        # Devanagari.
        min_nepali_ratio=0.50,

        # Fraction of Devanagari words matching our small
        # Nepali vocabulary list.
        min_nepali_word_ratio=0.02,
    )

    filter_pipeline.process_file()