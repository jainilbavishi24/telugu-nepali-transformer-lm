import json
import re
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "Nepalese/data.nosync/"
    "nepalese_combined_corpus_clean.jsonl"
)

OUTPUT_FILE = (
    "Nepalese/data.nosync/"
    "nepalese_combined_corpus_quality_filtered.jsonl"
)

STATS_FILE = (
    "Nepalese/data.nosync/"
    "nepalese_advanced_quality_filter_stats.json"
)

PRINT_EVERY = 10_000


# ============================================================
# ADVANCED QUALITY FILTER
# ============================================================

class AdvancedQualityFilter:

    def __init__(
        self,
        input_file,
        output_file,
        stats_file="advanced_quality_filter_stats.json",

        max_digit_ratio=0.30,
        max_long_token_ratio=0.20,

        min_lexical_diversity=0.20,

        ngram_size=5,
        max_ngram_repetition_ratio=0.30,

        max_sentence_repetition_ratio=0.30,

        long_token_length=40,
    ):

        self.input_file = input_file
        self.output_file = output_file
        self.stats_file = stats_file

        # ----------------------------------------------------
        # QUALITY THRESHOLDS
        # ----------------------------------------------------

        self.max_digit_ratio = max_digit_ratio

        self.max_long_token_ratio = (
            max_long_token_ratio
        )

        self.min_lexical_diversity = (
            min_lexical_diversity
        )

        self.ngram_size = ngram_size

        self.max_ngram_repetition_ratio = (
            max_ngram_repetition_ratio
        )

        self.max_sentence_repetition_ratio = (
            max_sentence_repetition_ratio
        )

        self.long_token_length = (
            long_token_length
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        self.stats = Counter()

        self.input_characters = 0
        self.output_characters = 0

    # ========================================================
    # TOKENIZE TEXT
    # ========================================================

    def tokenize(self, text):
        """
        Approximate word-level tokenization.

        Whitespace tokenization is sufficient for the quality
        heuristics used here.
        """

        return re.findall(
            r"\S+",
            text
        )

    # ========================================================
    # DIGIT RATIO
    # ========================================================

    def calculate_digit_ratio(self, text):
        """
        Ratio of digit characters to all non-whitespace
        characters.

        This catches documents dominated by:
            - tables
            - dates
            - statistics
            - IDs
            - numeric garbage
        """

        relevant_characters = [
            char
            for char in text
            if not char.isspace()
        ]

        if not relevant_characters:
            return 0.0

        digit_count = sum(
            char.isdigit()
            for char in relevant_characters
        )

        return (
            digit_count
            / len(relevant_characters)
        )

    # ========================================================
    # LONG TOKEN RATIO
    # ========================================================

    def calculate_long_token_ratio(
        self,
        tokens
    ):
        """
        Detect documents containing a large proportion of
        unusually long whitespace-separated tokens.

        These often indicate:
            - broken HTML
            - encoded data
            - URLs
            - malformed markup
            - concatenated text
        """

        if not tokens:
            return 0.0

        long_tokens = sum(
            len(token) >= self.long_token_length
            for token in tokens
        )

        return (
            long_tokens
            / len(tokens)
        )

    # ========================================================
    # LEXICAL DIVERSITY
    # ========================================================

    def calculate_lexical_diversity(
        self,
        tokens
    ):
        """
        unique tokens / total tokens

        Lower values can indicate repetitive or corrupted text.

        For Nepali, this is computed over whitespace-separated
        tokens. This is intentionally simple because this is a
        quality filter, not a linguistic analyzer.
        """

        if not tokens:
            return 0.0

        normalized_tokens = [
            token.lower()
            for token in tokens
        ]

        unique_tokens = len(
            set(normalized_tokens)
        )

        return (
            unique_tokens
            / len(tokens)
        )

    # ========================================================
    # N-GRAM REPETITION
    # ========================================================

    def calculate_ngram_repetition_ratio(
        self,
        tokens
    ):
        """
        Measures repeated word n-grams.

        Example:

            नेपाल सरकार नेपाल सरकार नेपाल सरकार

        will contain repeated n-grams.

        Returns:

            repeated_ngrams / total_ngrams
        """

        n = self.ngram_size

        if len(tokens) < n:
            return 0.0

        ngrams = []

        for i in range(
            len(tokens) - n + 1
        ):

            ngram = tuple(
                tokens[i:i + n]
            )

            ngrams.append(
                ngram
            )

        if not ngrams:
            return 0.0

        counts = Counter(
            ngrams
        )

        repeated_count = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        return (
            repeated_count
            / len(ngrams)
        )

    # ========================================================
    # SENTENCE SPLITTING
    # ========================================================

    def split_sentences(self, text):
        """
        Approximate Nepali sentence splitting.

        Supports:

            .
            !
            ?
            ॥
            ।

        Also handles multiple punctuation characters.
        """

        sentences = re.split(
            r"[.!?।॥]+",
            text
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        return sentences

    # ========================================================
    # SENTENCE REPETITION
    # ========================================================

    def calculate_sentence_repetition_ratio(
        self,
        text
    ):
        """
        Measures repeated sentences.

        Returns:

            repeated_sentences / total_sentences
        """

        sentences = self.split_sentences(
            text
        )

        if len(sentences) < 2:
            return 0.0

        normalized_sentences = [
            re.sub(
                r"\s+",
                " ",
                sentence.lower()
            ).strip()
            for sentence in sentences
        ]

        counts = Counter(
            normalized_sentences
        )

        repeated_count = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        return (
            repeated_count
            / len(sentences)
        )

    # ========================================================
    # CHECK DOCUMENT QUALITY
    # ========================================================

    def check_quality(self, text):

        tokens = self.tokenize(
            text
        )

        # ----------------------------------------------------
        # Digit ratio
        # ----------------------------------------------------

        digit_ratio = (
            self.calculate_digit_ratio(
                text
            )
        )

        if (
            digit_ratio
            > self.max_digit_ratio
        ):

            return (
                False,
                "high_digit_ratio",
                {
                    "digit_ratio":
                        digit_ratio
                }
            )

        # ----------------------------------------------------
        # Long token ratio
        # ----------------------------------------------------

        long_token_ratio = (
            self.calculate_long_token_ratio(
                tokens
            )
        )

        if (
            long_token_ratio
            > self.max_long_token_ratio
        ):

            return (
                False,
                "long_token_ratio",
                {
                    "long_token_ratio":
                        long_token_ratio
                }
            )

        # ----------------------------------------------------
        # Lexical diversity
        # ----------------------------------------------------

        if len(tokens) >= 20:

            lexical_diversity = (
                self.calculate_lexical_diversity(
                    tokens
                )
            )

            if (
                lexical_diversity
                < self.min_lexical_diversity
            ):

                return (
                    False,
                    "low_lexical_diversity",
                    {
                        "lexical_diversity":
                            lexical_diversity
                    }
                )

        # ----------------------------------------------------
        # N-gram repetition
        # ----------------------------------------------------

        if len(tokens) >= self.ngram_size:

            ngram_repetition_ratio = (
                self.calculate_ngram_repetition_ratio(
                    tokens
                )
            )

            if (
                ngram_repetition_ratio
                > self.max_ngram_repetition_ratio
            ):

                return (
                    False,
                    "ngram_repetition",
                    {
                        "ngram_repetition_ratio":
                            ngram_repetition_ratio
                    }
                )

        # ----------------------------------------------------
        # Sentence repetition
        # ----------------------------------------------------

        sentence_repetition_ratio = (
            self.calculate_sentence_repetition_ratio(
                text
            )
        )

        if (
            sentence_repetition_ratio
            > self.max_sentence_repetition_ratio
        ):

            return (
                False,
                "sentence_repetition",
                {
                    "sentence_repetition_ratio":
                        sentence_repetition_ratio
                }
            )

        # ----------------------------------------------------
        # Passed all checks
        # ----------------------------------------------------

        return (
            True,
            None,
            {}
        )

    # ========================================================
    # PROCESS FILE
    # ========================================================

    def process_file(self):

        print("=" * 70)
        print("ADVANCED NEPALI CORPUS QUALITY FILTERING")
        print("=" * 70)

        print(
            f"\nInput file : "
            f"{self.input_file}"
        )

        print(
            f"Output file: "
            f"{self.output_file}"
        )

        print("\nThresholds:")

        print(
            f"Maximum digit ratio         : "
            f"{self.max_digit_ratio}"
        )

        print(
            f"Maximum long token ratio    : "
            f"{self.max_long_token_ratio}"
        )

        print(
            f"Minimum lexical diversity   : "
            f"{self.min_lexical_diversity}"
        )

        print(
            f"N-gram size                 : "
            f"{self.ngram_size}"
        )

        print(
            f"Maximum n-gram repetition   : "
            f"{self.max_ngram_repetition_ratio}"
        )

        print(
            f"Maximum sentence repetition : "
            f"{self.max_sentence_repetition_ratio}"
        )

        print(
            f"Long token length           : "
            f"{self.long_token_length}"
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

            for line in infile:

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

                # ------------------------------------------------
                # Get text
                # ------------------------------------------------

                text = record.get(
                    "text",
                    ""
                )

                if (
                    not text
                    or not text.strip()
                ):

                    self.stats[
                        "empty"
                    ] += 1

                    continue

                self.input_characters += len(
                    text
                )

                # ------------------------------------------------
                # Quality check
                # ------------------------------------------------

                (
                    passed,
                    reason,
                    metrics
                ) = self.check_quality(
                    text
                )

                # Keep track of metric values for
                # diagnostics if desired.
                if metrics:

                    for key, value in metrics.items():

                        self.stats[
                            f"metric_{key}"
                        ] += value

                # ------------------------------------------------
                # Remove low-quality document
                # ------------------------------------------------

                if not passed:

                    self.stats[
                        reason
                    ] += 1

                    self.stats[
                        "removed"
                    ] += 1

                    continue

                # ------------------------------------------------
                # Keep document
                # ------------------------------------------------

                outfile.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                self.stats[
                    "kept"
                ] += 1

                self.output_characters += len(
                    text
                )

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    self.stats[
                        "total"
                    ]
                    % PRINT_EVERY
                    == 0
                ):

                    print(
                        f"Processed: "
                        f"{self.stats['total']:,} | "
                        f"Kept: "
                        f"{self.stats['kept']:,} | "
                        f"Removed: "
                        f"{self.stats['removed']:,}"
                    )

        self.save_statistics()
        self.print_summary()

    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    def save_statistics(self):

        total = self.stats[
            "total"
        ]

        removed = (
            total
            - self.stats["kept"]
        )

        removal_reasons = {

            "invalid_json":
                self.stats[
                    "invalid_json"
                ],

            "empty":
                self.stats[
                    "empty"
                ],

            "high_digit_ratio":
                self.stats[
                    "high_digit_ratio"
                ],

            "long_token_ratio":
                self.stats[
                    "long_token_ratio"
                ],

            "low_lexical_diversity":
                self.stats[
                    "low_lexical_diversity"
                ],

            "ngram_repetition":
                self.stats[
                    "ngram_repetition"
                ],

            "sentence_repetition":
                self.stats[
                    "sentence_repetition"
                ],
        }

        stats = {

            "input_file":
                self.input_file,

            "output_file":
                self.output_file,

            "thresholds": {

                "max_digit_ratio":
                    self.max_digit_ratio,

                "max_long_token_ratio":
                    self.max_long_token_ratio,

                "min_lexical_diversity":
                    self.min_lexical_diversity,

                "ngram_size":
                    self.ngram_size,

                "max_ngram_repetition_ratio":
                    self.max_ngram_repetition_ratio,

                "max_sentence_repetition_ratio":
                    self.max_sentence_repetition_ratio,

                "long_token_length":
                    self.long_token_length,
            },

            "documents": {

                "total":
                    total,

                "kept":
                    self.stats[
                        "kept"
                    ],

                "removed":
                    removed,

                "removal_reasons":
                    removal_reasons,
            },

            "characters": {

                "input":
                    self.input_characters,

                "output":
                    self.output_characters,

                "removed":
                    (
                        self.input_characters
                        - self.output_characters
                    ),
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
            "ADVANCED NEPALI QUALITY FILTERING COMPLETE"
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
            "empty",
            "high_digit_ratio",
            "long_token_ratio",
            "low_lexical_diversity",
            "ngram_repetition",
            "sentence_repetition",
        ]

        for reason in removal_reasons:

            count = self.stats[
                reason
            ]

            if count == 0:
                continue

            percentage = (
                count
                / total
                * 100
                if total > 0
                else 0
            )

            print(
                f"{reason:35} "
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
            f"Removed characters: "
            f"{self.input_characters - self.output_characters:,}"
        )

        print(
            f"\nStats saved to: "
            f"{self.stats_file}"
        )

        print(
            f"Filtered corpus saved to: "
            f"{self.output_file}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    quality_filter = AdvancedQualityFilter(

        input_file=INPUT_FILE,

        output_file=OUTPUT_FILE,

        stats_file=STATS_FILE,

        # ----------------------------------------------------
        # Recommended starting thresholds
        # ----------------------------------------------------

        max_digit_ratio=0.30,

        max_long_token_ratio=0.20,

        min_lexical_diversity=0.20,

        ngram_size=5,

        max_ngram_repetition_ratio=0.30,

        max_sentence_repetition_ratio=0.30,

        long_token_length=40,
    )

    quality_filter.process_file()