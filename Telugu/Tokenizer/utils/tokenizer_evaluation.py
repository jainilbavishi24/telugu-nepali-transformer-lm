import json
import math
import re
from collections import Counter, defaultdict

import sentencepiece as spm


class TokenizerEvaluator:
    """
    Evaluates a SentencePiece tokenizer on a JSONL corpus.

    Expected JSONL format:

    {
        "text": "...",
        "source": "wikipedia"
    }

    Each metric can be called independently.
    """

    def __init__(
        self,
        input_file,
        model_file,
        text_field="text",
        source_field="source"
    ):
        self.input_file = input_file
        self.model_file = model_file

        self.text_field = text_field
        self.source_field = source_field

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor(
            model_file=model_file
        )

        self.vocab_size = self.sp.get_piece_size()

        # Cache corpus statistics
        self._processed = False

        self.total_documents = 0
        self.total_characters = 0
        self.total_tokens = 0

        self.documents_with_unk = 0
        self.total_unk_tokens = 0

        self.token_frequency = Counter()

        self.source_documents = Counter()
        self.source_tokens = Counter()
        self.source_characters = Counter()

        self.document_token_lengths = []

        self.min_document_tokens = None
        self.max_document_tokens = 0

    # ============================================================
    # PROCESS CORPUS
    # ============================================================

    def process_corpus(
        self,
        print_every=10000
    ):
        """
        Processes the corpus once and caches all required
        statistics.

        Must be called before metric routines.
        """

        if self._processed:

            print(
                "Corpus already processed. "
                "Using cached statistics."
            )

            return

        print("=" * 65)
        print("PROCESSING CORPUS")
        print("=" * 65)

        unk_id = self.sp.unk_id()

        with open(
            self.input_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1
            ):

                try:

                    record = json.loads(line)

                except json.JSONDecodeError:

                    continue

                text = record.get(
                    self.text_field,
                    ""
                )

                if not text or not text.strip():

                    continue

                source = record.get(
                    self.source_field,
                    "unknown"
                )

                token_ids = self.sp.encode(
                    text,
                    out_type=int
                )

                num_tokens = len(token_ids)

                # ------------------------------------------------
                # Corpus statistics
                # ------------------------------------------------

                self.total_documents += 1

                self.total_characters += len(text)

                self.total_tokens += num_tokens

                self.document_token_lengths.append(
                    num_tokens
                )

                # ------------------------------------------------
                # Source statistics
                # ------------------------------------------------

                self.source_documents[source] += 1

                self.source_tokens[source] += num_tokens

                self.source_characters[source] += len(
                    text
                )

                # ------------------------------------------------
                # Token frequency
                # ------------------------------------------------

                self.token_frequency.update(
                    token_ids
                )

                # ------------------------------------------------
                # Unknown tokens
                # ------------------------------------------------

                unk_count = token_ids.count(
                    unk_id
                )

                if unk_count > 0:

                    self.documents_with_unk += 1

                    self.total_unk_tokens += unk_count

                # ------------------------------------------------
                # Min / max document length
                # ------------------------------------------------

                if self.min_document_tokens is None:

                    self.min_document_tokens = (
                        num_tokens
                    )

                else:

                    self.min_document_tokens = min(
                        self.min_document_tokens,
                        num_tokens
                    )

                self.max_document_tokens = max(
                    self.max_document_tokens,
                    num_tokens
                )

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    self.total_documents
                    % print_every
                    == 0
                ):

                    print(
                        f"Processed: "
                        f"{self.total_documents:,} | "
                        f"Tokens: "
                        f"{self.total_tokens:,}"
                    )

        self._processed = True

        print("\nProcessing complete.")

    # ============================================================
    # 1. TOTAL TOKEN COUNT
    # ============================================================

    def get_total_token_count(self):

        return {
            "documents": self.total_documents,
            "characters": self.total_characters,
            "tokens": self.total_tokens
        }

    # ============================================================
    # 2. TOKEN PERCENTAGE BY SOURCE
    # ============================================================

    def get_token_percentage_by_source(self):

        results = {}

        for source, token_count in (
            self.source_tokens.items()
        ):

            percentage = (
                token_count
                / self.total_tokens
                * 100
                if self.total_tokens > 0
                else 0
            )

            results[source] = {
                "documents": (
                    self.source_documents[source]
                ),

                "characters": (
                    self.source_characters[source]
                ),

                "tokens": token_count,

                "token_percentage": percentage
            }

        return results

    # ============================================================
    # 3. UNKNOWN TOKEN USAGE
    # ============================================================

    def get_unk_statistics(self):

        unk_percentage = (
            self.total_unk_tokens
            / self.total_tokens
            * 100
            if self.total_tokens > 0
            else 0
        )

        document_percentage = (
            self.documents_with_unk
            / self.total_documents
            * 100
            if self.total_documents > 0
            else 0
        )

        return {
            "unk_token_id": self.sp.unk_id(),

            "documents_with_unk": (
                self.documents_with_unk
            ),

            "documents_with_unk_percentage": (
                document_percentage
            ),

            "total_unk_tokens": (
                self.total_unk_tokens
            ),

            "unk_token_percentage": (
                unk_percentage
            )
        }

    # ============================================================
    # 4. FERTILITY
    # ============================================================

    def get_fertility(self):
        """
        Fertility = average number of tokenizer pieces
        per whitespace-separated word.

        A lower value generally indicates that the tokenizer
        represents words using fewer subword pieces.
        """

        total_words = 0

        total_token_pieces = 0

        with open(
            self.input_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                try:

                    record = json.loads(line)

                except json.JSONDecodeError:

                    continue

                text = record.get(
                    self.text_field,
                    ""
                )

                if not text or not text.strip():

                    continue

                words = text.split()

                if not words:

                    continue

                total_words += len(words)

                token_ids = self.sp.encode(
                    text,
                    out_type=int
                )

                total_token_pieces += len(
                    token_ids
                )

        fertility = (
            total_token_pieces
            / total_words
            if total_words > 0
            else 0
        )

        return {
            "total_words": total_words,

            "total_token_pieces": (
                total_token_pieces
            ),

            "fertility": fertility
        }

    # ============================================================
    # 5. VOCABULARY UTILISATION
    # ============================================================

    def get_vocabulary_utilisation(self):
        """
        Percentage of vocabulary tokens that appear at least once.
        """

        used_tokens = len(
            self.token_frequency
        )

        unused_tokens = (
            self.vocab_size
            - used_tokens
        )

        utilisation_percentage = (
            used_tokens
            / self.vocab_size
            * 100
            if self.vocab_size > 0
            else 0
        )

        return {
            "vocabulary_size": self.vocab_size,

            "used_tokens": used_tokens,

            "unused_tokens": unused_tokens,

            "vocabulary_utilisation_percentage": (
                utilisation_percentage
            )
        }

    # ============================================================
    # 6. TOKEN DISTRIBUTION
    # ============================================================

    def get_token_distribution(
        self,
        top_k=100
    ):
        """
        Returns the most frequent tokens.
        """

        distribution = []

        for token_id, frequency in (
            self.token_frequency.most_common(
                top_k
            )
        ):

            token = self.sp.id_to_piece(
                token_id
            )

            percentage = (
                frequency
                / self.total_tokens
                * 100
                if self.total_tokens > 0
                else 0
            )

            distribution.append(
                {
                    "token_id": token_id,

                    "token": token,

                    "frequency": frequency,

                    "percentage": percentage
                }
            )

        return distribution

    # ============================================================
    # 7. RARE TOKENS
    # ============================================================

    def get_rare_tokens(
        self,
        max_frequency=10
    ):
        """
        Returns vocabulary tokens occurring
        <= max_frequency times.
        """

        rare_tokens = []

        for token_id, frequency in (
            self.token_frequency.items()
        ):

            if frequency <= max_frequency:

                rare_tokens.append(
                    {
                        "token_id": token_id,

                        "token": (
                            self.sp.id_to_piece(
                                token_id
                            )
                        ),

                        "frequency": frequency
                    }
                )

        rare_tokens.sort(
            key=lambda x: x["frequency"]
        )

        return {
            "max_frequency": max_frequency,

            "rare_token_count": (
                len(rare_tokens)
            ),

            "rare_tokens": rare_tokens
        }

    # ============================================================
    # 8. AVERAGE TOKEN LENGTH
    # ============================================================

    def get_average_token_length(self):
        """
        Computes average character length of vocabulary pieces.

        SentencePiece whitespace marker ▁ is ignored.
        """

        token_lengths = []

        for token_id in range(
            self.vocab_size
        ):

            piece = self.sp.id_to_piece(
                token_id
            )

            # Remove SentencePiece word boundary marker
            clean_piece = piece.replace(
                "▁",
                ""
            )

            # Skip special tokens
            if clean_piece.startswith("<") and (
                clean_piece.endswith(">")
            ):

                continue

            token_lengths.append(
                len(clean_piece)
            )

        average_length = (
            sum(token_lengths)
            / len(token_lengths)
            if token_lengths
            else 0
        )

        return {
            "average_token_length": (
                average_length
            ),

            "minimum_token_length": (
                min(token_lengths)
                if token_lengths
                else 0
            ),

            "maximum_token_length": (
                max(token_lengths)
                if token_lengths
                else 0
            ),

            "tokens_considered": (
                len(token_lengths)
            )
        }

    # ============================================================
    # 9. SEQUENCE LENGTH DISTRIBUTION
    # ============================================================

    def get_sequence_length_distribution(
        self,
        bin_size=128
    ):
        """
        Groups documents according to tokenized sequence length.

        Example:

        0-127
        128-255
        256-383
        ...
        """

        bins = Counter()

        for length in (
            self.document_token_lengths
        ):

            lower = (
                length // bin_size
            ) * bin_size

            upper = (
                lower
                + bin_size
                - 1
            )

            label = (
                f"{lower}-{upper}"
            )

            bins[label] += 1

        distribution = []

        for label, count in sorted(
            bins.items(),
            key=lambda x: int(
                x[0].split("-")[0]
            )
        ):

            percentage = (
                count
                / self.total_documents
                * 100
                if self.total_documents > 0
                else 0
            )

            distribution.append(
                {
                    "range": label,

                    "documents": count,

                    "percentage": percentage
                }
            )

        return {
            "bin_size": bin_size,

            "distribution": distribution,

            "minimum_sequence_length": (
                self.min_document_tokens
            ),

            "maximum_sequence_length": (
                self.max_document_tokens
            ),

            "average_sequence_length": (
                self.total_tokens
                / self.total_documents
                if self.total_documents > 0
                else 0
            )
        }

    # ============================================================
    # OPTIONAL: SAVE RESULTS
    # ============================================================

    def save_results(
        self,
        output_file,
        rare_threshold=10,
        top_k_tokens=100,
        sequence_bin_size=128
    ):
        """
        Computes all metrics and saves them as JSON.
        """

        results = {

            "total_token_count":
                self.get_total_token_count(),

            "token_percentage_by_source":
                self.get_token_percentage_by_source(),

            "unk_statistics":
                self.get_unk_statistics(),

            "fertility":
                self.get_fertility(),

            "vocabulary_utilisation":
                self.get_vocabulary_utilisation(),

            "token_distribution":
                self.get_token_distribution(
                    top_k=top_k_tokens
                ),

            "rare_tokens":
                self.get_rare_tokens(
                    max_frequency=rare_threshold
                ),

            "average_token_length":
                self.get_average_token_length(),

            "sequence_length_distribution":
                self.get_sequence_length_distribution(
                    bin_size=sequence_bin_size
                )
        }

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=4
            )

        print(
            f"Results saved to: {output_file}"
        )

        return results