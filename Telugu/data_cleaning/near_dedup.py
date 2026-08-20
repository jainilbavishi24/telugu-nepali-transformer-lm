import json
import re
import unicodedata
from collections import Counter

from datasketch import MinHash, MinHashLSH


# ============================================================
# FAST NEAR DEDUPLICATION
# ============================================================

class NearDeduplicator:

    def __init__(
        self,
        input_file,
        output_file,
        stats_file,
        num_perm=64,
        lsh_threshold=0.95,
        remove_threshold=0.98,
        shingle_size=5,
        max_tokens_per_document=50000
    ):

        self.input_file = input_file
        self.output_file = output_file
        self.stats_file = stats_file

        self.num_perm = num_perm
        self.lsh_threshold = lsh_threshold
        self.remove_threshold = remove_threshold
        self.shingle_size = shingle_size

        # Prevent extremely large documents from dominating
        # processing time.
        self.max_tokens_per_document = (
            max_tokens_per_document
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize_text(self, text):

        text = unicodedata.normalize(
            "NFC",
            text
        )

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    # ========================================================
    # WORD TOKENIZATION
    # ========================================================

    def tokenize(self, text):

        text = self.normalize_text(text)

        tokens = re.findall(
            r"\S+",
            text
        )

        # Limit pathological huge documents
        if len(tokens) > self.max_tokens_per_document:

            tokens = tokens[
                :self.max_tokens_per_document
            ]

        return tokens


    # ========================================================
    # WORD SHINGLES
    # ========================================================

    def get_shingles(self, text):

        tokens = self.tokenize(text)

        if not tokens:
            return set()

        # Short documents
        if len(tokens) <= self.shingle_size:

            return {
                " ".join(tokens)
            }

        shingles = set()

        for i in range(
            len(tokens)
            - self.shingle_size
            + 1
        ):

            shingle = " ".join(
                tokens[
                    i:i + self.shingle_size
                ]
            )

            shingles.add(
                shingle
            )

        return shingles


    # ========================================================
    # CREATE MINHASH
    # ========================================================

    def create_minhash(self, shingles):

        minhash = MinHash(
            num_perm=self.num_perm
        )

        for shingle in shingles:

            minhash.update(
                shingle.encode("utf-8")
            )

        return minhash


    # ========================================================
    # SIMILARITY BUCKET
    # ========================================================

    def get_similarity_bucket(
        self,
        similarity
    ):

        if similarity < 0.90:
            return "< 0.90"

        elif similarity < 0.95:
            return "0.90 - 0.95"

        elif similarity < 0.98:
            return "0.95 - 0.98"

        elif similarity < 0.99:
            return "0.98 - 0.99"

        else:
            return "0.99 - 1.00"


    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    def load_documents(self):

        print("=" * 65)
        print("LOADING DOCUMENTS")
        print("=" * 65)

        documents = []

        counts = Counter()

        with open(
            self.input_file,
            "r",
            encoding="utf-8"
        ) as infile:

            for line in infile:

                counts["total"] += 1

                try:

                    record = json.loads(line)

                except json.JSONDecodeError:

                    counts[
                        "invalid_json"
                    ] += 1

                    continue

                text = record.get(
                    "text",
                    ""
                )

                if not text.strip():

                    counts["empty"] += 1

                    continue

                documents.append(
                    record
                )

                counts["loaded"] += 1

                if (
                    counts["total"]
                    % 10000 == 0
                ):

                    print(
                        f"Loaded: "
                        f"{counts['loaded']:,}"
                    )

        return documents, counts


    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build_index(
        self,
        documents
    ):

        print("\n" + "=" * 65)
        print("BUILDING MINHASH INDEX")
        print("=" * 65)

        lsh = MinHashLSH(
            threshold=self.lsh_threshold,
            num_perm=self.num_perm
        )

        minhashes = []

        for i, record in enumerate(
            documents
        ):

            text = record.get(
                "text",
                ""
            )

            shingles = self.get_shingles(
                text
            )

            minhash = self.create_minhash(
                shingles
            )

            lsh.insert(
                str(i),
                minhash
            )

            minhashes.append(
                minhash
            )

            if (i + 1) % 10000 == 0:

                print(
                    f"Indexed: "
                    f"{i + 1:,}"
                )

        return lsh, minhashes


    # ========================================================
    # FIND DUPLICATES
    # ========================================================

    def find_duplicates(
        self,
        documents,
        lsh,
        minhashes
    ):

        print("\n" + "=" * 65)
        print("SEARCHING FOR NEAR DUPLICATES")
        print("=" * 65)

        counts = Counter()

        similarity_buckets = Counter()

        documents_to_remove = set()

        processed_pairs = set()

        for i in range(
            len(documents)
        ):

            if i in documents_to_remove:
                continue

            candidates = lsh.query(
                minhashes[i]
            )

            for candidate in candidates:

                j = int(candidate)

                if i == j:
                    continue

                pair = (
                    min(i, j),
                    max(i, j)
                )

                if pair in processed_pairs:
                    continue

                processed_pairs.add(
                    pair
                )

                counts[
                    "candidate_pairs"
                ] += 1

                # ------------------------------------------------
                # Fast MinHash similarity estimate
                # ------------------------------------------------

                similarity = (
                    minhashes[i].jaccard(
                        minhashes[j]
                    )
                )

                counts[
                    "verified_pairs"
                ] += 1

                bucket = (
                    self.get_similarity_bucket(
                        similarity
                    )
                )

                similarity_buckets[
                    bucket
                ] += 1

                # ------------------------------------------------
                # Only remove extremely similar documents
                # ------------------------------------------------

                if (
                    similarity
                    >= self.remove_threshold
                ):

                    counts[
                        "pairs_above_threshold"
                    ] += 1

                    # Keep earlier document
                    documents_to_remove.add(
                        max(i, j)
                    )

            if (i + 1) % 10000 == 0:

                print(
                    f"Processed: {i + 1:,} | "
                    f"Candidates: "
                    f"{counts['candidate_pairs']:,} | "
                    f"Marked: "
                    f"{len(documents_to_remove):,}"
                )

        return (
            documents_to_remove,
            counts,
            similarity_buckets
        )


    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    def write_output(
        self,
        documents,
        documents_to_remove
    ):

        print("\n" + "=" * 65)
        print("WRITING OUTPUT")
        print("=" * 65)

        input_characters = 0
        output_characters = 0

        kept = 0

        with open(
            self.output_file,
            "w",
            encoding="utf-8"
        ) as outfile:

            for i, record in enumerate(
                documents
            ):

                text = record.get(
                    "text",
                    ""
                )

                input_characters += len(
                    text
                )

                if i in documents_to_remove:
                    continue

                outfile.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                kept += 1

                output_characters += len(
                    text
                )

        return {
            "documents_kept": kept,
            "documents_removed": (
                len(documents_to_remove)
            ),
            "input_characters": (
                input_characters
            ),
            "output_characters": (
                output_characters
            ),
            "removed_characters": (
                input_characters
                - output_characters
            )
        }


    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    def save_statistics(
        self,
        load_counts,
        dedup_counts,
        similarity_buckets,
        output_stats
    ):

        stats = {

            "input_file": self.input_file,

            "output_file": self.output_file,

            "configuration": {

                "num_perm": (
                    self.num_perm
                ),

                "lsh_threshold": (
                    self.lsh_threshold
                ),

                "remove_threshold": (
                    self.remove_threshold
                ),

                "shingle_size": (
                    self.shingle_size
                )
            },

            "documents_processed": (
                load_counts["total"]
            ),

            "documents_loaded": (
                load_counts["loaded"]
            ),

            "candidate_pairs": (
                dedup_counts[
                    "candidate_pairs"
                ]
            ),

            "pairs_verified": (
                dedup_counts[
                    "verified_pairs"
                ]
            ),

            "pairs_above_threshold": (
                dedup_counts[
                    "pairs_above_threshold"
                ]
            ),

            "similarity_buckets": dict(
                similarity_buckets
            ),

            "output": output_stats
        }

        with open(
            self.stats_file,
            "w",
            encoding="utf-8"
        ) as outfile:

            json.dump(
                stats,
                outfile,
                ensure_ascii=False,
                indent=4
            )


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        # Step 1
        documents, load_counts = (
            self.load_documents()
        )

        # Step 2
        lsh, minhashes = (
            self.build_index(
                documents
            )
        )

        # Step 3
        (
            documents_to_remove,
            dedup_counts,
            similarity_buckets
        ) = self.find_duplicates(
            documents,
            lsh,
            minhashes
        )

        # Step 4
        output_stats = (
            self.write_output(
                documents,
                documents_to_remove
            )
        )

        # Step 5
        self.save_statistics(
            load_counts,
            dedup_counts,
            similarity_buckets,
            output_stats
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("NEAR DEDUPLICATION COMPLETE")
        print("=" * 65)

        print(
            f"\nDocuments processed      : "
            f"{load_counts['total']:,}"
        )

        print(
            f"Documents kept           : "
            f"{output_stats['documents_kept']:,}"
        )

        print(
            f"Documents removed        : "
            f"{output_stats['documents_removed']:,}"
        )

        print()

        print(
            f"LSH candidate pairs      : "
            f"{dedup_counts['candidate_pairs']:,}"
        )

        print(
            f"Pairs verified           : "
            f"{dedup_counts['verified_pairs']:,}"
        )

        print(
            f"Pairs >= "
            f"{self.remove_threshold:.2f}          : "
            f"{dedup_counts['pairs_above_threshold']:,}"
        )

        print("\nSimilarity buckets:")

        bucket_order = [

            "< 0.90",
            "0.90 - 0.95",
            "0.95 - 0.98",
            "0.98 - 0.99",
            "0.99 - 1.00"
        ]

        for bucket in bucket_order:

            print(
                f"{bucket:15}: "
                f"{similarity_buckets[bucket]:,}"
            )

        print()

        print(
            f"Input characters         : "
            f"{output_stats['input_characters']:,}"
        )

        print(
            f"Output characters        : "
            f"{output_stats['output_characters']:,}"
        )

        print(
            f"Removed characters       : "
            f"{output_stats['removed_characters']:,}"
        )

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
        "telugu_wiktionary_quality_filtered.jsonl"
    )

    OUTPUT_FILE = (
        "telugu_wiktionary_near_dedup.jsonl"
    )

    STATS_FILE = (
        "telugu_wiktionary_near_dedup_stats.json"
    )

    deduplicator = NearDeduplicator(

        input_file=INPUT_FILE,

        output_file=OUTPUT_FILE,

        stats_file=STATS_FILE,

        # Faster than 128 permutations
        num_perm=64,

        # Only generate highly similar candidates
        lsh_threshold=0.95,

        # Automatically remove these
        remove_threshold=0.98,

        # 5-word shingles
        shingle_size=5
    )

    deduplicator.run()