import json
import re
import unicodedata
from collections import Counter, defaultdict

from datasketch import MinHash, MinHashLSH


# ============================================================
# CONFIGURATION
# ============================================================

WIKIPEDIA_FILE = "telugu_combined_corpus_v2.jsonl"

SANGRAHA_FILE = (
    "telugu_wikitionary_cross_dedup.jsonl"
)

PAIRS_FILE = "wikitionary_cross_near_duplicate_pairs.json"
STATS_FILE = "wikitionary_cross_near_dedup_stats.json"


# ------------------------------------------------------------
# Similarity settings
# ------------------------------------------------------------

NUM_PERM = 128

# LSH candidate-generation threshold.
# This is intentionally a little lower than the audit threshold
# so that potentially interesting pairs are not missed.
LSH_THRESHOLD = 0.80

# Only save verified pairs at or above this similarity.
AUDIT_THRESHOLD = 0.90

# Word shingle size.
SHINGLE_SIZE = 5


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for near-duplicate comparison.

    We intentionally do not remove Telugu punctuation or words.
    The goal is to compare lexical content while making Unicode
    and whitespace differences irrelevant.
    """

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def get_words(text):
    """
    Split normalized text into whitespace-separated units.

    This works reasonably well for Telugu because our goal here
    is duplicate detection rather than linguistic tokenization.
    """

    text = normalize_text(text)

    return text.split()


# ============================================================
# SHINGLES
# ============================================================

def get_shingles(text, k=SHINGLE_SIZE):
    """
    Generate a set of word-level k-shingles.

    Example:

    words:
        [a, b, c, d, e]

    3-shingles:
        {
            (a, b, c),
            (b, c, d),
            (c, d, e)
        }
    """

    words = get_words(text)

    # Very short documents need special handling.
    if len(words) < k:

        if words:
            return {tuple(words)}

        return set()

    shingles = set()

    for i in range(len(words) - k + 1):

        shingle = tuple(
            words[i:i + k]
        )

        shingles.add(shingle)

    return shingles


# ============================================================
# MINHASH
# ============================================================

def create_minhash(shingles):
    """
    Create a MinHash signature from a shingle set.
    """

    minhash = MinHash(
        num_perm=NUM_PERM
    )

    for shingle in shingles:

        # Convert tuple to a stable UTF-8 representation.
        value = " ".join(shingle)

        minhash.update(
            value.encode("utf-8")
        )

    return minhash


# ============================================================
# EXACT JACCARD VERIFICATION
# ============================================================

def jaccard_similarity(set_a, set_b):
    """
    Compute exact Jaccard similarity.
    """

    if not set_a and not set_b:
        return 0.0

    intersection = len(
        set_a.intersection(set_b)
    )

    union = len(
        set_a.union(set_b)
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# SIMILARITY BUCKET
# ============================================================

def get_similarity_bucket(similarity):

    if similarity < 0.90:
        return "< 0.90"

    elif similarity < 0.92:
        return "0.90 - 0.92"

    elif similarity < 0.95:
        return "0.92 - 0.95"

    elif similarity < 0.98:
        return "0.95 - 0.98"

    elif similarity < 0.99:
        return "0.98 - 0.99"

    else:
        return "0.99 - 1.00"


# ============================================================
# BUILD WIKIPEDIA LSH INDEX
# ============================================================

def build_wikipedia_index():

    print("=" * 60)
    print("BUILDING WIKIPEDIA MINHASH INDEX")
    print("=" * 60)

    lsh = MinHashLSH(
        threshold=LSH_THRESHOLD,
        num_perm=NUM_PERM
    )

    wikipedia_data = {}

    counts = Counter()

    with open(
        WIKIPEDIA_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for index, line in enumerate(infile):

            counts["total"] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                counts["invalid_json"] += 1
                continue

            text = record.get("text", "")

            shingles = get_shingles(text)

            if not shingles:

                counts["empty_shingles"] += 1
                continue

            minhash = create_minhash(
                shingles
            )

            key = str(index)

            # Store metadata needed for exact verification.
            wikipedia_data[key] = {
                "id": record.get("id"),
                "title": record.get("title", ""),
                "text_length": len(text),
                "shingles": shingles
            }

            lsh.insert(
                key,
                minhash
            )

            counts["indexed"] += 1

            if counts["total"] % 10_000 == 0:

                print(
                    f"Processed: {counts['total']:,} | "
                    f"Indexed: {counts['indexed']:,}"
                )

    print()

    print(
        f"Wikipedia documents indexed: "
        f"{counts['indexed']:,}"
    )

    return lsh, wikipedia_data, counts


# ============================================================
# QUERY SANGRAHA
# ============================================================

def query_sangraha(
    lsh,
    wikipedia_data
):

    print("\n" + "=" * 60)
    print("QUERYING SANGRAHA AGAINST WIKIPEDIA")
    print("=" * 60)

    counts = Counter()

    similarity_buckets = Counter()

    pairs = []

    # Prevent saving the same pair multiple times.
    seen_pairs = set()

    with open(
        SANGRAHA_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for sangraha_index, line in enumerate(infile):

            counts["total"] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                counts["invalid_json"] += 1
                continue

            text = record.get("text", "")

            shingles = get_shingles(text)

            if not shingles:

                counts["empty_shingles"] += 1
                continue

            minhash = create_minhash(
                shingles
            )

            # ------------------------------------------------
            # LSH candidate generation
            # ------------------------------------------------

            candidate_keys = lsh.query(
                minhash
            )

            counts["candidate_pairs"] += len(
                candidate_keys
            )

            # ------------------------------------------------
            # Exact Jaccard verification
            # ------------------------------------------------

            for wiki_key in candidate_keys:

                pair_key = (
                    wiki_key,
                    sangraha_index
                )

                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)

                wiki_shingles = (
                    wikipedia_data[wiki_key]["shingles"]
                )

                similarity = jaccard_similarity(
                    shingles,
                    wiki_shingles
                )

                counts["pairs_verified"] += 1

                # --------------------------------------------
                # Keep only interesting audit pairs
                # --------------------------------------------

                if similarity >= AUDIT_THRESHOLD:

                    counts["pairs_above_threshold"] += 1

                    bucket = get_similarity_bucket(
                        similarity
                    )

                    similarity_buckets[bucket] += 1

                    pair = {
                        "similarity": similarity,

                        "wikipedia": {
                            "index": int(wiki_key),
                            "id": (
                                wikipedia_data[wiki_key]["id"]
                            ),
                            "title": (
                                wikipedia_data[wiki_key]["title"]
                            ),
                            "characters": (
                                wikipedia_data[wiki_key]["text_length"]
                            )
                        },

                        "sangraha": {
                            "index": sangraha_index,
                            "id": record.get("id"),
                            "source": record.get(
                                "source",
                                ""
                            ),
                            "type": record.get(
                                "type",
                                ""
                            ),
                            "characters": len(text)
                        }
                    }

                    pairs.append(pair)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if counts["total"] % 1_000 == 0:

                print(
                    f"Processed: {counts['total']:,} | "
                    f"Candidates: "
                    f"{counts['candidate_pairs']:,} | "
                    f"Verified: "
                    f"{counts['pairs_verified']:,} | "
                    f"Audit pairs: "
                    f"{counts['pairs_above_threshold']:,}"
                )

    return (
        counts,
        similarity_buckets,
        pairs
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    wikipedia_counts,
    sangraha_counts,
    similarity_buckets,
    pairs
):

    # Sort highest similarity first.
    pairs.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    with open(
        PAIRS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            pairs,
            f,
            ensure_ascii=False,
            indent=2
        )

    stats = {
        "configuration": {
            "num_perm": NUM_PERM,
            "lsh_threshold": LSH_THRESHOLD,
            "audit_threshold": AUDIT_THRESHOLD,
            "shingle_size": SHINGLE_SIZE
        },

        "wikipedia": {
            "total_documents": (
                wikipedia_counts["total"]
            ),

            "indexed_documents": (
                wikipedia_counts["indexed"]
            ),

            "invalid_json": (
                wikipedia_counts["invalid_json"]
            ),

            "empty_shingles": (
                wikipedia_counts["empty_shingles"]
            )
        },

        "sangraha": {
            "total_documents": (
                sangraha_counts["total"]
            ),

            "candidate_pairs": (
                sangraha_counts["candidate_pairs"]
            ),

            "pairs_verified": (
                sangraha_counts["pairs_verified"]
            ),

            "pairs_above_audit_threshold": (
                sangraha_counts[
                    "pairs_above_threshold"
                ]
            ),

            "invalid_json": (
                sangraha_counts["invalid_json"]
            ),

            "empty_shingles": (
                sangraha_counts["empty_shingles"]
            )
        },

        "similarity_buckets": dict(
            similarity_buckets
        ),

        "total_saved_pairs": len(pairs)
    }

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1: Build Wikipedia index
    # --------------------------------------------------------

    (
        lsh,
        wikipedia_data,
        wikipedia_counts
    ) = build_wikipedia_index()

    # --------------------------------------------------------
    # STEP 2: Query Sangraha
    # --------------------------------------------------------

    (
        sangraha_counts,
        similarity_buckets,
        pairs
    ) = query_sangraha(
        lsh,
        wikipedia_data
    )

    # --------------------------------------------------------
    # STEP 3: Save
    # --------------------------------------------------------

    save_results(
        wikipedia_counts,
        sangraha_counts,
        similarity_buckets,
        pairs
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-DATASET NEAR-DEDUP AUDIT COMPLETE")
    print("=" * 60)

    print("\nWikipedia:")

    print(
        f"Documents indexed        : "
        f"{wikipedia_counts['indexed']:,}"
    )

    print("\nSangraha:")

    print(
        f"Documents processed      : "
        f"{sangraha_counts['total']:,}"
    )

    print(
        f"LSH candidate pairs      : "
        f"{sangraha_counts['candidate_pairs']:,}"
    )

    print(
        f"Pairs exactly verified   : "
        f"{sangraha_counts['pairs_verified']:,}"
    )

    print(
        f"Pairs >= {AUDIT_THRESHOLD:.2f}          : "
        f"{sangraha_counts['pairs_above_threshold']:,}"
    )

    print("\nSimilarity buckets:")

    bucket_order = [
        "0.90 - 0.92",
        "0.92 - 0.95",
        "0.95 - 0.98",
        "0.98 - 0.99",
        "0.99 - 1.00"
    ]

    for bucket in bucket_order:

        print(
            f"{bucket:15} : "
            f"{similarity_buckets[bucket]:,}"
        )

    print(
        f"\nPairs saved to: {PAIRS_FILE}"
    )

    print(
        f"Statistics saved to: {STATS_FILE}"
    )


if __name__ == "__main__":
    main()