import json
import re
import unicodedata
from datasketch import MinHash, MinHashLSH


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_wikipedia_exact_dedup.jsonl"
OUTPUT_FILE = "telugu_wikipedia_near_dedup.jsonl"
STATS_FILE = "near_dedup_stats.json"

NUM_PERM = 128

# Character shingle size
SHINGLE_SIZE = 5

# LSH candidate threshold.
# Use a slightly lower threshold than the final verification
# threshold so we don't miss likely candidates.
LSH_THRESHOLD = 0.85

# Final exact Jaccard similarity required to remove a document.
SIMILARITY_THRESHOLD = 0.90


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text before generating shingles.
    """

    text = unicodedata.normalize("NFC", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHARACTER SHINGLES
# ============================================================

def get_shingles(text, k=SHINGLE_SIZE):
    """
    Generate unique character-level k-shingles.

    Example:

    text = "తెలుగు భాష"

    → overlapping character sequences of length k
    """

    text = normalize_text(text)

    if len(text) <= k:
        return {text}

    return {
        text[i:i + k]
        for i in range(len(text) - k + 1)
    }


# ============================================================
# MINHASH
# ============================================================

def create_minhash(shingles):
    """
    Create a MinHash signature from a set of shingles.
    """

    minhash = MinHash(num_perm=NUM_PERM)

    for shingle in shingles:
        minhash.update(
            shingle.encode("utf-8")
        )

    return minhash


# ============================================================
# EXACT JACCARD SIMILARITY
# ============================================================

def jaccard_similarity(set_a, set_b):

    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for index, line in enumerate(infile):

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get("text", "")

            documents.append({
                "index": index,
                "record": record,
                "text": text
            })

    return documents


# ============================================================
# NEAR DEDUPLICATION
# ============================================================

def near_deduplicate():

    print("Loading documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents):,} documents"
    )

    # --------------------------------------------------------
    # LSH index
    # --------------------------------------------------------

    lsh = MinHashLSH(
        threshold=LSH_THRESHOLD,
        num_perm=NUM_PERM
    )

    # Store shingles for exact verification.
    shingle_sets = {}

    # Documents marked as duplicates.
    removed = set()

    candidate_pairs = 0
    verified_pairs = 0

    print("\nProcessing documents...\n")

    # --------------------------------------------------------
    # Process documents sequentially
    # --------------------------------------------------------

    for i, document in enumerate(documents):

        text = document["text"]

        # Generate shingles
        shingles = get_shingles(text)

        # Save for later exact verification
        shingle_sets[i] = shingles

        # Generate MinHash
        minhash = create_minhash(shingles)

        # ----------------------------------------------------
        # Find candidate near duplicates
        # ----------------------------------------------------

        candidates = lsh.query(minhash)

        candidate_pairs += len(candidates)

        is_duplicate = False

        # ----------------------------------------------------
        # Verify candidates using exact Jaccard similarity
        # ----------------------------------------------------

        for candidate_id in candidates:

            candidate_index = int(candidate_id)

            # Candidate may already be removed.
            if candidate_index in removed:
                continue

            verified_pairs += 1

            similarity = jaccard_similarity(
                shingles,
                shingle_sets[candidate_index]
            )

            if similarity >= SIMILARITY_THRESHOLD:

                removed.add(i)
                is_duplicate = True
                break

        # ----------------------------------------------------
        # Only insert unique documents into the LSH index
        # ----------------------------------------------------

        if not is_duplicate:

            lsh.insert(
                str(i),
                minhash
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (i + 1) % 1000 == 0:

            print(
                f"Processed: {i + 1:,} | "
                f"Kept: {i + 1 - len(removed):,} | "
                f"Removed: {len(removed):,} | "
                f"Candidates checked: {verified_pairs:,}"
            )

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    print("\nWriting output...")

    kept_documents = 0
    removed_characters = 0
    output_characters = 0
    input_characters = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for i, document in enumerate(documents):

            text = document["text"]

            input_characters += len(text)

            if i in removed:

                removed_characters += len(text)
                continue

            outfile.write(
                json.dumps(
                    document["record"],
                    ensure_ascii=False
                ) + "\n"
            )

            kept_documents += 1
            output_characters += len(text)

    # ========================================================
    # STATISTICS
    # ========================================================

    total_documents = len(documents)
    duplicates_removed = len(removed)

    stats = {
        "input_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,

        "configuration": {
            "num_perm": NUM_PERM,
            "shingle_size": SHINGLE_SIZE,
            "lsh_threshold": LSH_THRESHOLD,
            "similarity_threshold":
                SIMILARITY_THRESHOLD
        },

        "document_counts": {
            "input": total_documents,
            "kept": kept_documents,
            "near_duplicates_removed":
                duplicates_removed
        },

        "comparison_statistics": {
            "candidate_pairs_found":
                candidate_pairs,
            "pairs_verified":
                verified_pairs
        },

        "characters": {
            "input": input_characters,
            "output": output_characters,
            "removed": removed_characters
        }
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

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("NEAR DEDUPLICATION COMPLETE")
    print("=" * 60)

    print(f"Input documents        : {total_documents:,}")
    print(f"Kept documents         : {kept_documents:,}")
    print(
        f"Near duplicates removed: "
        f"{duplicates_removed:,}"
    )

    if total_documents > 0:

        percentage = (
            duplicates_removed
            / total_documents
            * 100
        )

        print(
            f"Removal percentage     : "
            f"{percentage:.2f}%"
        )

    print("\nComparison statistics:")

    print(
        f"LSH candidate pairs    : "
        f"{candidate_pairs:,}"
    )

    print(
        f"Pairs exactly verified : "
        f"{verified_pairs:,}"
    )

    print("\nCharacter statistics:")

    print(f"Input characters       : {input_characters:,}")
    print(f"Output characters      : {output_characters:,}")
    print(f"Characters removed     : {removed_characters:,}")

    print(f"\nStats saved to: {STATS_FILE}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    near_deduplicate()