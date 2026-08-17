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

# New audit file
AUDIT_FILE = "near_duplicate_pairs.jsonl"


# MinHash configuration
NUM_PERM = 128

# Character shingle size
SHINGLE_SIZE = 5

# Used only to retrieve possible candidates
LSH_THRESHOLD = 0.85

# Final exact Jaccard threshold
SIMILARITY_THRESHOLD = 0.98

# Number of characters to save as preview
PREVIEW_LENGTH = 500


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize("NFC", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHARACTER SHINGLES
# ============================================================

def get_shingles(text, k=SHINGLE_SIZE):

    text = normalize_text(text)

    if not text:
        return set()

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
# HELPER: CREATE TEXT PREVIEW
# ============================================================

def get_preview(text):

    text = normalize_text(text)

    if len(text) <= PREVIEW_LENGTH:
        return text

    return text[:PREVIEW_LENGTH] + "..."


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
    # Create LSH index
    # --------------------------------------------------------

    lsh = MinHashLSH(
        threshold=LSH_THRESHOLD,
        num_perm=NUM_PERM
    )

    # Store shingles for exact verification
    shingle_sets = {}

    # Documents to remove
    removed = set()

    candidate_pairs = 0
    verified_pairs = 0

    print("\nProcessing documents...\n")

    # --------------------------------------------------------
    # Open audit file while processing
    # --------------------------------------------------------

    with open(
        AUDIT_FILE,
        "w",
        encoding="utf-8"
    ) as audit_file:

        # ----------------------------------------------------
        # Process each document
        # ----------------------------------------------------

        for i, document in enumerate(documents):

            text = document["text"]

            # Generate character shingles
            shingles = get_shingles(text)

            # Save shingles for future exact comparisons
            shingle_sets[i] = shingles

            # Generate MinHash
            minhash = create_minhash(shingles)

            # ------------------------------------------------
            # Find candidate duplicates
            # ------------------------------------------------

            candidates = lsh.query(minhash)

            candidate_pairs += len(candidates)

            is_duplicate = False

            # ------------------------------------------------
            # Verify candidates
            # ------------------------------------------------

            for candidate_id in candidates:

                candidate_index = int(candidate_id)

                # This normally shouldn't happen because removed
                # documents are not inserted into the LSH, but
                # we keep the check for safety.
                if candidate_index in removed:
                    continue

                verified_pairs += 1

                candidate_shingles = (
                    shingle_sets[candidate_index]
                )

                similarity = jaccard_similarity(
                    shingles,
                    candidate_shingles
                )

                # --------------------------------------------
                # Near duplicate found
                # --------------------------------------------

                if similarity >= SIMILARITY_THRESHOLD:

                    removed.add(i)

                    is_duplicate = True

                    candidate_document = (
                        documents[candidate_index]
                    )

                    # ----------------------------------------
                    # Save audit record
                    # ----------------------------------------

                    audit_record = {

                        # Removed document
                        "removed_index": i,
                        "removed_id":
                            document["record"].get("id"),

                        "removed_title":
                            document["record"].get("title"),

                        "removed_characters":
                            len(text),

                        "removed_preview":
                            get_preview(text),


                        # Document that was kept
                        "matched_index":
                            candidate_index,

                        "matched_id":
                            candidate_document[
                                "record"
                            ].get("id"),

                        "matched_title":
                            candidate_document[
                                "record"
                            ].get("title"),

                        "matched_characters":
                            len(
                                candidate_document["text"]
                            ),

                        "matched_preview":
                            get_preview(
                                candidate_document["text"]
                            ),


                        # Similarity information
                        "jaccard_similarity":
                            round(similarity, 6)
                    }

                    audit_file.write(
                        json.dumps(
                            audit_record,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                    break

            # ------------------------------------------------
            # Insert only non-duplicates into LSH
            # ------------------------------------------------

            if not is_duplicate:

                lsh.insert(
                    str(i),
                    minhash
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (i + 1) % 1000 == 0:

                print(
                    f"Processed: {i + 1:,} | "
                    f"Kept: {i + 1 - len(removed):,} | "
                    f"Removed: {len(removed):,} | "
                    f"Verified pairs: {verified_pairs:,}"
                )

    # ========================================================
    # WRITE FINAL CORPUS
    # ========================================================

    print("\nWriting deduplicated corpus...")

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
                )
                + "\n"
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

        "audit_file": AUDIT_FILE,


        "configuration": {

            "num_perm": NUM_PERM,

            "shingle_size": SHINGLE_SIZE,

            "lsh_threshold": LSH_THRESHOLD,

            "similarity_threshold":
                SIMILARITY_THRESHOLD
        },


        "document_counts": {

            "input":
                total_documents,

            "kept":
                kept_documents,

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

            "input":
                input_characters,

            "output":
                output_characters,

            "removed":
                removed_characters
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
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)

    print("NEAR DEDUPLICATION COMPLETE")

    print("=" * 60)

    print(
        f"Input documents        : "
        f"{total_documents:,}"
    )

    print(
        f"Kept documents         : "
        f"{kept_documents:,}"
    )

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

    print(
        f"Input characters       : "
        f"{input_characters:,}"
    )

    print(
        f"Output characters      : "
        f"{output_characters:,}"
    )

    print(
        f"Characters removed     : "
        f"{removed_characters:,}"
    )

    print(
        f"\nAudit saved to: "
        f"{AUDIT_FILE}"
    )

    print(
        f"Stats saved to: "
        f"{STATS_FILE}"
    )

    print(
        f"Output saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    near_deduplicate()