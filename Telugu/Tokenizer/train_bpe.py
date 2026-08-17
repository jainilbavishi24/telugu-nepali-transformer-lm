import json
import re
import unicodedata
import heapq
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_wikipedia_quality_filtered.jsonl"

OUTPUT_DIR = Path("telugu_tokenizer_wiki_v1")

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2

END_OF_WORD = "</w>"

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]

# Rebuild global pair statistics periodically.
# This prevents too many stale heap entries.
REBUILD_EVERY = 1000

# Progress display
PRINT_EVERY = 100


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_words(text):
    """
    Extract Telugu, Latin, and numeric sequences.

    We deliberately preserve English words and numbers because
    they occur naturally in Telugu text.
    """
    return re.findall(
        r"[\u0C00-\u0C7FA-Za-z0-9]+",
        text
    )


# ============================================================
# READ CORPUS
# ============================================================

def build_word_frequencies():

    word_freqs = Counter()

    documents = 0

    print("Reading corpus...")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = normalize_text(
                record.get("text", "")
            )

            words = get_words(text)

            word_freqs.update(words)

            documents += 1

            if documents % 10000 == 0:
                print(
                    f"Processed {documents:,} documents | "
                    f"Unique words: {len(word_freqs):,}"
                )

    return word_freqs


# ============================================================
# INITIAL VOCABULARY
# ============================================================

def build_vocab(word_freqs):
    """
    Each unique word is represented as:

        tuple(word) + (</w>,)

    and weighted by its corpus frequency.
    """

    vocab = {}

    for word, freq in word_freqs.items():

        symbols = tuple(word) + (END_OF_WORD,)

        vocab[symbols] = freq

    return vocab


# ============================================================
# PAIR STATISTICS
# ============================================================

def compute_pair_statistics(vocab):
    """
    Compute:

        pair -> weighted frequency

    Also compute:

        pair -> set of words containing pair

    The second structure lets us update only affected words.
    """

    pair_counts = defaultdict(int)

    pair_to_words = defaultdict(set)

    for word_symbols, word_freq in vocab.items():

        for i in range(len(word_symbols) - 1):

            pair = (
                word_symbols[i],
                word_symbols[i + 1]
            )

            pair_counts[pair] += word_freq

            pair_to_words[pair].add(
                word_symbols
            )

    return pair_counts, pair_to_words


# ============================================================
# HEAP
# ============================================================

def build_heap(pair_counts):
    """
    Python heapq is a min-heap.

    Store negative frequencies to simulate max-heap.
    """

    heap = [
        (-count, pair)
        for pair, count
        in pair_counts.items()
        if count >= MIN_FREQUENCY
    ]

    heapq.heapify(heap)

    return heap


def get_best_pair(heap, pair_counts):
    """
    Retrieve the highest-frequency valid pair.

    Because counts may become stale after updates,
    verify the heap entry against pair_counts.
    """

    while heap:

        negative_count, pair = heapq.heappop(heap)

        count = -negative_count

        current_count = pair_counts.get(
            pair,
            0
        )

        if current_count == count:

            return pair, count

    return None, 0


# ============================================================
# MERGE HELPERS
# ============================================================

def merge_symbols(symbols, pair):

    first, second = pair

    merged = first + second

    result = []

    i = 0

    while i < len(symbols):

        if (
            i < len(symbols) - 1
            and symbols[i] == first
            and symbols[i + 1] == second
        ):

            result.append(merged)

            i += 2

        else:

            result.append(symbols[i])

            i += 1

    return tuple(result)


def get_pairs(symbols):

    return [
        (
            symbols[i],
            symbols[i + 1]
        )
        for i in range(len(symbols) - 1)
    ]


# ============================================================
# APPLY ONE MERGE
# ============================================================

def apply_merge(
    pair,
    vocab,
    pair_counts,
    pair_to_words,
    heap
):
    """
    Only update words that actually contain the selected pair.
    """

    affected_words = list(
        pair_to_words.get(pair, set())
    )

    if not affected_words:
        return vocab

    updates = []

    # --------------------------------------------------------
    # STEP 1:
    # Compute all old/new pair contributions.
    # --------------------------------------------------------

    for old_word in affected_words:

        if old_word not in vocab:
            continue

        freq = vocab[old_word]

        new_word = merge_symbols(
            old_word,
            pair
        )

        updates.append(
            (
                old_word,
                new_word,
                freq
            )
        )

    # --------------------------------------------------------
    # STEP 2:
    # Remove old pair contributions.
    # --------------------------------------------------------

    for old_word, new_word, freq in updates:

        old_pairs = get_pairs(
            old_word
        )

        for old_pair in old_pairs:

            pair_counts[old_pair] -= freq

            if pair_counts[old_pair] <= 0:

                pair_counts.pop(
                    old_pair,
                    None
                )

            if old_pair in pair_to_words:

                pair_to_words[
                    old_pair
                ].discard(old_word)

                if not pair_to_words[old_pair]:

                    pair_to_words.pop(
                        old_pair,
                        None
                    )

    # --------------------------------------------------------
    # STEP 3:
    # Remove old words and insert merged words.
    # --------------------------------------------------------

    for old_word, new_word, freq in updates:

        vocab.pop(
            old_word,
            None
        )

        vocab[new_word] = (
            vocab.get(new_word, 0)
            + freq
        )

    # --------------------------------------------------------
    # STEP 4:
    # Add new pair contributions.
    # --------------------------------------------------------

    for old_word, new_word, freq in updates:

        new_pairs = get_pairs(
            new_word
        )

        for new_pair in new_pairs:

            pair_counts[new_pair] += freq

            pair_to_words[
                new_pair
            ].add(new_word)

            if pair_counts[new_pair] >= MIN_FREQUENCY:

                heapq.heappush(
                    heap,
                    (
                        -pair_counts[new_pair],
                        new_pair
                    )
                )

    # Selected pair no longer needs to be tracked.
    pair_counts.pop(
        pair,
        None
    )

    pair_to_words.pop(
        pair,
        None
    )

    return vocab


# ============================================================
# TRAIN BPE
# ============================================================

def train_bpe(vocab):

    symbols = set()

    for word in vocab:
        symbols.update(word)

    symbols.update(SPECIAL_TOKENS)

    print(
        f"\nInitial vocabulary size: "
        f"{len(symbols):,}"
    )

    print(
        "\nComputing initial pair statistics..."
    )

    pair_counts, pair_to_words = (
        compute_pair_statistics(vocab)
    )

    print(
        f"Unique pairs: "
        f"{len(pair_counts):,}"
    )

    heap = build_heap(
        pair_counts
    )

    merges = []

    merge_number = 0

    print("\nTraining BPE...\n")

    while len(symbols) < VOCAB_SIZE:

        # ----------------------------------------------------
        # Periodic rebuild
        # ----------------------------------------------------

        if (
            merge_number > 0
            and merge_number % REBUILD_EVERY == 0
        ):

            print(
                "\nRebuilding pair statistics..."
            )

            pair_counts, pair_to_words = (
                compute_pair_statistics(vocab)
            )

            heap = build_heap(
                pair_counts
            )

        # ----------------------------------------------------
        # Get best pair
        # ----------------------------------------------------

        best_pair, frequency = get_best_pair(
            heap,
            pair_counts
        )

        if best_pair is None:
            print(
                "\nNo more valid pairs."
            )
            break

        if frequency < MIN_FREQUENCY:
            print(
                f"\nStopping: "
                f"best frequency {frequency} "
                f"< {MIN_FREQUENCY}"
            )
            break

        # ----------------------------------------------------
        # Apply merge
        # ----------------------------------------------------

        vocab = apply_merge(
            best_pair,
            vocab,
            pair_counts,
            pair_to_words,
            heap
        )

        merged_token = (
            best_pair[0]
            + best_pair[1]
        )

        symbols.add(
            merged_token
        )

        merges.append({
            "pair": [
                best_pair[0],
                best_pair[1]
            ],
            "merged": merged_token,
            "frequency": frequency
        })

        merge_number += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if merge_number % PRINT_EVERY == 0:

            print(
                f"Merges: {merge_number:,} | "
                f"Vocab: {len(symbols):,} | "
                f"Best freq: {frequency:,} | "
                f"Unique words: {len(vocab):,}"
            )

    return symbols, merges


# ============================================================
# SAVE TOKENIZER
# ============================================================

def save_tokenizer(symbols, merges):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    regular_tokens = sorted(
        token
        for token in symbols
        if token not in SPECIAL_TOKENS
    )

    all_tokens = (
        SPECIAL_TOKENS
        + regular_tokens
    )

    token_to_id = {
        token: i
        for i, token
        in enumerate(all_tokens)
    }

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "vocab.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            token_to_id,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Merge rules
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "merges.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            merges,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = {
        "type": "BPE",
        "vocab_size": len(token_to_id),
        "special_tokens": SPECIAL_TOKENS,
        "end_of_word": END_OF_WORD,
        "min_frequency": MIN_FREQUENCY,
        "input_file": INPUT_FILE
    }

    with open(
        OUTPUT_DIR / "tokenizer_config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "EFFICIENT TELUGU BPE TOKENIZER TRAINING"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Read corpus
    # --------------------------------------------------------

    word_freqs = build_word_frequencies()

    print("\nCorpus statistics:")

    print(
        f"Unique words: "
        f"{len(word_freqs):,}"
    )

    print(
        f"Total word occurrences: "
        f"{sum(word_freqs.values()):,}"
    )

    # --------------------------------------------------------
    # Initial BPE vocabulary
    # --------------------------------------------------------

    vocab = build_vocab(
        word_freqs
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    symbols, merges = train_bpe(
        vocab
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_tokenizer(
        symbols,
        merges
    )

    print("\n" + "=" * 65)

    print(
        "BPE TRAINING COMPLETE"
    )

    print("=" * 65)

    print(
        f"\nFinal vocabulary size: "
        f"{len(symbols):,}"
    )

    print(
        f"Total merges: "
        f"{len(merges):,}"
    )

    print(
        f"\nTokenizer saved in:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()