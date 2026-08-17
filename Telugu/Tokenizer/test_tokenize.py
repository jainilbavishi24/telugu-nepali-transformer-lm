import sentencepiece as spm


MODEL_FILE = (
    "telugu_sentencepiece_bpe_8k/"
    "telugu_bpe_8k.model"
)


sp = spm.SentencePieceProcessor(
    model_file=MODEL_FILE
)


test_sentences = [
    "తెలుగు భారతదేశంలోని ఒక ప్రధాన ద్రావిడ భాష.",
    "ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో తెలుగు ప్రధానంగా మాట్లాడబడుతుంది.",
    "హైదరాబాద్ భారతదేశంలోని ప్రముఖ నగరాలలో ఒకటి.",
    "2024 సంవత్సరంలో భారతదేశ జనాభా గురించి సమాచారం అందుబాటులో ఉంది.",
    "Wikipedia లో తెలుగు భాషలో అనేక వ్యాసాలు ఉన్నాయి.",
]


print("=" * 70)
print("TELUGU SENTENCEPIECE TOKENIZER TEST")
print("=" * 70)

print(f"\nVocabulary size: {sp.get_piece_size():,}")


for i, text in enumerate(test_sentences, start=1):

    pieces = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)
    decoded = sp.decode(ids)

    print("\n" + "-" * 70)
    print(f"TEST {i}")
    print("-" * 70)

    print("\nOriginal:")
    print(text)

    print("\nPieces:")
    print(pieces)

    print(f"\nNumber of tokens: {len(ids)}")

    print("\nIDs:")
    print(ids)

    print("\nDecoded:")
    print(decoded)

    print("\nRound-trip correct:")
    print(text == decoded)