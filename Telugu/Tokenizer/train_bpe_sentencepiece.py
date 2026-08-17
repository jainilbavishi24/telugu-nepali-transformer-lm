import sentencepiece as spm
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "telugu_tokenizer_corpus_v2.txt"

OUTPUT_DIR = Path("telugu_sentencepiece_bpe_24k")

VOCAB_SIZE = 24_000

MODEL_PREFIX = OUTPUT_DIR / "telugu_bpe_24k"


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("TRAINING TELUGU SENTENCEPIECE BPE TOKENIZER")
    print("=" * 60)

    print(f"\nInput file       : {INPUT_FILE}")
    print(f"Vocabulary size  : {VOCAB_SIZE:,}")
    print(f"Model type       : BPE")
    print(f"Output directory : {OUTPUT_DIR}")

    print("\nTraining...\n")

    spm.SentencePieceTrainer.train(
        input=INPUT_FILE,
        model_prefix=str(MODEL_PREFIX),
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        num_threads=16,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",

        hard_vocab_limit=False,
    )

    print("\n" + "=" * 60)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nModel: "
        f"{MODEL_PREFIX}.model"
    )

    print(
        f"Vocabulary: "
        f"{MODEL_PREFIX}.vocab"
    )


if __name__ == "__main__":
    main()