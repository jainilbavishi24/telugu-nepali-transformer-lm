import json
import os

import sentencepiece as spm
from sacrebleu.metrics import BLEU, CHRF
import evaluate


# =========================================================
# Configuration
# =========================================================

FILES = {
    "Greedy": "Nepalese/generation_examples_greedy_finetuned.json",
    "Temp 0.5": "Nepalese/generation_examples_0_5_finetuned.json",
    "Temp 1.0": "Nepalese/generation_examples_1_0_finetuned.json",
    "Temp 1.5": "Nepalese/generation_examples_1_5_finetuned.json",
}

TOKENIZER_MODEL = "Nepalese/tokenizer/models/nepalese_sentencepiece_bpe_6k/nepalese_bpe_6k.model"


# =========================================================
# Load SentencePiece tokenizer
# =========================================================

sp = spm.SentencePieceProcessor(
    model_file=TOKENIZER_MODEL
)


# =========================================================
# Load generation examples
# =========================================================

def load_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# SentencePiece tokenization
# =========================================================

def tokenize(text):
    """
    Tokenize generated text using the same SentencePiece
    tokenizer used by the language model.
    """
    return sp.encode(text, out_type=int)


# =========================================================
# Distinct-N
# =========================================================

def distinct_n(tokens, n):
    """
    Distinct-N =
        number of unique n-grams / total number of n-grams

    Higher is more diverse.
    """

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    if len(ngrams) == 0:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


# =========================================================
# Repetition-N
# =========================================================

def repetition_n(tokens, n):
    """
    Repetition-N =
        1 - Distinct-N

    This represents the fraction of n-gram occurrences
    that are repetitions rather than unique n-grams.

    Lower is better.
    """

    return 1.0 - distinct_n(tokens, n)


# =========================================================
# Evaluate one generation file
# =========================================================

def evaluate_file(path, rouge_metric, bleu_metric, chrf_metric):

    examples = load_examples(path)

    predictions = [
        example["generated"]
        for example in examples
    ]

    references = [
        example["reference"]
        for example in examples
    ]

    # -----------------------------------------------------
    # BLEU
    # -----------------------------------------------------

    bleu_score = bleu_metric.corpus_score(
        predictions,
        [references]
    ).score

    # -----------------------------------------------------
    # chrF
    # -----------------------------------------------------

    chrf_score = chrf_metric.corpus_score(
        predictions,
        [references]
    ).score

    # -----------------------------------------------------
    # ROUGE-L
    # -----------------------------------------------------

    rouge_scores = rouge_metric.compute(
        predictions=predictions,
        references=references,
    )

    rouge_l = rouge_scores["rougeL"]

    # -----------------------------------------------------
    # Tokenize generated texts with SentencePiece
    # -----------------------------------------------------

    tokenized_predictions = [
        tokenize(text)
        for text in predictions
    ]

    # -----------------------------------------------------
    # Repetition / Diversity
    # -----------------------------------------------------

    repetition_2 = [
        repetition_n(tokens, 2)
        for tokens in tokenized_predictions
    ]

    repetition_3 = [
        repetition_n(tokens, 3)
        for tokens in tokenized_predictions
    ]

    distinct_1 = [
        distinct_n(tokens, 1)
        for tokens in tokenized_predictions
    ]

    distinct_2 = [
        distinct_n(tokens, 2)
        for tokens in tokenized_predictions
    ]

    # Average over examples
    repetition_2 = sum(repetition_2) / len(repetition_2)
    repetition_3 = sum(repetition_3) / len(repetition_3)

    distinct_1 = sum(distinct_1) / len(distinct_1)
    distinct_2 = sum(distinct_2) / len(distinct_2)

    return {
        "BLEU": bleu_score,
        "chrF": chrf_score,
        "ROUGE-L": rouge_l,
        "Repetition-2": repetition_2,
        "Repetition-3": repetition_3,
        "Distinct-1": distinct_1,
        "Distinct-2": distinct_2,
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # Load metrics once
    rouge_metric = evaluate.load("rouge")

    bleu_metric = BLEU()

    chrf_metric = CHRF()

    results = {}

    # -----------------------------------------------------
    # Evaluate each decoding strategy
    # -----------------------------------------------------

    for name, path in FILES.items():

        if not os.path.exists(path):
            print(f"WARNING: {path} not found")
            continue

        print(f"Evaluating {name}...")

        results[name] = evaluate_file(
            path,
            rouge_metric,
            bleu_metric,
            chrf_metric,
        )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n")
    print("=" * 115)
    print("GENERATION EVALUATION")
    print("=" * 115)

    header = (
        f"{'Decoding':<12}"
        f"{'BLEU':>10}"
        f"{'chrF':>10}"
        f"{'ROUGE-L':>12}"
        f"{'Rep-2':>12}"
        f"{'Rep-3':>12}"
        f"{'Distinct-1':>15}"
        f"{'Distinct-2':>15}"
    )

    print(header)
    print("-" * 115)

    for name, scores in results.items():

        print(
            f"{name:<12}"
            f"{scores['BLEU']:>10.2f}"
            f"{scores['chrF']:>10.2f}"
            f"{scores['ROUGE-L']:>12.4f}"
            f"{scores['Repetition-2']:>12.4f}"
            f"{scores['Repetition-3']:>12.4f}"
            f"{scores['Distinct-1']:>15.4f}"
            f"{scores['Distinct-2']:>15.4f}"
        )

    print("=" * 115)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    with open(
        "Nepalese/outputs/finetuned/generation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nSaved results to generation_metrics.json")