"""Reasoning evaluation: pretrained vs finetuned.

Scoring is strict: the model must emit an "उत्तर:" / "సమాధానం:" line and the
normalized text after it must match the gold answer exactly. Loose substring
matching is still computed, but only as a generous upper bound -- it is not the
headline number, because premises in the prompt often contain the gold answer
token for the relation families.
"""

import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch
import yaml
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------- setup

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

CURRENT_FILE = Path(__file__).resolve()
EVAL_DIR = CURRENT_FILE.parent
LANGUAGE_DIR = EVAL_DIR.parent
LANGUAGE = LANGUAGE_DIR.name.lower()

if LANGUAGE == "nepalese":
    LANGUAGE = "nepali"
    DISPLAY_NAME = "Nepali"
else:
    DISPLAY_NAME = "Telugu" if LANGUAGE == "telugu" else LANGUAGE.capitalize()

print(f"Evaluating reasoning for: {DISPLAY_NAME}")

PROJECT_ROOT = EVAL_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REASONING_DATASET_DIR = PROJECT_ROOT / "reasoning_dataset" / LANGUAGE
TEST_JSONL = REASONING_DATASET_DIR / "test.jsonl"
CONFIG_FILE = LANGUAGE_DIR / "configs" / "config1.yaml"
OUTPUT_DIR = LANGUAGE_DIR / "outputs" / "finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO = "iconically-mine/LMA-individual-project"

# Generation budget. Targets in this dataset are short (~60-80 tokens); 120
# leaves headroom for a rambling pretrained model without wasting compute.
MAX_NEW_TOKENS = 120
EVAL_LIMIT = None  # None = use the whole test file

# Answer markers, tried in order. Add your Telugu marker here if it differs.
ANSWER_MARKERS = ("उत्तर:", "సమాధానం:", "జవాబు:")

# Random-guess baselines. Keyed by family; adjust if you add templates.
CHANCE_LEVEL = {
    "two_hop": 1.0 / 3.0,       # which of three entities is the middle
    "three_hop": 1.0 / 3.0,     # greater / less / equal
    "ranking": 1.0 / 6.0,       # orderings of three entities
}

# ---------------------------------------------------------------- loading


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer(config):
    tokenizer_path = config["tokenizer"]["model_path"]
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if sp.vocab_size() != config["model"]["vocab_size"]:
        raise RuntimeError(
            f"Tokenizer vocab size {sp.vocab_size()} does not match config "
            f"{config['model']['vocab_size']}"
        )
    return sp


def load_model(language, config, checkpoint_path, device):
    if language == "telugu":
        from Telugu.model.transformer import LanguageModel
    else:
        from Nepalese.model.transformer import LanguageModel

    model_cfg = config["model"]
    model = LanguageModel(
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        embed_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ff_hidden_dim=model_cfg["ff_hidden_dim"],
        dropout=model_cfg["dropout"],
        attn_dropout=model_cfg["attn_dropout"],
        rope_base=model_cfg["rope_base"],
        pad_token_id=model_cfg["pad_token_id"],
        tie_embeddings=model_cfg["tie_embeddings"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def download_checkpoint(language, finetuned):
    if language == "telugu":
        lang_lower, base = "telugu", "Telugu"
    else:
        lang_lower, base = "nepalese", "Nepalese"

    if finetuned:
        filename = f"finetuned/{base}/best_{lang_lower}_finetuned.pt"
    else:
        filename = f"{base}/best_{lang_lower}.pt"

    cache_dir = PROJECT_ROOT / ".hf_cache" / base
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {'finetuned' if finetuned else 'pretrained'} checkpoint...")
    return str(
        Path(
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                repo_type="model",
                local_dir=str(cache_dir),
            )
        )
    )


def load_test_data(limit=None):
    examples = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_num} invalid JSON: {e}", file=sys.stderr)
                continue
            if "prompt" not in example or "answer" not in example:
                print(f"Warning: line {line_num} missing prompt/answer", file=sys.stderr)
                continue
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    return examples


# ---------------------------------------------------------------- scoring


def normalize(text):
    """Normalize for exact comparison: NFC, strip danda, collapse list spacing."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("।", " ").replace("॥", " ")
    text = text.replace("\u060C", ",").replace("，", ",")
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;\u0964")


def extract_answer(generated_text):
    """Return the text on the answer line, or None if no marker was emitted."""
    for marker in ANSWER_MARKERS:
        idx = generated_text.find(marker)
        if idx != -1:
            tail = generated_text[idx + len(marker):]
            return tail.split("\n")[0].strip()
    return None


def score_one(generated_text, expected_answer):
    """Return (strict, loose, emitted_marker, predicted_answer_or_None)."""
    predicted = extract_answer(generated_text)
    emitted = predicted is not None
    strict = emitted and normalize(predicted) == normalize(expected_answer)
    loose = normalize(expected_answer) in normalize(generated_text)
    return strict, loose, emitted, predicted


# ---------------------------------------------------------------- generation


def greedy_generate(model, tokenizer, prompt, device, max_seq_len,
                    max_new_tokens=MAX_NEW_TOKENS):
    """Deterministic decoding with early stopping at EOS or end of answer line.

    Uses model.generate when available, and falls back to a manual argmax loop
    if greedy args are not supported. The manual loop is also what gives the
    early stop, which is most of the speedup.
    """
    prompt_ids = tokenizer.encode(prompt, out_type=int)

    budget = max_seq_len - len(prompt_ids)
    if budget <= 0:
        raise ValueError(
            f"Prompt of {len(prompt_ids)} tokens leaves no room under "
            f"max_seq_len={max_seq_len}. Shorten the prompt or raise max_seq_len."
        )
    n_new = min(max_new_tokens, budget)

    eos_id = tokenizer.eos_id()
    newline_ids = {
        i for i in (tokenizer.piece_to_id("<0x0A>"),) if i is not None and i > 0
    }

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    seen_marker = False

    with torch.no_grad():
        for _ in range(n_new):
            out = model(input_ids)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(logits, dict):
                logits = logits.get("logits", next(iter(logits.values())))
            next_id = int(torch.argmax(logits[0, -1, :]).item())

            if eos_id is not None and eos_id >= 0 and next_id == eos_id:
                break

            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )

            # Early stop: once an answer marker has appeared, end at the newline.
            text_so_far = tokenizer.decode(generated)
            if not seen_marker and any(m in text_so_far for m in ANSWER_MARKERS):
                seen_marker = True
            elif seen_marker and next_id in newline_ids:
                break

            if input_ids.shape[1] >= max_seq_len:
                break

    return tokenizer.decode(generated)


def evaluate_model(model, tokenizer, examples, device, max_seq_len, tag=""):
    """Run the model over the test set and return a results dict."""
    per_example = []
    by_family = defaultdict(lambda: {"strict": 0, "loose": 0, "emitted": 0, "n": 0})
    strict_total = loose_total = emitted_total = 0
    shown = 0

    for i, example in enumerate(examples):
        expected = example["answer"].strip()
        family = example.get("family", "unknown")

        generated = greedy_generate(
            model, tokenizer, example["prompt"], device, max_seq_len
        )
        strict, loose, emitted, predicted = score_one(generated, expected)

        strict_total += int(strict)
        loose_total += int(loose)
        emitted_total += int(emitted)

        fam = by_family[family]
        fam["strict"] += int(strict)
        fam["loose"] += int(loose)
        fam["emitted"] += int(emitted)
        fam["n"] += 1

        per_example.append(
            {
                "id": example.get("id", f"idx_{i}"),
                "family": family,
                "template_id": example.get("template_id"),
                "expected": expected,
                "predicted": predicted,
                "strict": bool(strict),
                "loose": bool(loose),
                "emitted_marker": bool(emitted),
            }
        )

        if not strict and shown < 3:
            shown += 1
            print(f"  [{tag}] miss #{shown}  family={family}")
            print(f"    expected : {expected}")
            print(f"    predicted: {predicted!r}")
            print(f"    raw      : {generated[:120]!r}...")

        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i + 1}/{len(examples)} done")

    n = len(examples)
    return {
        "status": "ok",
        "n": n,
        "strict_correct": strict_total,
        "strict_accuracy": strict_total / n if n else 0.0,
        "loose_correct": loose_total,
        "loose_accuracy": loose_total / n if n else 0.0,
        "format_compliance": emitted_total / n if n else 0.0,
        "by_family": {
            k: {
                "n": v["n"],
                "strict_accuracy": v["strict"] / v["n"],
                "loose_accuracy": v["loose"] / v["n"],
                "format_compliance": v["emitted"] / v["n"],
            }
            for k, v in sorted(by_family.items())
        },
        "per_example": per_example,
    }


def failed_result(n, reason):
    return {
        "status": "error",
        "error": reason,
        "n": n,
        "strict_correct": 0,
        "strict_accuracy": 0.0,
        "loose_correct": 0,
        "loose_accuracy": 0.0,
        "format_compliance": 0.0,
        "by_family": {},
        "per_example": [],
    }


# ---------------------------------------------------------------- statistics


def wilson_interval(correct, n, z=1.96):
    """95% Wilson score interval -- behaves sensibly near 0 and 1, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact(a_results, b_results):
    """Paired test on the same prompts. Returns (b01, b10, two-sided p)."""
    by_id_a = {r["id"]: r["strict"] for r in a_results}
    by_id_b = {r["id"]: r["strict"] for r in b_results}
    shared = set(by_id_a) & set(by_id_b)

    b01 = sum(1 for i in shared if not by_id_a[i] and by_id_b[i])  # only B right
    b10 = sum(1 for i in shared if by_id_a[i] and not by_id_b[i])  # only A right
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0

    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


# ---------------------------------------------------------------- reporting


def plot_results(pre, fine, language_name):
    families = sorted(set(pre["by_family"]) | set(fine["by_family"]))
    labels = ["Overall"] + families

    pre_vals = [pre["strict_accuracy"]] + [
        pre["by_family"].get(f, {}).get("strict_accuracy", 0.0) for f in families
    ]
    fine_vals = [fine["strict_accuracy"]] + [
        fine["by_family"].get(f, {}).get("strict_accuracy", 0.0) for f in families
    ]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 3, 6))
    b1 = ax.bar(x - width / 2, pre_vals, width, label="Pretrained",
                color="#8ecae6", edgecolor="#023047")
    b2 = ax.bar(x + width / 2, fine_vals, width, label="Finetuned",
                color="#fb8500", edgecolor="#023047")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    # Chance level per family, drawn as a short dashed segment over each group.
    for i, label in enumerate(labels):
        chance = CHANCE_LEVEL.get(label)
        if chance is None and label == "Overall":
            known = [CHANCE_LEVEL.get(f) for f in families]
            weights = [pre["by_family"].get(f, {}).get("n", 0) for f in families]
            if all(c is not None for c in known) and sum(weights):
                chance = sum(c * w for c, w in zip(known, weights)) / sum(weights)
        if chance is not None:
            ax.plot([i - 0.45, i + 0.45], [chance, chance],
                    linestyle="--", color="#d62828", linewidth=1.4,
                    label="Chance" if i == 0 else None)

    ax.set_ylabel("Strict accuracy (exact answer match)", fontsize=12)
    ax.set_title(f"Reasoning evaluation — {language_name}", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    path = OUTPUT_DIR / "reasoning_evaluation.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {path}")


def print_report(pre, fine):
    pre_lo, pre_hi = wilson_interval(pre["strict_correct"], pre["n"])
    fin_lo, fin_hi = wilson_interval(fine["strict_correct"], fine["n"])
    b01, b10, p = mcnemar_exact(pre["per_example"], fine["per_example"])

    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"{'':<14}{'strict':>10}{'95% CI':>18}{'loose':>10}{'format':>10}")
    print(f"{'Pretrained':<14}{pre['strict_accuracy']:>10.3f}"
          f"{f'[{pre_lo:.3f}, {pre_hi:.3f}]':>18}"
          f"{pre['loose_accuracy']:>10.3f}{pre['format_compliance']:>10.3f}")
    print(f"{'Finetuned':<14}{fine['strict_accuracy']:>10.3f}"
          f"{f'[{fin_lo:.3f}, {fin_hi:.3f}]':>18}"
          f"{fine['loose_accuracy']:>10.3f}{fine['format_compliance']:>10.3f}")

    print("\nPer family (strict accuracy):")
    for fam in sorted(set(pre["by_family"]) | set(fine["by_family"])):
        p_acc = pre["by_family"].get(fam, {}).get("strict_accuracy", 0.0)
        f_acc = fine["by_family"].get(fam, {}).get("strict_accuracy", 0.0)
        n_fam = fine["by_family"].get(fam, {}).get("n", 0)
        chance = CHANCE_LEVEL.get(fam)
        chance_str = f"chance {chance:.3f}" if chance is not None else "chance n/a"
        print(f"  {fam:<14} n={n_fam:<5} pre {p_acc:.3f}  fine {f_acc:.3f}  "
              f"({chance_str})")

    print(f"\nMcNemar (paired, exact): finetuned-only correct = {b01}, "
          f"pretrained-only correct = {b10}, p = {p:.4f}")
    if p >= 0.05:
        print("  Not significant at alpha=0.05 -- do not claim a difference.")
    print(f"\nAbsolute improvement (strict): "
          f"{fine['strict_accuracy'] - pre['strict_accuracy']:+.2%}")


# ---------------------------------------------------------------- main


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading config and tokenizer for {LANGUAGE}...")
    config = load_config()
    tokenizer = load_tokenizer(config)
    max_seq_len = config["model"]["max_seq_len"]

    print("Loading reasoning test dataset...")
    examples = load_test_data(limit=EVAL_LIMIT)
    print(f"Loaded {len(examples)} test examples from {TEST_JSONL}")
    if len(examples) < 300:
        print(f"  NOTE: n={len(examples)} gives roughly a +/-{1.96 * 0.5 / math.sqrt(len(examples)):.2f} "
              "confidence half-width. Consider a larger test split.")

    results = {}
    for tag, finetuned in (("pretrained", False), ("finetuned", True)):
        print("\n" + "=" * 68)
        print(f"Evaluating {tag.upper()} {DISPLAY_NAME} model")
        print("=" * 68)
        model = None
        try:
            ckpt = download_checkpoint(LANGUAGE, finetuned=finetuned)
            model = load_model(LANGUAGE, config, ckpt, device)
            results[tag] = evaluate_model(
                model, tokenizer, examples, device, max_seq_len, tag=tag
            )
            print(f"  strict={results[tag]['strict_accuracy']:.4f}  "
                  f"loose={results[tag]['loose_accuracy']:.4f}  "
                  f"format={results[tag]['format_compliance']:.4f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[tag] = failed_result(len(examples), f"{type(e).__name__}: {e}")
        finally:
            if model is not None:
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    pre, fine = results["pretrained"], results["finetuned"]

    if pre["status"] == "ok" and fine["status"] == "ok":
        print_report(pre, fine)
        plot_results(pre, fine, DISPLAY_NAME)
    else:
        print("\nOne or both evaluations failed; skipping comparison and plot.")

    payload = {
        "language": LANGUAGE,
        "display_name": DISPLAY_NAME,
        "seed": SEED,
        "decoding": "greedy (argmax), early stop at answer-line newline or EOS",
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_examples": len(examples),
        "scoring": {
            "strict": "exact match of normalized text after the answer marker",
            "loose": "normalized gold answer appears anywhere (upper bound only)",
            "format_compliance": "fraction of outputs that emitted an answer marker",
        },
        "chance_level": CHANCE_LEVEL,
        "pretrained": {k: v for k, v in pre.items() if k != "per_example"},
        "finetuned": {k: v for k, v in fine.items() if k != "per_example"},
    }
    if pre["status"] == "ok" and fine["status"] == "ok":
        b01, b10, p = mcnemar_exact(pre["per_example"], fine["per_example"])
        payload["comparison"] = {
            "strict_improvement": fine["strict_accuracy"] - pre["strict_accuracy"],
            "mcnemar": {
                "finetuned_only_correct": b01,
                "pretrained_only_correct": b10,
                "p_value": p,
                "significant_at_0.05": p < 0.05,
            },
            "pretrained_ci95": wilson_interval(pre["strict_correct"], pre["n"]),
            "finetuned_ci95": wilson_interval(fine["strict_correct"], fine["n"]),
        }

    results_path = OUTPUT_DIR / "reasoning_evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {results_path}")

    # Per-example records go in a separate file -- useful for error analysis.
    preds_path = OUTPUT_DIR / "reasoning_evaluation_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for tag in ("pretrained", "finetuned"):
            for row in results[tag]["per_example"]:
                f.write(json.dumps({"model": tag, **row}, ensure_ascii=False) + "\n")
    print(f"Saved per-example predictions to: {preds_path}")


if __name__ == "__main__":
    main()