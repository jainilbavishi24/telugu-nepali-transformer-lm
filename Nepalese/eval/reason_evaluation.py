import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download

# Determine language from directory structure: .../<language>/eval/
# Current file is in <language>/eval/reason_evaluation.py
CURRENT_FILE = Path(__file__).resolve()
EVAL_DIR = CURRENT_FILE.parent  # Telugu/eval or Nepalese/eval
LANGUAGE_DIR = EVAL_DIR.parent  # Telugu or Nepalese
LANGUAGE = LANGUAGE_DIR.name.lower()  # telugu or nepali
# Normalize: nepalese -> nepali (the dataset uses nepali)
if LANGUAGE == "nepalese":
    LANGUAGE = "nepali"
    DISPLAY_NAME = "Nepali"
else:
    DISPLAY_NAME = "Telugu" if LANGUAGE == "telugu" else LANGUAGE.capitalize()

print(f"Evaluating reasoning for: {DISPLAY_NAME}")

# Paths
PROJECT_ROOT = EVAL_DIR.parent.parent  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DATASET_DIR = PROJECT_ROOT / "reasoning_dataset" / LANGUAGE
TEST_JSONL = REASONING_DATASET_DIR / "test.jsonl"
CONFIG_FILE = LANGUAGE_DIR / "configs" / "config1.yaml"
OUTPUT_DIR = LANGUAGE_DIR / "outputs" / "finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HF configuration
HF_REPO = "iconically-mine/LMA-individual-project"

def load_config():
    """Load model config."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_tokenizer(config):
    """Load SentencePiece tokenizer."""
    tokenizer_path = config["tokenizer"]["model_path"]
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if sp.vocab_size() != config["model"]["vocab_size"]:
        raise RuntimeError(
            f"Tokenizer vocab size {sp.vocab_size()} does not match config "
            f"{config['model']['vocab_size']}"
        )
    return sp

def load_model(language, config, checkpoint_path, device):
    """Load LanguageModel from checkpoint."""
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Handle possible nested structure
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # Remove any prefixes that might have been added during training
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

def download_checkpoint(language, finetuned, device):
    """Download pretrained or finetuned checkpoint from HF Hub."""
    if language == "telugu":
        lang_lower = "telugu"
        base = "Telugu"
    else:
        lang_lower = "nepalese"
        base = "Nepalese"

    if finetuned:
        filename = f"finetuned/{base}/best_{lang_lower}_finetuned.pt"
    else:
        filename = f"{base}/best_{lang_lower}.pt"

    # Create cache directory
    cache_dir = PROJECT_ROOT / ".hf_cache" / base
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {'finetuned' if finetuned else 'pretrained'} checkpoint for {language}...")
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="model",
            local_dir=str(cache_dir),
        )
    )
    return str(checkpoint_path)

def load_test_data(limit=None):
    """Load reasoning test dataset."""
    examples = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                # Ensure required fields exist
                if "prompt" not in example or "answer" not in example:
                    print(f"Warning: line {line_num} missing prompt or answer", file=sys.stderr)
                    continue
                examples.append(example)
                if limit is not None and len(examples) >= limit:
                    break
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_num} invalid JSON: {e}", file=sys.stderr)
    return examples

def generate_answer(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 200, temperature: float = 0.8):
    """Generate answer from prompt and return generated text."""
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,  # disable top-k for now; can be added if needed
        )
    # Extract only the newly generated tokens (excluding input prompt)
    generated_ids = generated_ids[0, len(prompt_ids):].tolist()
    # Decode
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

def evaluate_model(model, tokenizer, examples, device: torch.device, max_new_tokens: int = 200, temperature: float = 0.8):
    """Evaluate model accuracy."""
    correct = 0
    total = len(examples)
    for i, example in enumerate(examples):
        prompt = example["prompt"]
        expected_answer = example["answer"].strip()
        # Generate answer
        generated_text = generate_answer(model, tokenizer, prompt, device, max_new_tokens, temperature)
        # Check if expected answer appears in generated text (simple substring match)
        if expected_answer in generated_text:
            correct += 1
        else:
            # Optional: print first few mismatches for debugging
            if i < 3:
                print(f"  Example {i} mismatch:")
                print(f"    Expected: {expected_answer}")
                print(f"    Generated: {generated_text[:100]}...")
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total} examples")
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def plot_accuracies(pretrained_acc, finetuned_acc, language_name):
    """Create and save bar chart comparing accuracies."""
    plt.figure(figsize=(8, 6))
    models = ['Pretrained', 'Finetuned']
    accuracies = [pretrained_acc, finetuned_acc]
    colors = ['skyblue', 'lightcoral']
    bars = plt.bar(models, accuracies, color=colors, edgecolor='navy', linewidth=1.2)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Reasoning Evaluation Accuracy - {language_name}', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    # Save figure
    output_path = OUTPUT_DIR / "reasoning_evaluation.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")

def main():
    # Choose device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load config and tokenizer
    print(f"Loading config and tokenizer for {LANGUAGE}...")
    config = load_config()
    tokenizer = load_tokenizer(config)

    # Load test data
    print("Loading reasoning test dataset...")
    examples = load_test_data(limit=1000)  
    print(f"Loaded {len(examples)} test examples "f"from {TEST_JSONL}") 

    # Evaluate pretrained model
    print("\n" + "=" * 60)
    print(f"Evaluating PRETRAINED {DISPLAY_NAME} model")
    print("=" * 60)
    try:
        pretrained_checkpoint = download_checkpoint(LANGUAGE, finetuned=False, device=device)
        pretrained_model = load_model(LANGUAGE, config, pretrained_checkpoint, device)
        pretrained_acc, pretrained_correct, pretrained_total = evaluate_model(
            pretrained_model, tokenizer, examples, device
        )
        print(f"Results for pretrained {LANGUAGE}:")
        print(f"  Accuracy: {pretrained_acc:.4f} ({pretrained_correct}/{pretrained_total})")
    except Exception as e:
        print(f"Error evaluating pretrained model: {e}")
        pretrained_acc, pretrained_correct, pretrained_total = 0.0, 0, len(examples)

    # Evaluate finetuned model
    print("\n" + "=" * 60)
    print(f"Evaluating FINETUNED {DISPLAY_NAME} model")
    print("=" * 60)
    try:
        finetuned_checkpoint = download_checkpoint(LANGUAGE, finetuned=True, device=device)
        finetuned_model = load_model(LANGUAGE, config, finetuned_checkpoint, device)
        finetuned_acc, finetuned_correct, finetuned_total = evaluate_model(
            finetuned_model, tokenizer, examples, device
        )
        print(f"Results for finetuned {LANGUAGE}:")
        print(f"  Accuracy: {finetuned_acc:.4f} ({finetuned_correct}/{finetuned_total})")
    except Exception as e:
        print(f"Error evaluating finetuned model: {e}")
        finetuned_acc, finetuned_correct, finetuned_total = 0.0, 0, len(examples)

    # Create visualization
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)
    plot_accuracies(pretrained_acc, finetuned_acc, DISPLAY_NAME)

    # Save results to JSON
    results = {
        "language": LANGUAGE,
        "display_name": DISPLAY_NAME,
        "pretrained": {
            "accuracy": pretrained_acc,
            "correct": int(pretrained_correct),
            "total": int(pretrained_total)
        },
        "finetuned": {
            "accuracy": finetuned_acc,
            "correct": int(finetuned_correct),
            "total": int(finetuned_total)
        },
        "improvement": finetuned_acc - pretrained_acc
    }
    results_path = OUTPUT_DIR / "reasoning_evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Pretrained accuracy: {pretrained_acc:.2%}")
    print(f"Finetuned accuracy:  {finetuned_acc:.2%}")
    print(f"Improvement:         {(finetuned_acc - pretrained_acc):.2%}")

if __name__ == "__main__":
    main()