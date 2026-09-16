#!/usr/bin/env python3
"""
Reasoning evaluation for Telugu and Nepali language models.
Evaluates accuracy on reasoning test set by generating answers from prompts
and checking if the expected answer appears in the generated text.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download

# Add project root to sys.path so we can import model modules
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(language: str):
    """Load model config for given language."""
    config_path = PROJECT_ROOT / f"{language.title() if language == 'telugu' else 'Nepalese'}/configs/config1.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer(language: str, config):
    """Load SentencePiece tokenizer."""
    tokenizer_path = PROJECT_ROOT / config["tokenizer"]["model_path"]
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if sp.vocab_size() != config["model"]["vocab_size"]:
        raise RuntimeError(
            f"Tokenizer vocab size {sp.vocab_size()} does not match config "
            f"{config['model']['vocab_size']}"
        )
    return sp


def load_model(language: str, config, checkpoint_path: str, device: torch.device):
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


def download_checkpoint(language: str, finetuned: bool, device: torch.device, hf_repo: str = "iconically-mine/LMA-individual-project"):
    """Download pretrained or finetuned checkpoint from HF Hub."""
    if language == "telugu":
        base = "Telugu"
        lang_lower = "telugu"
    else:
        base = "Nepalese"
        lang_lower = "nepali"

    if finetuned:
        filename = f"finetuned/{base}/{lang_lower}_finetuned.pt"
    else:
        filename = f"{base}/best_{lang_lower}.pt"

    # Create cache directory
    cache_dir = PROJECT_ROOT / ".hf_cache" / base
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {'finetuned' if finetuned else 'pretrained'} checkpoint for {language}...")
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="model",
            local_dir=str(cache_dir),
        )
    )
    return str(checkpoint_path)


def load_test_data(language: str, limit: int | None = None):
    """Load reasoning test dataset."""
    if language == "telugu":
        test_path = PROJECT_ROOT / "reasoning_dataset" / "telugu" / "test.jsonl"
    else:
        test_path = PROJECT_ROOT / "reasoning_dataset" / "nepali" / "test.jsonl"

    examples = []
    with open(test_path, "r", encoding="utf-8") as f:
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


def evaluate_reasoning(model, tokenizer, examples, device: torch.device, max_new_tokens: int = 200, temperature: float = 0.8):
    """Evaluate reasoning accuracy."""
    correct = 0
    total = len(examples)
    for i, example in enumerate(examples):
        prompt = example["prompt"]
        expected_answer = example["answer"].strip()
        # Generate answer
        generated_text = generate_answer(model, tokenizer, prompt, device, max_new_tokens, temperature)
        # Check if expected answer appears in generated text (simple substring match)
        # We'll also strip whitespace from generated text for matching
        if expected_answer in generated_text:
            correct += 1
        else:
            # Optional: print first few mismatches for debugging
            if i < 5:
                print(f"Example {i}:")
                print(f"  Prompt: {prompt[:100]}...")
                print(f"  Expected answer: {expected_answer}")
                print(f"  Generated text: {generated_text[:200]}...")
                print()
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total} examples")
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning accuracy")
    parser.add_argument(
        "--language",
        choices=["telugu", "nepali"],
        required=True,
        help="Language to evaluate",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Evaluate pretrained model"
    )
    parser.add_argument(
        "--finetuned", action="store_true", help="Evaluate finetuned model"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Evaluate both pretrained and finetuned models",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing)",
    )
    parser.add_argument(
        "--hf-repo",
        default="iconically-mine/LMA-individual-project",
        help="Hugging Face repository ID",
    )
    args = parser.parse_args()

    if not (args.pretrained or args.finetuned or args.both):
        parser.error("At least one of --pretrained, --finetuned, or --both must be specified")

    # Choose device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load config and tokenizer
    print(f"Loading config and tokenizer for {args.language}...")
    config = load_config(args.language)
    tokenizer = load_tokenizer(args.language, config)

    # Load test data
    print("Loading reasoning test dataset...")
    examples = load_test_data(args.language, limit=args.limit)
    print(f"Loaded {len(examples)} test examples")

    # Evaluate pretrained model
    if args.pretrained or args.both:
        print("\n" + "=" * 60)
        print(f"Evaluating PRETRAINED {args.language.upper()} model")
        print("=" * 60)
        checkpoint_path = download_checkpoint(args.language, finetuned=False, device=device, hf_repo=args.hf_repo)
        model = load_model(args.language, config, checkpoint_path, device)
        accuracy, correct, total = evaluate_reasoning(
            model, tokenizer, examples, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"Results for pretrained {args.language}:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Evaluate finetuned model
    if args.finetuned or args.both:
        print("\n" + "=" * 60)
        print(f"Evaluating FINETUNED {args.language.upper()} model")
        print("=" * 60)
        checkpoint_path = download_checkpoint(args.language, finetuned=True, device=device, hf_repo=args.hf_repo)
        model = load_model(args.language, config, checkpoint_path, device)
        accuracy, correct, total = evaluate_reasoning(
            model, tokenizer, examples, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"Results for finetuned {args.language}:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.both:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        # Re-run both to get accuracies (or we could store them)
        # For simplicity, we'll just note that the user can compare above.


if __name__ == "__main__":
    main()