#!/usr/bin/env python3
"""Finetune a pretrained Telugu or Nepali LM on the synthetic reasoning data.

This script downloads a *project checkpoint* from Hugging Face using
``hf_hub_download``.  It does not load a Hugging Face model or tokenizer.
The LanguageModel and SentencePiece tokenizer remain the ones you trained.

Example (run from the project root):

  python finetune_reasoning.py \
    --language telugu \
    --train-data reasoning_data/telugu/train.jsonl \
    --validation-data reasoning_data/telugu/validation.jsonl

  python finetune_reasoning.py \
    --language nepali \
    --train-data reasoning_data/nepali/train.jsonl \
    --validation-data reasoning_data/nepali/validation.jsonl

Set HF_TOKEN in the environment before running if the Hub model repository is
private.  The default Hub paths match the existing Phase 2 evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
LANGUAGE_SETTINGS = {
    "telugu": {
        "directory": "Telugu",
        "module": "Telugu.model.transformer",
        "config": "Telugu/configs/config1.yaml",
        "hf_file": "Telugu/best_telugu.pt",
        "dataset_language": "telugu",
    },
    "nepali": {
        "directory": "Nepalese",
        "module": "Nepalese.model.transformer",
        "config": "Nepalese/configs/config1.yaml",
        "hf_file": "Nepalese/best_nepalese.pt",
        "dataset_language": "nepali",
    },
}
DEFAULT_HF_REPO = "iconically-mine/LMA-individual-project"


@dataclass
class Example:
    """A tokenized training row with loss labels only on reasoning + answer."""

    input_ids: list[int]
    labels: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=sorted(LANGUAGE_SETTINGS), required=True)
    parser.add_argument("--train-data", type=Path, required=True, help="JSONL split with prompt and target fields")
    parser.add_argument("--validation-data", type=Path, required=True, help="JSONL split with prompt and target fields")
    parser.add_argument("--hf-repo", default="iconically-mine/LMA-individual-project", help="Hub model repository containing the pretrained .pt file")
    parser.add_argument("--hf-file", default="Nepalese/best_nepalese.pt", help="Path of the pretrained .pt file inside --hf-repo")
    parser.add_argument("--hf-revision", default=None, help="Optional Hub branch, tag, or commit hash")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".hf_cache")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume a finetuning checkpoint written by this script")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Defaults to the pretrained model's context length")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu; defaults to the best available")
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(language: str, config: dict[str, Any]) -> torch.nn.Module:
    """Construct the exact from-scratch architecture used during pretraining."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if language == "telugu":
        from Telugu.model.transformer import LanguageModel
    else:
        from Nepalese.model.transformer import LanguageModel

    model_cfg = config["model"]
    return LanguageModel(
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


def normalized_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Accept the project's normal checkpoint wrapper or a plain state dict."""
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint is neither a state_dict nor a dict containing 'model_state_dict'.")
    clean: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        clean[key] = value
    return clean


def download_and_load_pretrained(
    model: torch.nn.Module,
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: Path,
    device: torch.device,
) -> Path:
    """Download the selected project checkpoint and fail fast on a config mismatch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/checking {repo_id}:{filename}")
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            local_dir=str(cache_dir),
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompat = model.load_state_dict(normalized_state_dict(checkpoint), strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/config mismatch.\n"
            f"Missing keys: {incompat.missing_keys}\nUnexpected keys: {incompat.unexpected_keys}\n"
            "Check that --language, config1.yaml, and --hf-file refer to the same pretrained model."
        )
    model.to(device)
    return checkpoint_path


class ReasoningDataset(Dataset[Example]):
    """JSONL dataset with loss masked on the prompt and enabled on the target.

    The prompt ends with the language-specific reasoning marker.  The target
    contains the reasoning trace and answer marker.  Therefore the model learns
    to generate the requested completion while not spending loss on copying the
    question back to the user.
    """

    def __init__(self, path: Path, tokenizer: spm.SentencePieceProcessor, max_seq_len: int, expected_language: str):
        self.examples: list[Example] = []
        eos_id = tokenizer.eos_id()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row.get("language") not in (None, expected_language):
                    raise ValueError(f"{path}:{line_number} has language={row.get('language')!r}, expected {expected_language!r}")
                if not row.get("prompt") or not row.get("target"):
                    raise ValueError(f"{path}:{line_number} must contain non-empty 'prompt' and 'target' fields")

                # Encode the pieces separately so the target boundary remains
                # explicit; labels are calculated over the resulting next-token
                # sequence and ignore every prompt prediction.
                prompt_ids = tokenizer.encode(row["prompt"], out_type=int)
                target_ids = tokenizer.encode(row["target"], out_type=int)
                if eos_id >= 0:
                    target_ids.append(eos_id)

                if len(target_ids) >= max_seq_len:
                    raise ValueError(f"{path}:{line_number} target is too long for max_seq_len={max_seq_len}")
                allowed_prompt = max_seq_len - len(target_ids)
                prompt_ids = prompt_ids[-allowed_prompt:]
                tokens = prompt_ids + target_ids
                if len(tokens) < 2:
                    continue

                # logits[t] predicts tokens[t + 1].  Only target tokens are
                # supervised; the final prompt token predicts target_ids[0].
                labels = tokens[1:]
                first_supervised = max(0, len(prompt_ids) - 1)
                labels[:first_supervised] = [-100] * first_supervised
                self.examples.append(Example(input_ids=tokens[:-1], labels=labels))
        if not self.examples:
            raise ValueError(f"No usable records found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def collate(batch: list[Example], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad input ids, attention mask, and -100 loss labels to one batch length."""
    max_len = max(len(item.input_ids) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(item.labels, dtype=torch.long)
        attention_mask[row, :length] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, autocast_enabled: bool) -> tuple[float, float]:
    """Return target-token cross-entropy and perplexity on the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                logits = model(input_ids, attention_mask=attention_mask)
                loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
            token_count = int((labels != -100).sum().item())
            total_loss += float(loss_sum.item())
            total_tokens += token_count
    loss = total_loss / max(total_tokens, 1)
    return loss, math.exp(loss) if loss < 20 else float("inf")


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, scheduler, scaler, step: int, epoch: int, best_val: float, args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Save enough state to resume finetuning safely after an interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "reasoning_finetune",
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "finetune_args": vars(args),
        "pretrain_config": config,
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("epochs, batch-size, and grad-accum-steps must all be positive")
    settings = LANGUAGE_SETTINGS[args.language]
    config_path = PROJECT_ROOT / settings["config"]
    config = load_yaml(config_path)
    max_seq_len = args.max_seq_len or int(config["model"]["max_seq_len"])
    args.output_dir = args.output_dir or PROJECT_ROOT / "checkpoints" / "reasoning_finetune" / args.language
    args.hf_file = args.hf_file or settings["hf_file"]
    args.train_data = args.train_data.resolve()
    args.validation_data = args.validation_data.resolve()
    if not args.train_data.exists() or not args.validation_data.exists():
        raise FileNotFoundError("Both --train-data and --validation-data must exist before finetuning.")

    device = choose_device(args.device)
    seed_everything(args.seed)
    tokenizer_path = PROJECT_ROOT / config["tokenizer"]["model_path"]
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() != config["model"]["vocab_size"]:
        raise RuntimeError(f"Tokenizer has {tokenizer.vocab_size()} pieces; config expects {config['model']['vocab_size']}.")
    pad_id = tokenizer.pad_id()
    if pad_id < 0:
        # Padding is only a batching detail.  Labels at padded positions are
        # ignored and attention_mask prevents them becoming usable keys.
        pad_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else 0

    print(f"Device: {device}; tokenizer vocabulary: {tokenizer.vocab_size():,}; context: {max_seq_len}")
    train_ds = ReasoningDataset(args.train_data, tokenizer, max_seq_len, settings["dataset_language"])
    val_ds = ReasoningDataset(args.validation_data, tokenizer, max_seq_len, settings["dataset_language"])
    print(f"Examples: train={len(train_ds):,}, validation={len(val_ds):,}")
    make_loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=lambda batch: collate(batch, pad_id))
    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)

    model = build_model(args.language, config)
    if args.resume is None:
        checkpoint_path = download_and_load_pretrained(model, args.hf_repo, args.hf_file, args.hf_revision, args.cache_dir, device)
        print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    else:
        model.to(device)

    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = round(total_updates * args.warmup_ratio)

    def lr_scale(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / max(warmup_updates, 1)
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    start_epoch = 0
    best_val = float("inf")
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(normalized_state_dict(resume_payload))
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict"):
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
        step = int(resume_payload.get("step", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
        best_val = float(resume_payload.get("best_val_loss", float("inf")))
        print(f"Resumed finetuning at step {step:,}, epoch {start_epoch}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()

            is_update = (batch_index + 1) % args.grad_accum_steps == 0 or batch_index + 1 == len(train_loader)
            if not is_update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % 25 == 0:
                print(f"epoch={epoch + 1}/{args.epochs} step={step:,}/{total_updates:,} loss={loss.item() * args.grad_accum_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
            if step % args.eval_interval == 0 or step == total_updates:
                val_loss, val_ppl = evaluate(model, val_loader, device, use_amp)
                print(f"validation step={step:,}: loss={val_loss:.4f} perplexity={val_ppl:.3f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(args.output_dir / "best.pt", model, optimizer, scheduler, scaler, step, epoch, best_val, args, config)
                    print("Saved new best checkpoint.")
                model.train()
            if step % args.save_interval == 0:
                save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scheduler, scaler, step, epoch, best_val, args, config)

        save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scheduler, scaler, step, epoch + 1, best_val, args, config)

    val_loss, val_ppl = evaluate(model, val_loader, device, use_amp)
    print(f"Finished. Final validation loss={val_loss:.4f}, perplexity={val_ppl:.3f}; best={best_val:.4f}")


if __name__ == "__main__":
    main()
