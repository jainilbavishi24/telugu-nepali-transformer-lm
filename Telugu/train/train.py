"""
train.py

Training loop for the from-scratch decoder-only LM (Phase 2). Talks to the
packed .bin files produced by tokenize_and_pack.py via data_utils/lm_dataset.py.

Covers what the assignment requires for this phase:
  - AdamW + LR scheduling (linear warmup -> cosine decay)
  - validation evaluation during training (loss + perplexity)
  - checkpointing that includes model weights, optimizer state, scheduler
    state, current step, and the config — and can resume from a crash
  - loss curves logged to CSV (plot them in your report/ notebook)

Usage:
    python train.py --config config.yaml
Resumes automatically if a checkpoint already exists in training.out_dir —
delete that directory (or pass --no-resume) to start fresh.
"""

import sys
import os

# Get the directory of the current script (train.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# The project root is one level up from the 'train' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


from model.transformer import LanguageModel
from lm_dataset import make_dataloader


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    return LanguageModel(
        vocab_size=m["vocab_size"],
        max_seq_len=m["max_seq_len"],
        embed_dim=m.get("embed_dim", 256),
        num_layers=m.get("num_layers", 6),
        num_heads=m.get("num_heads", 8),
        ff_hidden_dim=m.get("ff_hidden_dim"),
        dropout=m.get("dropout", 0.0),
        rope_base=m.get("rope_base", 10000.0),
        pad_token_id=m.get("pad_token_id"),
        tie_embeddings=m.get("tie_embeddings", True),
    )


def lr_lambda(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float):
    """Linear warmup, then cosine decay to `min_lr_ratio` of the peak LR."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, cfg: dict) -> None:
    """
    Checkpoint contains everything the assignment requires for resume:
    weights, optimizer state, scheduler state, step, and the config.
    Written to a temp file then renamed, so a crash mid-write can't
    corrupt the checkpoint you'd otherwise resume from.
    """
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"]


@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters: int, pad_token_id=None):
    """Average val loss + perplexity over up to `eval_iters` batches."""
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(eval_iters):
        try:
            input_ids, target_ids = next(it)
        except StopIteration:
            break
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=pad_token_id if pad_token_id is not None else -100,
        )
        losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan"), float("nan")
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")  # guard against overflow early in training
    return avg_loss, ppl


def infinite_loader(loader):
    """Wrap a DataLoader so the training loop is step-based, not epoch-based —
    convenient for huge streamed-then-packed corpora where 'one epoch' isn't
    a meaningful unit to reason about up front."""
    while True:
        for batch in loader:
            yield batch


def main(config_path: str, resume: bool = True):
    cfg = load_config(config_path)
    t = cfg["training"]
    out_dir = Path(t.get("out_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_log.csv"

    device = get_device()
    print(f"Using device: {device}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t.get("learning_rate", 3e-4),
        weight_decay=t.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )
    max_steps = t.get("max_steps", 20000)
    warmup_steps = t.get("warmup_steps", max(1, max_steps // 100))
    min_lr_ratio = t.get("min_lr_ratio", 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, warmup_steps, max_steps, min_lr_ratio)
    )

    data_cfg = cfg["data"]
    bin_dir = Path(data_cfg.get("out_dir", "data_bins"))
    train_loader = make_dataloader(
        str(bin_dir / "pretrain_train.bin"),
        seq_len=cfg["model"]["max_seq_len"],
        stride=data_cfg.get("stride", cfg["model"]["max_seq_len"]),
        batch_size=t["batch_size"],
        num_workers=t.get("num_workers", 2),
        shuffle=True,
    )
    val_loader = make_dataloader(
        str(bin_dir / "pretrain_val.bin"),
        seq_len=cfg["model"]["max_seq_len"],
        stride=cfg["model"]["max_seq_len"],  # non-overlapping windows for eval
        batch_size=t["batch_size"],
        num_workers=t.get("num_workers", 2),
        shuffle=False,
    )

    ckpt_path = out_dir / "latest.pt"
    start_step = 0
    if resume and ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        print(f"Resumed from step {start_step}")
        write_header = False
    else:
        write_header = True

    log_f = open(log_path, "a", newline="")
    log_writer = csv.writer(log_f)
    if write_header:
        log_writer.writerow(["step", "train_loss", "val_loss", "val_ppl", "lr", "seconds"])
        log_f.flush()

    grad_clip = t.get("grad_clip", 1.0)
    eval_interval = t.get("eval_interval", 500)
    eval_iters = t.get("eval_iters", 50)
    save_interval = t.get("save_interval", 500)
    pad_token_id = cfg["model"].get("pad_token_id")

    model.train()
    data_iter = infinite_loader(train_loader)
    t0 = time.time()

    for step in range(start_step, max_steps):
        input_ids, target_ids = next(data_iter)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=pad_token_id if pad_token_id is not None else -100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        val_loss, val_ppl = float("nan"), float("nan")
        if (step + 1) % eval_interval == 0 or step + 1 == max_steps:
            val_loss, val_ppl = evaluate(model, val_loader, device, eval_iters, pad_token_id)
            print(
                f"step {step + 1}/{max_steps} | train_loss {loss.item():.4f} "
                f"| val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f} "
                f"| lr {scheduler.get_last_lr()[0]:.2e}"
            )

        log_writer.writerow(
            [step + 1, loss.item(), val_loss, val_ppl, scheduler.get_last_lr()[0], round(time.time() - t0, 1)]
        )
        if (step + 1) % 50 == 0:
            log_f.flush()

        if (step + 1) % save_interval == 0 or step + 1 == max_steps:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step + 1, cfg)
            # also keep a numbered snapshot every save so you have a history,
            # not just the latest — useful if the latest gets corrupted
            save_checkpoint(out_dir / f"step_{step + 1}.pt", model, optimizer, scheduler, step + 1, cfg)

    log_f.close()
    print(f"Training finished. Loss log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the from-scratch decoder-only LM")
    parser.add_argument("--config", default="Telugu/configs/config1.yaml")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint and start fresh")
    args = parser.parse_args()
    main(args.config, resume=not args.no_resume)