"""
tokenize_and_pack.py

Pipeline: config.yaml -> stream JSONL from HF Hub -> deterministic train/val
split -> SentencePiece tokenize -> pack token ids into flat .bin files that
lm_dataset.py memory-maps at training time.

Why a flat .bin file instead of a list-of-lists / HF `datasets` object?
For ~500M tokens per language, keeping everything as Python objects (or as
an in-memory arrow table) is wasteful. A single flat array of token ids on
disk, opened with np.memmap, gives O(1) random access with ~0 RAM overhead,
which is the standard approach (nanoGPT-style) for from-scratch LM pretraining.

Resumability: streaming from the Hub means we can't byte-seek into the
middle of an HTTP response after a crash. Instead we persist how many
*source records* we've already packed (progress.json) and, on restart,
fast-forward through the stream (cheap: just json.loads + skip) until we
reach that point, then resume writing new tokens. This satisfies the
checkpoint-resume requirement without needing Range-request byte offsets.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import sentencepiece as spm
import yaml

from stream_jsonl import StreamStats, stream_hf_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# uint16 covers vocab sizes up to 65535; assert this matches config so a
# vocab-size change later doesn't silently corrupt ids.
TOKEN_DTYPE = np.uint16
MAX_VOCAB_FOR_DTYPE = 65536


def load_config(path: str) -> dict:
    """Load and lightly validate the project config.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "training", "data", "tokenizer"):
        if section not in cfg:
            raise ValueError(f"config.yaml is missing the '{section}' section")
    return cfg


def is_validation_record(text: str, seed: int, val_fraction: float) -> bool:
    """
    Deterministic, single-pass train/val split.

    Rather than buffering all records to do a random split (impossible while
    streaming a 500M-token file), we hash each record's text together with
    the seed and threshold the hash. This is reproducible across runs/resumes
    without needing to know the total record count in advance, and it won't
    put near-duplicate lines on both sides just because of ordering.
    """
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    # take a chunk of the hex digest as a uniform float in [0, 1)
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return frac < val_fraction


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"records_consumed": 0, "tokens_written": {"train": 0, "val": 0}}


def save_progress(progress_path: Path, progress: dict) -> None:
    tmp = progress_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f)
    tmp.replace(progress_path)  # atomic-ish swap so a crash mid-write can't corrupt it


def pack_split(
    repo_id: str,
    repo_type: str,
    filename: str,
    field_candidates,
    sp: spm.SentencePieceProcessor,
    out_dir: Path,
    split_name: str,
    seed: int,
    val_fraction: float,
    log_every: int = 50_000,
) -> dict:
    """
    Stream `filename`, tokenize each record, and append ids to
    `{split_name}_train.bin` / `{split_name}_val.bin` (skipped if val_fraction
    is 0, e.g. for the held-out test file). Resumable via progress.json.

    Returns the final StreamStats as a dict for reporting in your Phase 1/2
    dataset-statistics writeup.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / f"{split_name}_progress.json"
    progress = load_progress(progress_path)
    already_done = progress["records_consumed"]

    train_bin = out_dir / f"{split_name}_train.bin"
    val_bin = out_dir / f"{split_name}_val.bin"
    # append mode: on resume we keep whatever was already flushed to disk
    train_f = open(train_bin, "ab")
    val_f = open(val_bin, "ab") if val_fraction > 0 else None

    stats = StreamStats()
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else None

    try:
        text_stream = stream_hf_jsonl(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            field_candidates=field_candidates,
            stats=stats,
        )
        for i, text in enumerate(text_stream):
            if i < already_done:
                continue  # fast-forward past already-packed records on resume

            ids = sp.encode(text, out_type=int)
            if eos_id is not None:
                ids.append(eos_id)
            arr = np.asarray(ids, dtype=TOKEN_DTYPE)

            if val_fraction > 0 and is_validation_record(text, seed, val_fraction):
                arr.tofile(val_f)
                progress["tokens_written"]["val"] += len(arr)
            else:
                arr.tofile(train_f)
                progress["tokens_written"]["train"] += len(arr)

            progress["records_consumed"] = i + 1

            if (i + 1) % log_every == 0:
                save_progress(progress_path, progress)
                logger.info(
                    "[%s] %d records | train_tok=%d val_tok=%d",
                    split_name, i + 1,
                    progress["tokens_written"]["train"],
                    progress["tokens_written"]["val"],
                )
    finally:
        train_f.close()
        if val_f is not None:
            val_f.close()
        save_progress(progress_path, progress)

    meta = {
        "split_name": split_name,
        "vocab_size": sp.vocab_size(),
        "dtype": str(np.dtype(TOKEN_DTYPE)),
        "tokens_written": progress["tokens_written"],
        "stream_stats": stats.as_dict(),
    }
    with open(out_dir / f"{split_name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Finished packing '%s': %s", split_name, meta)
    return meta


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    vocab_size_cfg = cfg["model"]["vocab_size"]
    if vocab_size_cfg >= MAX_VOCAB_FOR_DTYPE:
        raise ValueError(
            f"vocab_size {vocab_size_cfg} won't fit in {TOKEN_DTYPE}; "
            "bump TOKEN_DTYPE to uint32 in tokenize_and_pack.py and lm_dataset.py"
        )

    sp = spm.SentencePieceProcessor(model_file=cfg["tokenizer"]["model_path"])
    if sp.vocab_size() != vocab_size_cfg:
        logger.warning(
            "config.model.vocab_size=%d but tokenizer file has vocab_size=%d — using the tokenizer's.",
            vocab_size_cfg, sp.vocab_size(),
        )

    data_cfg = cfg["data"]
    out_dir = Path(data_cfg.get("out_dir", "data_bins"))

    pack_split(
        repo_id=data_cfg["repo_id"],
        repo_type=data_cfg.get("repo_type", "dataset"),
        filename=data_cfg["train_file"],
        field_candidates=data_cfg.get("text_field_candidates", ["text"]),
        sp=sp,
        out_dir=out_dir,
        split_name="pretrain",  # produces pretrain_train.bin + pretrain_val.bin
        seed=data_cfg.get("seed", 42),
        val_fraction=data_cfg.get("val_fraction", 0.05),
    )

    pack_split(
        repo_id=data_cfg["repo_id"],
        repo_type=data_cfg.get("repo_type", "dataset"),
        filename=data_cfg["test_file"],
        field_candidates=data_cfg.get("text_field_candidates", ["text"]),
        sp=sp,
        out_dir=out_dir,
        split_name="test",     # produces test_train.bin only (val_fraction=0)
        seed=data_cfg.get("seed", 42),
        val_fraction=0.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and pack HF-hosted JSONL corpus into .bin files")
    parser.add_argument("--config", default="Telugu/configs/config1.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)