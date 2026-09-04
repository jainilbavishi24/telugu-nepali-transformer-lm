"""
train.py

Pretraining loop for the from-scratch decoder-only LM (Phase 2), written to
keep one large GPU saturated (developed against an NVIDIA L40S, 48 GB).

What the brief asks for, and where it lives here:
  * AdamW + LR schedule (linear warmup -> cosine decay) -> build_optimizer, lr_lambda
  * validation loss and perplexity during training      -> evaluate()
  * checkpoints holding weights, optimizer state, scheduler state, step and the
    config, resumable after an interruption             -> save/load_checkpoint()
  * loss curves on disk for the report                  -> <out_dir>/loss_log.csv
  * hyperparameters stored beside the checkpoints       -> <out_dir>/config_used.yaml
                                                           <out_dir>/run_meta.json

How it keeps the GPU busy (all of this is under `runtime:` in the config):
  * bf16 autocast. The L40S runs bf16 through tensor cores at roughly 8x its
    fp32 rate, and bf16 keeps fp32's exponent range, so there is no loss scale
    to tune. The softmax and the cross-entropy still run in fp32 because
    autocast promotes them; only the matmuls drop precision.
  * TF32 for whatever fp32 matmul work is left.
  * torch.compile over the whole model, which fuses the hand-written attention
    chain (scale -> mask -> softmax -> matmul) instead of writing the
    (B, H, T, T) tensor back to HBM between every one of those ops.
  * fused AdamW: one kernel for the update rather than a few hundred small ones.
  * the whole packed token file sits in VRAM as int16 (~2 bytes/token, so 500M
    tokens is ~1 GB). A batch is then one gather over memory already next to
    the weights - no DataLoader workers, no pinned staging buffer, no
    host-to-device copy anywhere inside a step.
  * one device->host sync per logging interval instead of one per step.
  * gradient accumulation, so the global batch is decoupled from what fits.

Usage:
    python train.py --config ../configs/config1.yaml
    python train.py --config ../configs/config1.yaml --benchmark 50

--benchmark runs a handful of timed steps, reports tokens/sec, peak memory and
the projected wall clock for the configured token budget, then exits without
touching any checkpoint. Worth running first to size batch_size for your card.

Resume is automatic from training.out_dir; pass --no-resume to start fresh.
Batch sampling is a pure function of (seed, step), so a resumed run sees
exactly the batches it would have seen had it never been interrupted.
"""

import argparse
import csv
import json
import math
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# train.py lives in Telugu/train/, the packages it imports live in Telugu/, so
# put the project root on sys.path before importing them. sys.path[0] is
# already this file's directory, which is how `lm_dataset` resolves.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.transformer import LanguageModel

# TOKEN_DTYPE is imported rather than redeclared so there is exactly one place
# that decides how packed ids are stored. lm_dataset.PackedTokenDataset is still
# the Dataset the eval scripts use; training indexes the .bin itself because it
# wants the tokens resident on the GPU, which a Dataset cannot express.
from lm_dataset import TOKEN_DTYPE

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# int16 and uint16 have the same bit width, so ids round-trip exactly as long as
# none of them sets the sign bit. torch has no usable uint16, hence the reinterpret.
MAX_VOCAB_FOR_INT16 = 1 << 15


# ---------------------------------------------------------------------------
# config, device, determinism
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Read the YAML config and check that the sections train.py needs exist."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "training", "data"):
        if section not in cfg:
            raise ValueError(f"{path} is missing the '{section}' section")
    cfg.setdefault("runtime", {})
    return cfg


def get_device(rt: dict) -> torch.device:
    """Pick a device and switch on the fast matmul paths available on it.

    'high' matmul precision lets fp32 matmuls use TF32 tensor cores. At 25M
    parameters the accuracy cost is not measurable, and it applies to the
    fp32 work autocast deliberately leaves alone.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(str(rt.get("matmul_precision", "high")))
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed every RNG the run touches, so two launches with the same config and
    the same step count produce the same weights."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_hms(seconds: float) -> str:
    """Seconds -> H:MM:SS, for progress lines."""
    seconds = int(max(0.0, seconds))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# model, optimizer, schedule
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """Instantiate the decoder-only LM from the `model:` section of the config."""
    m = cfg["model"]
    return LanguageModel(
        vocab_size=m["vocab_size"],
        max_seq_len=m["max_seq_len"],
        embed_dim=m.get("embed_dim", 512),
        num_layers=m.get("num_layers", 8),
        num_heads=m.get("num_heads", 8),
        ff_hidden_dim=m.get("ff_hidden_dim"),
        dropout=m.get("dropout", 0.0),
        attn_dropout=m.get("attn_dropout", 0.0),
        rope_base=m.get("rope_base", 10000.0),
        pad_token_id=m.get("pad_token_id"),
        tie_embeddings=m.get("tie_embeddings", True),
    )


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable params, params that actually do matmul work).

    The second figure is what the MFU estimate needs. Looking a token up in the
    embedding table is a gather, not a matmul, so an untied embedding
    contributes parameters but no FLOPs. When the head is tied the same matrix
    *is* used as a real (d x V) projection at the output, so it counts.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if getattr(model, "tie_embeddings", True):
        return total, total
    return total, total - model.token_embedding.weight.numel()


def build_optimizer(model: nn.Module, t: dict, device: torch.device):
    """AdamW with the two usual parameter groups, fused where possible.

    Weight decay goes on the >=2-D parameters (embeddings, projections) and not
    on the 1-D ones (biases, LayerNorm gains) — decaying a LayerNorm gain toward
    zero just fights the normalization it is there to undo.

    The fused implementation folds the whole update into one kernel. At 25M
    parameters spread over ~100 tensors the elementwise Adam math is otherwise
    launch-bound, and this is a few percent of total step time.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": float(t.get("weight_decay", 0.1))},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(
        lr=float(t.get("learning_rate", 6e-4)),
        betas=(float(t.get("beta1", 0.9)), float(t.get("beta2", 0.95))),
        eps=float(t.get("eps", 1e-8)),
    )

    impl = "foreach"
    optimizer = None
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(groups, fused=True, **kwargs)
            impl = "fused"
        except (RuntimeError, TypeError, ValueError):
            optimizer = None
    if optimizer is None:
        optimizer = torch.optim.AdamW(groups, foreach=True, **kwargs)

    n_decay = sum(p.numel() for p in decay)
    n_plain = sum(p.numel() for p in no_decay)
    print(
        f"optimizer: AdamW ({impl}) | decayed {n_decay:,} params in {len(decay)} tensors "
        f"| undecayed {n_plain:,} in {len(no_decay)}"
    )
    return optimizer


def lr_lambda(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    """Linear warmup, then cosine decay down to `min_lr_ratio` of the peak LR.

    Warmup exists because Adam's second-moment estimate is garbage for the first
    few dozen steps, so a full-size update then can knock the model somewhere it
    takes a long time to climb out of. The cosine floor keeps the last stretch of
    training from becoming a no-op.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def flops_per_token(compute_params: int, num_layers: int, seq_len: int, embed_dim: int) -> float:
    """Forward+backward FLOPs for one token, used only for the MFU readout.

    6N covers the matmuls (2N forward, 4N backward). The second term is the
    attention score/weighted-sum pair, which 6N misses because it scales with
    sequence length rather than with parameter count — at T=512 it is a real
    ~14% of the total here.
    """
    return 6.0 * compute_params + 12.0 * num_layers * seq_len * embed_dim


# ---------------------------------------------------------------------------
# data: getting the packed .bin files, then getting them onto the GPU
# ---------------------------------------------------------------------------

def resolve_bin(filename: str, data_cfg: dict, cache_dir: Path) -> Path:
    """Return a local path to `filename`, downloading from the Hub if needed.

    A local file always wins, so a box that already has the .bin never touches
    the network. Otherwise it is pulled from data.hf_bin_repo_id. The path
    *inside* that repo can be pinned in data.hf_bin_files; if it is not, any
    file in the repo whose basename matches is accepted, which means the bins
    can live in whatever subfolder they were uploaded to.
    """
    local = cache_dir / filename
    if local.exists():
        return local

    repo_id = data_cfg.get("hf_bin_repo_id") or data_cfg.get("repo_id")
    if not repo_id:
        raise FileNotFoundError(
            f"{local} does not exist and data.hf_bin_repo_id is unset — either copy the "
            "packed .bin files into that directory or point the config at the Hub repo."
        )

    from huggingface_hub import HfApi, hf_hub_download

    repo_type = data_cfg.get("hf_bin_repo_type", "dataset")
    remote = (data_cfg.get("hf_bin_files") or {}).get(filename)
    if not remote:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
        matches = [f for f in files if Path(f).name == filename]
        if not matches:
            candidates = [f for f in files if f.endswith(".bin")] or sorted(files)[:40]
            raise FileNotFoundError(
                f"'{filename}' is not in {repo_type} repo '{repo_id}'.\n"
                f"  .bin files that are there: {candidates}\n"
                "  Pin the right path via data.hf_bin_files in the config."
            )
        remote = min(matches, key=len)  # shallowest match

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetching {repo_id}:{remote}")
    return Path(
        hf_hub_download(
            repo_id=repo_id, filename=remote, repo_type=repo_type, local_dir=str(cache_dir)
        )
    )


def want_resident(paths, device: torch.device, mode) -> bool:
    """Decide whether the token files should be held in VRAM.

    'auto' keeps them there when they take under a quarter of the free memory,
    leaving the rest for activations. At 2 bytes/token a 500M-token corpus is
    ~1 GB against the L40S's 48 GB, so in practice this is always true; the
    host fallback is there for smaller cards.
    """
    if device.type != "cuda":
        return False
    if isinstance(mode, bool):
        return mode
    text = str(mode).strip().lower()
    if text in ("true", "yes", "1"):
        return True
    if text in ("false", "no", "0"):
        return False
    total_bytes = sum(Path(p).stat().st_size for p in paths)
    free_bytes, _ = torch.cuda.mem_get_info()
    fits = total_bytes < 0.25 * free_bytes
    print(
        f"gpu_resident_data=auto -> {fits} ({total_bytes / 1024 ** 3:.2f} GB of tokens, "
        f"{free_bytes / 1024 ** 3:.1f} GB free)"
    )
    return fits


class TokenStream:
    """Fixed-length causal-LM windows over a packed token .bin file.

    Same contract as lm_dataset.PackedTokenDataset — target_ids is input_ids
    shifted one position left, so the logits at position t are trained against
    input token t+1 — but two things differ, both about throughput:

    1. When `resident`, the whole file is uploaded once and lives in VRAM as
       int16. A batch is then ``tokens[starts[:, None] + arange(T + 1)]``: one
       gather over memory that already sits next to the weights. No worker
       processes, no pinned staging buffer, no PCIe traffic inside a step. The
       non-resident path gathers from the memmap on the host and uploads
       through pinned memory instead.

    2. Window starts are drawn i.i.d. from a generator re-seeded to
       f(seed, step), rather than walked through a shuffled permutation. That
       makes the data a run sees a pure function of its step counter, so resume
       needs no serialised sampler state and two runs that reach step 4000 have
       read the same text. Sampling with replacement instead of by epoch is
       immaterial here: drawing 300M tokens from a ~500M-token corpus repeats
       windows about as often as a shuffled 0.6-epoch pass would.
    """

    def __init__(self, bin_path, seq_len: int, device: torch.device, vocab_size: int,
                 seed: int, name: str, resident: bool):
        self.path = Path(bin_path)
        self.seq_len = int(seq_len)
        self.device = device
        self.name = name
        self.resident = bool(resident)
        self._seed = int(seed)

        mm = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        self.n_tokens = int(mm.shape[0])
        # a window needs seq_len inputs plus one extra token to be the last target
        self.n_windows = self.n_tokens - self.seq_len - 1
        if self.n_windows <= 0:
            raise ValueError(
                f"{self.path} holds {self.n_tokens} tokens, too few for seq_len={self.seq_len}"
            )

        if self.resident:
            self._init_resident(mm, vocab_size)
            where = f"{self.n_tokens * 2 / 1024 ** 3:.2f} GB in VRAM"
        else:
            self.mm = mm
            where = "memmapped on host"

        print(f"{self.name}: {self.n_tokens:,} tokens, {self.n_windows:,} windows ({where})")

    def _init_resident(self, mm, vocab_size: int) -> None:
        """Upload the whole file to VRAM as int16, in bounded host-side blocks.

        The .bin is uint16 and torch has no uint16, so the bits are
        reinterpreted as int16: identical storage, and ids round-trip exactly
        as long as none of them sets the sign bit — checked here once, on the
        host, rather than anywhere near the training loop.
        """
        if vocab_size > MAX_VOCAB_FOR_INT16:
            raise ValueError(
                f"vocab_size={vocab_size} does not fit int16 token storage. Set "
                "runtime.gpu_resident_data: false, or widen TOKEN_DTYPE and this path."
            )

        self.tokens = torch.empty(self.n_tokens, dtype=torch.int16, device=self.device)
        block = 1 << 26  # 64M tokens = 128 MB per hop, so host RAM stays flat
        max_id = 0
        for start in range(0, self.n_tokens, block):
            host_u16 = np.array(mm[start : start + block])  # forces a real read
            if host_u16.size:
                max_id = max(max_id, int(host_u16.max()))
            as_i16 = torch.from_numpy(host_u16.view(np.int16))
            self.tokens[start : start + as_i16.numel()] = as_i16.to(self.device)
        del mm

        if max_id >= MAX_VOCAB_FOR_INT16:
            raise ValueError(
                f"{self.path} contains token id {max_id} >= {MAX_VOCAB_FOR_INT16}, which "
                "would come back negative out of int16 storage."
            )
        if max_id >= vocab_size:
            raise ValueError(
                f"{self.path} contains token id {max_id}, but model.vocab_size is "
                f"{vocab_size} — the .bin and the tokenizer disagree."
            )

        # Precomputed once: added to a column of starts to materialise windows.
        self._offsets = torch.arange(self.seq_len + 1, device=self.device)
        # A CUDA generator keeps index sampling on-device; MPS has none, so fall back.
        gen_device = self.device if self.device.type in ("cpu", "cuda") else torch.device("cpu")
        self._gen = torch.Generator(device=gen_device)

    def batch(self, step: int, micro_batch: int, accum: int):
        """Yield the `accum` (input_ids, target_ids) micro-batches for step `step`.

        Both tensors come back (micro_batch, seq_len) int64 on `self.device`.
        The starts depend only on (seed, step), never on how many steps have
        already run, which is what makes resume exact.
        """
        if self.resident:
            self._gen.manual_seed(self._seed * 1_000_003 + step)
            starts = torch.randint(
                0, self.n_windows, (accum, micro_batch),
                device=self._gen.device, generator=self._gen,
            )
            if starts.device != self.device:
                starts = starts.to(self.device)
            for a in range(accum):
                # (micro_batch, 1) + (T + 1,) -> (micro_batch, T + 1)
                window = self.tokens[starts[a].unsqueeze(1) + self._offsets].long()
                yield window[:, :-1].contiguous(), window[:, 1:].contiguous()
        else:
            rng = np.random.default_rng(abs(self._seed * 1_000_003 + step))
            for _ in range(accum):
                sel = rng.integers(0, self.n_windows, size=micro_batch)
                window = self._gather_host(sel).pin_memory().to(self.device, non_blocking=True)
                yield window[:, :-1].contiguous(), window[:, 1:].contiguous()

    def _gather_host(self, starts) -> torch.Tensor:
        """(len(starts), T + 1) int64 window tensor read off the host memmap."""
        rows = np.stack(
            [np.asarray(self.mm[int(s) : int(s) + self.seq_len + 1]) for s in starts]
        )
        return torch.from_numpy(rows.astype(np.int64))

    def eval_batches(self, micro_batch: int, n_batches: int) -> list:
        """A fixed, evenly spaced set of windows spanning the whole file.

        Materialised once and reused at every eval, so the val loss at step 250
        and at step 9000 are measured on exactly the same text — the curve in
        the report then shows the model moving rather than the sample changing.
        It is also nearly free to keep: 1,536 windows of int64 is ~13 MB.
        """
        n_batches = max(1, min(int(n_batches), self.n_windows // max(1, micro_batch)))
        starts = np.linspace(0, self.n_windows - 1, n_batches * micro_batch).astype(np.int64)
        out = []
        for b in range(n_batches):
            sel = starts[b * micro_batch : (b + 1) * micro_batch]
            if self.resident:
                idx = torch.from_numpy(sel).to(self.device).unsqueeze(1) + self._offsets
                window = self.tokens[idx].long()
            else:
                window = self._gather_host(sel).to(self.device)
            out.append((window[:, :-1].contiguous(), window[:, 1:].contiguous()))
        return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, batches, autocast_ctx, ignore_index) -> tuple[float, float]:
    """Mean cross-entropy over `batches`, and its exponential (perplexity).

    Dropout is off (model.eval()) and no graph is built, so this runs at
    roughly forward-only cost — the whole 1,536-window eval set is a second or
    two. Losses accumulate in a float64 device scalar and are read back once at
    the end, which keeps the eval to a single sync no matter how many batches.
    Perplexity is exp(mean token loss); the guard just stops an early,
    still-random model from overflowing to inf in the CSV.
    """
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    count = 0
    for x, y in batches:
        with autocast_ctx():
            logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
            ignore_index=ignore_index,
        )
        total += loss.double()
        count += 1
    model.train()
    avg = float(total.item() / max(1, count))
    return avg, float(math.exp(avg)) if avg < 20 else float("inf")


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, raw_model, optimizer, scheduler, step: int,
                    tokens_seen: int, best_val: float, cfg: dict) -> None:
    """Write a checkpoint that is enough to resume the run bit-for-bit.

    Weights come off `raw_model`, never the torch.compile wrapper, so the keys
    have no '_orig_mod.' prefix and load cleanly into a plain LanguageModel at
    eval time. The write goes to a .tmp beside the target and is then renamed:
    os.replace is atomic on POSIX, so a crash mid-save cannot leave a truncated
    latest.pt where the resume logic will find it.
    """
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, raw_model, optimizer, scheduler,
                    device: torch.device) -> tuple[int, int, float]:
    """Restore weights, optimizer, schedule and RNG state. Returns where to resume.

    weights_only=False is required because the payload holds the pickled RNG
    states, not just tensors; torch >= 2.6 defaults that flag to True. This is
    our own file, written by save_checkpoint above.

    RNG restore is best-effort: a checkpoint moved between machines can carry a
    CUDA RNG state for a different device count, and that must not be fatal —
    batch selection is driven by (seed, step), so it does not depend on this.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    rng = ckpt.get("rng") or {}
    try:
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except (RuntimeError, ValueError, TypeError) as exc:
        print(f"  (rng state not restored: {exc})")

    step = int(ckpt.get("step", 0))
    tokens_seen = int(ckpt.get("tokens_seen", 0))
    best_val = float(ckpt.get("best_val_loss", float("inf")))
    print(f"resumed from {path} at step {step:,} ({tokens_seen:,} tokens seen)")
    return step, tokens_seen, best_val


def prune_snapshots(out_dir: Path, keep: int) -> None:
    """Keep only the newest `keep` numbered step_*.pt files.

    latest.pt and best.pt are separate files and are never touched. Without
    this, a 9,155-step run snapshotting every 2,000 steps is fine, but a longer
    or more frequently snapshotted one quietly fills the disk — each checkpoint
    is the model plus two AdamW moments, so ~300 MB at 25M params.
    """
    if keep is None or keep <= 0:
        return
    snaps = []
    for p in out_dir.glob("step_*.pt"):
        try:
            snaps.append((int(p.stem.split("_")[1]), p))
        except (IndexError, ValueError):
            continue
    for _, p in sorted(snaps)[:-keep]:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def main(config_path: str, resume: bool = True, benchmark_steps: int = 0,
         override_out: str | None = None) -> None:
    cfg = load_config(config_path)
    m, t, dcfg = cfg["model"], cfg["training"], cfg["data"]
    rt = cfg.get("runtime", {})

    device = get_device(rt)
    seed = int(t.get("seed", 42))
    seed_everything(seed)

    # ---- precision -------------------------------------------------------
    # autocast only changes the dtype the matmuls run in. It keeps a master
    # copy of the weights in fp32 and promotes the reduction-heavy ops
    # (softmax, LayerNorm, cross-entropy) back to fp32 on its own, so this is
    # not the same thing as calling .half() on the model.
    precision = str(rt.get("precision", "bf16")).lower()
    if precision not in AMP_DTYPES:
        raise ValueError(f"runtime.precision must be one of {sorted(AMP_DTYPES)}")
    if device.type != "cuda" and precision != "fp32":
        print(f"no CUDA device -> falling back from {precision} to fp32")
        precision = "fp32"
    elif precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("bf16 not supported here -> falling back to fp16")
        precision = "fp16"
    amp_dtype = AMP_DTYPES[precision]
    use_amp = device.type == "cuda" and precision != "fp32"

    def autocast_ctx():
        if not use_amp:
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    # fp16's exponent range is narrow enough that small gradients underflow to
    # zero, so it needs a loss scale. bf16 keeps fp32's exponent range and does
    # not — which is the whole reason to prefer it on a card with native bf16.
    scaler = None
    if precision == "fp16":
        try:
            scaler = torch.amp.GradScaler(device.type)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    # ---- batch geometry -------------------------------------------------
    batch_size = int(t.get("batch_size", 64))
    max_seq_len = int(m.get("max_seq_len", 512))
    global_batch_tokens = int(t.get("global_batch_tokens", batch_size * max_seq_len))
    if global_batch_tokens % (batch_size * max_seq_len) != 0:
        raise ValueError(
            f"global_batch_tokens={global_batch_tokens} is not divisible by "
            f"(batch_size={batch_size} * max_seq_len={max_seq_len})"
        )
    grad_accum_steps = global_batch_tokens // (batch_size * max_seq_len)

    # ---- step count ------------------------------------------------------
    train_tokens = int(t.get("train_tokens", 300_000_000))
    max_steps = t.get("max_steps")
    if max_steps is None:
        max_steps = math.ceil(train_tokens / global_batch_tokens)
    else:
        max_steps = int(max_steps)
    warmup_steps = int(t.get("warmup_steps", 300))

    # ---- model -----------------------------------------------------------
    raw_model = build_model(cfg).to(device)
    total_params, compute_params = count_params(raw_model)
    print(f"model: {total_params:,} total params, {compute_params:,} compute params")

    # ---- optimiser & schedule --------------------------------------------
    optimizer = build_optimizer(raw_model, t, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(s, warmup_steps, max_steps, float(t.get("min_lr_ratio", 0.1))),
    )

    # ---- torch.compile ---------------------------------------------------
    # dynamic=False promises that every input shape is the one we compiled
    # with, which lets Inductor fuse the attention mask path. The compiled
    # wrapper only appears on `model`; `raw_model` stays plain for checkpoints.
    model = raw_model
    compiled = False
    if device.type == "cuda" and rt.get("compile", True):
        mode = rt.get("compile_mode", "default")
        try:
            model = torch.compile(raw_model, mode=mode, dynamic=False)
            compiled = True
            print(f"torch.compile enabled (mode={mode})")
        except Exception as exc:
            print(f"torch.compile failed, staying eager: {exc}")

    # ---- forward/backward closure ----------------------------------------
    V = m["vocab_size"]
    ignore_index = m.get("pad_token_id", -100)

    def fwd_bwd(x, y, scale: float):
        """One forward/backward under autocast. Returns the *scaled* loss."""
        with autocast_ctx():
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index
            )
        if scaler is not None:
            scaler.scale(loss * scale).backward()
        else:
            (loss * scale).backward()
        return loss

    # ---- data: locate and optionally upload the .bin files ---------------
    cache_dir = Path(dcfg.get("out_dir", "data_bins/telugu"))
    train_bin = resolve_bin(dcfg.get("train_bin", "pretrain_train.bin"), dcfg, cache_dir)
    val_bin = resolve_bin(dcfg.get("val_bin", "pretrain_val.bin"), dcfg, cache_dir)

    resident = want_resident([train_bin, val_bin], device, rt.get("gpu_resident_data", "auto"))
    train_stream = TokenStream(
        train_bin, max_seq_len, device, V, seed, "train", resident
    )
    val_stream = TokenStream(
        val_bin, max_seq_len, device, V, seed, "val", resident
    )

    # ---- validation windows ----------------------------------------------
    eval_iters = int(t.get("eval_iters", 24))
    val_batches = val_stream.eval_batches(batch_size, eval_iters)

    # ---- output directory ------------------------------------------------
    out_dir = Path(override_out or t.get("out_dir", "checkpoints/telugu"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- resume or start fresh -------------------------------------------
    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"

    if resume and latest_path.exists():
        start_step, tokens_seen, best_val_loss = load_checkpoint(
            latest_path, raw_model, optimizer, scheduler, device
        )
        scheduler.last_epoch = start_step

    # ---- warmup step to trigger torch.compile ----------------------------
    # Inductor lazily builds the graph on the first call, which takes a while.
    # Running one tiny batch now means the actual timing loop sees a compiled
    # model from step 0. The offset -1 signals "warmup only" to the sampler.
    if compiled and start_step == 0:
        print("warming up torch.compile (first call triggers the compile)...")
        try:
            x0, y0 = next(train_stream.batch(-1, batch_size, grad_accum_steps))
            _ = fwd_bwd(x0, y0, 1.0 / grad_accum_steps)
            optimizer.zero_grad(set_to_none=True)
        except Exception as exc:
            print(f"compile warmup failed, reverting to eager: {exc}")
            model = raw_model
            compiled = False
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    # ---- benchmark mode --------------------------------------------------
    if benchmark_steps > 0:
        print(f"\n-- benchmarking {benchmark_steps} steps --")
        torch.cuda.synchronize()
        t0 = time.time()
        for bs in range(benchmark_steps):
            step = start_step + bs
            for x, y in train_stream.batch(step, batch_size, grad_accum_steps):
                _ = fwd_bwd(x, y, 1.0 / grad_accum_steps)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tok_per_sec = benchmark_steps * global_batch_tokens / elapsed
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
        projected = fmt_hms(max_steps * elapsed / benchmark_steps)
        print(f"  {tok_per_sec:,.0f} tokens/sec | peak {peak_gb:.1f} GB | projected {projected}")
        return

    # ---- logging ---------------------------------------------------------
    log_interval = int(rt.get("log_interval", 10))
    print_interval = int(rt.get("print_interval", 100))
    csv_path = out_dir / "loss_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "tokens", "train_loss", "val_loss", "val_ppl", "lr",
        "grad_norm", "tokens_per_sec", "mfu", "elapsed_s", "peak_mem_gb"
    ])
    csv_file.flush()

    # ---- training loop ---------------------------------------------------
    grad_clip = float(t.get("grad_clip", 1.0))
    eval_interval = int(t.get("eval_interval", 250))
    save_interval = int(t.get("save_interval", 500))
    snapshot_interval = int(t.get("snapshot_interval", 2000))
    keep_snapshots = int(t.get("keep_snapshots", 3))
    gpu_peak_tflops = float(rt.get("gpu_peak_tflops", 181.05))
    flops_tok = flops_per_token(compute_params, m.get("num_layers", 8), max_seq_len, m.get("embed_dim", 512))

    # Accumulate loss on-device to avoid a sync per step. The sync happens once
    # per log_interval via .tolist(), which is one small host pull.
    window_loss = torch.zeros((), dtype=torch.float64, device=device)
    window_n = 0

    start_time = time.time()
    interrupted = False

    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\ncaught signal {signum}, checkpointing and exiting...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"\nstarting training from step {start_step:,} to {max_steps:,}")
    print(f"  global_batch_tokens={global_batch_tokens:,} | grad_accum={grad_accum_steps} | warmup={warmup_steps}")
    print(f"  precision={precision} | compiled={compiled} | device={device}\n")

    for step in range(start_step, max_steps):
        if interrupted:
            break

        lr = scheduler.get_last_lr()[0]

        # ---- gradient accumulation ---------------------------------------
        for micro_x, micro_y in train_stream.batch(step, batch_size, grad_accum_steps):
            loss = fwd_bwd(micro_x, micro_y, 1.0 / grad_accum_steps)
            window_loss += loss.detach().double()
            window_n += 1

        # ---- gradient clipping & optimizer step ---------------------------
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += global_batch_tokens

        # ---- logging ------------------------------------------------------
        if (step + 1) % log_interval == 0:
            # One sync per log_interval, not per step
            train_loss, gn = torch.stack([window_loss / max(1, window_n), grad_norm]).tolist()
            window_loss.zero_()
            window_n = 0
            if math.isnan(train_loss):
                raise RuntimeError("training loss is NaN — halting")

            elapsed = time.time() - start_time
            tok_per_sec = (step + 1 - start_step) * global_batch_tokens / max(1e-9, elapsed)
            mfu = (tok_per_sec * flops_tok) / (gpu_peak_tflops * 1e12) if device.type == "cuda" else 0.0
            peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0

            csv_writer.writerow([
                step + 1, tokens_seen, train_loss, "", "", lr, gn,
                tok_per_sec, mfu, elapsed, peak_gb
            ])
            csv_file.flush()

            if (step + 1) % print_interval == 0:
                print(
                    f"step {step + 1:,}/{max_steps:,} | tok {tokens_seen:,} | "
                    f"loss {train_loss:.4f} | lr {lr:.2e} | {tok_per_sec:,.0f} tok/s"
                )

        # ---- evaluation ---------------------------------------------------
        if (step + 1) % eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_batches, autocast_ctx, ignore_index)
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1 - start_step) * global_batch_tokens / max(1e-9, elapsed)
            mfu = (tok_per_sec * flops_tok) / (gpu_peak_tflops * 1e12) if device.type == "cuda" else 0.0
            peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0

            csv_writer.writerow([
                step + 1, tokens_seen, "", val_loss, val_ppl, scheduler.get_last_lr()[0], "",
                tok_per_sec, mfu, elapsed, peak_gb
            ])
            csv_file.flush()
            print(f"  eval @ step {step + 1:,}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_path, raw_model, optimizer, scheduler, step + 1, tokens_seen, best_val_loss, cfg)
                print(f"  new best -> {best_path}")

        # ---- checkpointing ------------------------------------------------
        if (step + 1) % save_interval == 0:
            save_checkpoint(latest_path, raw_model, optimizer, scheduler, step + 1, tokens_seen, best_val_loss, cfg)

        if (step + 1) % snapshot_interval == 0:
            snap_path = out_dir / f"step_{step + 1}.pt"
            save_checkpoint(snap_path, raw_model, optimizer, scheduler, step + 1, tokens_seen, best_val_loss, cfg)
            prune_snapshots(out_dir, keep_snapshots)

    # ---- final save -------------------------------------------------------
    save_checkpoint(latest_path, raw_model, optimizer, scheduler, max_steps, tokens_seen, best_val_loss, cfg)
    csv_file.close()

    # ---- write config and metadata ----------------------------------------
    config_copy = out_dir / "config_used.yaml"
    shutil.copy(config_path, config_copy)

    meta = {
        "total_params": total_params,
        "compute_params": compute_params,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "precision": precision,
        "compiled": compiled,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "global_batch_tokens": global_batch_tokens,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "train_tokens": train_tokens,
        "torch_version": torch.__version__,
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "train_tokens_available": train_stream.n_tokens,
        "val_tokens_available": val_stream.n_tokens,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\ndone: {max_steps:,} steps, {tokens_seen:,} tokens in {fmt_hms(elapsed)}")
    print(f"  final val_loss={best_val_loss:.4f} | checkpoints in {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the decoder-only LM")
    parser.add_argument(
        "--config", type=str, default="Telugu/configs/config1.yaml",
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh even if latest.pt exists"
    )
    parser.add_argument(
        "--benchmark", type=int, default=0,
        help="Run N timed steps and exit (no checkpointing)"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override training.out_dir"
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        resume=not args.no_resume,
        benchmark_steps=args.benchmark,
        override_out=args.out_dir,
    )
