"""
lm_dataset.py

Turns a flat, on-disk .bin file of token ids (written by tokenize_and_pack.py)
into a PyTorch Dataset of fixed-length (input, target) windows for causal LM
training. Uses np.memmap so opening a 500M-token file costs ~0 RAM — pages
are faulted in from disk lazily as __getitem__ is called.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TOKEN_DTYPE = np.uint16  # must match tokenize_and_pack.py


class PackedTokenDataset(Dataset):
    """
    Fixed-length windows over a flat token-id .bin file.

    Args:
        bin_path: path to a .bin file of TOKEN_DTYPE token ids.
        seq_len: length of each training example (== model.max_seq_len).
        stride: step between consecutive window start positions. stride ==
            seq_len means non-overlapping chunks (fastest, most common for
            pretraining); stride < seq_len means overlapping windows (more
            examples per token, more compute).

    Each item is (input_ids, target_ids), both length `seq_len`, where
    target_ids is input_ids shifted left by one position — the standard
    causal-LM next-token setup described in the assignment.
    """

    def __init__(self, bin_path: str, seq_len: int, stride: int):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"{bin_path} not found — run tokenize_and_pack.py first")
        self.seq_len = seq_len
        self.stride = stride

        # +1 because each window needs seq_len input tokens plus 1 extra
        # token to serve as the final target.
        self.data = np.memmap(self.bin_path, dtype=TOKEN_DTYPE, mode="r")
        n_tokens = len(self.data)
        usable = n_tokens - (seq_len + 1)
        if usable < 0:
            raise ValueError(
                f"{bin_path} has only {n_tokens} tokens, fewer than seq_len+1={seq_len + 1}"
            )
        self.n_windows = usable // stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        target_ids = torch.from_numpy(chunk[1:])
        return input_ids, target_ids


def make_dataloader(
    bin_path: str, seq_len: int, stride: int, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Convenience wrapper matching the training/model sections of config.yaml."""
    ds = PackedTokenDataset(bin_path, seq_len, stride)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )