from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.size(-1) % 2 != 0:
        raise ValueError("The last dimension of the input tensor must be even.")

    x = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)

def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: Optional[int] = None
) -> torch.Tensor:
    """Apply precomputed RoPE cosines and sines to ``x``.

    ``x`` must end in ``(sequence_length, head_dim)``. ``cos`` and ``sin``
    are broadcastable to ``x`` and normally have shape ``(batch, 1, T, D)``.
    Features past ``rotary_dim`` are intentionally left unchanged.
    """
    rotary_dim = x.size(-1) if rotary_dim is None else rotary_dim
    if not 0 < rotary_dim <= x.size(-1) or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive, even value no larger than head_dim")

    rotated, remainder = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rotated * cos + rotate_half(rotated) * sin
    return torch.cat((rotated, remainder), dim=-1)

class RotaryPositionalEmbedding(nn.Module):
    """Apply Rotary Position Embeddings (RoPE) to attention queries and keys.

    The expected query/key layout is ``(batch, heads, sequence_length, head_dim)``,
    which is the layout produced in :mod:`src.models.attention`.  A single
    position vector or batch-specific ``position_ids`` of shape ``(batch, T)``
    may be supplied.  If omitted, positions begin at ``offset``.
    """

    def __init__(self, head_dim: int, max_len: int = 512, base: float = 10000.0, rotary_dim: Optional[int] = None):
        super().__init__()
        rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if not 0 < rotary_dim <= head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive, even, and no larger than head_dim")
        if max_len <= 0 or base <= 0:
            raise ValueError("max_len and base must be positive")

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cache(max_len, device=inv_freq.device)

    def _set_cache(self, length: int, device: torch.device) -> None:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()

    def _ensure_cache(self, length: int, device: torch.device) -> None:
        if length > self.cos_cached.size(0) or self.cos_cached.device != device:
            self._set_cache(length, device)

    def _cos_sin_range(
        self, offset: int, length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosines/sines for the contiguous positions ``[offset, offset+length)``.

        Numerically identical to ``_cos_sin(arange(offset, offset+length))``,
        but driven entirely by Python ints, so it never reads a value back off
        the accelerator. ``_cos_sin`` has to call ``.item()`` on
        ``position_ids`` to validate them and size the cache, and each of those
        calls blocks until the device catches up — three syncs per RoPE call,
        once per layer, on every training step. It also stops ``torch.compile``
        from capturing the block as a single graph.

        Returns two tensors of shape ``(1, 1, length, rotary_dim)`` so they
        broadcast over ``(batch, heads, length, head_dim)``.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if length <= 0:
            raise ValueError("length must be positive")

        self._ensure_cache(offset + length, device)
        cos = self.cos_cached[offset : offset + length].to(dtype=dtype)
        sin = self.sin_cached[offset : offset + length].to(dtype=dtype)
        return cos[None, None], sin[None, None]

    def _cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integer positions")
        if position_ids.numel() == 0:
            raise ValueError("position_ids cannot be empty")
        if position_ids.min().item() < 0:
            raise ValueError("position_ids must be non-negative")

        self._ensure_cache(int(position_ids.max().item()) + 1, position_ids.device)
        cos = self.cos_cached[position_ids].to(dtype=dtype).unsqueeze(1)
        sin = self.sin_cached[position_ids].to(dtype=dtype).unsqueeze(1)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must have shape (batch, heads, sequence_length, head_dim)")
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
            raise ValueError("q and k must have the same batch and sequence dimensions")
        if q.size(-1) != self.head_dim or k.size(-1) != self.head_dim:
            raise ValueError(f"q and k must have head_dim={self.head_dim}")
        if q.device != k.device:
            raise ValueError("q and k must be on the same device")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        batch_size, _, sequence_length, _ = q.shape
        if position_ids is None:
            # Training always lands here: positions are just
            # offset .. offset + T - 1, which the cache can be sliced for
            # directly. Building an arange and pushing it through _cos_sin()
            # would work, but it costs a device sync per layer per step for
            # no benefit. Same numbers either way.
            cos, sin = self._cos_sin_range(offset, sequence_length, q.device, q.dtype)
        else:
            position_ids = position_ids.to(device=q.device)
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.ndim != 2 or position_ids.size(1) != sequence_length:
                raise ValueError("position_ids must have shape (sequence_length,) or (batch, sequence_length)")
            if position_ids.size(0) not in (1, batch_size):
                raise ValueError("position_ids batch size must be 1 or match q and k")
            cos, sin = self._cos_sin(position_ids, q.dtype)

        return (
            apply_rotary_pos_emb(q, cos, sin, self.rotary_dim),
            apply_rotary_pos_emb(k, cos, sin, self.rotary_dim),
        )


__all__ = [
    "RotaryPositionalEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
]
