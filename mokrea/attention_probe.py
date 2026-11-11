"""Diagnostic attention extraction for the official Krea-2 MMDiT.

Krea's mmdit.attention() helper normally calls PyTorch SDPA and returns only
P@V. This module temporarily replaces that helper with an equivalent explicit
attention calculation so probabilities can be inspected. It does not modify
model weights and the original fast SDPA path is restored on context exit.

Use only for short diagnostic runs: full attention maps scale quadratically
with sequence length and can consume substantial RAM/VRAM.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor


CaptureMode = Literal["full", "sampled", "summary"]


@dataclass
class AttentionRecord:
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    probabilities: Tensor


class KreaAttentionProbe:
    """Collect attention probabilities while preserving Krea's output contract."""

    def __init__(
        self,
        mode: CaptureMode = "sampled",
        head_stride: int = 1,
        query_stride: int = 8,
        key_stride: int = 8,
        device: str = "cpu",
        callback: Callable[[AttentionRecord], None] | None = None,
    ) -> None:
        if mode not in ("full", "sampled", "summary"):
            raise ValueError(f"Unsupported capture mode: {mode}")
        if min(head_stride, query_stride, key_stride) < 1:
            raise ValueError("Capture strides must be positive")
        self.mode = mode
        self.head_stride = head_stride
        self.query_stride = query_stride
        self.key_stride = key_stride
        self.device = torch.device(device)
        self.callback = callback
        self.records: list[AttentionRecord] = []

    def record(self, q: Tensor, k: Tensor, probabilities: Tensor) -> None:
        if self.mode == "full":
            selected = probabilities
        elif self.mode == "sampled":
            selected = probabilities[
                :, :: self.head_stride, :: self.query_stride, :: self.key_stride
            ]
        else:
            # Mean attention distribution per head, preserving key-token axis.
            selected = probabilities.mean(dim=-2, keepdim=True)
            selected = selected[:, :: self.head_stride, :, :: self.key_stride]
        record = AttentionRecord(
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            probabilities=selected.detach().to(device=self.device, dtype=torch.float16),
        )
        if self.callback is not None:
            self.callback(record)
        else:
            self.records.append(record)


def explicit_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return (flattened attention output, probabilities), matching Krea SDPA.

    q: (B, Hq, Lq, D); k/v: (B, Hkv, Lk, D).
    Boolean masks use SDPA semantics: True means the position is allowed.
    Floating masks are additive.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 (B,H,L,D) tensors")
    if k.shape != v.shape:
        raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share batch and head dimensions")
    hq, hkv = q.shape[1], k.shape[1]
    if hq != hkv:
        if not gqa:
            raise ValueError("Different query/KV head counts require gqa=True")
        if hq % hkv:
            raise ValueError("Query head count must be divisible by KV head count")
        repeats = hq // hkv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    qf, kf, vf = q.float(), k.float(), v.float()
    logits = torch.matmul(qf, kf.transpose(-2, -1))
    logits = logits * (scale if scale is not None else q.shape[-1] ** -0.5)
    if mask is not None:
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        else:
            logits = logits + mask.float()
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, vf).to(dtype=q.dtype)
    output = output.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
    return output, probabilities


@contextmanager
def capture_krea_attention(mmdit_module, probe: KreaAttentionProbe):
    """Patch the imported Krea mmdit module's attention helper temporarily.

    Example:
        import mmdit
        probe = KreaAttentionProbe(mode="sampled")
        with capture_krea_attention(mmdit, probe):
            output = model(...)
    """
    original = mmdit_module.attention

    def instrumented_attention(q, k, v, mask=None, scale=None, gqa=False):
        output, probabilities = explicit_attention(
            q, k, v, mask=mask, scale=scale, gqa=gqa
        )
        probe.record(q, k, probabilities)
        return output

    mmdit_module.attention = instrumented_attention
    try:
        yield probe
    finally:
        mmdit_module.attention = original
