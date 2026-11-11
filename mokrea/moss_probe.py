"""Diagnostic attention-probability probe for the official MOSS-Image transformer."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Literal
import torch
from torch import Tensor

@dataclass
class MossAttentionRecord:
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    probabilities: Tensor

class MossAttentionProbe:
    def __init__(self, mode: Literal["full", "sampled", "summary"]="sampled",
                 head_stride: int=1, query_stride: int=8, key_stride: int=8,
                 device: str="cpu",
                 callback: Callable[[MossAttentionRecord], None] | None=None):
        if mode not in ("full", "sampled", "summary"):
            raise ValueError(f"Unsupported capture mode: {mode}")
        if min(head_stride, query_stride, key_stride) < 1:
            raise ValueError("Capture strides must be positive")
        self.mode, self.head_stride = mode, head_stride
        self.query_stride, self.key_stride = query_stride, key_stride
        self.device, self.callback = torch.device(device), callback
        self.records: list[MossAttentionRecord] = []

    def record(self, q: Tensor, k: Tensor, probabilities: Tensor) -> None:
        if self.mode == "full":
            selected = probabilities
        elif self.mode == "sampled":
            selected = probabilities[:, ::self.head_stride, ::self.query_stride, ::self.key_stride]
        else:
            selected = probabilities.mean(dim=-2, keepdim=True)
            selected = selected[:, ::self.head_stride, :, ::self.key_stride]
        record = MossAttentionRecord(tuple(q.shape), tuple(k.shape),
            selected.detach().to(device=self.device, dtype=torch.float16))
        if self.callback is None:
            self.records.append(record)
        else:
            self.callback(record)

def _rotate(x: Tensor, rope: Tensor) -> Tensor:
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin

def moss_explicit_attention(q: Tensor, k: Tensor, v: Tensor,
                             rope: Tensor, heads: int) -> tuple[Tensor, Tensor]:
    """Match upstream MOSS _attention; q/k/v inputs are (B,L,dim)."""
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have identical (B,L,dim) shapes")
    if q.shape[-1] % heads:
        raise ValueError("hidden dimension must be divisible by heads")
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    qh = _rotate(q.reshape(shape), rope).transpose(1, 2)
    kh = _rotate(k.reshape(shape), rope).transpose(1, 2)
    vh = v.reshape(shape).transpose(1, 2)
    logits = torch.matmul(qh.float(), kh.float().transpose(-2, -1))
    probabilities = torch.softmax(logits * (qh.shape[-1] ** -0.5), dim=-1)
    output = torch.matmul(probabilities, vh.float()).to(q.dtype)
    return output.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1), probabilities

@contextmanager
def capture_moss_attention(model_module, probe: MossAttentionProbe):
    """Patch the official model module's _attention helper for a diagnostic forward."""
    original = model_module._attention
    def instrumented(q, k, v, rope, heads):
        output, probabilities = moss_explicit_attention(q, k, v, rope, heads)
        probe.record(q, k, probabilities)
        return output
    model_module._attention = instrumented
    try:
        yield probe
    finally:
        model_module._attention = original
