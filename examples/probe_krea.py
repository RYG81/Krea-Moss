"""Minimal example after installing/cloning the official krea-2 repository."""
import sys

# Add the official Krea-2 repository root to PYTHONPATH before running.
import torch
import mmdit

from mokrea.attention_probe import KreaAttentionProbe, capture_krea_attention

# This snippet demonstrates the patch around any real Krea model forward.
# Construct/load the official model using Krea's own inference.py _pipeline().
# Then call the model with the exact img/context/t/pos/mask tensors from sampling.
model = None  # Replace with the loaded Krea SingleStreamDiT.
if model is None:
    raise SystemExit("Load the official Krea SingleStreamDiT before running this example.")

probe = KreaAttentionProbe(
    mode="sampled", head_stride=4, query_stride=16, key_stride=16, device="cpu"
)
# Use inference_mode only for teacher extraction; no gradients are required.
with torch.inference_mode(), capture_krea_attention(mmdit, probe):
    raise NotImplementedError(
        "Call model(img=..., context=..., t=..., pos=..., mask=...) here. "
        "Use the same conditioned latent/timestep inputs for teacher-student pairs."
    )
