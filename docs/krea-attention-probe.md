# Krea attention probe

## Why this exists

The official Krea-2 implementation defines an `Attention` module that projects
Q/K/V, applies QK normalization and RoPE, then calls the module-level
`mmdit.attention()` helper. That helper invokes PyTorch SDPA and returns the
attended output. It does not return attention probabilities. Registering a
forward hook on `Attention` therefore cannot recover the probability matrix.

This probe temporarily replaces only the module-level helper. It receives the
already projected, normalized, RoPE-transformed Q/K/V tensors and computes
softmax(QKᵀ / sqrt(d)) explicitly, including grouped-query head expansion and
the optional mask. The original function is restored even if the forward pass
raises.

## Use

1. Clone the official Krea-2 repository and install its dependencies.
2. Install this project in the same Python environment.
3. Load the Krea RAW model using the official pipeline.
4. Import the exact `mmdit` module used by the loaded model.
5. Wrap a short model forward in `capture_krea_attention(mmdit, probe)`.

Use `mode="sampled"` initially. Full maps have shape
[B, Hq, Lq, Lk] and grow quadratically with token count. A 1024-pixel generation
can create very large maps. The sampled mode strides over heads, query tokens
and key tokens before moving results to CPU. Summary mode averages over query
positions and retains the key-token distribution.

## Numerical validation

Run:

```powershell
python -m pytest -q tests/test_attention_probe.py
```

The tests compare explicit attention against PyTorch SDPA for ordinary
attention, GQA and boolean masks. Use a tolerance appropriate to the selected
dtype/backend when validating CUDA BF16/FP16.

## Scope and caveats

- This is an extraction primitive, not yet teacher/student training.
- It records the attention matrix but does not infer semantic correspondences
  between Krea heads and MOSS heads.
- The current Krea model uses a combined text+image sequence. You must preserve
  the sequence boundary and spatial grid metadata from the sampler to label
  text and image token regions in the cache.
- The explicit diagnostic path is slower and more memory-intensive than SDPA.
  Do not use it for normal image generation.
- The module-level patch is process-global. Do not run concurrent Krea forwards
  in other threads while the context manager is active.
