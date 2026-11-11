# Krea-Moss / MOKREA-MOSS

Research project: cross-model attention and feature distillation from Krea 2 RAW into MOSS-Image-Base.

## Verified upstream implementations

- MOSS: https://github.com/OpenMOSS/MOSS-Image (not OpenMOSS-Team/MOSS-Image). Official implementation uses a Diffusers pipeline; native ComfyUI support is available in sufficiently recent ComfyUI builds. MOSS ComfyUI guidance: https://github.com/OpenMOSS/MOSS-Image/tree/main/comfyui
- Krea 2: https://github.com/krea-ai/krea-2. Its inference.py builds a 28-layer SingleStreamDiT with features=6144, heads=48, kvheads=12, patch=2, channels=16.
- Krea mmdit.py uses grouped-query SDPA. The public attention() helper returns attention output, not the attention probability matrix. Therefore attention probabilities require an instrumented attention implementation; a forward hook on the output cannot recover them.

## Objective

Train a MOSS-side student adapter using Krea 2 RAW as teacher. Do not average raw parameter tensors across models: their dimensions, positional encodings, text encoders, latent representations and attention implementations differ.

## Target machine

- Windows 10
- NVIDIA RTX 4070 12GB
- System RAM 32GB
- D: free space 20GB

Use staged model loading. Keep teacher and student offloaded / loaded sequentially. Start with a low-resolution, small-batch pilot. Store compressed FP16/BF16 feature caches and delete intermediate caches after validated exports.

## Implementation stages

1. Source pinning and model inspection: record upstream commit SHAs and model revisions; inspect Krea mmdit.py and MOSS Diffusers transformer implementation; map image tokens, text spans, positional encoding, timestep embedding and block outputs.
2. Teacher attention instrumentation: instrument Krea immediately before SDPA, or add an optional diagnostic path that explicitly computes probabilities. Preserve production SDPA when capture is disabled. Capture selected layers/timesteps and pool heads to control cache size.
3. MOSS student instrumentation: instrument the actual MOSS attention modules from its loaded Diffusers transformer.
4. Cross-model alignment: align image tokens by 2D latent-grid coordinates, text by tokenizer offsets/phrase spans, and project Krea's 48 query heads / 12 KV heads and MOSS heads into a shared representation. Do not assume head-index correspondence.
5. Adaptive fusion training: train a MOSS-side adapter/router conditioned on block, timestep, prompt embedding and spatial token. Combine MOSS flow-matching objective, attention distillation, projected hidden-feature alignment and optional perceptual loss.
6. Standalone export: train MOSS-compatible LoRA deltas against actual MOSS linear targets. Merge only matching LoRA A/B deltas into the exact base checkpoint with strict name/shape checks. A sidecar fusion module is not itself a merged checkpoint; integrate its forward path into the MOSS transformer before exporting standalone weights.
7. ComfyUI: prefer native MOSS support in current ComfyUI nightly/master or the official MOSS ComfyUI plugin workflow. Export in the MOSS-compatible layout and compare base vs adapted outputs.

## Hardware limits

12GB VRAM is suitable for an offloaded adapter-training pilot, not simultaneous residency of both full models. 32GB RAM means avoid keeping both full state dicts plus caches resident. 20GB free disk is tight for two model snapshots and attention caches. Check actual snapshot sizes before downloading; use an external SSD or cloud scratch space for temporary teacher assets if required.

## Critical technical issue

Krea's public mmdit.py attention helper calls torch.nn.functional.scaled_dot_product_attention and returns the attended value tensor. It does not expose attention probabilities. To distill maps, add an optional diagnostic backend that computes scaled QK^T, applies the same mask/scale/GQA semantics, softmaxes, captures probabilities, then computes P@V. Validate this path numerically against original SDPA before trusting it. Inspect MOSS separately; generic output hooks are not attention-map extraction.

## Status

- [x] Located user-owned GitHub repository RYG81/Krea-Moss
- [x] Verified official MOSS and Krea source repositories
- [x] Verified Krea architecture entry point and SDPA attention helper
- [ ] Pin exact upstream revisions and inspect MOSS transformer internals
- [ ] Implement Krea diagnostic attention backend
- [ ] Implement MOSS attention capture and native LoRA targets
- [ ] Implement coordinate/text alignment and adaptive fusion trainer
- [ ] Train pilot and validate against SDPA baseline
- [ ] Train final adapter and merge
- [ ] Validate ComfyUI workflow

## License

Krea 2 RAW is gated and governed by Krea's community license. Obtain written clarification from Krea before redistributing or commercially using any model/adapter derived from its attention/features. MOSS source code and weights have their own terms. This repository contains no upstream weights.
