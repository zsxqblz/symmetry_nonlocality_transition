# Generation

This directory contains the standalone image-generation entrypoints for the public release. These scripts do not depend on the original private repo layout; they assume this directory lives inside the repo root alongside `models/`.

## Recommended Workflow

1. Choose the model family:
   - use `run_dit_local_global.py` for ImageNet DiT
   - use `run_dit_local_global_clip.py` for SD3
2. Set the local-attention schedule:
   - choose `RADIUS`
   - choose `LOCAL_INTERVALS`
   - choose conditioning or prompt settings
3. Run the generator once per configuration.
4. Inspect the PNG written under `generation/outputs/`.

## Files

### `run_dit_local_global.py`
Function:
- Generates a single ImageNet DiT sample with timestep-dependent switching between global attention and local attention.
- Uses the DiT class-conditional model and optional classifier-free guidance.

Inputs:
- Configuration is controlled by top-level constants plus optional environment overrides.
- Common knobs:
  - `MODEL_ID`: DiT checkpoint, default `facebook/DiT-XL-2-256`
  - `RADIUS`: local-attention radius
  - `LOCAL_INTERVALS`: list of normalized timestep intervals where local attention is active
  - `CONDITION`: whether to run conditioned or effectively unconditional sampling
  - `CLASS_ID`: ImageNet class id for conditioned runs
  - `GUIDANCE`: classifier-free guidance scale
  - `STEPS`, `SEED`
  - `OUT_NAME`: output PNG path

Outputs:
- A single PNG under `generation/outputs/` by default.
- Console summary reporting steps, radius, intervals, and output path.

Exact invocation:
```bash
conda run -n arena-env python -m generation.run_dit_local_global
```

Example override:
```bash
RADIUS=7 LOCAL_INTERVALS='[[0.0, 0.2], [0.7, 1.0]]' OUT_NAME=generation/outputs/dit_r7.png \
conda run -n arena-env python -m generation.run_dit_local_global
```

### `run_dit_local_global_clip.py`
Function:
- Generates a single SD3 text-to-image sample with timestep-dependent switching between global attention and local attention.
- Supports either the gather-based local attention implementation or the masked-SDPA implementation.

Inputs:
- Driven by environment variables rather than argparse.
- Common knobs:
  - `MODEL_ID`: SD3 checkpoint, default `stabilityai/stable-diffusion-3-medium-diffusers`
  - `PROMPT`, `PROMPT2`, `PROMPT3`
  - `NEGATIVE_PROMPT`, `NEGATIVE_PROMPT2`, `NEGATIVE_PROMPT3`
  - `RADIUS`, `LOCAL_INTERVALS`
  - `GUIDANCE`, `STEPS`, `SEED`
  - `HEIGHT`, `WIDTH`
  - `LOCAL_ATTN_IMPL`: `masked_sdpa` or `gather`
  - `DISABLE_T5`, `CPU_OFFLOAD`, `SEQUENTIAL_CPU_OFFLOAD`, `VARIANT`
  - `OUT_NAME`

Outputs:
- A single PNG under `generation/outputs/` by default.
- Console summary reporting geometry, attention implementation, and output path.

Exact invocation:
```bash
conda run -n arena-env python -m generation.run_dit_local_global_clip
```

Example override:
```bash
PROMPT='a golden retriever playing in a park' \
RADIUS=5 LOCAL_INTERVALS='[[0.0, 0.6]]' LOCAL_ATTN_IMPL=masked_sdpa \
OUT_NAME=generation/outputs/sd3_r5.png \
conda run -n arena-env python -m generation.run_dit_local_global_clip
```
