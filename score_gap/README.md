# Score Gap

This directory contains the score-gap experiment runners plus the post-processing and plotting scripts used to turn saved metrics into paper-ready plots.

## Bundled data

Bundled DiT score-gap summaries:
- `data/score_gap_facebook_cond`
- `data/score_gap_facebook_uncond`
- `data/score_gap_facebook_train`

Bundled SD3 sampling-only summaries:
- `data/score_gap_sd3_cond`
- `data/score_gap_sd3_uncond`

Not bundled:
- The aggregate SD3 `sd3_score_gap_metrics.json` dataset expected by `plot_score_gap_aggregate_sd3.py`

## Recommended Workflow

1. Generate metrics:
   - run `run_score_gap_experiment.py` for DiT train/sampling score-gap experiments
   - run `run_score_gap_experiment_sd3_sampling_only.py` for SD3 sampling-only score-gap experiments
2. Inspect the saved metric JSON:
   - `score_gap_summary.json` for DiT-style runs
   - `sd3_sampling_score_gap_metrics.json` for SD3 sampling-only runs
3. Run the matching plot script:
   - DiT wrappers for bundled DiT datasets
   - SD3 sampling-only wrappers for bundled SD3 datasets
   - aggregate plotters when you want a radius-wide summary across runs
4. Read `run_info.txt` in the visualization directory to confirm which roots and plot settings were used.

## Runners

### `run_score_gap_experiment.py`
Function:
- Runs the DiT score-gap experiment on both training trajectories and sampling trajectories.
- Compares local versus global scores and also measures conditional versus unconditional score gaps.

Inputs:
- CLI arguments via `argparse`.
- Main inputs:
  - `--dataset-dir`: required image directory for training-trajectory probes
  - `--model-id`
  - `--radius`
  - `--local-intervals`
  - `--num-samples`
  - `--sampling-steps`, `--train-steps`, `--max-train-t`
  - `--class-id`, `--guidance-scale`
  - `--output-dir`
  - `--local-attn-impl`: `gather` or `masked_sdpa`
  - `--seed`, `--batch-size`

Outputs:
- A run directory under `score_gap/outputs/score_gap_dit/` unless overridden.
- Per-radius `score_gap_summary.json`
- Per-radius line plots such as `train_local_vs_global.png` and `sample_conditioning_gap.png`

Example:
```bash
conda run -n arena-env python -m score_gap.run_score_gap_experiment \
  --dataset-dir /path/to/imagenet_subset \
  --output-dir score_gap/outputs/score_gap_dit
```

### `run_score_gap_experiment_sd3_sampling_only.py`
Function:
- Runs the SD3 sampling-only score-gap experiment.
- Measures conditional and unconditional locality gaps along real evolving SD3 denoising trajectories.

Inputs:
- CLI arguments via `argparse`.
- Main inputs:
  - `--model-id`
  - `--radius`
  - `--local-intervals`
  - `--sampling-steps`
  - `--num-samples`
  - `--guidance-scale`
  - `--prompt`
  - `--sampling-trajectory`: `guided` or `unconditional`
  - `--local-attn-impl`
  - `--output-dir`

Outputs:
- A run directory under `score_gap/outputs/score_gap_sd3_sampling_only/` unless overridden.
- `sd3_sampling_score_gap_metrics.json`

Example:
```bash
conda run -n arena-env python -m score_gap.run_score_gap_experiment_sd3_sampling_only \
  --output-dir score_gap/outputs/score_gap_sd3_sampling_only/r7
```

## Plotting and post-processing

### `plot_score_gap_dit_generic.py`
Function:
- Shared plotting backend for DiT `score_gap_summary.json` directories.
- Produces per-radius overlays, heatmaps, and heatmap-plus-curve figures.

Inputs:
- `--root`: directory containing `r*/score_gap_summary.json`
- `--outdir`
- `--dpi`

Outputs:
- Plot families under the requested output directory plus `run_info.txt`

This file is usually called through the wrappers below.

### `plot_score_gap_dit_masked_sdpa_cond.py`
Function:
- Plots the bundled conditioned DiT summaries.

Inputs:
- Optional `--root`, `--outdir`, `--dpi` forwarded through the generic backend.

Outputs:
- `score_gap/visualization/score_gap_facebook_cond/`

Exact command:
```bash
conda run -n arena-env python -m score_gap.plot_score_gap_dit_masked_sdpa_cond
```

### `plot_score_gap_dit_masked_sdpa_uncond.py`
Function:
- Plots the bundled unconditioned DiT summaries.

Inputs:
- Same interface as the conditioned wrapper.

Outputs:
- `score_gap/visualization/score_gap_facebook_uncond/`

Exact command:
```bash
conda run -n arena-env python -m score_gap.plot_score_gap_dit_masked_sdpa_uncond
```

### `plot_score_gap_dit_masked_sdpa_cond_trainsample.py`
Function:
- Plots the bundled DiT training-trajectory summaries.

Inputs:
- Same interface as the conditioned wrapper.

Outputs:
- `score_gap/visualization/score_gap_facebook_train/`

Exact command:
```bash
conda run -n arena-env python -m score_gap.plot_score_gap_dit_masked_sdpa_cond_trainsample
```

### `plot_score_gap_aggregate.py`
Function:
- Aggregates DiT score-gap summaries across radii and emits combined line plots and heatmaps.

Inputs:
- `--root`
- `--outdir`

Outputs:
- Aggregate plots under `score_gap/visualization/score_gap_facebook_cond_aggregate/` by default

Exact command:
```bash
conda run -n arena-env python -m score_gap.plot_score_gap_aggregate
```

### `plot_score_gap_sampling_only_generic.py`
Function:
- Shared plotting backend for sampling-only metrics with per-timestep rows.
- Aggregates repeated trajectories into mean/std/SEM plots.

Inputs:
- `--root`
- `--outdir`
- `--metrics-name`
- `--dpi`

Outputs:
- Radius-subset plot directories plus `run_info.txt`

This file is usually called through the wrappers below.

### `plot_score_gap_sd3_sampling_only_masked_sdpa.py`
Function:
- Plots the bundled conditional SD3 sampling-only dataset.

Inputs:
- `--root`, default `score_gap/data/score_gap_sd3_cond`
- `--outdir`, default `score_gap/visualization/score_gap_sd3_cond`
- `--dpi`

Outputs:
- Conditional SD3 plot families under `score_gap/visualization/score_gap_sd3_cond/`

Exact command:
```bash
conda run -n arena-env python -m score_gap.plot_score_gap_sd3_sampling_only_masked_sdpa \
  --root score_gap/data/score_gap_sd3_cond \
  --outdir score_gap/visualization/score_gap_sd3_cond
```

### `plot_score_gap_sd3_sampling_only_masked_sdpa_unconditional.py`
Function:
- Plots the bundled unconditional SD3 sampling-only dataset.

Inputs:
- `--root`, default `score_gap/data/score_gap_sd3_uncond`
- `--outdir`, default `score_gap/visualization/score_gap_sd3_uncond`
- `--metrics-name`, default `sd3_sampling_score_gap_metrics.json`

Outputs:
- Unconditional SD3 plot families under `score_gap/visualization/score_gap_sd3_uncond/`

Exact command:
```bash
conda run -n arena-env python -m score_gap.plot_score_gap_sd3_sampling_only_masked_sdpa_unconditional \
  --root score_gap/data/score_gap_sd3_uncond \
  --outdir score_gap/visualization/score_gap_sd3_uncond
```

### `plot_score_gap_aggregate_sd3.py`
Function:
- Aggregates SD3 `sd3_score_gap_metrics.json` runs across radii into bar plots and heatmaps.

Inputs:
- `--root`: directory containing `r*/sd3_score_gap_metrics.json`
- `--outdir`

Outputs:
- Aggregate SD3 plots under the requested output directory

Exact command once that dataset exists:
```bash
conda run -n arena-env python -m score_gap.plot_score_gap_aggregate_sd3 \
  --root score_gap/data/score_gap_sd3 \
  --outdir score_gap/visualization/score_gap_sd3_aggregate
```
