# Forward / Backward

This directory contains the partial-denoising experiments and the aggregate plotting scripts that summarize their saved metrics.

## Bundled data

Available metric roots:
- `data/noise_sweep_global`
- `data/noise_sweep_local`
- `data/noise_sweep_sd3_global`
- `data/noise_sweep_sd3_local`

These copies keep metrics, plots, and example images. 

## Recommended Workflow

1. Run the sweep:
   - use the global script for the global baseline
   - use the local script for each radius or local schedule you want to compare
2. Verify each run directory contains:
   - `metrics.json`
   - example images
   - per-run MSE and classifier-error plots
3. After collecting the local and global runs, execute the aggregate plotter:
   - `aggregate_noise_sweep_plots.py` for DiT
   - `aggregate_noise_sweep_plots_sd3.py` for SD3
4. Use the aggregate visualization directory as the summary artifact for release or paper figures.

## Runners

### `noise_sweep_global_denoising.py`
Function:
- Generates global-attention DiT clean samples, injects noise at many timesteps, denoises globally, and records reconstruction MSE and classifier behavior.

Inputs:
- CLI arguments via `argparse`
- Main inputs:
  - `--model-id`
  - `--class-id`
  - `--guidance`
  - `--clean-steps`, `--denoise-steps`
  - `--num-noise-levels`, `--min-t`, `--max-t`
  - `--base-seed`, `--noise-seed-offset`
  - `--n-clean`, `--n-denoise`
  - `--output-dir`, `--metrics-path`, `--raw-mse-path`

Outputs:
- A run directory containing `metrics.json`, optional raw per-example JSON, example images, and single-run MSE/classification plots.

### `noise_sweep_local_denoising.py`
Function:
- Same protocol as the global DiT sweep, but denoises with local attention.

Inputs:
- All inputs from the global DiT sweep plus:
  - `--radius`
  - `--local-intervals`

Outputs:
- A local sweep directory, typically one per radius, containing `metrics.json`, examples, and single-run plots.

### `noise_sweep_global_sd3_denoising.py`
Function:
- SD3 version of the global partial-denoising sweep using text prompts instead of class labels.

Inputs:
- Main inputs:
  - `--model-id`
  - `--prompt`, `--prompt2`, `--prompt3`
  - `--negative-prompt`, `--negative-prompt2`, `--negative-prompt3`
  - `--guidance`
  - `--clean-steps`, `--denoise-steps`
  - `--num-noise-levels`, `--min-t`, `--max-t`
  - `--base-seed`, `--noise-seed-offset`
  - `--n-clean`, `--n-denoise`
  - `--output-dir`, `--metrics-path`, `--raw-mse-path`
  - `--use-dpms`

Outputs:
- `metrics.json`, examples, and per-run MSE/classification plots for SD3 global denoising.

### `noise_sweep_local_sd3_denoising.py`
Function:
- SD3 version of the local partial-denoising sweep.

Inputs:
- All global SD3 sweep inputs plus:
  - `--radius`
  - `--local-intervals`

Outputs:
- A local SD3 sweep directory with `metrics.json`, examples, and per-run plots.

## Plotting

### `aggregate_noise_sweep_plots.py`
Function:
- Aggregates DiT local and global sweep metrics across radii into shared MSE and classifier-error plots.

Inputs:
- `--local-root`
- `--global-root`
- `--out-dir`
- `--mse-error-stat`
- `--class-error-stat`

Outputs:
- Aggregate figures under `forward_backward/visualization/noise_sweep_aggregate/`

Exact command:
```bash
conda run -n arena-env python -m forward_backward.aggregate_noise_sweep_plots
```

### `aggregate_noise_sweep_plots_sd3.py`
Function:
- Aggregates SD3 local and global sweep metrics across radii into shared MSE and classifier-error plots.

Inputs:
- `--local-root`
- `--global-root`
- `--out-dir`

Outputs:
- Aggregate figures under `forward_backward/visualization/noise_sweep_sd3_aggregate/`

Exact command:
```bash
conda run -n arena-env python -m forward_backward.aggregate_noise_sweep_plots_sd3
```
