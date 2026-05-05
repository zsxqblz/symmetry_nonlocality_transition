# Windowed Sampling

This directory contains the conditioning-window and sliding global-window/local benchmark runners, the error-bar post-processing utility, and the plot regeneration scripts.

## Bundled results

Bundled benchmark result roots live under `results/` and include:
- `conditioning_window_fid_500`
- `sd3_conditioning_window_fid_500`
- `dit_sliding_global_window_local_r5_L0p4_unconditional_n50`
- `sd3_sliding_global_window_local_r5_L0p4_unconditional_n50`
- `dit_sliding_global_window_local_masked_sdpa_r*_L0p4_unconditional_n500`
- `sd3_sliding_global_window_local_masked_sdpa_r*_L0p4_unconditional_n500`
- `sd3_sliding_global_window_local_masked_sdpa_r*_L0p4_unconditional_n62`

Bundled paper figures live under `paper_plots/`.

## Recommended Workflow

1. Run the benchmark:
   - use `run_schedule_fid_benchmark.py` or `run_sd3_schedule_fid_benchmark.py` for the main experiments
   - use the masked-SDPA benchmark runners for radius sweeps over explicit `t_i` windows
2. Confirm the raw benchmark directory contains:
   - metrics JSON
   - summary CSV
   - examples
   - plots
3. Post-process with `postprocess_benchmark_error_bars.py` to produce:
   - `*_summary_with_error_bars.csv`
   - uncertainty JSON outputs
4. Regenerate figures:
   - use `replot_paper_plots_with_error_bars.py` for the main paper figure families
   - use `replot_masked_sdpa_radius_aggregate.py` for the DiT masked-SDPA radius aggregate
   - use `replot_sd3_masked_sdpa_radius_aggregate.py` for the SD3 masked-SDPA radius aggregate
   - use `replot_conditioning_window_paper_benchmarks.py` when you want one figure family from one summary CSV

## Benchmark runners

### `run_schedule_fid_benchmark.py`
Function:
- Runs the ImageNet DiT benchmark for:
  - `local_attention`
  - `conditioning_window`
  - `sliding_global_window_local`
- Generates samples, computes classifier error, and measures FID to a global-conditioned baseline.

Inputs:
- CLI arguments via `argparse`
- Main inputs:
  - `--experiment`
  - `--model-id`
  - `--class-id`
  - `--guidance`
  - `--steps`
  - `--radius`
  - `--window-length`
  - `--outside-conditioning`
  - `--num-samples`
  - `--base-seed`
  - `--schedule-indices`
  - `--output-dir`
  - `--classifier-model-id`, `--classifier-device`
  - `--fid-device`, `--fid-batch-size`
  - `--local-files-only`
  - `--save-all-images`, `--append-existing`, `--skip-existing-schedules`, `--allow-cpu`

Outputs:
- A benchmark directory containing:
  - `metrics/*_metrics.json`
  - `metrics/*_summary.csv`
  - `examples/`
  - `plots/`
  - optional saved images and FID features

### `run_dit_sliding_global_window_local_masked_sdpa_benchmark.py`
Function:
- Runs the DiT masked-SDPA sliding-window benchmark over explicit `t_i` start values.
- Reuses the main DiT benchmark utilities but swaps in the masked-SDPA local attention kernel.

Inputs:
- Main inputs:
  - `--radius` (required)
  - `--window-length`
  - `--outside-conditioning`
  - `--ti-start`, `--ti-end`, `--ti-step`, `--schedule-index-offset`
  - `--num-samples`, `--base-seed`, `--schedule-indices`
  - `--output-dir`
  - the same classifier/FID/runtime controls as the base DiT benchmark

Outputs:
- A masked-SDPA benchmark directory per radius, including summary CSVs and saved plots.

### `run_sd3_schedule_fid_benchmark.py`
Function:
- SD3 analogue of the main benchmark runner.
- Supports `local_attention`, `conditioning_window`, and `sliding_global_window_local` experiments with text prompts and SD3 runtime controls.

Inputs:
- Main inputs:
  - `--experiment`
  - `--model-id`
  - `--prompt`, `--prompt2`, `--prompt3`
  - `--negative-prompt`, `--negative-prompt2`, `--negative-prompt3`
  - `--guidance`
  - `--steps`
  - `--radius`
  - `--window-length`
  - `--num-samples`, `--num-examples`, `--base-seed`
  - `--schedule-indices`
  - `--output-dir`
  - `--height`, `--width`
  - `--classifier-model-id`, `--classifier-device`
  - `--fid-device`, `--fid-batch-size`
  - `--max-sequence-length`, `--variant`
  - `--cpu-offload`, `--sequential-cpu-offload`, `--vae-tiling`, `--vae-slicing`, `--disable-t5`, `--use-dpms`
  - `--allow-cpu`

Outputs:
- SD3 benchmark directories analogous to the DiT runner: metrics, summaries, examples, plots, and optional cached features.

### `run_sd3_sliding_global_window_local_masked_sdpa_benchmark.py`
Function:
- SD3 masked-SDPA sliding-window benchmark over explicit `t_i` start values.

Inputs:
- Main inputs:
  - `--radius` (required)
  - `--window-length`
  - `--outside-conditioning`
  - `--ti-start`, `--ti-end`, `--ti-step`
  - `--num-samples`, `--num-examples`, `--base-seed`
  - `--schedule-indices`
  - `--output-dir`
  - `--height`, `--width`
  - prompt and SD3 runtime options inherited from the SD3 benchmark runner

Outputs:
- SD3 masked-SDPA benchmark directories with metrics, summaries, examples, and plots.

## Post-processing

### `postprocess_benchmark_error_bars.py`
Function:
- Adds classifier-confidence intervals and FID uncertainty estimates to existing benchmark result directories.

Inputs:
- Positional `result_dirs`: one or more benchmark result directories
- `--ci-level`
- `--fid-interval-method`
- `--fid-bootstrap-reps`
- `--fid-fold-permutations`
- `--fid-min-fold-size`
- `--bootstrap-seed`
- `--recompute-fid-point`
- `--summary-suffix`

Outputs:
- New `*_summary_with_error_bars.csv` files
- JSON payloads describing classifier and FID uncertainty

## Plot regeneration

### `replot_conditioning_window_paper_benchmarks.py`
Function:
- Replots a single summary CSV into the wide-format paper figures used for conditioning-window and selected sliding-window experiments.

Inputs:
- `--summary-csv`
- `--output-dir`
- `--plot-prefix`
- `--error-ymin`, `--error-ymax`
- `--x-shift`
- `--max-schedule-index`
- `--xmin`, `--xmax`

Outputs:
- Two PNGs in the target directory:
  - classifier error vs `t_i`
  - FID to global-conditioned baseline vs `t_i`

### `replot_paper_plots_with_error_bars.py`
Function:
- Batch driver that regenerates the four committed paper-plot families from saved `*_summary_with_error_bars.csv`.

Inputs:
- No CLI arguments; uses hard-coded `PLOT_CONFIGS` pointing at the bundled result directories.

Outputs:
- Regenerates:
  - `paper_plots/conditioning_window_fid_500_ti_axis_wide`
  - `paper_plots/sd3_conditioning_window_fid_500_ti_axis_wide`
  - `paper_plots/dit_sliding_global_window_local_r5_L0p4_unconditional_n50_ti_axis_full`
  - `paper_plots/sd3_sliding_global_window_local_r5_L0p4_unconditional_n50_ti_axis_full`

Exact command:
```bash
conda run -n arena-env python -m windowed_sampling.replot_paper_plots_with_error_bars
```

### `replot_masked_sdpa_radius_aggregate.py`
Function:
- Aggregates the masked-SDPA sliding-window benchmark across multiple radii into two paper plots.

Inputs:
- `--results-glob`
- `--output-dir`
- `--xmin`, `--xmax`
- `--error-ymin`, `--error-ymax`
- `--legend-loc`

Outputs:
- Radius-aggregate classifier-error plot
- Radius-aggregate FID plot

Exact command:
```bash
conda run -n arena-env python -m windowed_sampling.replot_masked_sdpa_radius_aggregate
```

### `replot_sd3_masked_sdpa_radius_aggregate.py`
Function:
- Aggregates the SD3 masked-SDPA sliding-window benchmark across multiple radii into two paper plots.

Inputs:
- `--results-glob`
- `--output-dir`
- `--xmin`, `--xmax`
- `--error-ymin`, `--error-ymax`

Outputs:
- SD3 radius-aggregate classifier-error plot
- SD3 radius-aggregate FID plot

Exact command:
```bash
conda run -n arena-env python -m windowed_sampling.replot_sd3_masked_sdpa_radius_aggregate
```
