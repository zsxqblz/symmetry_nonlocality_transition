# Symmetry Breaking and Nonlocality Phase Transitions

This directory contains code and plot data accompanying "Concurrence of Symmetry Breaking and Nonlocality Phase Transitions in Diffusion Models". The file structure is organized as follows.

- `models/`: local-attention modules used by the copied experiments
- `generation/`: DiT and SD3 sample-generation entrypoints
- `score_gap/`: score-gap experiments, aggregate plots, and bundled DiT summaries
- `forward_backward/`: forward-backward experiments and plotting entrypoints
- `windowed_sampling/`: conditioning-window and sliding-window benchmarks, replot scripts

Run experiments from the repository root with module execution, for example:

```bash
python -m generation.run_dit_local_global
python -m score_gap.plot_score_gap_dit_masked_sdpa_cond
python -m forward_backward.aggregate_noise_sweep_plots
python -m windowed_sampling.replot_paper_plots_with_error_bars
```

Bundled data is sufficient to regenerate the committed paper plots from the repository root. Run:

```bash
python -m score_gap.plot_score_gap_dit_masked_sdpa_cond
python -m score_gap.plot_score_gap_dit_masked_sdpa_uncond
python -m score_gap.plot_score_gap_dit_masked_sdpa_cond_trainsample
python -m score_gap.plot_score_gap_aggregate
python -m forward_backward.aggregate_noise_sweep_plots
python -m forward_backward.aggregate_noise_sweep_plots_sd3
python -m windowed_sampling.replot_paper_plots_with_error_bars
python -m windowed_sampling.replot_masked_sdpa_radius_aggregate
python -m windowed_sampling.replot_sd3_masked_sdpa_radius_aggregate
```
