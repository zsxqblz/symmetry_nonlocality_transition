#!/usr/bin/env python3
from score_gap.plot_score_gap_sampling_only_generic import run_plotting


if __name__ == "__main__":
    run_plotting(
        default_root="data/score_gap_sd3_uncond",
        default_outdir="visualization/score_gap_sd3_uncond",
        default_metrics_name="sd3_sampling_score_gap_metrics.json",
    )
