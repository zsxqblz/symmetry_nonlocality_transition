#!/usr/bin/env python3
from score_gap.plot_score_gap_dit_generic import run_plotting


if __name__ == "__main__":
    run_plotting(
        default_root="data/score_gap_facebook_train",
        default_outdir="visualization/score_gap_facebook_train",
    )
