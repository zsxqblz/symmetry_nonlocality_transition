#!/usr/bin/env python
"""Regenerate the committed paper plot directories using saved error bars."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


PERTURB_DIR = Path(__file__).resolve().parent
REPLOT_SCRIPT = PERTURB_DIR / "replot_conditioning_window_paper_benchmarks.py"

PLOT_CONFIGS = [
    {
        "summary_csv": PERTURB_DIR / "results" / "conditioning_window_fid_500" / "metrics" / "conditioning_window_summary_with_error_bars.csv",
        "output_dir": PERTURB_DIR / "paper_plots" / "conditioning_window_fid_500_ti_axis_wide",
        "plot_prefix": "conditioning_window",
        "error_ymin": 0.7,
        "error_ymax": 1.0,
    },
    {
        "summary_csv": PERTURB_DIR / "results" / "sd3_conditioning_window_fid_500" / "metrics" / "sd3_conditioning_window_summary_with_error_bars.csv",
        "output_dir": PERTURB_DIR / "paper_plots" / "sd3_conditioning_window_fid_500_ti_axis_wide",
        "plot_prefix": "sd3_conditioning_window",
        "error_ymin": 0.0,
        "error_ymax": 1.0,
    },
    {
        "summary_csv": PERTURB_DIR / "results" / "dit_sliding_global_window_local_r5_L0p4_unconditional_n50" / "metrics" / "sliding_global_window_local_summary_with_error_bars.csv",
        "output_dir": PERTURB_DIR / "paper_plots" / "dit_sliding_global_window_local_r5_L0p4_unconditional_n50_ti_axis_full",
        "plot_prefix": "sliding_global_window_local",
        "error_ymin": 0.0,
        "error_ymax": 1.0,
        "x_shift": 0.2,
        "max_schedule_index": 6,
        "xmin": 0.0,
        "xmax": 1.0,
    },
    {
        "summary_csv": PERTURB_DIR / "results" / "sd3_sliding_global_window_local_r5_L0p4_unconditional_n50" / "metrics" / "sd3_sliding_global_window_local_summary_with_error_bars.csv",
        "output_dir": PERTURB_DIR / "paper_plots" / "sd3_sliding_global_window_local_r5_L0p4_unconditional_n50_ti_axis_full",
        "plot_prefix": "sd3_sliding_global_window_local",
        "error_ymin": 0.0,
        "error_ymax": 1.0,
        "x_shift": 0.2,
        "max_schedule_index": 6,
        "xmin": 0.0,
        "xmax": 1.0,
    },
]


def count_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    for cfg in PLOT_CONFIGS:
        cmd = [
            sys.executable,
            str(REPLOT_SCRIPT),
            "--summary-csv",
            str(cfg["summary_csv"]),
            "--output-dir",
            str(cfg["output_dir"]),
            "--plot-prefix",
            str(cfg["plot_prefix"]),
            "--error-ymin",
            str(cfg["error_ymin"]),
            "--error-ymax",
            str(cfg["error_ymax"]),
        ]
        if "x_shift" in cfg:
            cmd.extend(["--x-shift", str(cfg["x_shift"])])
        if "max_schedule_index" in cfg:
            cmd.extend(["--max-schedule-index", str(cfg["max_schedule_index"])])
        if "xmin" in cfg:
            cmd.extend(["--xmin", str(cfg["xmin"])])
        if "xmax" in cfg:
            cmd.extend(["--xmax", str(cfg["xmax"])])

        row_count = count_rows(cfg["summary_csv"])
        print(
            f"[replot] output_dir={cfg['output_dir'].name} "
            f"rows={row_count} summary={cfg['summary_csv'].name}"
        )
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
