#!/usr/bin/env python
"""Replot benchmark outputs for paper figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


PERTURB_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_CSV = (
    PERTURB_DIR
    / "results"
    / "conditioning_window_fid_500"
    / "metrics"
    / "conditioning_window_summary_with_error_bars.csv"
)
DEFAULT_OUTPUT_DIR = PERTURB_DIR / "paper_plots" / "conditioning_window_fid_500_ti_axis_wide"
DEFAULT_PLOT_PREFIX = "conditioning_window"
FIGSIZE = (10.5, 8.0 / 3.0)
LINEWIDTH = 1.8
MARKERSIZE = 5.5
DPI = 200
FONT_SCALE = 1.5


def apply_font_scale(scale: float) -> None:
    keys = [
        "font.size",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
    ]
    updated = {}
    for key in keys:
        value = plt.rcParams[key]
        if isinstance(value, str):
            continue
        updated[key] = value * scale
    plt.rcParams.update(updated)


apply_font_scale(FONT_SCALE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot conditioning-window benchmark curves for paper figures.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(DEFAULT_SUMMARY_CSV),
        help="Path to benchmark summary CSV, preferably *_with_error_bars.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for the reformatted plots.",
    )
    parser.add_argument(
        "--plot-prefix",
        default=DEFAULT_PLOT_PREFIX,
        help="Filename prefix for generated plots.",
    )
    parser.add_argument(
        "--error-ymin",
        type=float,
        default=0.7,
        help="Lower bound for classifier-error y-axis.",
    )
    parser.add_argument(
        "--error-ymax",
        type=float,
        default=1.0,
        help="Upper bound for classifier-error y-axis.",
    )
    parser.add_argument(
        "--x-shift",
        type=float,
        default=0.0,
        help="Additive shift applied to interval_start before plotting.",
    )
    parser.add_argument(
        "--max-schedule-index",
        type=int,
        default=None,
        help="If set, keep only rows with schedule_index <= this value.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        help="Lower bound for x-axis.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Upper bound for x-axis.",
    )
    return parser.parse_args()


def load_summary(path: Path, max_schedule_index: int | None = None) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            schedule_index = float(row["schedule_index"])
            if max_schedule_index is not None and schedule_index > max_schedule_index:
                continue
            rows.append(
                {
                    "schedule_index": schedule_index,
                    "interval_start": float(row["interval_start"]),
                    "interval_end": float(row["interval_end"]),
                    "error_rate": float(row["error_rate"]),
                    "fid_to_global_conditioned": float(row["fid_to_global_conditioned"]),
                    "error_ci_low": float(row["error_ci_low"]) if row.get("error_ci_low") else None,
                    "error_ci_high": float(row["error_ci_high"]) if row.get("error_ci_high") else None,
                    "fid_ci_low": float(row["fid_ci_low"]) if row.get("fid_ci_low") else None,
                    "fid_ci_high": float(row["fid_ci_high"]) if row.get("fid_ci_high") else None,
                }
            )
    rows.sort(key=lambda row: row["schedule_index"])
    return rows


def plot_metric(
    rows: list[dict[str, float]],
    y_key: str,
    y_label: str,
    output_path: Path,
    *,
    x_shift: float = 0.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ci_low_key: str | None = None,
    ci_high_key: str | None = None,
) -> None:
    xs = [row["interval_start"] + x_shift for row in rows]
    ys = [row[y_key] for row in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    show_error_bars = (
        ci_low_key is not None
        and ci_high_key is not None
        and all(row.get(ci_low_key) is not None and row.get(ci_high_key) is not None for row in rows)
    )
    if show_error_bars:
        lows = [row[ci_low_key] for row in rows]
        highs = [row[ci_high_key] for row in rows]
        ax.vlines(xs, lows, highs, linewidth=1.1, alpha=0.9)
        cap_half_width = 0.012
        for x, low, high in zip(xs, lows, highs):
            ax.hlines(low, x - cap_half_width, x + cap_half_width, linewidth=1.1, alpha=0.9)
            ax.hlines(high, x - cap_half_width, x + cap_half_width, linewidth=1.1, alpha=0.9)
        ax.plot(xs, ys, marker="o", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    else:
        ax.plot(xs, ys, marker="o", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if xlim is not None:
        start, end = xlim
        tick_start = math.ceil(start * 10.0) / 10.0
        tick_end = math.floor(end * 10.0) / 10.0
        num_ticks = int(round((tick_end - tick_start) / 0.1)) + 1
        xticks = [round(tick_start + 0.1 * idx, 10) for idx in range(max(0, num_ticks))]
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"$t_i$")
    ax.set_ylabel(y_label)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.2, top=0.96)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_summary(args.summary_csv, max_schedule_index=args.max_schedule_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        rows,
        "error_rate",
        "classifier error",
        args.output_dir / f"{args.plot_prefix}_classifier_error_vs_t_i.png",
        x_shift=args.x_shift,
        xlim=(args.xmin, args.xmax) if args.xmin is not None and args.xmax is not None else None,
        ylim=(args.error_ymin, args.error_ymax),
        ci_low_key="error_ci_low",
        ci_high_key="error_ci_high",
    )
    plot_metric(
        rows,
        "fid_to_global_conditioned",
        "FID to baseline",
        args.output_dir / f"{args.plot_prefix}_fid_to_global_conditioned_vs_t_i.png",
        x_shift=args.x_shift,
        xlim=(args.xmin, args.xmax) if args.xmin is not None and args.xmax is not None else None,
        ci_low_key="fid_ci_low",
        ci_high_key="fid_ci_high",
    )

    print(f"Saved paper plots to {args.output_dir}")


if __name__ == "__main__":
    main()
