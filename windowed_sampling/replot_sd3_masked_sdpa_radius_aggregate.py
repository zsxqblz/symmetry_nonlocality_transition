#!/usr/bin/env python
"""Aggregate masked-SDPA SD3 sliding-window radius sweep plots for local-perturbation runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


PERTURB_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_GLOB = "sd3_sliding_global_window_local_masked_sdpa_r*_L0p4_unconditional_n62"
DEFAULT_OUTPUT_DIR = PERTURB_DIR / "paper_plots" / "sd3_sliding_global_window_local_masked_sdpa_unconditional_n62_radius_aggregate"
FIGSIZE = (10.5, 4.3)
LINEWIDTH = 1.7
MARKERSIZE = 4.8
DPI = 200
FONT_SCALE = 1.35


def apply_font_scale(scale: float) -> None:
    keys = [
        "font.size",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
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
        description="Aggregate masked-SDPA sliding-window radius sweep plots.",
    )
    parser.add_argument(
        "--results-glob",
        default=DEFAULT_RESULTS_GLOB,
        help="Glob under results/ selecting radius directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the aggregate plots.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=0.2,
        help="Minimum x-axis value.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=0.8,
        help="Maximum x-axis value.",
    )
    parser.add_argument(
        "--error-ymin",
        type=float,
        default=0.0,
        help="Minimum classifier-error y-axis value.",
    )
    parser.add_argument(
        "--error-ymax",
        type=float,
        default=1.0,
        help="Maximum classifier-error y-axis value.",
    )
    return parser.parse_args()


def extract_radius(result_dir: Path) -> int:
    name = result_dir.name
    marker = "_r"
    start = name.index(marker) + len(marker)
    end = name.index("_L", start)
    return int(name[start:end])


def load_summary(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "interval_start": float(row["interval_start"]),
                    "interval_end": float(row["interval_end"]),
                    "interval_center": 0.5 * (float(row["interval_start"]) + float(row["interval_end"])),
                    "error_rate": float(row["error_rate"]),
                    "fid_to_global_conditioned": float(row["fid_to_global_conditioned"]),
                    "error_ci_low": float(row["error_ci_low"]),
                    "error_ci_high": float(row["error_ci_high"]),
                    "fid_ci_low": float(row["fid_ci_low"]),
                    "fid_ci_high": float(row["fid_ci_high"]),
                }
            )
    rows.sort(key=lambda row: row["interval_center"])
    return rows


def discover_runs(results_glob: str, summary_name: str) -> list[tuple[int, Path, list[dict[str, float]]]]:
    result_dirs = sorted((PERTURB_DIR / "results").glob(results_glob))
    runs: list[tuple[int, Path, list[dict[str, float]]]] = []
    for result_dir in result_dirs:
        summary_path = result_dir / "metrics" / summary_name
        if not summary_path.is_file():
            continue
        runs.append((extract_radius(result_dir), result_dir, load_summary(summary_path)))
    runs.sort(key=lambda item: item[0])
    if not runs:
        raise FileNotFoundError(f"No postprocessed summary CSVs matched results/{results_glob}")
    return runs


def plot_metric(
    runs: list[tuple[int, Path, list[dict[str, float]]]],
    *,
    y_key: str,
    ci_low_key: str,
    ci_high_key: str,
    y_label: str,
    output_path: Path,
    xmin: float,
    xmax: float,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    cmap = plt.get_cmap("viridis", len(runs) + 1)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for idx, (radius, _result_dir, rows) in enumerate(runs):
        xs = [row["interval_center"] for row in rows]
        ys = [row[y_key] for row in rows]
        lows = [row[ci_low_key] for row in rows]
        highs = [row[ci_high_key] for row in rows]
        yerr = [
            [max(0.0, y - low) for y, low in zip(ys, lows)],
            [max(0.0, high - y) for y, high in zip(ys, highs)],
        ]
        color = cmap(idx + 1)
        ax.errorbar(xs, ys, yerr=yerr, color=color, marker="o", linewidth=LINEWIDTH, markersize=MARKERSIZE, elinewidth=1.0, capsize=2.5, label=f"r={radius}")

    xticks = sorted({round(row["interval_center"], 4) for _radius, _result_dir, rows in runs for row in rows})
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("window center")
    ax.set_ylabel(y_label)
    ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.18, top=0.78)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render(results_glob: str, summary_name: str, output_dir: Path, error_name: str, fid_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(results_glob, summary_name)
    plot_metric(runs, y_key="error_rate", ci_low_key="error_ci_low", ci_high_key="error_ci_high", y_label="classifier error", output_path=output_dir / error_name, xmin=0.2, xmax=0.8, ymin=0.0, ymax=1.0)
    plot_metric(runs, y_key="fid_to_global_conditioned", ci_low_key="fid_ci_low", ci_high_key="fid_ci_high", y_label="FID to baseline", output_path=output_dir / fid_name, xmin=0.2, xmax=0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate masked-SDPA SD3 sliding-window radius sweep plots.")
    parser.add_argument("--results-glob", default=DEFAULT_RESULTS_GLOB, help="Glob under results/ selecting radius directories.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save the aggregate plots.")
    parser.add_argument("--xmin", type=float, default=0.2, help="Minimum x-axis value.")
    parser.add_argument("--xmax", type=float, default=0.8, help="Maximum x-axis value.")
    parser.add_argument("--error-ymin", type=float, default=0.0, help="Minimum classifier-error y-axis value.")
    parser.add_argument("--error-ymax", type=float, default=1.0, help="Maximum classifier-error y-axis value.")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.results_glob, "sd3_sliding_global_window_local_summary_with_error_bars.csv")
    plot_metric(runs, y_key="error_rate", ci_low_key="error_ci_low", ci_high_key="error_ci_high", y_label="classifier error", output_path=output_dir / "sd3_masked_sdpa_radius_aggregate_classifier_error_vs_t_i.png", xmin=args.xmin, xmax=args.xmax, ymin=args.error_ymin, ymax=args.error_ymax)
    plot_metric(runs, y_key="fid_to_global_conditioned", ci_low_key="fid_ci_low", ci_high_key="fid_ci_high", y_label="FID to baseline", output_path=output_dir / "sd3_masked_sdpa_radius_aggregate_fid_vs_t_i.png", xmin=args.xmin, xmax=args.xmax)
    print(f"Saved aggregate plots to {output_dir}")


if __name__ == "__main__":
    main()
