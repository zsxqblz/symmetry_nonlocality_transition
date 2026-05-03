#!/usr/bin/env python3
"""
Generic aggregate plots for DiT score-gap summaries in the masked-SDPA runs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent


RADIUS_SUBSETS = {
    "r1_to_r8_step1": list(range(1, 9)),
    "r2_to_r16_step2": list(range(2, 17, 2)),
}

BAND_MODES = {
    "plain": "none",
    "trajstd": "std",
    "sem": "sem",
}


def build_parser(default_root: str, default_outdir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic DiT score-gap aggregate plots.")
    parser.add_argument("--root", type=str, default=str(MODULE_DIR / default_root))
    parser.add_argument("--outdir", type=str, default=str(MODULE_DIR / default_outdir))
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def find_runs(root: Path) -> Dict[int, Dict]:
    runs: Dict[int, Dict] = {}
    for metrics_path in root.glob("r*/score_gap_summary.json"):
        match = re.fullmatch(r"r(\d+)", metrics_path.parent.name)
        if not match:
            continue
        with open(metrics_path, "r", encoding="utf-8") as handle:
            runs[int(match.group(1))] = json.load(handle)
    return runs


def extract_series_stats(summary: Dict, metric_root: str, metric_key: str) -> Tuple[List[int], List[float], List[float], List[int]]:
    bucket = summary.get(metric_root, {}).get(metric_key, {})
    xs = sorted(int(t) for t in bucket.keys())
    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []
    for t in xs:
        entry = bucket.get(str(t), bucket.get(t, {}))
        means.append(float(entry.get("mean", np.nan)))
        stds.append(float(entry.get("std", 0.0)))
        counts.append(int(entry.get("count", 0)))
    return xs, means, stds, counts


def align_series(xs: Sequence[int], values: Sequence[float], timesteps: Sequence[int], fill_value: float = np.nan) -> np.ndarray:
    mapping = dict(zip(xs, values))
    return np.array([mapping.get(t, fill_value) for t in timesteps], dtype=float)


def band_from_stats(stds: np.ndarray, counts: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros_like(stds, dtype=float)

    band = np.full(stds.shape, np.nan, dtype=float)
    valid = counts > 0
    if mode == "std":
        band[valid] = stds[valid]
    elif mode == "sem":
        band[valid] = stds[valid] / np.sqrt(counts[valid])
    else:
        raise ValueError(f"Unsupported band mode: {mode}")
    return band


def compute_normalized_ticks(timesteps: Sequence[int]) -> Tuple[List[int], List[str]]:
    if not timesteps:
        return [], []

    t_min = min(timesteps)
    t_max = max(timesteps)
    t_range = max(1e-9, float(t_max - t_min))
    tick_norms = np.linspace(0.0, 1.0, 11)
    xticks_idx: List[int] = []
    xtick_labels: List[str] = []
    for t_norm in tick_norms:
        target_t = t_min + t_norm * t_range
        nearest_idx = min(range(len(timesteps)), key=lambda i: abs(timesteps[i] - target_t))
        if xticks_idx and nearest_idx == xticks_idx[-1]:
            continue
        xticks_idx.append(nearest_idx)
        xtick_labels.append(f"{t_norm:.1f}")
    return xticks_idx, xtick_labels


def collect_heatmap_data(
    subset_runs: List[Tuple[int, Dict]],
    metric_root: str,
    metric_key: str,
) -> Tuple[List[int], List[int], np.ndarray]:
    all_timesteps = set()
    per_run_series: List[Tuple[int, Dict[int, float]]] = []
    for radius, summary in subset_runs:
        xs, means, _, _ = extract_series_stats(summary, metric_root, metric_key)
        per_run_series.append((radius, dict(zip(xs, means))))
        all_timesteps.update(xs)

    timesteps = sorted(all_timesteps)
    if not timesteps:
        return [], [], np.zeros((0, 0), dtype=float)

    radii = [radius for radius, _ in per_run_series]
    arr = np.full((len(radii), len(timesteps)), np.nan, dtype=float)
    for i, (_, series) in enumerate(per_run_series):
        for j, timestep in enumerate(timesteps):
            if timestep in series:
                arr[i, j] = series[timestep]
    return radii, timesteps, arr


def representative_curve_stats(
    subset_runs: List[Tuple[int, Dict]],
    metric_root: str,
    metric_key: str,
    timesteps: Sequence[int],
    band_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if not timesteps:
        return np.array([]), np.array([])

    for _, summary in subset_runs:
        xs, means, stds, counts = extract_series_stats(summary, metric_root, metric_key)
        if not xs:
            continue
        mean_arr = align_series(xs, means, timesteps)
        std_arr = align_series(xs, stds, timesteps)
        count_arr = align_series(xs, [float(c) for c in counts], timesteps, fill_value=0.0).astype(int)
        return mean_arr, band_from_stats(std_arr, count_arr, band_mode)
    return np.array([]), np.array([])


def plot_curve_with_error_band(
    timesteps: Sequence[int],
    mean: np.ndarray,
    band: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
    line_color: str = "black",
) -> None:
    if mean.size == 0 or not timesteps:
        return

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    x_positions = np.arange(len(timesteps), dtype=float)
    xticks_idx, xtick_labels = compute_normalized_ticks(timesteps)

    ax.plot(x_positions, mean, color=line_color, linewidth=2.2)
    if np.nanmax(np.abs(band)) > 0:
        ax.fill_between(x_positions, mean - band, mean + band, color=line_color, alpha=0.18)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlim(0, max(0, len(timesteps) - 1))
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_overlay(
    subset_runs: List[Tuple[int, Dict]],
    metric_root: str,
    metric_key: str,
    band_mode: str,
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    _, timesteps, _ = collect_heatmap_data(subset_runs, metric_root, metric_key)
    if not timesteps:
        return

    fig, ax = plt.subplots(figsize=(12, 5.3), constrained_layout=True)
    x_positions = np.arange(len(timesteps), dtype=float)
    xticks_idx, xtick_labels = compute_normalized_ticks(timesteps)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(subset_runs))))

    plotted = False
    for color, (radius, summary) in zip(colors, subset_runs):
        xs, means, stds, counts = extract_series_stats(summary, metric_root, metric_key)
        if not xs:
            continue
        mean_arr = align_series(xs, means, timesteps)
        std_arr = align_series(xs, stds, timesteps)
        count_arr = align_series(xs, [float(c) for c in counts], timesteps, fill_value=0.0).astype(int)
        band_arr = band_from_stats(std_arr, count_arr, band_mode)
        ax.plot(x_positions, mean_arr, color=color, linewidth=2.0, label=f"r={radius}")
        if np.nanmax(np.abs(band_arr)) > 0:
            ax.fill_between(x_positions, mean_arr - band_arr, mean_arr + band_arr, color=color, alpha=0.12)
        plotted = True

    if not plotted:
        plt.close()
        return

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlim(0, max(0, len(timesteps) - 1))
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=14)
    ax.legend(ncol=2, fontsize=12, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_heatmap_with_curve(
    subset_runs: List[Tuple[int, Dict]],
    heatmap_metric_root: str,
    heatmap_metric_key: str,
    curve_metric_root: str,
    curve_metric_key: str,
    band_mode: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    radii, timesteps, arr = collect_heatmap_data(subset_runs, heatmap_metric_root, heatmap_metric_key)
    if arr.size == 0:
        return

    curve, curve_band = representative_curve_stats(
        subset_runs,
        curve_metric_root,
        curve_metric_key,
        timesteps,
        band_mode,
    )

    fig, (ax_curve, ax_heat) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 11.7),
        gridspec_kw={"height_ratios": [1.2, 2]},
        constrained_layout=True,
    )

    curve_label_fontsize = 30
    heatmap_label_fontsize = 30
    heatmap_tick_fontsize = 30
    colorbar_fontsize = 30
    title_fontsize = 30

    x_positions = np.arange(len(timesteps), dtype=float)
    xticks_idx, xtick_labels = compute_normalized_ticks(timesteps)

    if curve.size > 0:
        ax_curve.plot(x_positions, curve, color="black", linewidth=2.2)
        if np.nanmax(np.abs(curve_band)) > 0:
            ax_curve.fill_between(x_positions, curve - curve_band, curve + curve_band, color="black", alpha=0.18)
    ax_curve.set_title(title, fontsize=title_fontsize)
    ax_curve.set_ylabel(r"$\|\Delta \vec{s}_{cond}\|$", fontsize=curve_label_fontsize)
    ax_curve.yaxis.set_label_position("right")
    ax_curve.yaxis.tick_right()
    ax_curve.set_xlim(0, max(0, len(timesteps) - 1))
    ax_curve.set_xticks(xticks_idx)
    ax_curve.set_xticklabels([])
    ax_curve.tick_params(labelbottom=False, labelsize=heatmap_tick_fontsize)

    image = ax_heat.imshow(
        np.ma.array(arr, mask=np.isnan(arr)),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        extent=[0, len(timesteps), 0, len(radii)],
    )
    ax_heat.set_xlabel("t", fontsize=heatmap_label_fontsize)
    ax_heat.set_ylabel("r", fontsize=heatmap_label_fontsize)
    ax_heat.set_xticks(xticks_idx)
    ax_heat.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax_heat.set_yticks([i + 0.5 for i in range(len(radii))])
    ax_heat.set_yticklabels([str(radius) for radius in radii])
    ax_heat.tick_params(labelsize=heatmap_tick_fontsize)
    cbar = fig.colorbar(image, ax=ax_heat)
    cbar.set_label(r"$\|\Delta \vec{s}_{loc}\|$", fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def available_subset(runs: Dict[int, Dict], radii: Sequence[int]) -> Tuple[List[Tuple[int, Dict]], List[int]]:
    present = [(radius, runs[radius]) for radius in radii if radius in runs]
    missing = [radius for radius in radii if radius not in runs]
    return present, missing


def run_plotting(default_root: str, default_outdir: str) -> None:
    parser = build_parser(default_root, default_outdir)
    args = parser.parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    runs = find_runs(root)
    if not runs:
        raise FileNotFoundError(f"No score_gap_summary.json files found under {root}")

    summary_lines = [
        f"root={root}",
        f"available_radii={','.join(str(radius) for radius in sorted(runs))}",
    ]

    split_specs = [
        ("train", "local_vs_global_time", "conditioning_gap_time"),
        ("sample", "local_vs_global_time", "conditioning_gap_time"),
    ]

    for subset_name, radii in RADIUS_SUBSETS.items():
        subset_runs, missing = available_subset(runs, radii)
        summary_lines.append(f"{subset_name}:available={','.join(str(radius) for radius, _ in subset_runs) or 'none'}")
        summary_lines.append(f"{subset_name}:missing={','.join(str(radius) for radius in missing) or 'none'}")
        if not subset_runs:
            continue

        subset_dir = outdir / subset_name
        for mode_suffix, band_mode in BAND_MODES.items():
            for split_name, locality_root, conditioning_root in split_specs:
                conditioning_key = f"{split_name}_global"
                curve_timesteps = collect_heatmap_data(subset_runs, locality_root, f"{split_name}_cond")[1]
                curve, curve_band = representative_curve_stats(
                    subset_runs,
                    conditioning_root,
                    conditioning_key,
                    curve_timesteps,
                    band_mode,
                )
                plot_curve_with_error_band(
                    timesteps=curve_timesteps,
                    mean=curve,
                    band=curve_band,
                    title=f"{split_name.capitalize()} conditioning gap",
                    ylabel=r"$\|\Delta \vec{s}_{cond}\|$",
                    out_path=subset_dir / f"{split_name}_conditioning_gap_global_{mode_suffix}.png",
                    dpi=args.dpi,
                )

                for branch_name, locality_key in (("cond", f"{split_name}_cond"), ("uncond", f"{split_name}_uncond")):
                    plot_overlay(
                        subset_runs=subset_runs,
                        metric_root=locality_root,
                        metric_key=locality_key,
                        band_mode=band_mode,
                        title=f"{split_name.capitalize()} locality gap ({branch_name})",
                        ylabel=r"$\|\Delta \vec{s}_{loc}\|$",
                        out_path=subset_dir / f"{split_name}_locality_gap_{branch_name}_overlay_{mode_suffix}.png",
                        dpi=args.dpi,
                    )
                    plot_heatmap_with_curve(
                        subset_runs=subset_runs,
                        heatmap_metric_root=locality_root,
                        heatmap_metric_key=locality_key,
                        curve_metric_root=conditioning_root,
                        curve_metric_key=conditioning_key,
                        band_mode=band_mode,
                        title=f"{split_name.capitalize()} locality gap ({branch_name}) + global conditioning gap",
                        out_path=subset_dir / f"{split_name}_locality_gap_{branch_name}_heatmap_curve_{mode_suffix}.png",
                        dpi=args.dpi,
                    )

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_info.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote plots to {outdir}")


if __name__ == "__main__":
    run_plotting(
        default_root="data/score_gap_facebook_cond",
        default_outdir="visualization/score_gap_facebook_cond",
    )
