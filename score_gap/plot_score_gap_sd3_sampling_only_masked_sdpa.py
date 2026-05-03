"""
Plots for the sampling-only SD3 masked-SDPA score-gap runs.

Generates three 1D plot variants for each requested radius subset:
- plain lines without error bands
- trajectory std bands (mean +/- std)
- SEM bands (mean +/- std / sqrt(n))

For each subset, writes:
- a global conditioning-gap curve
- a conditional locality-gap overlay across radii
- an unconditional locality-gap overlay across radii
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot masked-SDPA SD3 sampling-only score-gap runs.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(MODULE_DIR / "data" / "score_gap_sd3_cond"),
        help="Root directory containing r*/sd3_sampling_score_gap_metrics.json outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(MODULE_DIR / "visualization" / "score_gap_sd3_cond"),
        help="Output directory for plots.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def find_runs(root: Path) -> Dict[int, Dict]:
    runs: Dict[int, Dict] = {}
    for metrics_path in root.glob("r*/sd3_sampling_score_gap_metrics.json"):
        match = re.fullmatch(r"r(\d+)", metrics_path.parent.name)
        if not match:
            continue
        radius = int(match.group(1))
        with open(metrics_path, "r", encoding="utf-8") as handle:
            runs[radius] = json.load(handle)
    return runs


def aggregate_stats(rows: Iterable[Dict], metric_key: str) -> Dict[int, Dict[str, float]]:
    buckets: Dict[int, List[float]] = {}
    t_norms: Dict[int, float] = {}
    for row in rows:
        value = row.get(metric_key)
        if value is None:
            continue
        t_idx = int(row["t_index"])
        buckets.setdefault(t_idx, []).append(float(value))
        t_norms[t_idx] = float(row["t_norm"])

    stats: Dict[int, Dict[str, float]] = {}
    for t_idx, values in sorted(buckets.items()):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stats[t_idx] = {
            "mean": mean,
            "std": std,
            "count": len(values),
            "t_norm": t_norms[t_idx],
        }
    return stats


def get_sample_rows(data: Dict) -> List[Dict]:
    return data.get("results", {}).get("sample", [])


def canonical_axis(data: Dict, metric_key: str) -> Tuple[List[int], np.ndarray]:
    stats = aggregate_stats(get_sample_rows(data), metric_key)
    if not stats:
        raise ValueError(f"No stats found for metric '{metric_key}'.")
    timesteps = sorted(stats.keys())
    t_norm = np.array([float(stats[t_idx]["t_norm"]) for t_idx in timesteps], dtype=float)
    return timesteps, t_norm


def align_series(
    stats: Dict[int, Dict[str, float]],
    timesteps: List[int],
    key: str,
    fill: float = np.nan,
) -> np.ndarray:
    return np.array([float(stats.get(t_idx, {}).get(key, fill)) for t_idx in timesteps], dtype=float)


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


def compute_ticks(t_norm_plot: np.ndarray) -> Tuple[List[int], List[str]]:
    if t_norm_plot.size == 0:
        return [], []
    candidate_indices = np.linspace(0, len(t_norm_plot) - 1, 11)
    xticks: List[int] = []
    labels: List[str] = []
    for raw_idx in candidate_indices:
        idx = int(round(float(raw_idx)))
        if xticks and idx == xticks[-1]:
            continue
        xticks.append(idx)
        labels.append(f"{t_norm_plot[idx]:.1f}")
    return xticks, labels


def plot_single_curve(
    means: np.ndarray,
    bands: np.ndarray,
    t_norm: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
    color: str = "black",
) -> None:
    means_plot = means[::-1]
    bands_plot = bands[::-1]
    t_norm_plot = t_norm[::-1]
    xs = np.arange(len(means_plot), dtype=float)
    xticks, labels = compute_ticks(t_norm_plot)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(xs, means_plot, color=color, linewidth=2.2)
    if np.nanmax(np.abs(bands_plot)) > 0:
        ax.fill_between(xs, means_plot - bands_plot, means_plot + bands_plot, color=color, alpha=0.18)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlim(0, max(0, len(means_plot) - 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_overlay(
    subset_runs: List[Tuple[int, Dict]],
    metric_key: str,
    band_mode: str,
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    if not subset_runs:
        return

    timesteps, t_norm = canonical_axis(subset_runs[0][1], metric_key)
    t_norm_plot = t_norm[::-1]
    xs = np.arange(len(t_norm_plot), dtype=float)
    xticks, labels = compute_ticks(t_norm_plot)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(subset_runs)))

    fig, ax = plt.subplots(figsize=(12, 5.3), constrained_layout=True)
    for color, (radius, data) in zip(colors, subset_runs):
        stats = aggregate_stats(get_sample_rows(data), metric_key)
        means = align_series(stats, timesteps, "mean")[::-1]
        stds = align_series(stats, timesteps, "std")[::-1]
        counts = align_series(stats, timesteps, "count", fill=0.0)[::-1].astype(int)
        bands = band_from_stats(stds, counts, band_mode)
        ax.plot(xs, means, color=color, linewidth=2.0, label=f"r={radius}")
        if np.nanmax(np.abs(bands)) > 0:
            ax.fill_between(xs, means - bands, means + bands, color=color, alpha=0.12)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlim(0, max(0, len(t_norm_plot) - 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=14)
    ax.legend(ncol=2, fontsize=12, frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def collect_heatmap_data(
    subset_runs: List[Tuple[int, Dict]],
    metric_key: str,
) -> Tuple[List[int], List[int], np.ndarray]:
    all_timesteps = set()
    per_run_series: List[Tuple[int, Dict[int, float]]] = []
    for radius, data in subset_runs:
        stats = aggregate_stats(get_sample_rows(data), metric_key)
        per_run_series.append((radius, {t_idx: row["mean"] for t_idx, row in stats.items()}))
        all_timesteps.update(stats.keys())

    timesteps = sorted(all_timesteps)
    if not timesteps:
        return [], [], np.zeros((0, 0), dtype=float)

    radii = [radius for radius, _ in per_run_series]
    arr = np.full((len(radii), len(timesteps)), np.nan, dtype=float)
    for i, (_, series) in enumerate(per_run_series):
        for j, t_idx in enumerate(timesteps):
            if t_idx in series:
                arr[i, j] = series[t_idx]
    return radii, timesteps, arr


def plot_heatmap_with_curve(
    subset_runs: List[Tuple[int, Dict]],
    heatmap_metric_key: str,
    band_mode: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    radii, timesteps, arr = collect_heatmap_data(subset_runs, heatmap_metric_key)
    if arr.size == 0:
        return

    reference_data = subset_runs[0][1]
    curve_means, curve_bands, t_norm = conditioning_curve_from_reference(reference_data, band_mode)
    arr_plot = arr[:, ::-1]
    curve_plot = curve_means[::-1]
    band_plot = curve_bands[::-1]
    t_norm_plot = t_norm[::-1]
    xs = np.arange(len(t_norm_plot), dtype=float)
    xticks, labels = compute_ticks(t_norm_plot)

    fig, (ax_curve, ax_heat) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 11.7),
        gridspec_kw={"height_ratios": [1.2, 2]},
        constrained_layout=True,
    )

    # Match typography to the violin paper plots.
    ax_curve.tick_params(labelsize=30)
    ax_heat.tick_params(labelsize=30)

    ax_curve.plot(xs, curve_plot, color="black", linewidth=2.2)
    if np.nanmax(np.abs(band_plot)) > 0:
        ax_curve.fill_between(xs, curve_plot - band_plot, curve_plot + band_plot, color="black", alpha=0.18)
    ax_curve.set_title(title, fontsize=22)
    ax_curve.set_ylabel(r"$\|\Delta \vec{s}_{cond}\|$", fontsize=30)
    ax_curve.yaxis.set_label_position("right")
    ax_curve.yaxis.tick_right()
    ax_curve.set_xlim(0, max(0, len(t_norm_plot) - 1))
    ax_curve.set_xticks(xticks)
    ax_curve.set_xticklabels([])
    ax_curve.tick_params(labelbottom=False)

    image = ax_heat.imshow(
        np.ma.array(arr_plot, mask=np.isnan(arr_plot)),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        extent=[0, len(t_norm_plot), 0, len(radii)],
    )
    ax_heat.set_xlabel("t", fontsize=30)
    ax_heat.set_ylabel("r", fontsize=30)
    ax_heat.set_xticks(xticks)
    ax_heat.set_xticklabels(labels, rotation=45, ha="right")
    ax_heat.set_yticks([i + 0.5 for i in range(len(radii))])
    ax_heat.set_yticklabels([str(radius) for radius in radii])
    cbar = fig.colorbar(image, ax=ax_heat)
    cbar.set_label(r"$\|\Delta \vec{s}_{loc}\|$", fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def require_subset(runs: Dict[int, Dict], radii: List[int], subset_name: str) -> List[Tuple[int, Dict]]:
    missing = [radius for radius in radii if radius not in runs]
    if missing:
        missing_str = ", ".join(str(radius) for radius in missing)
        raise FileNotFoundError(f"Missing radii for {subset_name}: {missing_str}")
    return [(radius, runs[radius]) for radius in radii]


def conditioning_curve_from_reference(reference_data: Dict, band_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    timesteps, t_norm = canonical_axis(reference_data, "conditioning_gap_global")
    stats = aggregate_stats(get_sample_rows(reference_data), "conditioning_gap_global")
    means = align_series(stats, timesteps, "mean")
    stds = align_series(stats, timesteps, "std")
    counts = align_series(stats, timesteps, "count", fill=0.0).astype(int)
    bands = band_from_stats(stds, counts, band_mode)
    return means, bands, t_norm


def mode_description(mode_key: str) -> str:
    if mode_key == "plain":
        return "no error band"
    if mode_key == "trajstd":
        return "trajectory std"
    if mode_key == "sem":
        return "SEM"
    raise ValueError(mode_key)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    runs = find_runs(root)
    if not runs:
        raise FileNotFoundError(f"No sd3_sampling_score_gap_metrics.json files found under {root}")

    summary_lines = [
        f"root={root}",
        f"available_radii={','.join(str(radius) for radius in sorted(runs))}",
    ]

    for subset_name, radii in RADIUS_SUBSETS.items():
        subset_runs = require_subset(runs, radii, subset_name)
        subset_dir = outdir / subset_name
        reference_data = subset_runs[0][1]

        for mode_suffix, band_mode in BAND_MODES.items():
            means, bands, t_norm = conditioning_curve_from_reference(reference_data, band_mode)
            plot_single_curve(
                means=means,
                bands=bands,
                t_norm=t_norm,
                title="Global conditioning gap",
                ylabel=r"$\|\Delta \vec{s}_{cond}\|$",
                out_path=subset_dir / f"conditioning_gap_global_{mode_suffix}.png",
                dpi=args.dpi,
            )
            plot_overlay(
                subset_runs=subset_runs,
                metric_key="mse_local_vs_global_cond",
                band_mode=band_mode,
                title="Conditional locality gap",
                ylabel=r"$\|\Delta \vec{s}_{loc}\|$",
                out_path=subset_dir / f"locality_gap_cond_overlay_{mode_suffix}.png",
                dpi=args.dpi,
            )
            plot_overlay(
                subset_runs=subset_runs,
                metric_key="mse_local_vs_global_uncond",
                band_mode=band_mode,
                title="Unconditional locality gap",
                ylabel=r"$\|\Delta \vec{s}_{loc}\|$",
                out_path=subset_dir / f"locality_gap_uncond_overlay_{mode_suffix}.png",
                dpi=args.dpi,
            )
            plot_heatmap_with_curve(
                subset_runs=subset_runs,
                heatmap_metric_key="mse_local_vs_global_cond",
                band_mode=band_mode,
                title="Conditioning gap + conditional locality heatmap",
                out_path=subset_dir / f"locality_gap_cond_heatmap_curve_{mode_suffix}.png",
                dpi=args.dpi,
            )
            plot_heatmap_with_curve(
                subset_runs=subset_runs,
                heatmap_metric_key="mse_local_vs_global_uncond",
                band_mode=band_mode,
                title="Conditioning gap + unconditional locality heatmap",
                out_path=subset_dir / f"locality_gap_uncond_heatmap_curve_{mode_suffix}.png",
                dpi=args.dpi,
            )
            summary_lines.append(
                f"{subset_name}:{mode_suffix}:{mode_description(mode_suffix)}"
            )

    run_info = outdir / "run_info.txt"
    run_info.parent.mkdir(parents=True, exist_ok=True)
    run_info.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote plots to {outdir}")


if __name__ == "__main__":
    main()
