#!/usr/bin/env python3
"""
Aggregate SD3 score-gap metrics (sd3_score_gap_metrics.json) across radii and emit simple plots.

Expected layout:
score_gap_outputs_sd3/
  r3/sd3_score_gap_metrics.json
  r5/sd3_score_gap_metrics.json

Metrics schema (from run_score_gap_experiment_sd3.py):
- results.train: list of entries with mse_local_vs_global (few timesteps)
- results.sample: same for sampling trajectory
- train_mse_mean/max, sample_mse_mean/max
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SD3 score gap plots across radii.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(MODULE_DIR / "data" / "score_gap_sd3"),
        help="Root directory containing r*/sd3_score_gap_metrics.json.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(MODULE_DIR / "visualization" / "score_gap_sd3_aggregate"),
        help="Where to write combined plots.",
    )
    return parser.parse_args()


def infer_radius(path: Path, data: Dict) -> float:
    if "radius" in data:
        try:
            return float(data["radius"])
        except Exception:
            pass
    m = re.search(r"r(\d+)", path.parent.name)
    if m:
        return float(m.group(1))
    raise ValueError(f"Could not infer radius for metrics at {path}")


def find_runs(root: Path) -> List[Tuple[float, Dict]]:
    runs: List[Tuple[float, Dict]] = []
    for metrics in root.glob("r*/sd3_score_gap_metrics.json"):
        with open(metrics, "r", encoding="utf-8") as f:
            data = json.load(f)
        radius = infer_radius(metrics, data)
        runs.append((radius, data))
    runs.sort(key=lambda x: x[0])
    return runs


def plot_bar(runs: List[Tuple[float, Dict]], key: str, title: str, ylabel: str, out_path: Path):
    xs, ys = [], []
    for r, data in runs:
        if key in data.get("results", {}):
            xs.append(r)
            ys.append(float(data["results"][key]))
    if not xs:
        return
    plt.figure(figsize=(6, 3))
    plt.bar(xs, ys, width=0.6)
    plt.xlabel("radius (r)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap_timesteps(
    runs: List[Tuple[float, Dict]],
    traj_key: str,
    out_path: Path,
    title: str,
    metric_key: str = "mse_local_vs_global",
):
    radii = []
    ts_set = set()
    series = []
    for r, data in runs:
        vals = data.get("results", {}).get(traj_key, [])
        ts = {int(e["t_index"]) for e in vals if metric_key in e}
        ts_set |= ts
        series.append((r, {int(e["t_index"]): float(e[metric_key]) for e in vals if metric_key in e}))
        radii.append(r)
    # Reverse so clean (late steps) are on the left of the heatmap.
    ts_sorted = sorted(ts_set, reverse=True)
    if not ts_sorted:
        return
    arr = np.full((len(series), len(ts_sorted)), np.nan)
    for i, (_, sdict) in enumerate(series):
        for j, t_idx in enumerate(ts_sorted):
            if t_idx in sdict:
                arr[i, j] = sdict[t_idx]
    fig, ax = plt.subplots(figsize=(8, 4))
    data = np.ma.array(arr, mask=np.isnan(arr))
    cax = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("t index (clean \u2192 noise)")
    ax.set_ylabel("radius (r)")
    ax.set_xticks(range(len(ts_sorted)))
    ax.set_xticklabels(ts_sorted)
    ax.set_yticks(range(len(radii)))
    ax.set_yticklabels([int(r) for r in radii])
    fig.colorbar(cax, ax=ax, label="mse(local vs global)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    runs = find_runs(root)
    if not runs:
        raise FileNotFoundError(f"No sd3_score_gap_metrics.json under {root}")

    # Bar plots of aggregate means
    plot_bar(runs, "train_mse_mean", "Train: mean MSE local vs global", "mse", outdir / "train_mse_mean.png")
    plot_bar(runs, "sample_mse_mean", "Sample: mean MSE local vs global", "mse", outdir / "sample_mse_mean.png")
    plot_bar(
        runs,
        "train_conditioning_gap_global_mean",
        "Train: conditioning gap (global)",
        "mse",
        outdir / "train_conditioning_gap_global_mean.png",
    )
    plot_bar(
        runs,
        "train_conditioning_gap_local_mean",
        "Train: conditioning gap (local)",
        "mse",
        outdir / "train_conditioning_gap_local_mean.png",
    )
    plot_bar(
        runs,
        "sample_conditioning_gap_global_mean",
        "Sample: conditioning gap (global)",
        "mse",
        outdir / "sample_conditioning_gap_global_mean.png",
    )
    plot_bar(
        runs,
        "sample_conditioning_gap_local_mean",
        "Sample: conditioning gap (local)",
        "mse",
        outdir / "sample_conditioning_gap_local_mean.png",
    )

    # Heatmaps over timesteps (start/mid/end)
    plot_heatmap_timesteps(
        runs, "train", outdir / "train_mse_heatmap.png", "Train: MSE(local vs global) by timestep"
    )
    plot_heatmap_timesteps(
        runs, "sample", outdir / "sample_mse_heatmap.png", "Sample: MSE(local vs global) by timestep"
    )
    plot_heatmap_timesteps(
        runs,
        "train",
        outdir / "train_conditioning_gap_global_heatmap.png",
        "Train: conditioning gap (global) by timestep",
        metric_key="conditioning_gap_global",
    )
    plot_heatmap_timesteps(
        runs,
        "train",
        outdir / "train_conditioning_gap_local_heatmap.png",
        "Train: conditioning gap (local) by timestep",
        metric_key="conditioning_gap_local",
    )
    plot_heatmap_timesteps(
        runs,
        "sample",
        outdir / "sample_conditioning_gap_global_heatmap.png",
        "Sample: conditioning gap (global) by timestep",
        metric_key="conditioning_gap_global",
    )
    plot_heatmap_timesteps(
        runs,
        "sample",
        outdir / "sample_conditioning_gap_local_heatmap.png",
        "Sample: conditioning gap (local) by timestep",
        metric_key="conditioning_gap_local",
    )
    print(f"[OK] wrote plots to {outdir}")


if __name__ == "__main__":
    main()
