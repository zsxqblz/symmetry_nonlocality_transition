"""
Aggregate score-gap summaries across different radii and emit combined plots.

Expected layouts:
score_gap_outputs/
  run_r3/
    score_gap_summary.json
  run_r5/
    score_gap_summary.json

score_gap_outputs_flux2/
  global/
    flux2_score_gap_metrics.json
  r3/
    flux2_score_gap_metrics.json

This script scans for supported summary files, extracts the radius,
normalizes them into a common schema, and plots per-timestep means.
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
    parser = argparse.ArgumentParser(description="Aggregate score gap plots across radii.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(MODULE_DIR / "data" / "score_gap_facebook_cond"),
        help="Root directory containing run subfolders with score_gap_summary.json.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(MODULE_DIR / "visualization" / "score_gap_facebook_cond_aggregate"),
        help="Where to write combined plots.",
    )
    return parser.parse_args()


def infer_radius(summary_path: Path, data: Dict) -> float:
    # Prefer explicit radius in summary
    if "radius" in data:
        try:
            return float(data["radius"])
        except Exception:
            pass
    # Fallback to config radius if present
    cfg_radius = data.get("config", {}).get("radius", None)
    if cfg_radius is not None:
        try:
            return float(cfg_radius)
        except Exception:
            pass
    # Try to parse from parent folder name like r3
    m = re.search(r"r(\d+)", summary_path.parent.name)
    if m:
        return float(m.group(1))
    # If we reach here, we cannot infer radius; surface an error to help debugging.
    raise ValueError(f"Could not infer radius for summary at {summary_path}. Parent dir: {summary_path.parent}")


def find_runs(root: Path) -> List[Tuple[float, Path]]:
    runs: List[Tuple[float, Path]] = []
    for pattern in ("score_gap_summary.json", "flux2_score_gap_metrics.json"):
        for summary in root.rglob(pattern):
            if not summary.is_file():
                continue
            with open(summary, "r", encoding="utf-8") as f:
                data = json.load(f)
            radius = infer_radius(summary, data)
            runs.append((radius, summary))
    runs.sort(key=lambda x: x[0])
    return runs


def bucket_mean(rows: List[Dict], value_key: str) -> Dict[int, Dict[str, float]]:
    buckets: Dict[int, List[float]] = {}
    for row in rows:
        if value_key not in row or row[value_key] is None:
            continue
        t_index = int(row["t_index"])
        buckets.setdefault(t_index, []).append(float(row[value_key]))
    return {
        t_index: {"mean": float(sum(vals) / len(vals))}
        for t_index, vals in sorted(buckets.items())
        if vals
    }


def normalize_flux2_metrics(data: Dict) -> Dict:
    results = data.get("results", {})
    train_rows = results.get("train", [])
    sample_rows = results.get("sample", [])
    return {
        "config": data.get("config", {}),
        "radius": data.get("config", {}).get("radius", data.get("radius")),
        "local_vs_global_time": {
            "train_cond": bucket_mean(train_rows, "mse_local_vs_global"),
            "sample_cond": bucket_mean(sample_rows, "mse_local_vs_global"),
        },
        "conditioning_gap_time": {
            "train_global": bucket_mean(train_rows, "conditioning_gap_global"),
            "train_local": bucket_mean(train_rows, "conditioning_gap_local"),
            "sample_global": bucket_mean(sample_rows, "conditioning_gap_global"),
            "sample_local": bucket_mean(sample_rows, "conditioning_gap_local"),
        },
    }


def extract_series(summary: Dict, key: str) -> Tuple[List[int], List[float]]:
    bucket = summary.get(key, {})
    # keys might be strings in JSON
    xs = sorted(int(t) for t in bucket.keys())
    ys: List[float] = []
    for t in xs:
        entry = bucket.get(str(t), bucket.get(t, {}))
        ys.append(float(entry.get("mean", 0.0)))
    return xs, ys


def plot_multi(
    runs: List[Tuple[float, Dict]],
    metric_root: str,
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 4))
    plotted = False
    for radius, summary in runs:
        series = summary.get(metric_root, {})
        xs, ys = extract_series(series, metric_key)
        if not xs:
            continue
        plt.plot(xs, ys, label=f"r={int(radius)}")
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def collect_heatmap_data(
    runs: List[Tuple[float, Dict]],
    metric_root: str,
    metric_key: str,
) -> Tuple[List[float], List[int], np.ndarray]:
    # Collect union of timesteps
    all_ts: set[int] = set()
    series_per_run: List[Tuple[float, Dict]] = []
    for radius, summary in runs:
        series = summary.get(metric_root, {})
        xs, ys = extract_series(series, metric_key)
        series_per_run.append((radius, dict(zip(xs, ys))))
        all_ts.update(xs)
    ts_sorted = sorted(all_ts)
    if not ts_sorted:
        return [], [], np.zeros((0, 0))

    radii = [r for r, _ in series_per_run]
    arr = np.full((len(radii), len(ts_sorted)), np.nan, dtype=float)
    for i, (_, series) in enumerate(series_per_run):
        for j, t in enumerate(ts_sorted):
            if t in series:
                arr[i, j] = series[t]
    return radii, ts_sorted, arr


def plot_heatmap(
    runs: List[Tuple[float, Dict]],
    metric_root: str,
    metric_key: str,
    title: str,
    out_path: Path,
) -> None:
    radii, timesteps, arr = collect_heatmap_data(runs, metric_root, metric_key)
    if arr.size == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    data = np.ma.array(arr, mask=np.isnan(arr))
    cax = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("radius (r)")
    # Thin xticks if too many
    if len(timesteps) > 12:
        step = max(1, len(timesteps) // 12)
        xticks_idx = list(range(0, len(timesteps), step))
    else:
        xticks_idx = list(range(len(timesteps)))
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels([timesteps[i] for i in xticks_idx], rotation=45, ha="right")
    ax.set_yticks(list(range(len(radii))))
    ax.set_yticklabels([int(r) for r in radii])
    fig.colorbar(cax, ax=ax, label="mean |Δ score| per pixel")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    runs = find_runs(root)
    if not runs:
        raise FileNotFoundError(f"No score_gap_summary.json files found under {root}")

    loaded: List[Tuple[float, Dict]] = []
    for radius, path in runs:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if path.name == "flux2_score_gap_metrics.json":
            summary = normalize_flux2_metrics(summary)
        loaded.append((radius, summary))

    # local vs global gap plots
    plot_multi(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="train_cond",
        title="Train: local vs global gap (cond)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "train_local_vs_global_cond.png",
    )
    plot_heatmap(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="train_cond",
        title="Train: local vs global gap (cond, heatmap)",
        out_path=outdir / "train_local_vs_global_cond_heatmap.png",
    )
    plot_multi(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="train_uncond",
        title="Train: local vs global gap (uncond)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "train_local_vs_global_uncond.png",
    )
    plot_heatmap(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="train_uncond",
        title="Train: local vs global gap (uncond, heatmap)",
        out_path=outdir / "train_local_vs_global_uncond_heatmap.png",
    )
    plot_multi(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="train_global",
        title="Train: conditioning gap (global)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "train_conditioning_gap_global.png",
    )
    plot_heatmap(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="train_global",
        title="Train: conditioning gap (global, heatmap)",
        out_path=outdir / "train_conditioning_gap_global_heatmap.png",
    )
    plot_multi(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="train_local",
        title="Train: conditioning gap (local)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "train_conditioning_gap_local.png",
    )
    plot_heatmap(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="train_local",
        title="Train: conditioning gap (local, heatmap)",
        out_path=outdir / "train_conditioning_gap_local_heatmap.png",
    )

    plot_multi(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="sample_cond",
        title="Sample: local vs global gap (cond)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "sample_local_vs_global_cond.png",
    )
    plot_heatmap(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="sample_cond",
        title="Sample: local vs global gap (cond, heatmap)",
        out_path=outdir / "sample_local_vs_global_cond_heatmap.png",
    )
    plot_multi(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="sample_uncond",
        title="Sample: local vs global gap (uncond)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "sample_local_vs_global_uncond.png",
    )
    plot_heatmap(
        loaded,
        metric_root="local_vs_global_time",
        metric_key="sample_uncond",
        title="Sample: local vs global gap (uncond, heatmap)",
        out_path=outdir / "sample_local_vs_global_uncond_heatmap.png",
    )
    plot_multi(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="sample_global",
        title="Sample: conditioning gap (global)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "sample_conditioning_gap_global.png",
    )
    plot_heatmap(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="sample_global",
        title="Sample: conditioning gap (global, heatmap)",
        out_path=outdir / "sample_conditioning_gap_global_heatmap.png",
    )
    plot_multi(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="sample_local",
        title="Sample: conditioning gap (local)",
        ylabel="mean |Δ score| per pixel",
        out_path=outdir / "sample_conditioning_gap_local.png",
    )
    plot_heatmap(
        loaded,
        metric_root="conditioning_gap_time",
        metric_key="sample_local",
        title="Sample: conditioning gap (local, heatmap)",
        out_path=outdir / "sample_conditioning_gap_local_heatmap.png",
    )


if __name__ == "__main__":
    main()
