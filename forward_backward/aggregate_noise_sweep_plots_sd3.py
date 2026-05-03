"""
Aggregate local + global SD3 noise sweep results and plot combined curves.
"""
import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ROOT_DEFAULT = os.path.join(MODULE_DIR, "data", "noise_sweep_sd3_local")
GLOBAL_ROOT_DEFAULT = os.path.join(MODULE_DIR, "data", "noise_sweep_sd3_global")
OUT_DIR_DEFAULT = os.path.join(MODULE_DIR, "visualization", "noise_sweep_sd3_aggregate")


def load_metrics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_curve(metrics: Dict, key: str) -> Tuple[List[float], List[float], List[float]]:
    summary = metrics.get("summary", [])
    t_norm = [row["t_norm"] for row in summary]
    cond = [row[f"cond_{key}"] for row in summary]
    uncond = [row[f"uncond_{key}"] for row in summary]
    return t_norm, cond, uncond


def get_available_r(local_root: str) -> List[int]:
    if not os.path.isdir(local_root):
        return []
    r_vals = []
    for name in os.listdir(local_root):
        match = re.fullmatch(r"r(\d+)", name)
        if not match:
            continue
        metrics_path = os.path.join(local_root, name, "metrics.json")
        if os.path.isfile(metrics_path):
            r_vals.append(int(match.group(1)))
    return sorted(r for r in r_vals if r != 0)


def plot_curve(
    out_path: str,
    title: str,
    ylabel: str,
    global_series: Tuple[List[float], List[float]],
    local_series: Dict[int, Tuple[List[float], List[float]]],
):
    plt.figure(figsize=(6, 5))
    
    # Create color gradient from tab:blue to tab:orange
    num_r = len(local_series)
    blue_rgb = mcolors.to_rgb("tab:blue")
    orange_rgb = mcolors.to_rgb("tab:orange")
    colors = []
    for i in range(num_r):
        t = i / max(num_r - 1, 1)
        r = blue_rgb[0] + t * (orange_rgb[0] - blue_rgb[0])
        g = blue_rgb[1] + t * (orange_rgb[1] - blue_rgb[1])
        b = blue_rgb[2] + t * (orange_rgb[2] - blue_rgb[2])
        colors.append((r, g, b))
    
    # Plot local series with gradient colors
    for idx, (r, (t_vals, y_vals)) in enumerate(sorted(local_series.items())):
        plt.plot(t_vals, y_vals, marker="o", markersize=6, color=colors[idx], label=f"r={r}")
    
    # Plot global as tab:orange
    t_global, y_global = global_series
    plt.plot(t_global, y_global, color="tab:orange", linewidth=3.0, marker="o", markersize=6, label="global")
    plt.xlabel("t", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_legend_only(
    out_path: str,
    local_series: Dict[int, Tuple[List[float], List[float]]],
):
    # Mirror the color mapping used in plot_curve so legend colors match plots.
    num_r = len(local_series)
    blue_rgb = mcolors.to_rgb("tab:blue")
    orange_rgb = mcolors.to_rgb("tab:orange")
    colors = []
    for i in range(num_r):
        t = i / max(num_r - 1, 1)
        r = blue_rgb[0] + t * (orange_rgb[0] - blue_rgb[0])
        g = blue_rgb[1] + t * (orange_rgb[1] - blue_rgb[1])
        b = blue_rgb[2] + t * (orange_rgb[2] - blue_rgb[2])
        colors.append((r, g, b))

    fig, ax = plt.subplots(figsize=(6, 4))
    handles = []
    labels = []
    for idx, (r, _) in enumerate(sorted(local_series.items())):
        handle = ax.plot([], [], marker="o", markersize=6, color=colors[idx], linestyle="-")[0]
        handles.append(handle)
        labels.append(f"r={r}")
    handles.append(ax.plot([], [], color="tab:orange", linewidth=3.0, marker="o", markersize=6)[0])
    labels.append("global")

    ax.legend(handles, labels, ncol=2, fontsize=16, frameon=False, loc="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    global_metrics_path = os.path.join(args.global_root, "metrics.json")
    if not os.path.isfile(global_metrics_path):
        raise FileNotFoundError(f"Missing global metrics.json at {global_metrics_path}")

    global_metrics = load_metrics(global_metrics_path)
    t_global, cond_global, uncond_global = extract_curve(global_metrics, "mean")
    _, cond_agree, uncond_agree = extract_curve(global_metrics, "agree_rate")
    cond_err_global = [None if v is None else 1.0 - v for v in cond_agree]
    uncond_err_global = [None if v is None else 1.0 - v for v in uncond_agree]

    r_vals = get_available_r(args.local_root)
    if not r_vals:
        raise FileNotFoundError(f"No local r*/metrics.json found in {args.local_root}")

    local_cond_mse = {}
    local_uncond_mse = {}
    local_cond_err = {}
    local_uncond_err = {}

    for r in r_vals:
        metrics_path = os.path.join(args.local_root, f"r{r}", "metrics.json")
        metrics = load_metrics(metrics_path)
        t_vals, cond_vals, uncond_vals = extract_curve(metrics, "mean")
        _, cond_agree_vals, uncond_agree_vals = extract_curve(metrics, "agree_rate")
        cond_err_vals = [None if v is None else 1.0 - v for v in cond_agree_vals]
        uncond_err_vals = [None if v is None else 1.0 - v for v in uncond_agree_vals]
        local_cond_mse[r] = (t_vals, cond_vals)
        local_uncond_mse[r] = (t_vals, uncond_vals)
        local_cond_err[r] = (t_vals, cond_err_vals)
        local_uncond_err[r] = (t_vals, uncond_err_vals)

    plot_curve(
        os.path.join(args.out_dir, "mse_cond_all.png"),
        "Per-pixel MSE vs noise level (conditional)",
        "Per-pixel MSE",
        (t_global, cond_global),
        local_cond_mse,
    )
    plot_curve(
        os.path.join(args.out_dir, "mse_uncond_all.png"),
        "Per-pixel MSE vs noise level (unconditional)",
        "Per-pixel MSE",
        (t_global, uncond_global),
        local_uncond_mse,
    )
    plot_curve(
        os.path.join(args.out_dir, "classification_error_cond_all.png"),
        "Classifier error vs noise level (conditional)",
        "Classifier error",
        (t_global, cond_err_global),
        local_cond_err,
    )
    plot_curve(
        os.path.join(args.out_dir, "classification_error_uncond_all.png"),
        "Classifier error vs noise level (unconditional)",
        "Classifier error",
        (t_global, uncond_err_global),
        local_uncond_err,
    )
    save_legend_only(os.path.join(args.out_dir, "legend.png"), local_cond_mse)

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate local/global SD3 noise sweep plots.")
    parser.add_argument("--local-root", type=str, default=LOCAL_ROOT_DEFAULT)
    parser.add_argument("--global-root", type=str, default=GLOBAL_ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()
    main(args)
