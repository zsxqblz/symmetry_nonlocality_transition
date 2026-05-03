"""
Aggregate local + global noise sweep results and plot combined curves.
"""
import argparse
import json
import math
import os
import re
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ROOT_DEFAULT = os.path.join(MODULE_DIR, "data", "noise_sweep_local")
GLOBAL_ROOT_DEFAULT = os.path.join(MODULE_DIR, "data", "noise_sweep_global")
OUT_DIR_DEFAULT = os.path.join(MODULE_DIR, "visualization", "noise_sweep_aggregate")


def load_metrics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_curve(metrics: Dict, key: str) -> Tuple[List[float], List[float], List[float]]:
    summary = metrics.get("summary", [])
    t_norm = [row["t_norm"] for row in summary]
    cond = [row[f"cond_{key}"] for row in summary]
    uncond = [row[f"uncond_{key}"] for row in summary]
    return t_norm, cond, uncond


def noise_key(t_val: float) -> str:
    return f"{float(t_val):.4f}"


def compute_mean_and_error(values: Iterable[float], error_stat: str) -> Tuple[float | None, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None, None
    mean = float(arr.mean())
    std = float(arr.std()) if arr.size > 1 else 0.0
    if error_stat == "std":
        err = std
    elif error_stat == "sem":
        err = std / math.sqrt(arr.size)
    else:
        raise ValueError(f"Unsupported error stat: {error_stat}")
    return mean, float(err)


def extract_series(
    metrics: Dict,
    metric_kind: str,
    branch: str,
    error_stat: str,
) -> Tuple[List[float], List[float], List[float]]:
    t_vals = [float(t) for t in metrics.get("noise_levels", [])]
    means: List[float] = []
    errs: List[float] = []
    per_level = metrics.get("per_level", {})
    per_level_agree = metrics.get("per_level_agree", {})

    for t_val in t_vals:
        key = noise_key(t_val)
        if metric_kind == "mse":
            raw_vals = per_level.get(key, {}).get(branch, [])
        elif metric_kind == "class_error":
            raw_agree = per_level_agree.get(key, {}).get(branch, [])
            raw_vals = [1.0 - float(v) for v in raw_agree]
        else:
            raise ValueError(f"Unsupported metric kind: {metric_kind}")

        mean, err = compute_mean_and_error(raw_vals, error_stat)
        means.append(np.nan if mean is None else mean)
        errs.append(np.nan if err is None else err)

    return t_vals, means, errs


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
    return sorted(r_vals)


def plot_curve(
    out_path: str,
    title: str,
    ylabel: str,
    global_series: Tuple[List[float], List[float], List[float]],
    local_series: Dict[int, Tuple[List[float], List[float], List[float]]],
    use_errorbars: bool = False,
):
    plt.figure(figsize=(6, 5))
    local_linewidth = 1.8
    global_linewidth = 3.0
    local_markersize = 6
    global_markersize = 6
    
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
    for idx, (r, (t_vals, y_vals, y_errs)) in enumerate(sorted(local_series.items())):
        t_arr = np.asarray(t_vals, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64)
        e_arr = np.asarray(y_errs, dtype=np.float64)
        mask = np.isfinite(t_arr) & np.isfinite(y_arr)
        if not np.any(mask):
            continue
        if use_errorbars:
            plt.errorbar(
                t_arr[mask],
                y_arr[mask],
                yerr=e_arr[mask],
                marker="o",
                markersize=local_markersize,
                linewidth=local_linewidth,
                elinewidth=0.8,
                capsize=2,
                color=colors[idx],
                label=f"local r={r}",
            )
        else:
            plt.plot(
                t_arr[mask],
                y_arr[mask],
                marker="o",
                markersize=local_markersize,
                linewidth=local_linewidth,
                color=colors[idx],
                label=f"local r={r}",
            )
    
    # Plot global as tab:orange
    t_global, y_global, yerr_global = global_series
    t_arr = np.asarray(t_global, dtype=np.float64)
    y_arr = np.asarray(y_global, dtype=np.float64)
    e_arr = np.asarray(yerr_global, dtype=np.float64)
    mask = np.isfinite(t_arr) & np.isfinite(y_arr)
    if use_errorbars:
        plt.errorbar(
            t_arr[mask],
            y_arr[mask],
            yerr=e_arr[mask],
            color="tab:orange",
            linewidth=global_linewidth,
            marker="o",
            markersize=global_markersize,
            elinewidth=1.0,
            capsize=2,
            label="global",
        )
    else:
        plt.plot(
            t_arr[mask],
            y_arr[mask],
            color="tab:orange",
            linewidth=global_linewidth,
            marker="o",
            markersize=global_markersize,
            label="global",
        )
    
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


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    global_metrics_path = os.path.join(args.global_root, "metrics.json")
    if not os.path.isfile(global_metrics_path):
        raise FileNotFoundError(f"Missing global metrics.json at {global_metrics_path}")

    global_metrics = load_metrics(global_metrics_path)
    global_cond_mse = extract_series(global_metrics, "mse", "cond", args.mse_error_stat)
    global_uncond_mse = extract_series(global_metrics, "mse", "uncond", args.mse_error_stat)
    global_cond_err = extract_series(global_metrics, "class_error", "cond", args.class_error_stat)
    global_uncond_err = extract_series(global_metrics, "class_error", "uncond", args.class_error_stat)

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
        local_cond_mse[r] = extract_series(metrics, "mse", "cond", args.mse_error_stat)
        local_uncond_mse[r] = extract_series(metrics, "mse", "uncond", args.mse_error_stat)
        local_cond_err[r] = extract_series(metrics, "class_error", "cond", args.class_error_stat)
        local_uncond_err[r] = extract_series(metrics, "class_error", "uncond", args.class_error_stat)

    plot_curve(
        os.path.join(args.out_dir, "mse_cond_all.png"),
        "Per-pixel MSE vs noise level (conditional)",
        "Per-pixel MSE",
        global_cond_mse,
        local_cond_mse,
    )
    plot_curve(
        os.path.join(args.out_dir, "mse_uncond_all.png"),
        "Per-pixel MSE vs noise level (unconditional)",
        "Per-pixel MSE",
        global_uncond_mse,
        local_uncond_mse,
    )
    plot_curve(
        os.path.join(args.out_dir, "classification_error_cond_all.png"),
        "Classifier error vs noise level (conditional)",
        "Classifier error",
        global_cond_err,
        local_cond_err,
    )
    plot_curve(
        os.path.join(args.out_dir, "classification_error_uncond_all.png"),
        "Classifier error vs noise level (unconditional)",
        "Classifier error",
        global_uncond_err,
        local_uncond_err,
    )

    mse_suffix = f"errorbar_{args.mse_error_stat}"
    plot_curve(
        os.path.join(args.out_dir, f"mse_cond_all_{mse_suffix}.png"),
        f"Per-pixel MSE vs noise level (conditional, {args.mse_error_stat})",
        "Per-pixel MSE",
        global_cond_mse,
        local_cond_mse,
        use_errorbars=True,
    )
    plot_curve(
        os.path.join(args.out_dir, f"mse_uncond_all_{mse_suffix}.png"),
        f"Per-pixel MSE vs noise level (unconditional, {args.mse_error_stat})",
        "Per-pixel MSE",
        global_uncond_mse,
        local_uncond_mse,
        use_errorbars=True,
    )

    class_suffix = f"errorbar_{args.class_error_stat}"
    plot_curve(
        os.path.join(args.out_dir, f"classification_error_cond_all_{class_suffix}.png"),
        f"Classifier error vs noise level (conditional, {args.class_error_stat})",
        "Classifier error",
        global_cond_err,
        local_cond_err,
        use_errorbars=True,
    )
    plot_curve(
        os.path.join(args.out_dir, f"classification_error_uncond_all_{class_suffix}.png"),
        f"Classifier error vs noise level (unconditional, {args.class_error_stat})",
        "Classifier error",
        global_uncond_err,
        local_uncond_err,
        use_errorbars=True,
    )

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate local/global noise sweep plots.")
    parser.add_argument("--local-root", type=str, default=LOCAL_ROOT_DEFAULT)
    parser.add_argument("--global-root", type=str, default=GLOBAL_ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT)
    parser.add_argument(
        "--mse-error-stat",
        type=str,
        choices=("std", "sem"),
        default="std",
        help="Statistic used for MSE error bars in the *_errorbar_*.png outputs.",
    )
    parser.add_argument(
        "--class-error-stat",
        type=str,
        choices=("std", "sem"),
        default="sem",
        help="Statistic used for classifier-error error bars in the *_errorbar_*.png outputs.",
    )
    args = parser.parse_args()
    main(args)
