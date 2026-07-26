"""Schedule-weighted (instantaneous drift) replot of the Figure 3 score gaps.

Motivation: the raw gaps stored by the runners are in each model's native
output parameterization (epsilon-prediction gap for DiT, velocity gap for
SD3), so their magnitudes are not comparable across time or across models.
Rewriting any interpolant's probability-flow ODE in log-SNR time
lambda = log(alpha_t^2 / sigma_t^2) gives

    dx/dlambda = (dlog(alpha)/dlambda) x - (sigma_t^2 / 2) s,

so the schedule-invariant instantaneous effect of a score gap is
(sigma_t^2 / 2) * ||Delta s|| for every schedule.  Converted to the stored
quantities:

    DiT (VP, linear beta):  Delta eps = sigma_t * Delta s
        -> weight on stored gap:  sigma_t / 2 = sqrt(1 - alphabar_t) / 2
    SD3 (rectified flow):   Delta v = (t / (1 - t)) * Delta s,  t = timestep/1000
        -> weight on stored gap:  t (1 - t) / 2

The x-axis is time t (normalized to [0, 1]), as in the original Figure 3;
the SD3 panel's axis is reversed so that sampling proceeds left to right.

Inputs (bundled data):
- DiT sampling trajectory: score_gap/data/score_gap_facebook_cond/r*/score_gap_summary.json
- DiT training trajectory: score_gap/data/score_gap_facebook_train/r*/score_gap_summary.json
- SD3 sampling trajectory: score_gap/data/score_gap_sd3_cond/r*/sd3_sampling_score_gap_metrics.json

Notes:
- DiT gaps are stored as per-pixel MAE (amplitude-like) and are weighted
  directly; SD3 gaps are stored as MSE, so the per-row amplitude sqrt(MSE)
  is taken before weighting.
- alphabar_t is computed analytically from the facebook/DiT-XL-2-256
  scheduler config (linear betas 1e-4 -> 2e-2 over 1000 steps); override via
  CLI if using a different checkpoint.

Usage:
    conda run -n arena-env python -m score_gap.plot_score_gap_lambda_weighted
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent

HEATMAP_CMAP = "viridis"
CURVE_LABEL_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dit-sampling-root",
        type=Path,
        default=MODULE_DIR / "data" / "score_gap_facebook_cond",
        help="Root with r*/score_gap_summary.json for the DiT sampling trajectory.",
    )
    parser.add_argument(
        "--dit-train-root",
        type=Path,
        default=MODULE_DIR / "data" / "score_gap_facebook_train",
        help="Root with r*/score_gap_summary.json for the DiT training trajectory.",
    )
    parser.add_argument(
        "--sd3-root",
        type=Path,
        default=MODULE_DIR / "data" / "score_gap_sd3_cond",
        help="Root with r*/sd3_sampling_score_gap_metrics.json for SD3 sampling.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=MODULE_DIR / "visualization" / "score_gap_lambda_weighted",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Schedule quantities
# ---------------------------------------------------------------------------


def vp_alphas_cumprod(beta_start: float, beta_end: float, num_steps: int) -> np.ndarray:
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    return np.cumprod(1.0 - betas)


def vp_time_and_weight(timesteps: np.ndarray, alphas_cumprod: np.ndarray,
                       num_train_timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unit time and weight (on the stored epsilon gap) for VP timesteps."""
    abar = alphas_cumprod[timesteps.astype(int)]
    t_norm = timesteps / float(num_train_timesteps)
    weight = np.sqrt(1.0 - abar) / 2.0
    return t_norm, weight


def rf_time_and_weight(raw_timesteps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit time and weight (on the stored velocity gap) for rectified flow."""
    sigmas = raw_timesteps / 1000.0
    weight = sigmas * (1.0 - sigmas) / 2.0
    return sigmas, weight


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_radius_dirs(root: Path, metrics_name: str) -> Dict[int, Path]:
    radius_dirs: Dict[int, Path] = {}
    for child in sorted(root.iterdir()):
        match = re.fullmatch(r"r(\d+)", child.name)
        if match and (child / metrics_name).exists():
            radius_dirs[int(match.group(1))] = child / metrics_name
    if not radius_dirs:
        raise FileNotFoundError(f"No r*/{metrics_name} found under {root}")
    return radius_dirs


def load_dit_series(path: Path, section: str, series: str) -> Dict[int, Tuple[float, float]]:
    """Return {timestep: (mean, std)} for one series of a DiT summary."""
    data = json.loads(path.read_text())
    out = {}
    for t_key, stats in data[section][series].items():
        out[int(t_key)] = (float(stats["mean"]), float(stats["std"]))
    return out


def load_sd3_rows(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    rows = []
    for sample_rows in data["results"].values():
        rows.extend(sample_rows)
    return rows


def sd3_amplitude_by_t(rows: List[dict], key: str) -> Dict[float, Tuple[float, float]]:
    """Group per-sample sqrt(MSE) amplitudes by raw timestep -> (mean, std)."""
    buckets: Dict[float, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        buckets[float(row["t"])].append(math.sqrt(float(value)))
    out = {}
    for t, vals in buckets.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[t] = (float(arr.mean()), float(arr.std()))
    return out


# ---------------------------------------------------------------------------
# Panel assembly: each panel is a weighted curve + weighted heatmap over time
# ---------------------------------------------------------------------------


class Panel:
    def __init__(self, title: str, t: np.ndarray, curve_mean: np.ndarray, curve_std: np.ndarray,
                 radii: List[int], heat: np.ndarray, invert_x: bool = False):
        order = np.argsort(t)
        self.title = title
        self.t = t[order]
        self.curve_mean = curve_mean[order]
        self.curve_std = curve_std[order]
        self.radii = radii
        self.heat = heat[:, order]
        self.invert_x = invert_x


def build_dit_panel(root: Path, curve_series: str, heat_series: str, title: str,
                    alphas_cumprod: np.ndarray, num_train_timesteps: int) -> Panel:
    radius_dirs = discover_radius_dirs(root, "score_gap_summary.json")
    radii = sorted(radius_dirs)

    # The conditioning gap only involves the global model (identical across
    # radius dirs, same seed), so take it from the smallest radius dir.
    curve_stats = load_dit_series(radius_dirs[radii[0]], "conditioning_gap_time", curve_series)
    timesteps = np.asarray(sorted(curve_stats), dtype=np.float64)
    t_norm, weight = vp_time_and_weight(timesteps, alphas_cumprod, num_train_timesteps)
    curve_mean = weight * np.asarray([curve_stats[int(t)][0] for t in timesteps])
    curve_std = weight * np.asarray([curve_stats[int(t)][1] for t in timesteps])

    heat = np.full((len(radii), len(timesteps)), np.nan)
    for i, radius in enumerate(radii):
        stats = load_dit_series(radius_dirs[radius], "local_vs_global_time", heat_series)
        for j, t in enumerate(timesteps):
            if int(t) in stats:
                heat[i, j] = weight[j] * stats[int(t)][0]

    return Panel(title, t_norm, curve_mean, curve_std, radii, heat)


def build_sd3_panel(root: Path, curve_key: str, heat_key: str, title: str) -> Panel:
    radius_dirs = discover_radius_dirs(root, "sd3_sampling_score_gap_metrics.json")
    radii = sorted(radius_dirs)

    rows_by_radius = {r: load_sd3_rows(p) for r, p in radius_dirs.items()}
    curve_stats = sd3_amplitude_by_t(rows_by_radius[radii[0]], curve_key)

    raw_ts = np.asarray(sorted(curve_stats), dtype=np.float64)
    t_norm, weight = rf_time_and_weight(raw_ts)

    curve_mean = weight * np.asarray([curve_stats[t][0] for t in raw_ts])
    curve_std = weight * np.asarray([curve_stats[t][1] for t in raw_ts])

    heat = np.full((len(radii), len(raw_ts)), np.nan)
    for i, radius in enumerate(radii):
        stats = sd3_amplitude_by_t(rows_by_radius[radius], heat_key)
        for j, t in enumerate(raw_ts):
            if t in stats:
                heat[i, j] = weight[j] * stats[t][0]

    # Ascending t axis (0 -> 1); sampling proceeds right to left, as in the
    # DiT panels.
    return Panel(title, t_norm, curve_mean, curve_std, radii, heat, invert_x=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def time_edges(t: np.ndarray) -> np.ndarray:
    mids = 0.5 * (t[1:] + t[:-1])
    first = t[0] - (mids[0] - t[0])
    last = t[-1] + (t[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def draw_curve(ax: plt.Axes, panel: Panel) -> None:
    ax.plot(panel.t, panel.curve_mean, color="black", linewidth=2.0)
    ax.fill_between(panel.t, panel.curve_mean - panel.curve_std,
                    panel.curve_mean + panel.curve_std, color="black", alpha=0.18)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def draw_heatmap(ax: plt.Axes, panel: Panel) -> plt.cm.ScalarMappable:
    x_edges = time_edges(panel.t)
    y_edges = np.arange(len(panel.radii) + 1) - 0.5
    mesh = ax.pcolormesh(x_edges, y_edges, panel.heat, cmap=HEATMAP_CMAP, shading="flat")
    ax.set_yticks(np.arange(len(panel.radii)))
    ax.set_yticklabels([str(r) for r in panel.radii])
    ax.tick_params(labelsize=TICK_FONTSIZE)
    return mesh


def save_panel_figures(panel: Panel, slug: str, outdir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    draw_curve(ax, panel)
    if panel.invert_x:
        ax.invert_xaxis()
    ax.set_xlabel("t", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\frac{\sigma_t^2}{2}\,\|\Delta \vec{s}_{cond}\|$", fontsize=CURVE_LABEL_FONTSIZE)
    ax.set_title(panel.title, fontsize=AXIS_LABEL_FONTSIZE)
    fig.savefig(outdir / f"{slug}_conditioning_gap_weighted.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.3), constrained_layout=True)
    mesh = draw_heatmap(ax, panel)
    if panel.invert_x:
        ax.invert_xaxis()
    ax.set_xlabel("t", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("r", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(panel.title, fontsize=AXIS_LABEL_FONTSIZE)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$\frac{\sigma_t^2}{2}\,\|\Delta \vec{s}_{loc}\|$", fontsize=CURVE_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.savefig(outdir / f"{slug}_locality_gap_heatmap_weighted.png", dpi=dpi)
    plt.close(fig)


def save_combined_figure(panels: List[Panel], outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(
        2, len(panels), figsize=(6.0 * len(panels), 7.2),
        sharex="col", constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.3]},
    )
    for col, panel in enumerate(panels):
        ax_curve, ax_heat = axes[0][col], axes[1][col]
        draw_curve(ax_curve, panel)
        ax_curve.set_title(f"({chr(ord('a') + col)}) {panel.title}", fontsize=15)
        mesh = draw_heatmap(ax_heat, panel)
        if panel.invert_x:
            ax_heat.invert_xaxis()  # sharex="col" propagates to the curve axis
        ax_heat.set_xlabel("t", fontsize=15)
        cbar = fig.colorbar(mesh, ax=ax_heat, location="bottom", shrink=0.9, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        if col == 0:
            ax_curve.set_ylabel(r"$\frac{\sigma_t^2}{2}\,\|\Delta \vec{s}_{cond}\|$", fontsize=16)
            ax_heat.set_ylabel("r", fontsize=16)
    fig.savefig(outdir / "fig3_lambda_weighted.png", dpi=dpi)
    plt.close(fig)


def write_run_info(args: argparse.Namespace, panels: List[Panel], outdir: Path) -> None:
    lines = [
        "Schedule-weighted (instantaneous drift per unit log-SNR) replot of Figure 3.",
        "",
        "Weighted quantity: (sigma_t^2 / 2) * ||Delta s||, the perturbation of the",
        "probability-flow ODE drift per unit lambda = log SNR.",
        "",
        "Applied to stored gaps:",
        "  DiT   (eps-pred MAE): multiply by sqrt(1 - alphabar_t) / 2,",
        f"        alphabar from linear betas {args.beta_start} -> {args.beta_end}"
        f" over {args.num_train_timesteps} steps (facebook/DiT-XL-2-256 config).",
        "  SD3   (velocity MSE): amplitude sqrt(MSE) taken per row, then multiplied",
        "        by t (1 - t) / 2 with t = raw_timestep / 1000 (shift already in grid).",
        "",
        "x-axis: time t normalized to [0, 1] (DiT: timestep/1000; SD3: flow time),",
        "ascending 0 -> 1 in every panel (sampling proceeds right -> left).",
        "",
        "Caveats: DiT gaps are per-pixel MAE while SD3 gaps are RMS, so cross-model",
        "comparisons are up to a norm convention; the SD3 conditional prediction",
        "includes CFG at the guidance scale recorded in the run config.",
        "",
        f"dit_sampling_root={args.dit_sampling_root}",
        f"dit_train_root={args.dit_train_root}",
        f"sd3_root={args.sd3_root}",
    ]
    for panel in panels:
        lines.append(
            f"panel '{panel.title}': radii={panel.radii}, n_t={panel.t.size}, "
            f"t_range=[{panel.t.min():.3f}, {panel.t.max():.3f}], invert_x={panel.invert_x}"
        )
    (outdir / "run_info.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    alphas_cumprod = vp_alphas_cumprod(args.beta_start, args.beta_end, args.num_train_timesteps)

    panels = [
        build_dit_panel(args.dit_sampling_root, "sample_global", "sample_cond",
                        "DiT-XL sampling trajectory", alphas_cumprod, args.num_train_timesteps),
        build_dit_panel(args.dit_train_root, "train_global", "train_cond",
                        "DiT-XL training trajectory", alphas_cumprod, args.num_train_timesteps),
        build_sd3_panel(args.sd3_root, "conditioning_gap_global", "mse_local_vs_global_cond",
                        "SD3 medium sampling trajectory"),
    ]

    slugs = ["dit_sampling", "dit_train", "sd3_sampling"]
    for panel, slug in zip(panels, slugs):
        save_panel_figures(panel, slug, args.outdir, args.dpi)
    save_combined_figure(panels, args.outdir, args.dpi)
    write_run_info(args, panels, args.outdir)
    print(f"Wrote weighted Figure 3 plots to {args.outdir}")


if __name__ == "__main__":
    main()
