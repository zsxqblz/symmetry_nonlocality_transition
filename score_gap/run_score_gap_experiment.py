#!/usr/bin/env python3
"""
Score-gap experiment for local attention DiT variants.

Notation
--------
- Global denoiser: the pretrained DiT with standard global self-attention.
- Local denoiser: the same pretrained DiT where each attention head is constrained
                  to a Chebyshev window of radius ``R`` using
                  ``EfficientLocalSelfAttention``.
- Training trajectory: start from a clean dataset image, encode it to latent space,
                       and add diffusion noise using the DiT scheduler.
- Sampling trajectory: start from pure noise and follow the global denoiser's
                       generation path while probing both denoisers at every step.

The script compares local vs global score predictions (conditioned and
unconditioned) on both trajectories, and also measures the effect of conditioning
for each denoiser separately. Results are aggregated over multiple samples and
saved as bar plots alongside raw statistics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_AFFINITY", "disabled")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

from models.local_attention import current_t_norm, swap_in_efficient_local_attn
from models.local_attention_masked_sdpa import swap_in_masked_sdpa_local_attn


MODULE_DIR = Path(__file__).resolve().parent

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local vs global DiT scores on training and sampling trajectories."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/DiT-XL-2-256",
        help="Diffusers model id or local path for the pretrained DiT.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Local attention radius R (window = (2R+1)^2).",
    )
    parser.add_argument(
        "--local-intervals",
        type=str,
        default="[[0.0, 1.0]]",
        help="JSON list of [start, end] pairs for when to enable local attention (default: always local).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of dataset images / sampling trajectories to evaluate.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory with raw RGB images used for the training-trajectory probe.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODULE_DIR / "outputs" / "score_gap_dit"),
        help="Directory to store plots and metric JSON.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=40,
        help="Number of solver steps when following the sampling trajectory.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1000,
        help="Number of timesteps to cover along the training trajectory.",
    )
    parser.add_argument(
        "--max-train-t",
        type=int,
        default=1000,
        help="Maximum raw timestep to include for training-trajectory probes (inclusive).",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=207,
        help="ImageNet class id used for conditioned runs.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale for the sampling trajectory (global denoiser).",
    )
    parser.add_argument(
        "--uncond-sampling",
        action="store_true",
        help="Generate sampling trajectory without conditioning (use uncond score for the step).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed controlling dataset shuffling and latent noise.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for encoding dataset images to latents (use 1 to avoid OOM).",
    )
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Disable float16 even when CUDA is available.",
    )
    parser.add_argument(
        "--local-attn-impl",
        type=str,
        default="gather",
        choices=("gather", "masked_sdpa"),
        help="Local attention implementation for the local DiT transformer.",
    )
    parser.add_argument(
        "--allow-online-weights",
        action="store_false",
        dest="local_files_only",
        help="Permit downloading weights if they are not cached locally (default: offline only).",
    )
    parser.set_defaults(local_files_only=True)
    return parser.parse_args()


def parse_intervals(interval_str: str) -> List[List[float]]:
    try:
        intervals = json.loads(interval_str)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse --local-intervals: {err}") from err
    if not isinstance(intervals, list):
        raise ValueError("--local-intervals must decode to a list of [start, end] pairs.")
    parsed: List[List[float]] = []
    for pair in intervals:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(x, (int, float)) for x in pair)
        ):
            raise ValueError(f"Invalid interval entry: {pair}")
        start, end = float(pair[0]), float(pair[1])
        if not (0.0 <= start <= end <= 1.0):
            raise ValueError(f"Interval endpoints must satisfy 0 <= start <= end <= 1, got {pair}")
        parsed.append([start, end])
    return parsed


def make_t_normalizer(scheduler: DPMSolverMultistepScheduler):
    ts = scheduler.timesteps.to("cpu")
    tmax = float(ts.max().item())
    tmin = float(ts.min().item())
    rng = max(1.0, tmax - tmin)

    def to_unit_interval(t_int: float) -> float:
        return (float(t_int) - tmin) / rng

    return to_unit_interval


def ensure_timestep_tensor(timestep: torch.Tensor | int | float, batch: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(timestep):
        t = torch.tensor([int(timestep)], device=device, dtype=torch.long)
    else:
        t = timestep.to(device=device, dtype=torch.long)
        if t.ndim == 0:
            t = t.unsqueeze(0)
    if t.shape[0] == 1 and batch > 1:
        t = t.expand(batch)
    return t


def per_pixel_mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, a.ndim))
    return (a - b).float().abs().mean(dim=dims)


def chunk_list(items: Sequence, chunk_size: int) -> Iterable[Sequence]:
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def load_pipelines(model_id: str, device: torch.device, dtype: torch.dtype, local_only: bool):
    pipe_global = DiTPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_only,
    ).to(device)
    pipe_global.scheduler = DPMSolverMultistepScheduler.from_config(pipe_global.scheduler.config)

    pipe_local = DiTPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_only,
    ).to(device)
    pipe_local.scheduler = DPMSolverMultistepScheduler.from_config(pipe_local.scheduler.config)
    return pipe_global, pipe_local


def iter_dataset_latents(
    image_paths: Sequence[Path],
    pipe: DiTPipeline,
    batch_size: int,
    device: torch.device,
    image_size: int,
) -> Iterable[torch.Tensor]:
    dtype = pipe.vae.dtype
    scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    chunk = max(1, batch_size)
    for batch_paths in chunk_list(image_paths, chunk):
        pixel_batches: List[torch.Tensor] = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((image_size, image_size), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            img.close()
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor * 2.0 - 1.0
            pixel_batches.append(tensor)
        batch = torch.cat(pixel_batches, dim=0).to(device=device, dtype=dtype)
        with torch.no_grad():
            encoded = pipe.vae.encode(batch).latent_dist.mode() * scale
        for latent in encoded:
            yield latent.cpu()


def predict_cond_uncond(
    transformer: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor | int,
    class_id: int,
    null_class_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = latents.device
    batch = latents.shape[0]
    cond_labels = torch.full((batch,), class_id, device=device, dtype=torch.long)
    uncond_labels = torch.full((batch,), null_class_id, device=device, dtype=torch.long)
    latent_model_input = torch.cat([latents, latents], dim=0)
    labels = torch.cat([uncond_labels, cond_labels], dim=0)
    timestep_input = ensure_timestep_tensor(timestep, latent_model_input.shape[0], device)
    with torch.no_grad():
        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep_input,
            class_labels=labels,
        ).sample
    uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
    if cond_pred.shape[1] == latents.shape[1] * 2:
        cond_pred = cond_pred[:, : latents.shape[1]]
        uncond_pred = uncond_pred[:, : latents.shape[1]]
    return cond_pred, uncond_pred


def gather_image_paths(dataset_dir: Path, limit: int) -> List[Path]:
    image_paths: List[Path] = []
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                image_paths.append(Path(root) / name)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {dataset_dir}")
    random.shuffle(image_paths)
    if len(image_paths) < limit:
        raise ValueError(f"Found {len(image_paths)} images but --num-samples={limit}")
    return image_paths[:limit]


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = float(sum(values) / len(values))
    var = float(sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1))
    std = math.sqrt(max(var, 0.0))
    return {"mean": mean, "std": std, "count": len(values)}


def summarize_time_series(time_dict: Dict[int, List[float]]) -> Dict[int, Dict[str, float]]:
    return {int(t): summarize(vals) for t, vals in time_dict.items()}


def plot_time_series(
    metrics: Dict[str, Dict[int, List[float]]],
    norm_map: Dict[int, float],
    legend_order: List[Tuple[str, str]],
    title: str,
    ylabel: str,
    out_path: Path,
):
    plt.figure(figsize=(8, 4))
    for label, key in legend_order:
        time_buckets = metrics.get(key)
        if not time_buckets:
            continue
        xs = sorted(time_buckets.keys(), key=lambda t: norm_map.get(int(t), 0.0))
        if not xs:
            continue
        x_norm = [norm_map.get(int(t), 0.0) for t in xs]
        y_mean = [summarize(time_buckets[int(t)])["mean"] for t in xs]
        plt.plot(x_norm, y_mean, marker="o", label=label)
    plt.xlabel("normalized time (t_norm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" and not args.force_fp32 else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    intervals = parse_intervals(args.local_intervals)
    pipe_global, pipe_local = load_pipelines(args.model_id, device, dtype, args.local_files_only)

    patch = pipe_global.transformer.config.patch_size
    res = pipe_global.transformer.config.sample_size
    vae_scale = 2 ** (len(pipe_global.vae.config.block_out_channels) - 1)
    image_res = res * vae_scale
    h_tok, w_tok = res // patch, res // patch

    if args.local_attn_impl == "masked_sdpa":
        swap_in_masked_sdpa_local_attn(
            pipe_local.transformer,
            h_tok,
            w_tok,
            r=args.radius,
            local_intervals=intervals,
        )
    else:
        swap_in_efficient_local_attn(pipe_local.transformer, h_tok, w_tok, r=args.radius, local_intervals=intervals)

    base_scheduler_config = pipe_global.scheduler.config
    train_scheduler = DPMSolverMultistepScheduler.from_config(base_scheduler_config)
    train_scheduler.set_timesteps(args.train_steps, device=device)
    sampling_scheduler = DPMSolverMultistepScheduler.from_config(base_scheduler_config)
    sampling_scheduler.set_timesteps(args.sampling_steps, device=device)

    dataset_dir = Path(args.dataset_dir)
    image_paths = gather_image_paths(dataset_dir, args.num_samples)
    latent_iterator = iter_dataset_latents(
        image_paths,
        pipe_global,
        args.batch_size,
        device,
        image_res,
    )

    num_classes = getattr(pipe_global.transformer.config, "num_labels", 1000)
    null_class_id = num_classes

    def bucket_dict() -> DefaultDict[int, List[float]]:
        return defaultdict(list)

    local_vs_global_time: Dict[str, DefaultDict[int, List[float]]] = {
        "train_cond": bucket_dict(),
        "train_uncond": bucket_dict(),
        "sample_cond": bucket_dict(),
        "sample_uncond": bucket_dict(),
    }
    conditioning_gap_time: Dict[str, DefaultDict[int, List[float]]] = {
        "train_global": bucket_dict(),
        "train_local": bucket_dict(),
        "sample_global": bucket_dict(),
        "sample_local": bucket_dict(),
    }

    g_latent = torch.Generator(device=device).manual_seed(args.seed)
    g_cpu = torch.Generator().manual_seed(args.seed)

    train_t_norm = make_t_normalizer(train_scheduler)
    train_norm_map = {
        int(t.item() if isinstance(t, torch.Tensor) else t): train_t_norm(
            float(t.item() if isinstance(t, torch.Tensor) else t)
        )
        for t in train_scheduler.timesteps
    }
    for latent in latent_iterator:
        latent = latent.unsqueeze(0).to(device=device, dtype=dtype)
        for timestep in train_scheduler.timesteps:
            t_int = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
            if t_int > args.max_train_t:
                continue
            t_tensor = ensure_timestep_tensor(timestep, 1, device)
            noise = randn_tensor(latent.shape, generator=g_latent, device=device, dtype=dtype)
            noisy = train_scheduler.add_noise(latent, noise, t_tensor)
            t_value = int(t_tensor[0].item())
            t_norm = train_norm_map[t_value]

            current_t_norm["value"] = None
            global_cond, global_uncond = predict_cond_uncond(
                pipe_global.transformer, noisy, t_tensor, args.class_id, null_class_id
            )
            current_t_norm["value"] = t_norm
            local_cond, local_uncond = predict_cond_uncond(
                pipe_local.transformer, noisy, t_tensor, args.class_id, null_class_id
            )

            diff_train_cond = per_pixel_mae(local_cond, global_cond).mean().item()
            diff_train_uncond = per_pixel_mae(local_uncond, global_uncond).mean().item()
            gap_global = per_pixel_mae(global_cond, global_uncond).mean().item()
            gap_local = per_pixel_mae(local_cond, local_uncond).mean().item()
            local_vs_global_time["train_cond"][t_value].append(diff_train_cond)
            local_vs_global_time["train_uncond"][t_value].append(diff_train_uncond)
            conditioning_gap_time["train_global"][t_value].append(gap_global)
            conditioning_gap_time["train_local"][t_value].append(gap_local)

    sample_t_norm = make_t_normalizer(sampling_scheduler)
    sample_norm_map = {
        int(t.item() if isinstance(t, torch.Tensor) else t): sample_t_norm(
            float(t.item() if isinstance(t, torch.Tensor) else t)
        )
        for t in sampling_scheduler.timesteps
    }
    in_ch = pipe_global.transformer.config.in_channels
    latents_shape = (1, in_ch, res, res)

    for sample_idx in range(args.num_samples):
        sampling_scheduler.set_timesteps(args.sampling_steps, device=device)
        latents = randn_tensor(latents_shape, generator=g_latent, device=device, dtype=dtype)
        if hasattr(sampling_scheduler, "init_noise_sigma"):
            latents = latents * sampling_scheduler.init_noise_sigma
        for timestep in sampling_scheduler.timesteps:
            t_tensor = ensure_timestep_tensor(timestep, 1, device)
            t_norm = sample_t_norm(float(timestep.item() if isinstance(timestep, torch.Tensor) else timestep))
            current_t_norm["value"] = None
            global_cond, global_uncond = predict_cond_uncond(
                pipe_global.transformer, latents, t_tensor, args.class_id, null_class_id
            )
            current_t_norm["value"] = t_norm
            local_cond, local_uncond = predict_cond_uncond(
                pipe_local.transformer, latents, t_tensor, args.class_id, null_class_id
            )

            t_value = int(t_tensor[0].item())
            diff_sample_cond = per_pixel_mae(local_cond, global_cond).mean().item()
            diff_sample_uncond = per_pixel_mae(local_uncond, global_uncond).mean().item()
            gap_sample_global = per_pixel_mae(global_cond, global_uncond).mean().item()
            gap_sample_local = per_pixel_mae(local_cond, local_uncond).mean().item()
            local_vs_global_time["sample_cond"][t_value].append(diff_sample_cond)
            local_vs_global_time["sample_uncond"][t_value].append(diff_sample_uncond)
            conditioning_gap_time["sample_global"][t_value].append(gap_sample_global)
            conditioning_gap_time["sample_local"][t_value].append(gap_sample_local)

            if args.uncond_sampling:
                cfg = global_uncond
            else:
                cfg = global_uncond + args.guidance_scale * (global_cond - global_uncond)
            step_out = sampling_scheduler.step(cfg, t_tensor[0], latents)
            latents = step_out.prev_sample

    plot_time_series(
        metrics=local_vs_global_time,
        norm_map=train_norm_map,
        legend_order=[("Train cond", "train_cond"), ("Train uncond", "train_uncond")],
        title="Local vs Global score gap (training trajectory)",
        ylabel="mean |Δ score| per pixel",
        out_path=output_dir / "train_local_vs_global.png",
    )
    plot_time_series(
        metrics=local_vs_global_time,
        norm_map=sample_norm_map,
        legend_order=[("Sample cond", "sample_cond"), ("Sample uncond", "sample_uncond")],
        title="Local vs Global score gap (sampling trajectory)",
        ylabel="mean |Δ score| per pixel",
        out_path=output_dir / "sample_local_vs_global.png",
    )
    plot_time_series(
        metrics=conditioning_gap_time,
        norm_map=train_norm_map,
        legend_order=[("Train global", "train_global"), ("Train local", "train_local")],
        title="Conditioning gap (training trajectory)",
        ylabel="mean |Δ score| per pixel",
        out_path=output_dir / "train_conditioning_gap.png",
    )
    plot_time_series(
        metrics=conditioning_gap_time,
        norm_map=sample_norm_map,
        legend_order=[("Sample global", "sample_global"), ("Sample local", "sample_local")],
        title="Conditioning gap (sampling trajectory)",
        ylabel="mean |Δ score| per pixel",
        out_path=output_dir / "sample_conditioning_gap.png",
    )

    summary = {
        "config": vars(args),
        "local_vs_global_time": {
            k: summarize_time_series(v) for k, v in local_vs_global_time.items()
        },
        "conditioning_gap_time": {
            k: summarize_time_series(v) for k, v in conditioning_gap_time.items()
        },
    }
    with open(output_dir / "score_gap_summary.json", "w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2)

    print(f"Saved plots and metrics to {output_dir}")


if __name__ == "__main__":
    main()
