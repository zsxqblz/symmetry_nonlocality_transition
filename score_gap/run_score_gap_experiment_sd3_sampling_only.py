#!/usr/bin/env python3
"""
Sampling-only SD3 score-gap probe following a true evolving denoising trajectory.

- Uses multiple independently sampled trajectories.
- Supports either a CFG-guided global trajectory or a purely unconditional global trajectory.
- Compares local vs global scores for both conditional and unconditional branches.
- Keeps only the SD3 transformer on GPU; text encoders and VAE stay on CPU.
- Saves raw per-sample/per-timestep metrics and per-timestep std summaries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor

from models.local_attention import current_t_norm, swap_in_efficient_local_attn
from models.local_attention_masked_sdpa import swap_in_masked_sdpa_local_attn


DEFAULT_PROMPT = "a golden retriever playing in a park, high detail, soft lighting"
MODULE_DIR = Path(__file__).resolve().parent


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sampling-only SD3 score-gap probe.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Diffusers SD3 model id or local path.",
    )
    parser.add_argument("--radius", type=int, default=7, help="Local attention radius R.")
    parser.add_argument(
        "--local-intervals",
        type=str,
        default="[[0.0, 1.0]]",
        help="JSON list of [start, end] pairs for when to enable local attention.",
    )
    parser.add_argument("--sampling-steps", type=int, default=100, help="Number of scheduler steps to span.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of independent sampling trajectories.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt used for sampling trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODULE_DIR / "outputs" / "score_gap_sd3_sampling_only" / "r7"),
        help="Output directory.",
    )
    parser.add_argument(
        "--local-attn-impl",
        type=str,
        default="masked_sdpa",
        choices=("masked_sdpa", "gather"),
        help="Local attention implementation to swap into the SD3 transformer.",
    )
    parser.add_argument(
        "--sampling-trajectory",
        type=str,
        default="guided",
        choices=("guided", "unconditional"),
        help="Which global trajectory to evolve: CFG-guided or unconditional.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_false",
        dest="local_files_only",
        help="Allow downloading weights if missing (default: offline only).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )
    parser.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        help="Skip running if output json already exists.",
    )
    parser.set_defaults(local_files_only=True, do_continue=False)
    return parser.parse_args()


def make_t_normalizer(scheduler) -> callable:
    ts = scheduler.timesteps.to("cpu")
    tmax = float(ts.max().item())
    tmin = float(ts.min().item())
    rng = max(1.0, tmax - tmin)

    def to_unit_interval(t_int: float) -> float:
        return (float(t_int) - tmin) / rng

    return to_unit_interval


def maybe_scale_model_input(scheduler, latents, timestep):
    if hasattr(scheduler, "scale_model_input"):
        return scheduler.scale_model_input(latents, timestep)
    return latents


def select_timesteps_all(timesteps: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
    selected = []
    for idx, t in enumerate(timesteps):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=timesteps.device, dtype=timesteps.dtype)
        elif t.ndim == 0:
            t = t.unsqueeze(0)
        selected.append((idx, t))
    return selected


def compute_noise_pred(
    pipe: StableDiffusion3Pipeline,
    scheduler,
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    get_uncond: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
    latent_model_input = maybe_scale_model_input(scheduler, latent_model_input, t)

    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
    ).sample

    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
        noise_pred_cond = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred_cond, noise_pred_uncond if get_uncond else None
    return noise_pred, None


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a.float() - b.float()) ** 2).item()


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": None, "std": None, "max": None, "count": 0}
    mean = float(sum(vals) / len(vals))
    if len(vals) > 1:
        var = float(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    return {"mean": mean, "std": std, "max": float(max(vals)), "count": len(vals)}


def summarize_rows_by_t(rows: List[Dict[str, float]], metric_key: str) -> Dict[int, Dict[str, float]]:
    buckets: DefaultDict[int, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric_key)
        if value is None:
            continue
        buckets[int(row["t_index"])].append(float(value))
    return {t_idx: summarize_values(vals) for t_idx, vals in sorted(buckets.items())}


def move_non_transformer_modules_to_cpu(pipe: StableDiffusion3Pipeline) -> None:
    for attr_name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipe, attr_name, None)
        if module is not None:
            module.to(device="cpu", dtype=torch.float32)


def main() -> None:
    args = parse_args()
    local_intervals = parse_intervals(args.local_intervals)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sd3_sampling_score_gap_metrics.json"
    if args.do_continue and out_path.exists():
        print(f"[skip] {out_path} exists; --continue set.")
        return

    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    )
    move_non_transformer_modules_to_cpu(pipe)
    pipe.transformer.to(device=device, dtype=dtype)
    pipe.transformer.eval()

    patch = pipe.transformer.config.patch_size
    res = pipe.transformer.config.sample_size
    Htok, Wtok = res // patch, res // patch

    do_cfg = args.guidance_scale is not None and args.guidance_scale > 1.0
    if args.sampling_trajectory == "unconditional" and not do_cfg:
        raise ValueError("--sampling-trajectory unconditional requires --guidance-scale > 1.0.")
    prompt_device = torch.device("cpu")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        prompt_2=args.prompt,
        prompt_3=args.prompt,
        device=prompt_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.sampling_steps, device=device)
    t_to_unit = make_t_normalizer(scheduler)
    selected_ts = select_timesteps_all(scheduler.timesteps)

    if args.local_attn_impl == "masked_sdpa":
        swap_in_masked_sdpa_local_attn(pipe.transformer, Htok, Wtok, r=args.radius, local_intervals=local_intervals)
    else:
        swap_in_efficient_local_attn(pipe.transformer, Htok, Wtok, r=args.radius, local_intervals=local_intervals)

    in_ch = pipe.transformer.config.in_channels
    latents_shape = (1, in_ch, res, res)
    results: List[Dict[str, float]] = []

    for sample_idx in range(args.num_samples):
        scheduler.set_timesteps(args.sampling_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed + sample_idx)
        latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        for idx, t in selected_ts:
            t_norm = t_to_unit(float(t[0].item()))

            current_t_norm["value"] = None
            global_cond, global_uncond = compute_noise_pred(
                pipe, scheduler, latents, t, prompt_embeds, pooled_prompt_embeds, args.guidance_scale, get_uncond=True
            )

            current_t_norm["value"] = t_norm
            local_cond, local_uncond = compute_noise_pred(
                pipe, scheduler, latents, t, prompt_embeds, pooled_prompt_embeds, args.guidance_scale, get_uncond=True
            )

            gap_cond = mse(local_cond, global_cond)
            gap_uncond = (
                mse(local_uncond, global_uncond)
                if (local_uncond is not None and global_uncond is not None)
                else None
            )
            conditioning_gap_global = (
                mse(global_cond, global_uncond)
                if global_uncond is not None
                else None
            )
            conditioning_gap_local = (
                mse(local_cond, local_uncond)
                if local_uncond is not None
                else None
            )

            results.append(
                {
                    "sample_index": sample_idx,
                    "t_index": idx,
                    "t": float(t[0].item()),
                    "t_norm": float(t_norm),
                    "mse_local_vs_global_cond": gap_cond,
                    "mse_local_vs_global_uncond": gap_uncond,
                    "conditioning_gap_global": conditioning_gap_global,
                    "conditioning_gap_local": conditioning_gap_local,
                }
            )

            if args.sampling_trajectory == "unconditional":
                if global_uncond is None:
                    raise RuntimeError("Unconditional trajectory requested, but unconditional branch is unavailable.")
                step_pred = global_uncond
            elif global_uncond is None:
                step_pred = global_cond
            else:
                step_pred = global_uncond + args.guidance_scale * (global_cond - global_uncond)
            latents = scheduler.step(step_pred, t[0], latents).prev_sample

    per_timestep = {
        "mse_local_vs_global_cond": summarize_rows_by_t(results, "mse_local_vs_global_cond"),
        "mse_local_vs_global_uncond": summarize_rows_by_t(results, "mse_local_vs_global_uncond"),
        "conditioning_gap_global": summarize_rows_by_t(results, "conditioning_gap_global"),
        "conditioning_gap_local": summarize_rows_by_t(results, "conditioning_gap_local"),
    }
    summary = {
        "mse_local_vs_global_cond": summarize_values(
            row["mse_local_vs_global_cond"] for row in results if row.get("mse_local_vs_global_cond") is not None
        ),
        "mse_local_vs_global_uncond": summarize_values(
            row["mse_local_vs_global_uncond"] for row in results if row.get("mse_local_vs_global_uncond") is not None
        ),
        "conditioning_gap_global": summarize_values(
            row["conditioning_gap_global"] for row in results if row.get("conditioning_gap_global") is not None
        ),
        "conditioning_gap_local": summarize_values(
            row["conditioning_gap_local"] for row in results if row.get("conditioning_gap_local") is not None
        ),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model_id": args.model_id,
                    "radius": args.radius,
                    "local_intervals": local_intervals,
                    "sampling_steps": args.sampling_steps,
                    "num_samples": args.num_samples,
                    "seed": args.seed,
                    "guidance_scale": args.guidance_scale,
                    "prompt": args.prompt,
                    "output_dir": str(output_dir),
                    "local_files_only": args.local_files_only,
                    "local_attn_impl": args.local_attn_impl,
                    "sampling_trajectory": args.sampling_trajectory,
                },
                "token_grid": [Htok, Wtok],
                "trajectory_type": f"sampling_only_global_evolution_{args.sampling_trajectory}",
                "results": {"sample": results},
                "per_timestep": per_timestep,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"[OK] wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
