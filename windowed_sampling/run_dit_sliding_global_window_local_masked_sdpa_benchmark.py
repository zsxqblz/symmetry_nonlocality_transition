#!/usr/bin/env python3
"""
Masked-SDPA DiT sliding global-window/local benchmark.

- Uses the newer masked-SDPA local attention kernel.
- Keeps the DiT transformer on GPU and the VAE/classifier on CPU.
- Sweeps explicit global-window start times t_i rather than the legacy i/10 grid.
- Preserves the saved feature tensors required by postprocess_benchmark_error_bars.py.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from windowed_sampling import run_schedule_fid_benchmark as base
from models.local_attention_masked_sdpa import swap_in_masked_sdpa_local_attn


MODEL_ID = "facebook/DiT-XL-2-256"
CLASS_ID = 207
CLS_MODEL_ID = "microsoft/resnet-50"
GUIDANCE = 4.0
STEPS = 40
NUM_SAMPLES = 500
BASE_SEED = 123
WINDOW_LENGTH = 0.4
TI_START = 0.2
TI_END = 0.8
TI_STEP = 0.05

CPU_DEVICE = torch.device("cpu")
ACTIVE_TI_BY_INDEX: Dict[int, float] = {}


def build_ti_index_map(start: float, end: float, step: float, offset: int) -> Dict[int, float]:
    if step <= 0:
        raise ValueError("--ti-step must be positive")
    if end < start:
        raise ValueError("--ti-end must be >= --ti-start")
    if offset < 0:
        raise ValueError("--schedule-index-offset must be non-negative")

    num_steps = int(math.floor((end - start) / step + 1e-9)) + 1
    values = [round(start + idx * step, 4) for idx in range(num_steps)]
    if abs(values[-1] - end) > 1e-6 and values[-1] < end:
        values.append(round(end, 4))
    mapping = {}
    for idx, value in enumerate(values):
        mapping[offset + idx] = value
    for value in mapping.values():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"All t_i values must lie in [0, 1], got {value}")
    return mapping


def parse_schedule_indices(raw: str, _experiment: str) -> List[int]:
    valid_indices = sorted(ACTIVE_TI_BY_INDEX)
    if raw.strip().lower() in {"all", "*"}:
        return valid_indices

    indices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx not in valid_indices:
            raise ValueError(f"schedule index must be in [0, {valid_indices[-1]}], got {idx}")
        indices.append(idx)
    if not indices:
        raise ValueError("at least one schedule index is required")
    return sorted(dict.fromkeys(indices))


def sliding_window_interval(schedule_index: int, window_length: float) -> List[List[float]]:
    if schedule_index not in ACTIVE_TI_BY_INDEX:
        valid_indices = sorted(ACTIVE_TI_BY_INDEX)
        raise ValueError(f"sliding-window schedule index must be one of {valid_indices}, got {schedule_index}")
    start = ACTIVE_TI_BY_INDEX[schedule_index]
    end = min(1.0, start + float(window_length))
    return [[round(start, 4), round(end, 4)]]


def group_config(
    experiment: str,
    schedule_idx: int,
    radius: int,
    window_length: float,
    outside_conditioning: str,
):
    if experiment != "sliding_global_window_local":
        raise ValueError(f"Unsupported experiment for this script: {experiment}")
    window = sliding_window_interval(schedule_idx, window_length)
    conditioning_mode = "always" if outside_conditioning == "conditional" else "inside"
    outside_desc = "local conditioned" if outside_conditioning == "conditional" else "local unconditional"
    return {
        "local_intervals": base.complement_intervals(window),
        "conditioning_intervals": window,
        "conditioning_mode": conditioning_mode,
        "summary_interval": window[0],
        "description": f"global conditioned in {window[0]}, {outside_desc} otherwise",
    }


def decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    latents_cpu = latents.detach().to(device=CPU_DEVICE, dtype=torch.float32)
    with torch.no_grad():
        image = pipe.vae.decode(latents_cpu / scale).sample
    image = (image.clamp(-1, 1) + 1) / 2.0
    image_np = image[0].permute(1, 2, 0).float().cpu().detach().numpy()
    return Image.fromarray((image_np * 255).round().astype("uint8"))


def build_pipeline(model_id: str, local_files_only: bool, radius: int):
    pipe = DiTPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.transformer.eval()
    pipe.vae.eval()

    patch = pipe.transformer.config.patch_size
    res = pipe.transformer.config.sample_size
    h_tok, w_tok = res // patch, res // patch
    num_replaced = swap_in_masked_sdpa_local_attn(
        pipe.transformer,
        h_tok,
        w_tok,
        r=radius,
        local_intervals=[],
    )
    pipe.transformer.to(device=base.device, dtype=base.dtype)
    pipe.vae.to(device=CPU_DEVICE, dtype=torch.float32)

    print(
        f"[init] model={model_id} token_grid={h_tok}x{w_tok} dtype={base.dtype} "
        f"replaced_attn={num_replaced} transformer_device={base.device} vae_device={CPU_DEVICE}"
    )
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Masked-SDPA DiT sliding global-window/local FID/classifier benchmark."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["sliding_global_window_local"],
        default="sliding_global_window_local",
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--class-id", type=int, default=CLASS_ID)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--window-length", type=float, default=WINDOW_LENGTH)
    parser.add_argument(
        "--outside-conditioning",
        type=str,
        choices=["conditional", "unconditional"],
        default="unconditional",
    )
    parser.add_argument("--ti-start", type=float, default=TI_START)
    parser.add_argument("--ti-end", type=float, default=TI_END)
    parser.add_argument("--ti-step", type=float, default=TI_STEP)
    parser.add_argument("--schedule-index-offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--schedule-indices", type=str, default="all")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--classifier-model-id", type=str, default=CLS_MODEL_ID)
    parser.add_argument(
        "--classifier-device",
        type=str,
        choices=["same", "cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--fid-device", type=str, default="cpu")
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-all-images",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--skip-existing-schedules", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main():
    global ACTIVE_TI_BY_INDEX
    args = parse_args()
    ACTIVE_TI_BY_INDEX = build_ti_index_map(
        args.ti_start,
        args.ti_end,
        args.ti_step,
        args.schedule_index_offset,
    )

    base.parse_schedule_indices = parse_schedule_indices
    base.group_config = group_config
    base.decode_latents = decode_latents
    base.build_pipeline = build_pipeline

    print(
        f"[config] masked_sdpa=True transformer_only_gpu=True "
        f"schedule_to_ti={ACTIVE_TI_BY_INDEX} window_length={args.window_length}"
    )
    base.run_benchmark(args)


if __name__ == "__main__":
    main()
