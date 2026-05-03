#!/usr/bin/env python3
"""
Masked-SDPA SD3 Medium sliding global-window/local benchmark.

- Uses the newer masked-SDPA local attention kernel.
- Keeps only the SD3 transformer on GPU; VAE and text encoders stay on CPU.
- Sweeps explicit global-window start times t_i rather than the legacy i/10 grid.
- Preserves the saved feature tensors required by postprocess_benchmark_error_bars.py.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusion3Pipeline
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from windowed_sampling import run_sd3_schedule_fid_benchmark as base
from models.local_attention_masked_sdpa import swap_in_masked_sdpa_local_attn


MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "a golden retriever playing in a park, high detail, soft lighting"
NEGATIVE_PROMPT = ""
CLASS_ID = 207
CLS_MODEL_ID = "microsoft/resnet-50"
GUIDANCE = 3.0
STEPS = 40
NUM_SAMPLES = 500
NUM_EXAMPLES = 4
BASE_SEED = 123
WINDOW_LENGTH = 0.4
TI_START = 0.2
TI_END = 0.8
TI_STEP = 0.05

CPU_DEVICE = torch.device("cpu")
ACTIVE_TI_VALUES: List[float] = []


def build_ti_values(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--ti-step must be positive")
    if end < start:
        raise ValueError("--ti-end must be >= --ti-start")

    num_steps = int(math.floor((end - start) / step + 1e-9)) + 1
    values = [round(start + idx * step, 4) for idx in range(num_steps)]
    if abs(values[-1] - end) > 1e-6 and values[-1] < end:
        values.append(round(end, 4))
    for value in values:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"All t_i values must lie in [0, 1], got {value}")
    return values


def parse_schedule_indices(raw: str, _experiment: str) -> List[int]:
    valid_indices = list(range(len(ACTIVE_TI_VALUES)))
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
    if schedule_index < 0 or schedule_index >= len(ACTIVE_TI_VALUES):
        raise ValueError(
            f"sliding-window schedule index must be in [0, {len(ACTIVE_TI_VALUES) - 1}], got {schedule_index}"
        )
    start = ACTIVE_TI_VALUES[schedule_index]
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


def move_non_transformer_modules_to_cpu(pipe: StableDiffusion3Pipeline) -> None:
    for attr_name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipe, attr_name, None)
        if module is not None and hasattr(module, "to"):
            module.to(device=CPU_DEVICE, dtype=torch.float32)


def decode_to_pil(pipe, latents: torch.Tensor) -> Image.Image:
    scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents_cpu = latents.detach().to(device=CPU_DEVICE, dtype=torch.float32)
    latents_dec = latents_cpu / scale + shift
    pipe.vae.to(device=CPU_DEVICE, dtype=torch.float32)
    image = pipe.vae.decode(latents_dec, return_dict=False)[0]
    image = image.detach()
    if hasattr(pipe, "image_processor"):
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    image = (image.clamp(-1, 1) + 1) / 2.0
    image_np = image[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((image_np * 255).round().astype("uint8"))


def encode_prompt_parts(pipe, args):
    kwargs = dict(
        prompt=args.prompt,
        prompt_2=args.prompt2,
        prompt_3=args.prompt3 if base.has_component(pipe, "text_encoder_3") else None,
        device=CPU_DEVICE,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=args.negative_prompt,
        negative_prompt_2=args.negative_prompt2,
        negative_prompt_3=args.negative_prompt3 if base.has_component(pipe, "text_encoder_3") else None,
    )
    try:
        prompt_parts = pipe.encode_prompt(**kwargs, max_sequence_length=args.max_sequence_length)
    except TypeError as exc:
        if "max_sequence_length" not in str(exc):
            raise
        prompt_parts = pipe.encode_prompt(**kwargs)

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_parts
    return (
        prompt_embeds.to(device=base.device, dtype=base.dtype),
        negative_prompt_embeds.to(device=base.device, dtype=base.dtype),
        pooled_prompt_embeds.to(device=base.device, dtype=base.dtype),
        negative_pooled_prompt_embeds.to(device=base.device, dtype=base.dtype),
    )


def build_pipeline(args):
    load_kwargs = {
        "torch_dtype": torch.float32,
        "local_files_only": args.local_files_only,
    }
    if args.variant:
        load_kwargs["variant"] = args.variant
    if args.disable_t5:
        load_kwargs["text_encoder_3"] = None
        load_kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, **load_kwargs)
    base.validate_pipeline(pipe, args.model_id)
    if args.use_dpms:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    height, width, latent_h, latent_w, h_tok, w_tok = base.resolve_geometry(
        pipe,
        args.height,
        args.width,
    )
    num_replaced = swap_in_masked_sdpa_local_attn(
        pipe.transformer,
        h_tok,
        w_tok,
        r=args.radius,
        local_intervals=[],
    )

    if args.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if args.vae_slicing and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    move_non_transformer_modules_to_cpu(pipe)
    pipe.transformer.to(device=base.device, dtype=base.dtype)
    pipe.transformer.eval()
    pipe.vae.eval()

    print(
        f"[init] model={args.model_id} image={height}x{width} latent={latent_h}x{latent_w} "
        f"tokens={h_tok}x{w_tok} dtype={base.dtype} replaced_attn={num_replaced} "
        f"transformer_device={base.device} vae_device={CPU_DEVICE} disable_t5={args.disable_t5} "
        f"vae_tiling={args.vae_tiling} vae_slicing={args.vae_slicing}"
    )

    return pipe, {
        "height": height,
        "width": width,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "h_tok": h_tok,
        "w_tok": w_tok,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Masked-SDPA SD3 sliding global-window/local FID/classifier benchmark."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["sliding_global_window_local"],
        default="sliding_global_window_local",
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--prompt2", type=str, default=None)
    parser.add_argument("--prompt3", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--negative-prompt2", type=str, default=None)
    parser.add_argument("--negative-prompt3", type=str, default=None)
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
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--schedule-indices", type=str, default="all")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--classifier-model-id", type=str, default=CLS_MODEL_ID)
    parser.add_argument(
        "--classifier-device",
        type=str,
        choices=["same", "cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--fid-device", type=str, default="cpu")
    parser.add_argument("--fid-batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--disable-t5", action="store_true")
    parser.add_argument("--use-dpms", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if args.prompt2 is None:
        args.prompt2 = args.prompt
    if args.prompt3 is None:
        args.prompt3 = args.prompt
    if args.negative_prompt2 is None:
        args.negative_prompt2 = args.negative_prompt
    if args.negative_prompt3 is None:
        args.negative_prompt3 = args.negative_prompt
    return args


def main():
    global ACTIVE_TI_VALUES
    args = parse_args()
    ACTIVE_TI_VALUES = build_ti_values(args.ti_start, args.ti_end, args.ti_step)

    base.parse_schedule_indices = parse_schedule_indices
    base.group_config = group_config
    base.build_pipeline = build_pipeline
    base.encode_prompt_parts = encode_prompt_parts
    base.decode_to_pil = decode_to_pil

    print(
        f"[config] masked_sdpa=True transformer_only_gpu=True "
        f"ti_values={ACTIVE_TI_VALUES} window_length={args.window_length}"
    )
    base.run_benchmark(args)


if __name__ == "__main__":
    main()
