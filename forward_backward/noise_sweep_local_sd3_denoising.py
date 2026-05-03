"""
Noise sweep experiment (SD3 local attention): generate clean images with
global attention, inject noise at multiple timesteps, and denoise with local
attention (conditional + unconditional). Reports per-pixel MSE and classifier error.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoImageProcessor, AutoModelForImageClassification

from models.local_attention import current_t_norm, swap_in_efficient_local_attn


MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "a golden retriever."
PROMPT2 = PROMPT
PROMPT3 = PROMPT
NEGATIVE_PROMPT = ""
NEGATIVE_PROMPT2 = ""
NEGATIVE_PROMPT3 = ""
GUIDANCE = 3.0
CLASS_ID = 207
CLEAN_STEPS = 20
DENOISE_STEPS = 50
NUM_NOISE_LEVELS = 40
NOISE_MIN = 0.1
NOISE_MAX = 1.0
BASE_SEED = 123
N_DENOISE = 40
N_CLEAN = 1
MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = str(MODULE_DIR / "outputs" / "noise_sweep_sd3_local")
CLS_MODEL_ID = "microsoft/resnet-50"
LOCAL_INTERVALS = "[[0.0, 1.0]]"
USE_DPMS = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32


def make_t_normalizer(scheduler):
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


def build_scheduler(pipe, use_dpms: bool):
    if use_dpms:
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return type(pipe.scheduler).from_config(pipe.scheduler.config)


def setup_partial_timesteps(scheduler, start_timestep: int, device, num_inference_steps: int):
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    ts = timesteps.detach().cpu() if isinstance(timesteps, torch.Tensor) else torch.tensor(timesteps)
    start_idx = int(torch.argmin(torch.abs(ts - float(start_timestep))).item())
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "config") and "solver_order" in scheduler.config:
        scheduler.model_outputs = [None] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0
    return timesteps, start_idx


def select_timestep_from_t_norm(scheduler, t_norm: float) -> Tuple[float, int]:
    ts = scheduler.timesteps
    num = len(ts)
    idx = int(np.clip(round((1.0 - t_norm) * (num - 1)), 0, num - 1))
    t_val = float(ts[idx].item()) if isinstance(ts, torch.Tensor) else float(ts[idx])
    return t_val, idx


def decode_latents(pipe, latents: torch.Tensor) -> np.ndarray:
    scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
    with torch.no_grad():
        image = pipe.vae.decode(latents / scale).sample
    image = (image.clamp(-1, 1) + 1) / 2.0
    return image[0].permute(1, 2, 0).float().cpu().detach().numpy()


def parse_intervals(raw: str):
    try:
        intervals = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid local intervals JSON: {raw}") from exc
    return intervals


def build_pipeline(model_id: str, use_dpms: bool):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    if use_dpms:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def configure_attention(pipe, radius: int, local_intervals):
    patch = pipe.transformer.config.patch_size
    res = pipe.transformer.config.sample_size
    h_tok, w_tok = res // patch, res // patch
    swap_in_efficient_local_attn(
        pipe.transformer,
        h_tok,
        w_tok,
        r=radius,
        local_intervals=local_intervals,
    )
    return pipe


def load_classifier(cache_dir: str = None):
    cache_dir = cache_dir or os.environ.get("HF_HOME")
    processor = AutoImageProcessor.from_pretrained(CLS_MODEL_ID, cache_dir=cache_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(CLS_MODEL_ID, cache_dir=cache_dir, local_files_only=True)
    model = model.to(device)
    model.eval()
    return processor, model


def classify_image(image_np: np.ndarray, processor, model) -> int:
    """Return predicted ImageNet class id for an image in [0,1] np array."""
    pil = Image.fromarray((image_np * 255).round().astype("uint8"))
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = int(torch.argmax(logits, dim=-1).item())
    return pred


def encode_prompts(pipe, prompt, prompt2, prompt3, do_cfg, neg, neg2, neg3):
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
        pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt2,
            prompt_3=prompt3,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=neg if do_cfg else None,
            negative_prompt_2=neg2 if do_cfg else None,
            negative_prompt_3=neg3 if do_cfg else None,
        )
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    return prompt_embeds, pooled_prompt_embeds


def sample_clean_latents(
    pipe,
    seed: int,
    prompt_data: Tuple[torch.Tensor, torch.Tensor],
    guidance_scale: float,
    num_inference_steps: int,
    use_dpms: bool,
) -> Tuple[torch.Tensor, np.ndarray]:
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = build_scheduler(pipe, use_dpms)
    scheduler.set_timesteps(num_inference_steps, device=device)
    t_to_unit = make_t_normalizer(scheduler)
    prompt_embeds, pooled_prompt_embeds = prompt_data

    in_ch = pipe.transformer.config.in_channels
    res = pipe.transformer.config.sample_size
    latents = randn_tensor((1, in_ch, res, res), generator=generator, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma

    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_norm = t_to_unit(float(t.item() if isinstance(t, torch.Tensor) else t))
            current_t_norm["value"] = t_norm
            if not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=device, dtype=scheduler.timesteps.dtype)
            elif t.ndim == 0:
                t = t.unsqueeze(0)

            latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_model_input = maybe_scale_model_input(scheduler, latent_model_input, t)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
            ).sample
            if do_cfg:
                uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

            step_out = scheduler.step(noise_pred, t[0] if t.shape[0] > 1 else t, latents)
            latents = step_out.prev_sample

    latents_cpu = latents.cpu()
    clean_image = decode_latents(pipe, latents)
    return latents_cpu, clean_image


def add_noise_at_t(
    clean_latents: torch.Tensor,
    scheduler,
    timestep: float,
    generator: torch.Generator,
    step_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    latents = clean_latents.to(device, dtype=dtype)
    noise = torch.randn(
        latents.shape,
        device=device,
        dtype=latents.dtype,
        generator=generator,
    )
    timestep_tensor = torch.tensor([timestep], device=device, dtype=scheduler.timesteps.dtype)
    if hasattr(scheduler, "scale_noise"):
        scheduler.set_begin_index(step_index)
        scheduler._step_index = None
        noisy_latents = scheduler.scale_noise(latents, timestep_tensor, noise)
    else:
        alpha_bar = scheduler.alphas_cumprod[timestep_tensor.long()].to(device=device, dtype=latents.dtype)
        noisy_latents = alpha_bar.sqrt() * latents + (1 - alpha_bar).sqrt() * noise
    return noisy_latents, timestep_tensor


def denoise_local(
    pipe,
    noisy_latents: torch.Tensor,
    start_timestep: int,
    prompt_data: Tuple[torch.Tensor, torch.Tensor],
    guidance_scale: float,
    num_inference_steps: int,
    use_dpms: bool,
) -> np.ndarray:
    scheduler = build_scheduler(pipe, use_dpms)
    timesteps, start_idx = setup_partial_timesteps(scheduler, start_timestep, device, num_inference_steps)
    latents = noisy_latents.to(device, dtype=dtype)
    prompt_embeds, pooled_prompt_embeds = prompt_data
    do_cfg = guidance_scale is not None and guidance_scale > 1.0

    t_to_unit = make_t_normalizer(scheduler)
    with torch.no_grad():
        for t in timesteps[start_idx:]:
            t_norm = t_to_unit(float(t.item() if isinstance(t, torch.Tensor) else t))
            current_t_norm["value"] = t_norm
            if not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=device, dtype=scheduler.timesteps.dtype)
            elif t.ndim == 0:
                t = t.unsqueeze(0)

            latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_model_input = maybe_scale_model_input(scheduler, latent_model_input, t)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
            ).sample
            if do_cfg:
                uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

            step_out = scheduler.step(noise_pred, t[0] if t.shape[0] > 1 else t, latents)
            latents = step_out.prev_sample

    return decode_latents(pipe, latents)


def mse_per_pixel(clean_img: np.ndarray, denoised_img: np.ndarray) -> float:
    return float(np.mean((clean_img - denoised_img) ** 2))


def run_noise_sweep(args):
    local_intervals = parse_intervals(args.local_intervals)
    r_dir = os.path.join(args.output_dir, f"r{args.radius}")
    os.makedirs(r_dir, exist_ok=True)
    examples_dir = os.path.join(r_dir, "examples")
    cond_dir = os.path.join(examples_dir, "cond")
    uncond_dir = os.path.join(examples_dir, "uncond")
    clean_dir = os.path.join(examples_dir, "clean")
    noisy_dir = os.path.join(examples_dir, "noisy")
    raw_dir = os.path.join(r_dir, "raw")
    os.makedirs(cond_dir, exist_ok=True)
    os.makedirs(uncond_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    metrics_path = args.metrics_path or os.path.join(r_dir, "metrics.json")
    raw_mse_path = args.raw_mse_path or os.path.join(raw_dir, "raw_mse.json")

    global_pipe = build_pipeline(args.model_id, args.use_dpms)
    configure_attention(global_pipe, radius=max(1, args.radius), local_intervals=[])
    pipe = build_pipeline(args.model_id, args.use_dpms)
    configure_attention(pipe, args.radius, local_intervals)

    noise_scheduler = build_scheduler(pipe, args.use_dpms)
    noise_scheduler.set_timesteps(args.denoise_steps, device=device)
    noise_levels = np.linspace(args.min_t, args.max_t, args.num_noise_levels)
    per_level: Dict[str, Dict[str, list]] = {
        f"{lvl:.4f}": {"cond": [], "uncond": []} for lvl in noise_levels
    }
    per_level_agree: Dict[str, Dict[str, list]] = {
        f"{lvl:.4f}": {"cond": [], "uncond": []} for lvl in noise_levels
    }
    classifier_processor, classifier_model = load_classifier()

    cond_prompt_data = encode_prompts(
        pipe,
        args.prompt,
        args.prompt2 or args.prompt,
        args.prompt3 or args.prompt,
        True,
        args.negative_prompt,
        args.negative_prompt2 or args.negative_prompt,
        args.negative_prompt3 or args.negative_prompt,
    )
    uncond_prompt_data = encode_prompts(
        pipe,
        "",
        "",
        "",
        False,
        "",
        "",
        "",
    )

    for clean_idx in range(args.n_clean):
        clean_seed = args.base_seed + clean_idx
        print(f"[clean {clean_idx+1}/{args.n_clean}] seed={clean_seed}")
        clean_latents, clean_img = sample_clean_latents(
            global_pipe,
            seed=clean_seed,
            prompt_data=cond_prompt_data,
            guidance_scale=args.guidance,
            num_inference_steps=args.clean_steps,
            use_dpms=args.use_dpms,
        )
        Image.fromarray((clean_img * 255).round().astype("uint8")).save(
            os.path.join(clean_dir, f"clean_{clean_idx}.png")
        )

        for noise_idx, t_norm in enumerate(noise_levels):
            timestep, step_idx = select_timestep_from_t_norm(noise_scheduler, float(t_norm))
            print(
                f"  [noise {noise_idx+1}/{len(noise_levels)}] t_norm={t_norm:.3f} "
                f"timestep={timestep}"
            )
            for run_idx in range(args.n_denoise):
                noise_seed = args.noise_seed_offset + clean_idx * 10_000 + noise_idx * 100 + run_idx
                generator = torch.Generator(device=device).manual_seed(noise_seed)
                noisy_latents, _ = add_noise_at_t(clean_latents, noise_scheduler, timestep, generator, step_idx)

                if clean_idx == 0 and run_idx == 0:
                    noisy_img = decode_latents(pipe, noisy_latents)
                    Image.fromarray((noisy_img * 255).round().astype("uint8")).save(
                        os.path.join(noisy_dir, f"noisy_t{t_norm:.3f}.png")
                    )

                cond_img = denoise_local(
                    pipe,
                    noisy_latents,
                    start_timestep=timestep,
                    prompt_data=cond_prompt_data,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.denoise_steps,
                    use_dpms=args.use_dpms,
                )
                uncond_img = denoise_local(
                    pipe,
                    noisy_latents,
                    start_timestep=timestep,
                    prompt_data=uncond_prompt_data,
                    guidance_scale=1.0,
                    num_inference_steps=args.denoise_steps,
                    use_dpms=args.use_dpms,
                )

                if clean_idx == 0 and run_idx == 0:
                    Image.fromarray((cond_img * 255).round().astype("uint8")).save(
                        os.path.join(cond_dir, f"denoised_t{t_norm:.3f}.png")
                    )
                    Image.fromarray((uncond_img * 255).round().astype("uint8")).save(
                        os.path.join(uncond_dir, f"denoised_t{t_norm:.3f}.png")
                    )

                cond_mse = mse_per_pixel(clean_img, cond_img)
                uncond_mse = mse_per_pixel(clean_img, uncond_img)
                per_level[f"{t_norm:.4f}"]["cond"].append(cond_mse)
                per_level[f"{t_norm:.4f}"]["uncond"].append(uncond_mse)

                pred_cond = classify_image(cond_img, classifier_processor, classifier_model)
                pred_uncond = classify_image(uncond_img, classifier_processor, classifier_model)
                per_level_agree[f"{t_norm:.4f}"]["cond"].append(int(pred_cond == args.class_id))
                per_level_agree[f"{t_norm:.4f}"]["uncond"].append(int(pred_uncond == args.class_id))

    summary = []
    for t_norm in noise_levels:
        key = f"{t_norm:.4f}"
        cond_vals = np.array(per_level[key]["cond"])
        uncond_vals = np.array(per_level[key]["uncond"])
        cond_agree = np.array(per_level_agree[key]["cond"])
        uncond_agree = np.array(per_level_agree[key]["uncond"])
        timestep, _ = select_timestep_from_t_norm(noise_scheduler, float(t_norm))
        summary.append(
            {
                "t_norm": float(t_norm),
                "timestep": timestep,
                "cond_mean": float(cond_vals.mean()) if cond_vals.size else None,
                "cond_std": float(cond_vals.std()) if cond_vals.size else None,
                "uncond_mean": float(uncond_vals.mean()) if uncond_vals.size else None,
                "uncond_std": float(uncond_vals.std()) if uncond_vals.size else None,
                "count": int(cond_vals.size),
                "cond_agree_rate": float(cond_agree.mean()) if cond_agree.size else None,
                "uncond_agree_rate": float(uncond_agree.mean()) if uncond_agree.size else None,
            }
        )

    metrics = {
        "config": {
            "model_id": args.model_id,
            "prompt": args.prompt,
            "prompt2": args.prompt2,
            "prompt3": args.prompt3,
            "negative_prompt": args.negative_prompt,
            "negative_prompt2": args.negative_prompt2,
            "negative_prompt3": args.negative_prompt3,
            "guidance": args.guidance,
            "clean_steps": args.clean_steps,
            "denoise_steps": args.denoise_steps,
            "num_noise_levels": args.num_noise_levels,
            "min_t": args.min_t,
            "max_t": args.max_t,
            "n_denoise": args.n_denoise,
            "n_clean": args.n_clean,
            "base_seed": args.base_seed,
            "noise_seed_offset": args.noise_seed_offset,
            "classifier_model_id": CLS_MODEL_ID,
            "class_id": args.class_id,
            "radius": args.radius,
            "local_intervals": local_intervals,
            "use_dpms": args.use_dpms,
        },
        "noise_levels": noise_levels.tolist(),
        "per_level": per_level,
        "per_level_agree": per_level_agree,
        "summary": summary,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(raw_mse_path, "w") as f:
        json.dump(per_level, f, indent=2)

    # Plot MSE curves
    cond_means = [row["cond_mean"] for row in summary]
    uncond_means = [row["uncond_mean"] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(noise_levels, cond_means, marker="o", label="conditional")
    plt.plot(noise_levels, uncond_means, marker="s", label="unconditional")
    plt.xlabel("t_norm")
    plt.ylabel("Per-pixel MSE")
    plt.title("MSE vs noise level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(r_dir, "mse_curve.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Plot classifier error (1 - agreement)
    cond_agree = [row["cond_agree_rate"] for row in summary]
    uncond_agree = [row["uncond_agree_rate"] for row in summary]
    cond_error = [None if v is None else 1.0 - v for v in cond_agree]
    uncond_error = [None if v is None else 1.0 - v for v in uncond_agree]
    plt.figure(figsize=(7, 4))
    plt.plot(noise_levels, cond_error, marker="o", label="conditional")
    plt.plot(noise_levels, uncond_error, marker="s", label="unconditional")
    plt.xlabel("t_norm")
    plt.ylabel("Classifier error (top-1 != target class)")
    plt.title("Classifier error vs noise level")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    error_path = os.path.join(r_dir, "classification_error_curve.png")
    plt.savefig(error_path, dpi=200)
    plt.close()

    print("t_norm\tstep\tcond_mean\tcond_std\tuncond_mean\tuncond_std")
    for row in summary:
        print(
            f"{row['t_norm']:.3f}\t{row['timestep']}\t"
            f"{row['cond_mean']:.6f}\t{row['cond_std']:.6f}\t"
            f"{row['uncond_mean']:.6f}\t{row['uncond_std']:.6f}"
        )
    print(f"Metrics saved to {metrics_path}")
    print(f"MSE plot saved to {plot_path}")
    print(f"Classification error plot saved to {error_path}")
    print(f"Raw MSE saved to {raw_mse_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Noise sweep SD3 local denoising experiment")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--prompt2", type=str, default=PROMPT2)
    parser.add_argument("--prompt3", type=str, default=PROMPT3)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--negative-prompt2", type=str, default=NEGATIVE_PROMPT2)
    parser.add_argument("--negative-prompt3", type=str, default=NEGATIVE_PROMPT3)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--class-id", type=int, default=CLASS_ID)
    parser.add_argument("--clean-steps", type=int, default=CLEAN_STEPS)
    parser.add_argument("--denoise-steps", type=int, default=DENOISE_STEPS)
    parser.add_argument("--num-noise-levels", type=int, default=NUM_NOISE_LEVELS)
    parser.add_argument("--min-t", type=float, default=NOISE_MIN)
    parser.add_argument("--max-t", type=float, default=NOISE_MAX)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--noise-seed-offset", type=int, default=10_000)
    parser.add_argument("--n-denoise", type=int, default=N_DENOISE)
    parser.add_argument("--n-clean", type=int, default=N_CLEAN)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--raw-mse-path", type=str, default="")
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--local-intervals", type=str, default=LOCAL_INTERVALS)
    parser.add_argument("--use-dpms", action="store_true", default=USE_DPMS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_noise_sweep(args)
