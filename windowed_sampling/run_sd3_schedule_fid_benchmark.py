"""
SD3 schedule benchmark with FID to a global-conditioned baseline.

Modes:
- local_attention: schedule i uses local image attention with radius r only in
  [i / 10, (i + 1) / 10], with text conditioning active at every step.
- conditioning_window: schedule i uses global attention at every step, but
  text conditioning/CFG is active only in [i / 10, (i + 1) / 10]; outside the
  window the negative/unconditional prompt embedding is used.

This script streams Inception features and saves only a few example images per
group by default.
"""
import argparse
import csv
import inspect
import json
import os
import time
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoImageProcessor, AutoModelForImageClassification


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from models.local_attention import current_t_norm, swap_in_efficient_local_attn


MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
CLS_MODEL_ID = "microsoft/resnet-50"
PROMPT = "a golden retriever playing in a park, high detail, soft lighting"
NEGATIVE_PROMPT = ""
CLASS_ID = 207
GUIDANCE = 3.0
STEPS = 40
RADIUS = 3
NUM_SAMPLES = 500
BASE_SEED = 123


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32


def resolve_output_dir(path: str) -> str:
    if os.path.isabs(path):
        out_dir = os.path.abspath(path)
    else:
        out_dir = os.path.abspath(os.path.join(SCRIPT_DIR, path))
    root = os.path.abspath(SCRIPT_DIR)
    if os.path.commonpath([out_dir, root]) != root:
        raise ValueError(f"output-dir must stay under {root}; got {out_dir}")
    return out_dir


def make_t_normalizer(scheduler):
    ts = scheduler.timesteps.to("cpu")
    tmax = float(ts.max().item())
    tmin = float(ts.min().item())
    rng = max(1.0, tmax - tmin)

    def to_unit_interval(t_int: float) -> float:
        return (float(t_int) - tmin) / rng

    return to_unit_interval


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def ensure_t_tensor(t, scheduler):
    if not isinstance(t, torch.Tensor):
        return torch.tensor([t], device=device, dtype=scheduler.timesteps.dtype)
    if t.ndim == 0:
        return t.unsqueeze(0)
    return t


def maybe_scale_model_input(scheduler, latents, timestep):
    if hasattr(scheduler, "scale_model_input"):
        return scheduler.scale_model_input(latents, timestep)
    return latents


def fix_channels(noise_pred: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    if noise_pred.shape[1] == 2 * latents.shape[1]:
        noise_pred, _ = noise_pred.chunk(2, dim=1)
    return noise_pred


def parse_schedule_indices(raw: str, experiment: str) -> List[int]:
    if experiment == "sliding_global_window_local":
        valid_indices = list(range(8))
    else:
        valid_indices = list(range(10))

    if raw.strip().lower() in {"all", "*"}:
        return valid_indices

    indices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx not in valid_indices:
            lo = valid_indices[0]
            hi = valid_indices[-1]
            raise ValueError(f"schedule index must be in [{lo}, {hi}], got {idx}")
        indices.append(idx)
    if not indices:
        raise ValueError("at least one schedule index is required")
    return sorted(dict.fromkeys(indices))


def schedule_interval(schedule_index: int) -> List[List[float]]:
    start = schedule_index / 10.0
    end = (schedule_index + 1) / 10.0
    return [[round(start, 4), round(end, 4)]]


def sliding_window_interval(schedule_index: int, window_length: float) -> List[List[float]]:
    if schedule_index < 0 or schedule_index > 7:
        raise ValueError(f"sliding-window schedule index must be in [0, 7], got {schedule_index}")
    start = schedule_index / 10.0
    end = min(1.0, start + float(window_length))
    return [[round(start, 4), round(end, 4)]]


def complement_intervals(intervals: List[List[float]]) -> List[List[float]]:
    if not intervals:
        return [[0.0, 1.0]]

    intervals = sorted(intervals, key=lambda item: item[0])
    complement = []
    cursor = 0.0
    eps = 1e-6
    for start, end in intervals:
        if start > cursor:
            complement.append([cursor, max(cursor, start - eps)])
        cursor = max(cursor, min(1.0, end + eps))
    if cursor < 1.0:
        complement.append([cursor, 1.0])
    return [[round(start, 6), round(end, 6)] for start, end in complement if start <= end]


def in_intervals(t_norm: float, intervals: List[List[float]]) -> bool:
    return any(start <= t_norm <= end for start, end in intervals)


def should_use_conditioning(
    t_norm: float,
    conditioning_intervals: List[List[float]],
    conditioning_mode: str,
) -> bool:
    if conditioning_mode == "always":
        return True

    in_window = in_intervals(t_norm, conditioning_intervals)
    if conditioning_mode == "inside":
        return in_window
    if conditioning_mode == "outside":
        return not in_window
    raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")


def has_component(pipe, name: str) -> bool:
    return hasattr(pipe, name) and getattr(pipe, name) is not None


def validate_pipeline(pipe, model_id: str) -> None:
    required = ["transformer", "vae", "scheduler"]
    missing = [name for name in required if not has_component(pipe, name)]
    if missing:
        raise ValueError(f"Model '{model_id}' is missing required SD3 components: {missing}")
    text_pairs = [
        ("text_encoder", "tokenizer"),
        ("text_encoder_2", "tokenizer_2"),
        ("text_encoder_3", "tokenizer_3"),
    ]
    broken_pairs = [
        pair for pair in text_pairs if has_component(pipe, pair[0]) != has_component(pipe, pair[1])
    ]
    if broken_pairs:
        raise ValueError(f"Model '{model_id}' has incomplete text components: {broken_pairs}")
    if not any(has_component(pipe, enc) for enc, _tok in text_pairs):
        raise ValueError(f"Model '{model_id}' has no text encoder/tokenizer components")


def resolve_geometry(pipe, height: int, width: int) -> Tuple[int, int, int, int, int, int]:
    vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
    sample_size = int(getattr(pipe.transformer.config, "sample_size", 0) or 128)
    default_height = sample_size * vae_scale
    height = height or default_height
    width = width or height
    if height % vae_scale != 0 or width % vae_scale != 0:
        raise ValueError(f"HEIGHT/WIDTH must be divisible by VAE scale {vae_scale}")

    latent_h = height // vae_scale
    latent_w = width // vae_scale
    patch = int(getattr(pipe.transformer.config, "patch_size", 1))
    if latent_h % patch != 0 or latent_w % patch != 0:
        raise ValueError(f"Latent size {latent_h}x{latent_w} must divide patch_size={patch}")

    h_tok = latent_h // patch
    w_tok = latent_w // patch
    max_pos = getattr(pipe.transformer.config, "pos_embed_max_size", None)
    if max_pos is not None and (h_tok > int(max_pos) or w_tok > int(max_pos)):
        raise ValueError(f"Token grid {h_tok}x{w_tok} exceeds pos_embed_max_size={max_pos}")
    return height, width, latent_h, latent_w, h_tok, w_tok


def set_sd3_timesteps(scheduler, steps: int, latents: torch.Tensor, patch_size: int):
    kwargs = {}
    if scheduler.config.get("use_dynamic_shifting", None):
        _b, _c, latent_h, latent_w = latents.shape
        image_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
        kwargs["mu"] = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.16),
        )

    set_params = inspect.signature(scheduler.set_timesteps).parameters
    if "mu" not in set_params:
        kwargs.pop("mu", None)
    scheduler.set_timesteps(steps, device=device, **kwargs)
    return scheduler.timesteps


def prepare_dirs(output_dir: str) -> Dict[str, str]:
    dirs = {
        "root": output_dir,
        "examples": os.path.join(output_dir, "examples"),
        "features": os.path.join(output_dir, "features"),
        "metrics": os.path.join(output_dir, "metrics"),
        "plots": os.path.join(output_dir, "plots"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def build_pipeline(args):
    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if args.variant:
        load_kwargs["variant"] = args.variant
    if args.disable_t5:
        load_kwargs["text_encoder_3"] = None
        load_kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, **load_kwargs)
    validate_pipeline(pipe, args.model_id)
    if args.use_dpms:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    height, width, latent_h, latent_w, h_tok, w_tok = resolve_geometry(
        pipe,
        args.height,
        args.width,
    )
    num_replaced = swap_in_efficient_local_attn(
        pipe.transformer,
        h_tok,
        w_tok,
        r=args.radius,
        local_intervals=[],
    )

    print(
        f"[init] model={args.model_id} image={height}x{width} latent={latent_h}x{latent_w} "
        f"tokens={h_tok}x{w_tok} dtype={dtype} replaced_attn={num_replaced} "
        f"cpu_offload={args.cpu_offload} sequential={args.sequential_cpu_offload} "
        f"disable_t5={args.disable_t5} vae_tiling={args.vae_tiling} "
        f"vae_slicing={args.vae_slicing}"
    )

    if args.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if args.vae_slicing and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    if args.sequential_cpu_offload and device.type == "cuda":
        pipe.enable_sequential_cpu_offload()
    elif args.cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.transformer.eval()
    pipe.vae.eval()
    return pipe, {
        "height": height,
        "width": width,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "h_tok": h_tok,
        "w_tok": w_tok,
    }


def set_attention_schedule(transformer, radius: int, local_intervals: List[List[float]]) -> int:
    count = 0
    for module in transformer.modules():
        if hasattr(module, "local_intervals") and hasattr(module, "r"):
            module.r = radius
            module.local_intervals = local_intervals
            count += 1
    return count


def encode_prompt_parts(pipe, args):
    kwargs = dict(
        prompt=args.prompt,
        prompt_2=args.prompt2,
        prompt_3=args.prompt3 if has_component(pipe, "text_encoder_3") else None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=args.negative_prompt,
        negative_prompt_2=args.negative_prompt2,
        negative_prompt_3=args.negative_prompt3 if has_component(pipe, "text_encoder_3") else None,
    )
    try:
        return pipe.encode_prompt(**kwargs, max_sequence_length=args.max_sequence_length)
    except TypeError as exc:
        if "max_sequence_length" not in str(exc):
            raise
        return pipe.encode_prompt(**kwargs)


def decode_to_pil(pipe, latents: torch.Tensor) -> Image.Image:
    scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents_dec = latents / scale + shift
    if latents_dec.dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False):
        pipe.vae.to(dtype=torch.float32)
        latents_dec = latents_dec.float()
    image = pipe.vae.decode(latents_dec, return_dict=False)[0]
    image = image.detach()
    if hasattr(pipe, "image_processor"):
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    image = (image.clamp(-1, 1) + 1) / 2.0
    image_np = image[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((image_np * 255).round().astype("uint8"))


def generate_sample(
    pipe,
    geometry: Dict[str, int],
    prompt_parts,
    seed: int,
    guidance_scale: float,
    steps: int,
    conditioning_intervals: List[List[float]],
    conditioning_mode: str,
) -> Image.Image:
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_parts
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = pipe.scheduler

    in_ch = pipe.transformer.config.in_channels
    latents = randn_tensor(
        (1, in_ch, geometry["latent_h"], geometry["latent_w"]),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma

    timesteps = set_sd3_timesteps(scheduler, steps, latents, pipe.transformer.config.patch_size)
    t_to_unit = make_t_normalizer(scheduler)

    with torch.inference_mode():
        for t in timesteps:
            t_norm = t_to_unit(float(t.item() if isinstance(t, torch.Tensor) else t))
            current_t_norm["value"] = t_norm
            t = ensure_t_tensor(t, scheduler)
            use_conditioning = should_use_conditioning(
                t_norm=t_norm,
                conditioning_intervals=conditioning_intervals,
                conditioning_mode=conditioning_mode,
            )

            if use_conditioning and guidance_scale is not None and guidance_scale > 1.0:
                latent_model_input = torch.cat([latents, latents], dim=0)
                encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                latent_model_input = maybe_scale_model_input(scheduler, latent_model_input, t)
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred_uncond = fix_channels(noise_pred_uncond, latents)
                noise_pred_text = fix_channels(noise_pred_text, latents)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                latent_model_input = maybe_scale_model_input(scheduler, latents, t)
                timestep = t.expand(latent_model_input.shape[0])
                if use_conditioning:
                    encoder_hidden_states = prompt_embeds
                    pooled = pooled_prompt_embeds
                else:
                    encoder_hidden_states = negative_prompt_embeds
                    pooled = negative_pooled_prompt_embeds
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled,
                    return_dict=False,
                )[0]
                noise_pred = fix_channels(noise_pred, latents)

            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t[0] if t.numel() == 1 else t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

    return decode_to_pil(pipe, latents)


def resolve_classifier_device(raw: str) -> torch.device:
    raw = raw.lower()
    if raw == "same":
        return device
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--classifier-device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError(f"Unknown classifier device: {raw}")


def load_classifier(
    model_id: str,
    classifier_device: torch.device,
    cache_dir: str = None,
    local_files_only: bool = True,
):
    cache_dir = cache_dir or os.environ.get("HF_HOME")
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = model.to(classifier_device)
    model.eval()
    return processor, model


def classify_image(
    image: Image.Image,
    processor,
    model,
    classifier_device: torch.device,
) -> Tuple[int, str]:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(classifier_device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    pred = int(torch.argmax(logits, dim=-1).item())
    label = str(model.config.id2label.get(pred, pred))
    return pred, label


def build_inception(fid_device: torch.device):
    inception = torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
        aux_logits=True,
    )
    inception.fc = nn.Identity()
    inception.dropout = nn.Identity()
    inception.eval().to(fid_device)
    return inception


def pil_batch_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    arrays = []
    for image in images:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        arrays.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(arrays, dim=0)


@torch.no_grad()
def inception_features_from_pils(
    images: List[Image.Image],
    inception,
    fid_device: torch.device,
) -> torch.Tensor:
    x = pil_batch_to_tensor(images).clamp(0, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    feats = inception(x.to(fid_device))
    if hasattr(feats, "logits"):
        feats = feats.logits
    elif isinstance(feats, tuple):
        feats = feats[0]
    return feats.detach().cpu()


class FeatureCollector:
    def __init__(self, inception, fid_device: torch.device, batch_size: int):
        self.inception = inception
        self.fid_device = fid_device
        self.batch_size = batch_size
        self.pending: List[Image.Image] = []
        self.features: List[torch.Tensor] = []

    def add(self, image: Image.Image) -> None:
        self.pending.append(image.copy())
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.features.append(inception_features_from_pils(self.pending, self.inception, self.fid_device))
        self.pending = []

    def result(self) -> torch.Tensor:
        self.flush()
        if not self.features:
            raise RuntimeError("No features collected")
        return torch.cat(self.features, dim=0)


def feature_stats(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    features = features.to(torch.float32)
    mu = features.mean(dim=0)
    if features.shape[0] < 2:
        sigma = torch.zeros((features.shape[1], features.shape[1]), dtype=torch.float32)
    else:
        sigma = torch.cov(features.T)
    return mu, sigma


def trace_sqrt_product(sigma_a: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
    prod = (sigma_a @ sigma_b).to(torch.double)
    eigvals = torch.linalg.eigvals(prod).real.clamp(min=0)
    return torch.sqrt(eigvals).sum().to(torch.float32)


def fid_score(mu_ref, sigma_ref, mu_gen, sigma_gen) -> float:
    diff = mu_ref - mu_gen
    covmean_trace = trace_sqrt_product(sigma_ref, sigma_gen)
    fid = diff.dot(diff) + torch.trace(sigma_ref) + torch.trace(sigma_gen) - 2.0 * covmean_trace
    return float(fid.item())


def make_example_grid(example_paths: Iterable[str], output_path: str) -> None:
    paths = [p for p in example_paths if p and os.path.exists(p)]
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = images[0].size
    grid = Image.new("RGB", (width * len(images), height), color=(255, 255, 255))
    for idx, image in enumerate(images):
        grid.paste(image.resize((width, height)), (idx * width, 0))
    grid.save(output_path)


def write_summary_csv(summary: List[Dict], path: str) -> None:
    fieldnames = [
        "schedule_index",
        "interval_start",
        "interval_end",
        "num_samples",
        "correct_count",
        "error_rate",
        "fid_to_global_conditioned",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_error(summary: List[Dict], path: str, title: str) -> None:
    xs = [row["schedule_index"] for row in summary]
    ys = [row["error_rate"] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs)
    plt.xlabel("schedule index i")
    plt.ylabel("classifier error")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_fid(summary: List[Dict], path: str, title: str) -> None:
    xs = [row["schedule_index"] for row in summary]
    ys = [row["fid_to_global_conditioned"] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs)
    plt.xlabel("schedule index i")
    plt.ylabel("FID to global conditioned baseline")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def group_config(
    experiment: str,
    schedule_idx: int,
    radius: int,
    window_length: float,
    outside_conditioning: str,
):
    intervals = schedule_interval(schedule_idx)
    if experiment == "local_attention":
        return {
            "local_intervals": intervals,
            "conditioning_intervals": [],
            "conditioning_mode": "always",
            "summary_interval": intervals[0],
            "description": f"local r={radius} in {intervals[0]}, conditioned always",
        }
    if experiment == "conditioning_window":
        return {
            "local_intervals": [],
            "conditioning_intervals": intervals,
            "conditioning_mode": "inside",
            "summary_interval": intervals[0],
            "description": f"global attention, conditioned only in {intervals[0]}",
        }
    if experiment == "sliding_global_window_local":
        window = sliding_window_interval(schedule_idx, window_length)
        conditioning_mode = "always" if outside_conditioning == "conditional" else "inside"
        outside_desc = "local conditioned" if outside_conditioning == "conditional" else "local unconditional"
        return {
            "local_intervals": complement_intervals(window),
            "conditioning_intervals": window,
            "conditioning_mode": conditioning_mode,
            "summary_interval": window[0],
            "description": f"global conditioned in {window[0]}, {outside_desc} otherwise",
        }
    raise ValueError(f"Unknown experiment: {experiment}")


def generate_group(
    *,
    pipe,
    geometry,
    prompt_parts,
    classifier_processor,
    classifier_model,
    classifier_device: torch.device,
    inception,
    fid_device: torch.device,
    sample_seeds: List[int],
    examples_dir: str,
    class_id: int,
    guidance: float,
    steps: int,
    fid_batch_size: int,
    conditioning_intervals: List[List[float]],
    conditioning_mode: str,
    group_name: str,
    num_examples: int,
) -> Tuple[torch.Tensor, int, List[Dict], List[str]]:
    collector = FeatureCollector(inception, fid_device, fid_batch_size)
    predictions = []
    correct_count = 0
    example_paths = []

    for sample_idx, seed in enumerate(sample_seeds):
        image = generate_sample(
            pipe=pipe,
            geometry=geometry,
            prompt_parts=prompt_parts,
            seed=seed,
            guidance_scale=guidance,
            steps=steps,
            conditioning_intervals=conditioning_intervals,
            conditioning_mode=conditioning_mode,
        )
        collector.add(image)
        pred_class, pred_label = classify_image(
            image,
            classifier_processor,
            classifier_model,
            classifier_device,
        )
        correct = int(pred_class == class_id)
        correct_count += correct

        image_path = ""
        if sample_idx < num_examples:
            image_path = os.path.join(examples_dir, f"{group_name}_sample_{sample_idx:03d}_seed_{seed}.png")
            image.save(image_path)
            example_paths.append(image_path)

        predictions.append(
            {
                "sample_index": sample_idx,
                "seed": seed,
                "pred_class": pred_class,
                "pred_label": pred_label,
                "target_class": class_id,
                "correct": correct,
                "example_path": image_path,
            }
        )
        print(
            f"  [{group_name}] sample={sample_idx:04d} seed={seed} "
            f"pred={pred_class} correct={correct}"
        )

    return collector.result(), correct_count, predictions, example_paths


def run_benchmark(args):
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for debugging.")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.radius < 0:
        raise ValueError("--radius must be non-negative")
    if args.fid_batch_size < 1:
        raise ValueError("--fid-batch-size must be at least 1")
    if args.num_examples < 0:
        raise ValueError("--num-examples must be non-negative")

    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)
        torch.backends.cudnn.benchmark = False

    output_dir = resolve_output_dir(args.output_dir)
    dirs = prepare_dirs(output_dir)
    schedule_indices = parse_schedule_indices(args.schedule_indices, args.experiment)
    sample_seeds = [args.base_seed + sample_idx for sample_idx in range(args.num_samples)]
    classifier_device = resolve_classifier_device(args.classifier_device)
    fid_device = torch.device(args.fid_device)
    if fid_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--fid-device cuda requested but CUDA is unavailable")

    print(f"[config] experiment={args.experiment} output_dir={output_dir}")
    print(f"[config] schedules={schedule_indices} samples={args.num_samples} examples={args.num_examples}")
    print(f"[config] sample_seeds={sample_seeds[0]}..{sample_seeds[-1]}")
    print(f"[config] classifier_device={classifier_device} fid_device={fid_device}")

    pipe, geometry = build_pipeline(args)
    prompt_parts = encode_prompt_parts(pipe, args)
    classifier_processor, classifier_model = load_classifier(
        args.classifier_model_id,
        classifier_device=classifier_device,
        local_files_only=args.local_files_only,
    )
    inception = build_inception(fid_device)

    start_time = time.time()
    set_attention_schedule(pipe.transformer, args.radius, [])
    print("[baseline] global attention, conditioned always")
    baseline_features, baseline_correct, baseline_predictions, baseline_examples = generate_group(
        pipe=pipe,
        geometry=geometry,
        prompt_parts=prompt_parts,
        classifier_processor=classifier_processor,
        classifier_model=classifier_model,
        classifier_device=classifier_device,
        inception=inception,
        fid_device=fid_device,
        sample_seeds=sample_seeds,
        examples_dir=dirs["examples"],
        class_id=args.class_id,
        guidance=args.guidance,
        steps=args.steps,
        fid_batch_size=args.fid_batch_size,
        conditioning_intervals=[],
        conditioning_mode="always",
        group_name="baseline_global_conditioned",
        num_examples=args.num_examples,
    )
    baseline_mu, baseline_sigma = feature_stats(baseline_features)
    torch.save(
        {"features": baseline_features, "mu": baseline_mu, "sigma": baseline_sigma},
        os.path.join(dirs["features"], "baseline_global_conditioned_features.pt"),
    )
    baseline_error = 1.0 - baseline_correct / float(args.num_samples)
    print(f"[baseline] correct={baseline_correct}/{args.num_samples} error={baseline_error:.4f}")

    summary = []
    all_predictions: Dict[str, List[Dict]] = {"baseline_global_conditioned": baseline_predictions}
    all_example_paths = list(baseline_examples)

    for schedule_idx in schedule_indices:
        cfg = group_config(
            args.experiment,
            schedule_idx,
            args.radius,
            args.window_length,
            args.outside_conditioning,
        )
        interval = cfg["summary_interval"]
        set_attention_schedule(pipe.transformer, args.radius, cfg["local_intervals"])
        group_name = f"schedule_{schedule_idx:02d}"
        print(f"[schedule {schedule_idx}] {cfg['description']}")

        features, correct_count, predictions, example_paths = generate_group(
            pipe=pipe,
            geometry=geometry,
            prompt_parts=prompt_parts,
            classifier_processor=classifier_processor,
            classifier_model=classifier_model,
            classifier_device=classifier_device,
            inception=inception,
            fid_device=fid_device,
            sample_seeds=sample_seeds,
            examples_dir=dirs["examples"],
            class_id=args.class_id,
            guidance=args.guidance,
            steps=args.steps,
            fid_batch_size=args.fid_batch_size,
            conditioning_intervals=cfg["conditioning_intervals"],
            conditioning_mode=cfg["conditioning_mode"],
            group_name=group_name,
            num_examples=args.num_examples,
        )
        mu, sigma = feature_stats(features)
        fid = fid_score(baseline_mu, baseline_sigma, mu, sigma)
        torch.save(
            {"features": features, "mu": mu, "sigma": sigma, "fid": fid},
            os.path.join(dirs["features"], f"{group_name}_features.pt"),
        )
        error_rate = 1.0 - correct_count / float(args.num_samples)
        row = {
            "schedule_index": schedule_idx,
            "interval_start": interval[0],
            "interval_end": interval[1],
            "num_samples": args.num_samples,
            "correct_count": correct_count,
            "error_rate": error_rate,
            "fid_to_global_conditioned": fid,
        }
        summary.append(row)
        all_predictions[group_name] = predictions
        all_example_paths.extend(example_paths)
        print(
            f"[schedule {schedule_idx}] correct={correct_count}/{args.num_samples} "
            f"error={error_rate:.4f} fid={fid:.4f}"
        )

    summary.sort(key=lambda row: row["schedule_index"])
    metrics = {
        "config": {
            "experiment": args.experiment,
            "model_id": args.model_id,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "class_id": args.class_id,
            "guidance": args.guidance,
            "steps": args.steps,
            "radius": args.radius,
            "window_length": args.window_length,
            "outside_conditioning": args.outside_conditioning,
            "num_samples": args.num_samples,
            "num_examples": args.num_examples,
            "base_seed": args.base_seed,
            "sample_seeds": sample_seeds,
            "schedule_indices": schedule_indices,
            "classifier_model_id": args.classifier_model_id,
            "classifier_device": str(classifier_device),
            "fid_device": str(fid_device),
            "fid_batch_size": args.fid_batch_size,
            "geometry": geometry,
            "local_files_only": args.local_files_only,
            "cpu_offload": args.cpu_offload,
            "sequential_cpu_offload": args.sequential_cpu_offload,
            "disable_t5": args.disable_t5,
            "vae_tiling": args.vae_tiling,
            "vae_slicing": args.vae_slicing,
        },
        "baseline": {
            "name": "global_conditioned",
            "num_samples": args.num_samples,
            "correct_count": baseline_correct,
            "error_rate": baseline_error,
            "example_paths": [os.path.relpath(p, output_dir) for p in baseline_examples],
        },
        "summary": summary,
        "predictions": all_predictions,
        "elapsed_seconds": time.time() - start_time,
    }

    metrics_path = os.path.join(dirs["metrics"], f"sd3_{args.experiment}_metrics.json")
    summary_csv_path = os.path.join(dirs["metrics"], f"sd3_{args.experiment}_summary.csv")
    error_plot_path = os.path.join(dirs["plots"], f"sd3_{args.experiment}_classification_error.png")
    fid_plot_path = os.path.join(dirs["plots"], f"sd3_{args.experiment}_fid_to_baseline.png")
    grid_path = os.path.join(dirs["examples"], f"sd3_{args.experiment}_examples_grid.png")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    write_summary_csv(summary, summary_csv_path)
    plot_error(summary, error_plot_path, f"SD3 {args.experiment}: classifier error")
    plot_fid(summary, fid_plot_path, f"SD3 {args.experiment}: FID to global baseline")
    make_example_grid(all_example_paths, grid_path)

    print("schedule\twindow\tcorrect\terror\tfid")
    for row in summary:
        print(
            f"{row['schedule_index']}\t"
            f"[{row['interval_start']:.1f}, {row['interval_end']:.1f}]\t"
            f"{row['correct_count']}/{row['num_samples']}\t"
            f"{row['error_rate']:.4f}\t"
            f"{row['fid_to_global_conditioned']:.4f}"
        )
    print(f"Metrics saved to {metrics_path}")
    print(f"Summary CSV saved to {summary_csv_path}")
    print(f"Classifier error plot saved to {error_plot_path}")
    print(f"FID plot saved to {fid_plot_path}")
    print(f"Examples saved to {dirs['examples']}")


def parse_args():
    parser = argparse.ArgumentParser(description="SD3 schedule FID/classifier benchmark.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["local_attention", "conditioning_window", "sliding_global_window_local"],
        required=True,
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
    parser.add_argument("--radius", type=int, default=RADIUS)
    parser.add_argument("--window-length", type=float, default=0.3)
    parser.add_argument(
        "--outside-conditioning",
        type=str,
        choices=["conditional", "unconditional"],
        default="conditional",
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--schedule-indices", type=str, default="all")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
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


if __name__ == "__main__":
    run_benchmark(parse_args())
