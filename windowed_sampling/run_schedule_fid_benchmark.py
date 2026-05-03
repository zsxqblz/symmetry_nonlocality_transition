"""
Schedule benchmark for ImageNet DiT.

Modes:
- local_attention: schedule i uses local attention with radius r only in
  [i / 10, (i + 1) / 10], with class conditioning active at every step.
- conditioning_window: schedule i uses global attention at every step, but
  class conditioning is active only in [i / 10, (i + 1) / 10]; outside the
  window the null class is used without CFG.

Both modes generate an always-global, always-conditioned baseline with the
same sample seeds and compute FID from each schedule to that baseline.
"""
import argparse
import csv
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
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoImageProcessor, AutoModelForImageClassification


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from models.local_attention import current_t_norm, swap_in_efficient_local_attn


MODEL_ID = "facebook/DiT-XL-2-256"
CLASS_ID = 207
CLS_MODEL_ID = "microsoft/resnet-50"
GUIDANCE = 4.0
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


def ensure_t_tensor(t, scheduler):
    if not isinstance(t, torch.Tensor):
        return torch.tensor([t], device=device, dtype=scheduler.timesteps.dtype)
    if t.ndim == 0:
        return t.unsqueeze(0)
    return t


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


def prepare_dirs(output_dir: str) -> Dict[str, str]:
    dirs = {
        "root": output_dir,
        "baseline": os.path.join(output_dir, "baseline_global_conditioned"),
        "samples": os.path.join(output_dir, "samples"),
        "examples": os.path.join(output_dir, "examples"),
        "features": os.path.join(output_dir, "features"),
        "metrics": os.path.join(output_dir, "metrics"),
        "plots": os.path.join(output_dir, "plots"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def build_pipeline(model_id: str, local_files_only: bool, radius: int):
    pipe = DiTPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.transformer.eval()
    pipe.vae.eval()

    patch = pipe.transformer.config.patch_size
    res = pipe.transformer.config.sample_size
    h_tok, w_tok = res // patch, res // patch
    num_replaced = swap_in_efficient_local_attn(
        pipe.transformer,
        h_tok,
        w_tok,
        r=radius,
        local_intervals=[],
    )
    print(
        f"[init] model={model_id} token_grid={h_tok}x{w_tok} "
        f"dtype={dtype} replaced_attn={num_replaced}"
    )
    return pipe


def set_attention_schedule(transformer, radius: int, local_intervals: List[List[float]]) -> int:
    count = 0
    for module in transformer.modules():
        if hasattr(module, "local_intervals") and hasattr(module, "r"):
            module.r = radius
            module.local_intervals = local_intervals
            count += 1
    return count


def decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    with torch.no_grad():
        image = pipe.vae.decode(latents / scale).sample
    image = (image.clamp(-1, 1) + 1) / 2.0
    image_np = image[0].permute(1, 2, 0).float().cpu().detach().numpy()
    return Image.fromarray((image_np * 255).round().astype("uint8"))


def generate_sample(
    pipe,
    seed: int,
    class_id: int,
    guidance_scale: float,
    steps: int,
    conditioning_intervals: List[List[float]],
    conditioning_mode: str,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(steps, device=device)
    t_to_unit = make_t_normalizer(scheduler)

    in_ch = pipe.transformer.config.in_channels
    res = pipe.transformer.config.sample_size
    latents = randn_tensor(
        (1, in_ch, res, res),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma

    num_classes = getattr(pipe.transformer.config, "num_labels", 1000)
    cond_label = torch.tensor([class_id], device=device, dtype=torch.long)
    uncond_label = torch.full_like(cond_label, num_classes)

    with torch.inference_mode():
        for t in scheduler.timesteps:
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
                class_labels_input = torch.cat([uncond_label, cond_label], dim=0)
                timestep_input = torch.cat([t, t], dim=0)
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(
                        latent_model_input,
                        timestep_input,
                    )
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    class_labels=class_labels_input,
                ).sample
                uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
                model_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                class_label = cond_label if use_conditioning else uncond_label
                latent_model_input = latents
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                model_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t,
                    class_labels=class_label,
                ).sample

            if model_pred.shape[1] == 2 * latents.shape[1]:
                model_pred, _ = model_pred.chunk(2, dim=1)

            step_out = scheduler.step(model_pred, t[0] if t.shape[0] > 1 else t, latents)
            latents = step_out.prev_sample

    return decode_latents(pipe, latents)


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
        self.features.append(
            inception_features_from_pils(self.pending, self.inception, self.fid_device)
        )
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


def fid_score(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    mu_gen: torch.Tensor,
    sigma_gen: torch.Tensor,
) -> float:
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


def load_summary_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "schedule_index": int(row["schedule_index"]),
                    "interval_start": float(row["interval_start"]),
                    "interval_end": float(row["interval_end"]),
                    "num_samples": int(row["num_samples"]),
                    "correct_count": int(row["correct_count"]),
                    "error_rate": float(row["error_rate"]),
                    "fid_to_global_conditioned": float(row["fid_to_global_conditioned"]),
                }
            )
    return rows


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
    classifier_processor,
    classifier_model,
    classifier_device: torch.device,
    inception,
    fid_device: torch.device,
    sample_seeds: List[int],
    output_dir: str,
    class_id: int,
    guidance: float,
    steps: int,
    fid_batch_size: int,
    conditioning_intervals: List[List[float]],
    conditioning_mode: str,
    save_all_images: bool,
    group_name: str,
    target_class_for_error: int,
) -> Tuple[torch.Tensor, int, List[Dict], str]:
    os.makedirs(output_dir, exist_ok=True)
    collector = FeatureCollector(inception, fid_device, fid_batch_size)
    predictions = []
    correct_count = 0
    example_path = ""

    for sample_idx, seed in enumerate(sample_seeds):
        image = generate_sample(
            pipe=pipe,
            seed=seed,
            class_id=class_id,
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
        correct = int(pred_class == target_class_for_error)
        correct_count += correct

        image_name = f"sample_{sample_idx:04d}_seed_{seed}.png"
        image_path = os.path.join(output_dir, image_name)
        if save_all_images:
            image.save(image_path)
        if sample_idx == 0:
            example_path = os.path.join(output_dir, f"{group_name}_example.png")
            image.save(example_path)

        predictions.append(
            {
                "sample_index": sample_idx,
                "seed": seed,
                "pred_class": pred_class,
                "pred_label": pred_label,
                "target_class": target_class_for_error,
                "correct": correct,
                "image_path": image_path if save_all_images else "",
            }
        )
        print(
            f"  [{group_name}] sample={sample_idx:04d} seed={seed} "
            f"pred={pred_class} correct={correct}"
        )

    features = collector.result()
    return features, correct_count, predictions, example_path


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

    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)
        torch.backends.cudnn.benchmark = False

    output_dir = resolve_output_dir(args.output_dir)
    dirs = prepare_dirs(output_dir)
    schedule_indices = parse_schedule_indices(args.schedule_indices, args.experiment)
    sample_seeds = [args.base_seed + sample_idx for sample_idx in range(args.num_samples)]
    append_existing = bool(getattr(args, "append_existing", False))
    skip_existing_schedules = bool(getattr(args, "skip_existing_schedules", append_existing))
    classifier_device = resolve_classifier_device(args.classifier_device)
    fid_device = torch.device(args.fid_device)
    if fid_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--fid-device cuda requested but CUDA is unavailable")

    print(f"[config] experiment={args.experiment} output_dir={output_dir}")
    print(f"[config] schedules={schedule_indices} samples={args.num_samples}")
    print(f"[config] sample_seeds={sample_seeds[0]}..{sample_seeds[-1]}")
    print(f"[config] classifier_device={classifier_device} fid_device={fid_device}")

    pipe = build_pipeline(args.model_id, args.local_files_only, args.radius)
    classifier_processor, classifier_model = load_classifier(
        args.classifier_model_id,
        classifier_device=classifier_device,
        local_files_only=args.local_files_only,
    )
    inception = build_inception(fid_device)

    start_time = time.time()
    metrics_path = os.path.join(dirs["metrics"], f"{args.experiment}_metrics.json")
    summary_csv_path = os.path.join(dirs["metrics"], f"{args.experiment}_summary.csv")
    error_plot_path = os.path.join(dirs["plots"], f"{args.experiment}_classification_error.png")
    fid_plot_path = os.path.join(dirs["plots"], f"{args.experiment}_fid_to_baseline.png")
    grid_path = os.path.join(dirs["examples"], f"{args.experiment}_examples_grid.png")
    baseline_feature_path = os.path.join(dirs["features"], "baseline_global_conditioned_features.pt")

    baseline_correct = None
    baseline_error = None
    baseline_features = None
    baseline_mu = None
    baseline_sigma = None
    summary = []
    all_predictions: Dict[str, List[Dict]] = {}
    example_paths = []
    existing_metrics = None

    if append_existing and os.path.exists(metrics_path) and os.path.exists(summary_csv_path) and os.path.exists(baseline_feature_path):
        with open(metrics_path) as f:
            existing_metrics = json.load(f)
        existing_config = existing_metrics.get("config", {})
        if int(existing_config.get("num_samples", args.num_samples)) != args.num_samples:
            raise ValueError("append-existing requires matching num_samples")
        if int(existing_config.get("base_seed", args.base_seed)) != args.base_seed:
            raise ValueError("append-existing requires matching base_seed")
        if int(existing_config.get("steps", args.steps)) != args.steps:
            raise ValueError("append-existing requires matching steps")
        if float(existing_config.get("window_length", args.window_length)) != float(args.window_length):
            raise ValueError("append-existing requires matching window_length")
        if str(existing_config.get("outside_conditioning", args.outside_conditioning)) != str(args.outside_conditioning):
            raise ValueError("append-existing requires matching outside_conditioning")

        baseline_payload = torch.load(baseline_feature_path, map_location="cpu")
        baseline_features = baseline_payload["features"]
        baseline_mu = baseline_payload.get("mu")
        baseline_sigma = baseline_payload.get("sigma")
        if baseline_mu is None or baseline_sigma is None:
            baseline_mu, baseline_sigma = feature_stats(baseline_features)

        baseline = existing_metrics.get("baseline", {})
        baseline_correct = int(baseline["correct_count"])
        baseline_error = float(baseline["error_rate"])
        summary = load_summary_csv(summary_csv_path)
        all_predictions = existing_metrics.get("predictions", {})
        example_paths = sorted(
            path for path in [os.path.join(dirs["examples"], name) for name in os.listdir(dirs["examples"])] if os.path.isfile(path)
        )
        print(
            f"[append] reusing baseline and existing summary from {output_dir} "
            f"existing_schedules={[row['schedule_index'] for row in summary]}"
        )
    else:
        set_attention_schedule(pipe.transformer, args.radius, [])
        print("[baseline] global attention, conditioned always")
        baseline_features, baseline_correct, baseline_predictions, baseline_example = generate_group(
            pipe=pipe,
            classifier_processor=classifier_processor,
            classifier_model=classifier_model,
            classifier_device=classifier_device,
            inception=inception,
            fid_device=fid_device,
            sample_seeds=sample_seeds,
            output_dir=dirs["baseline"],
            class_id=args.class_id,
            guidance=args.guidance,
            steps=args.steps,
            fid_batch_size=args.fid_batch_size,
            conditioning_intervals=[],
            conditioning_mode="always",
            save_all_images=args.save_all_images,
            group_name="baseline_global_conditioned",
            target_class_for_error=args.class_id,
        )
        baseline_mu, baseline_sigma = feature_stats(baseline_features)
        torch.save(
            {"features": baseline_features, "mu": baseline_mu, "sigma": baseline_sigma},
            baseline_feature_path,
        )
        baseline_error = 1.0 - baseline_correct / float(args.num_samples)
        print(
            f"[baseline] correct={baseline_correct}/{args.num_samples} "
            f"error={baseline_error:.4f}"
        )
        summary = []
        all_predictions = {"baseline_global_conditioned": baseline_predictions}
        example_paths = [baseline_example]

    existing_schedule_indices = {int(row["schedule_index"]) for row in summary}
    if skip_existing_schedules:
        schedule_indices = [idx for idx in schedule_indices if idx not in existing_schedule_indices]
        print(f"[append] pending schedules={schedule_indices}")

    for schedule_idx in schedule_indices:
        cfg = group_config(
            args.experiment,
            schedule_idx,
            args.radius,
            args.window_length,
            args.outside_conditioning,
        )
        intervals = cfg["summary_interval"]
        set_attention_schedule(pipe.transformer, args.radius, cfg["local_intervals"])
        schedule_dir = os.path.join(dirs["samples"], f"schedule_{schedule_idx:02d}")
        print(f"[schedule {schedule_idx}] {cfg['description']}")

        features, correct_count, predictions, example_path = generate_group(
            pipe=pipe,
            classifier_processor=classifier_processor,
            classifier_model=classifier_model,
            classifier_device=classifier_device,
            inception=inception,
            fid_device=fid_device,
            sample_seeds=sample_seeds,
            output_dir=schedule_dir,
            class_id=args.class_id,
            guidance=args.guidance,
            steps=args.steps,
            fid_batch_size=args.fid_batch_size,
            conditioning_intervals=cfg["conditioning_intervals"],
            conditioning_mode=cfg["conditioning_mode"],
            save_all_images=args.save_all_images,
            group_name=f"schedule_{schedule_idx:02d}",
            target_class_for_error=args.class_id,
        )
        mu, sigma = feature_stats(features)
        fid = fid_score(baseline_mu, baseline_sigma, mu, sigma)
        torch.save(
            {"features": features, "mu": mu, "sigma": sigma, "fid": fid},
            os.path.join(dirs["features"], f"schedule_{schedule_idx:02d}_features.pt"),
        )

        error_rate = 1.0 - correct_count / float(args.num_samples)
        row = {
            "schedule_index": schedule_idx,
            "interval_start": intervals[0],
            "interval_end": intervals[1],
            "num_samples": args.num_samples,
            "correct_count": correct_count,
            "error_rate": error_rate,
            "fid_to_global_conditioned": fid,
        }
        summary.append(row)
        all_predictions[f"schedule_{schedule_idx:02d}"] = predictions
        example_paths.append(example_path)
        print(
            f"[schedule {schedule_idx}] correct={correct_count}/{args.num_samples} "
            f"error={error_rate:.4f} fid={fid:.4f}"
        )

    summary.sort(key=lambda row: row["schedule_index"])
    metrics = {
        "config": {
            "experiment": args.experiment,
            "model_id": args.model_id,
            "class_id": args.class_id,
            "guidance": args.guidance,
            "steps": args.steps,
            "radius": args.radius,
            "window_length": args.window_length,
            "outside_conditioning": args.outside_conditioning,
            "num_samples": args.num_samples,
            "base_seed": args.base_seed,
            "sample_seeds": sample_seeds,
            "schedule_indices": schedule_indices,
            "classifier_model_id": args.classifier_model_id,
            "classifier_device": str(classifier_device),
            "fid_device": str(fid_device),
            "fid_batch_size": args.fid_batch_size,
            "local_files_only": args.local_files_only,
            "save_all_images": args.save_all_images,
            "append_existing": append_existing,
            "skip_existing_schedules": skip_existing_schedules,
        },
        "baseline": {
            "name": "global_conditioned",
            "num_samples": args.num_samples,
            "correct_count": baseline_correct,
            "error_rate": baseline_error,
            "example_path": (
                existing_metrics.get("baseline", {}).get("example_path", "")
                if existing_metrics is not None
                else os.path.relpath(example_paths[0], output_dir)
            ),
        },
        "summary": summary,
        "predictions": all_predictions,
        "elapsed_seconds": time.time() - start_time,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    write_summary_csv(summary, summary_csv_path)
    plot_error(summary, error_plot_path, f"{args.experiment}: classifier error")
    plot_fid(summary, fid_plot_path, f"{args.experiment}: FID to global baseline")
    make_example_grid(example_paths, grid_path)

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
    parser = argparse.ArgumentParser(description="Schedule FID/classifier benchmark.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["local_attention", "conditioning_window", "sliding_global_window_local"],
        required=True,
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
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
        default=True,
    )
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--skip-existing-schedules", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
