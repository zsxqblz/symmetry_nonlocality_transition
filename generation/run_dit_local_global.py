# run_dit_local_global.py
# ------------------------------------------------------------
# One-file generator with efficient local attention for DiT.
# Replaces attention modules (no monkey-patching of methods).
# ------------------------------------------------------------
import os, time, math, warnings, types
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

# Import local attention module
from models.local_attention import (
    EfficientLocalSelfAttention,
    current_t_norm,
    swap_in_efficient_local_attn,
)

# ====================== USER PARAMS =========================
RADIUS    = 5       # <-- attention window radius r (r=2 => 5x5 window)
# LOCAL_INTERVALS: list of [start, end] pairs where local attention is used
# t_norm=0.0 is final step (clean), t_norm=1.0 is first step (pure noise)
# Examples:
#   [] - always global (default)
#   [[0.7, 1.0]] - local only in early/noisy phase
#   [[0.0, 0.2], [0.7, 1.0]] - local in early (noisy) and late (refinement)
#   [[0.0, 1.0]] - always local
LOCAL_INTERVALS = [[0.0,0.2]]
# LOCAL_INTERVALS = [[0.2, 0.5]]
# LOCAL_INTERVALS = []
STEPS     = 40      # Reduced for testing
SEED      = 123

# Conditioning
CONDITION = True    # True: class-conditional; False: effectively unconditional
CLASS_ID  = 207     # ImageNet "golden retriever" when CONDITION=True
GUIDANCE  = 4.0     # guidance scale; ignored when CONDITION=False (set to 1.0)

MODULE_DIR = Path(__file__).resolve().parent

MODEL_ID  = os.getenv("MODEL_ID", "facebook/DiT-XL-2-256")
OUT_NAME  = os.getenv(
    "OUT_NAME",
    str(MODULE_DIR / "outputs" / f"dit_local_r{RADIUS}{'_cond' if CONDITION else '_uncond'}.png"),
)
# ===========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float16 if device.type == "cuda" else torch.float32


# ---------------- Normalizer for scheduler timesteps ----------------
def make_t_normalizer(scheduler) -> callable:
    ts = scheduler.timesteps.to("cpu")
    tmax = float(ts.max().item())
    tmin = float(ts.min().item())
    rng  = max(1.0, tmax - tmin)
    def to_unit_interval(t_int: float) -> float:
        return (float(t_int) - tmin) / rng
    return to_unit_interval

# ---------------- Load pipeline components ----------------
pipe = DiTPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    local_files_only=True,
).to(device)

# Use a fast sampler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Compute token grid from DiT config
patch = pipe.transformer.config.patch_size
res   = pipe.transformer.config.sample_size
Htok, Wtok = res // patch, res // patch

# Replace attention modules with efficient local/global variant
num_replaced = swap_in_efficient_local_attn(pipe.transformer, Htok, Wtok, r=RADIUS, local_intervals=LOCAL_INTERVALS)
print(f"[init] replaced {num_replaced} attention modules (r={RADIUS}, intervals={LOCAL_INTERVALS})")

# ---------------- Manual sampling loop ----------------
g = torch.Generator(device=device).manual_seed(SEED)

# Class conditioning
if CONDITION:
    class_labels = torch.tensor([CLASS_ID], device=device, dtype=torch.long)
    guidance_scale = GUIDANCE
else:
    class_labels = torch.zeros(1, device=device, dtype=torch.long)  # dummy label
    guidance_scale = 1.0

scheduler = pipe.scheduler
scheduler.set_timesteps(STEPS, device=device)

# Latent shape is (B, in_channels, H, W) where in_channels is usually 4 (VAE latents)
in_ch = pipe.transformer.config.in_channels
latents = randn_tensor(
    (1, in_ch, res, res),
    generator=g, device=device, dtype=dtype
)
# Scale by sigma like SD pipelines (API parity)
if hasattr(scheduler, "init_noise_sigma"):
    latents = latents * scheduler.init_noise_sigma

# timestep → [0,1] normalizer
t_to_unit = make_t_normalizer(scheduler)

# Guidance handling: DiT is class-conditional; we implement CFG by
# one forward for cond and one for uncond (label = num_classes) if guidance_scale > 1.
use_cfg = guidance_scale is not None and guidance_scale > 1.0
num_classes = pipe.transformer.config.num_labels if hasattr(pipe.transformer.config, 'num_labels') else 1000
uncond_labels = torch.full_like(class_labels, num_classes) if use_cfg else None

t_start = time.time()
with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        t_norm = t_to_unit(float(t.item() if isinstance(t, torch.Tensor) else t))
        current_t_norm["value"] = t_norm
        
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=device, dtype=torch.long)
        elif t.ndim == 0:
            t = t.unsqueeze(0)

        if use_cfg:
            latent_model_input = torch.cat([latents, latents], dim=0)
            class_labels_input = torch.cat([uncond_labels, class_labels], dim=0)
            timestep_input = torch.cat([t, t], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep_input)
            
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                class_labels=class_labels_input,
            ).sample
            
            uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
            model_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            
            if model_pred.shape[1] == 2 * latents.shape[1]:
                model_pred, _ = model_pred.chunk(2, dim=1)
        else:
            latent_model_input = scheduler.scale_model_input(latents, t)
            model_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                class_labels=class_labels,
            ).sample
            
            if model_pred.shape[1] == 2 * latents.shape[1]:
                model_pred, _ = model_pred.chunk(2, dim=1)

        step_out = scheduler.step(model_pred, t[0] if t.shape[0] > 1 else t, latents)
        latents = step_out.prev_sample

# Decode through VAE
# Match typical scaling used in SD-like pipelines
scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
latents_dec = latents / scale
image = pipe.vae.decode(latents_dec).sample  # [B, 3, H, W] in [-1,1]
image = (image.clamp(-1, 1) + 1) / 2.0
image = image[0].permute(1, 2, 0).float().cpu().detach().numpy()

# Save with PIL
from PIL import Image

OUT_PATH = Path(OUT_NAME).expanduser()
if not OUT_PATH.is_absolute():
    OUT_PATH = (Path.cwd() / OUT_PATH).resolve()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
img = Image.fromarray((image * 255).round().astype("uint8"))
img.save(OUT_PATH)
dt = time.time() - t_start
print(f"[OK] saved {OUT_PATH} | steps={STEPS} | r={RADIUS} | intervals={LOCAL_INTERVALS} | time={dt:.2f}s")
