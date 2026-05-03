# run_dit_local_global_clip.py
# ------------------------------------------------------------
# SD3/MMDiT text-to-image sampler with efficient local attention
# during chosen timestep intervals.
# ------------------------------------------------------------
import json
import os
import time

import inspect
import torch
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

from models.local_attention import current_t_norm, swap_in_efficient_local_attn
from models.local_attention_masked_sdpa import swap_in_masked_sdpa_local_attn

# ====================== USER PARAMS =========================
SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
if os.path.basename(SCRIPT_DIR) == "slurm_logs":
    SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
RADIUS    = int(os.getenv("RADIUS", "5"))  # attention window radius r (r=2 => 5x5 window)
try:
    LOCAL_INTERVALS = json.loads(os.getenv("LOCAL_INTERVALS", "[]"))
except Exception:
    LOCAL_INTERVALS = []
# LOCAL_INTERVALS: list of [start, end] intervals for local attention
STEPS     = int(os.getenv("STEPS", "40"))
SEED      = int(os.getenv("SEED", "123"))
HEIGHT    = int(os.getenv("HEIGHT", "0"))  # 0 = model default
WIDTH     = int(os.getenv("WIDTH", "0"))   # 0 = HEIGHT/model default

PROMPT    = os.getenv("PROMPT", "a golden retriever playing in a park, high detail, soft lighting")
PROMPT2   = os.getenv("PROMPT2", PROMPT)  # some SD3 checkpoints require three prompts
PROMPT3   = os.getenv("PROMPT3", PROMPT)
NEGATIVE_PROMPT = os.getenv("NEGATIVE_PROMPT", "")
NEGATIVE_PROMPT2 = os.getenv("NEGATIVE_PROMPT2", NEGATIVE_PROMPT)
NEGATIVE_PROMPT3 = os.getenv("NEGATIVE_PROMPT3", NEGATIVE_PROMPT)
GUIDANCE  = float(os.getenv("GUIDANCE", "3.0"))

MODEL_ID  = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers")
OUT_NAME_RAW = os.getenv("OUT_NAME", f"outputs/dit_local_clip_r{RADIUS}.png")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"}
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0").lower() in {"1", "true", "yes"}
SEQUENTIAL_CPU_OFFLOAD = os.getenv("SEQUENTIAL_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes"}
DISABLE_T5 = os.getenv("DISABLE_T5", os.getenv("DISABLE_TEXT_ENCODER_3", "0")).lower() in {"1", "true", "yes"}
FORCE_ATTENTION_SWAP = os.getenv("FORCE_ATTENTION_SWAP", "0").lower() in {"1", "true", "yes"}
VARIANT = os.getenv("VARIANT", "")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
LOCAL_ATTN_IMPL = os.getenv("LOCAL_ATTN_IMPL", "masked_sdpa").strip().lower()
# ===========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float16 if device.type == "cuda" else torch.float32


def normalize_local_attn_impl(value: str) -> str:
    aliases = {
        "masked_sdpa": "masked_sdpa",
        "masked": "masked_sdpa",
        "sdpa": "masked_sdpa",
        "gather": "gather",
        "original": "gather",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LOCAL_ATTN_IMPL='{value}'. Expected one of: masked_sdpa, masked, sdpa, gather, original."
        ) from exc


def swap_local_attention(transformer, h_tok: int, w_tok: int, radius: int, local_intervals):
    if LOCAL_ATTN_IMPL == "masked_sdpa":
        return swap_in_masked_sdpa_local_attn(
            transformer,
            h_tok,
            w_tok,
            r=radius,
            local_intervals=local_intervals,
        )
    return swap_in_efficient_local_attn(
        transformer,
        h_tok,
        w_tok,
        r=radius,
        local_intervals=local_intervals,
    )


LOCAL_ATTN_IMPL = normalize_local_attn_impl(LOCAL_ATTN_IMPL)


def make_t_normalizer(scheduler):
    ts = scheduler.timesteps.to("cpu")
    tmax = float(ts.max().item())
    tmin = float(ts.min().item())
    rng  = max(1.0, tmax - tmin)
    def to_unit_interval(t_int: float) -> float:
        return (float(t_int) - tmin) / rng
    return to_unit_interval


def maybe_scale_model_input(scheduler, latents, timestep):
    if hasattr(scheduler, "scale_model_input"):
        return scheduler.scale_model_input(latents, timestep)
    return latents


def resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


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
        raise ValueError(f"Model '{model_id}' has no text encoder/tokenizer components for prompt encoding")


def resolve_geometry(pipe):
    vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
    sample_size = int(getattr(pipe.transformer.config, "sample_size", 0) or 128)
    default_height = sample_size * vae_scale
    height = HEIGHT or default_height
    width = WIDTH or height

    if height % vae_scale != 0 or width % vae_scale != 0:
        raise ValueError(
            f"HEIGHT/WIDTH must be divisible by VAE scale factor {vae_scale}; got {height}x{width}"
        )

    latent_h = height // vae_scale
    latent_w = width // vae_scale
    patch = int(getattr(pipe.transformer.config, "patch_size", 1))
    if latent_h % patch != 0 or latent_w % patch != 0:
        raise ValueError(
            f"Latent size {latent_h}x{latent_w} must be divisible by transformer patch_size={patch}"
        )

    h_tok = latent_h // patch
    w_tok = latent_w // patch
    max_pos = getattr(pipe.transformer.config, "pos_embed_max_size", None)
    if max_pos is not None and (h_tok > int(max_pos) or w_tok > int(max_pos)):
        raise ValueError(
            f"Token grid {h_tok}x{w_tok} exceeds pos_embed_max_size={max_pos}; "
            "reduce HEIGHT/WIDTH or choose a model that supports this resolution."
        )

    return height, width, latent_h, latent_w, h_tok, w_tok


def ensure_t_tensor(t, scheduler, device):
    if not isinstance(t, torch.Tensor):
        return torch.tensor([t], device=device, dtype=scheduler.timesteps.dtype)
    if t.ndim == 0:
        return t.unsqueeze(0)
    return t


def fix_channels(noise_pred: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    if noise_pred.shape[1] == 2 * latents.shape[1]:
        noise_pred, _ = noise_pred.chunk(2, dim=1)
    return noise_pred


def set_sd3_timesteps(scheduler, steps: int, latents: torch.Tensor, patch_size: int, device):
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


def encode_prompts(pipe, do_cfg: bool):
    kwargs = dict(
        prompt=PROMPT,
        prompt_2=PROMPT2 if has_component(pipe, "text_encoder_2") else None,
        prompt_3=PROMPT3 if has_component(pipe, "text_encoder_3") else None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=NEGATIVE_PROMPT if do_cfg else None,
        negative_prompt_2=NEGATIVE_PROMPT2 if do_cfg and has_component(pipe, "text_encoder_2") else None,
        negative_prompt_3=NEGATIVE_PROMPT3 if do_cfg and has_component(pipe, "text_encoder_3") else None,
    )

    try:
        return pipe.encode_prompt(**kwargs, max_sequence_length=MAX_SEQUENCE_LENGTH)
    except TypeError as exc:
        if "max_sequence_length" not in str(exc):
            raise
        return pipe.encode_prompt(**kwargs)


def decode_to_pil(pipe, latents: torch.Tensor):
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
    image = image[0].permute(1, 2, 0).float().cpu().detach().numpy()
    from PIL import Image

    return Image.fromarray((image * 255).round().astype("uint8"))


# ---------------- Load pipeline components ----------------
load_kwargs = {
    "torch_dtype": dtype,
    "local_files_only": LOCAL_FILES_ONLY,
}
if VARIANT:
    load_kwargs["variant"] = VARIANT
if DISABLE_T5:
    load_kwargs["text_encoder_3"] = None
    load_kwargs["tokenizer_3"] = None

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    **load_kwargs,
)

validate_pipeline(pipe, MODEL_ID)

# Keep the model's default scheduler unless explicitly overridden.
if os.getenv("USE_DPMS", "").lower() in {"1", "true", "yes"}:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

height, width, latent_h, latent_w, Htok, Wtok = resolve_geometry(pipe)
print(
    f"[init] model={MODEL_ID} | image={height}x{width} | latent={latent_h}x{latent_w} "
    f"| token_grid={Htok}x{Wtok} | dtype={dtype} | cpu_offload={CPU_OFFLOAD} "
    f"| sequential_offload={SEQUENTIAL_CPU_OFFLOAD} | disable_t5={DISABLE_T5} "
    f"| local_attn_impl={LOCAL_ATTN_IMPL}"
)

if LOCAL_INTERVALS or FORCE_ATTENTION_SWAP:
    num_replaced = swap_local_attention(
        pipe.transformer,
        Htok,
        Wtok,
        radius=RADIUS,
        local_intervals=LOCAL_INTERVALS,
    )
else:
    num_replaced = 0
print(
    f"[init] replaced {num_replaced} attention modules (impl={LOCAL_ATTN_IMPL}, "
    f"r={RADIUS}, intervals={LOCAL_INTERVALS})"
)

if SEQUENTIAL_CPU_OFFLOAD and device.type == "cuda":
    pipe.enable_sequential_cpu_offload()
elif CPU_OFFLOAD and device.type == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)


# ---------------- Encode text prompts ----------------
g = torch.Generator(device=device).manual_seed(SEED)
do_cfg = GUIDANCE is not None and GUIDANCE > 1.0

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_prompts(pipe, do_cfg)
if do_cfg:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

batch_size = prompt_embeds.shape[0] // (2 if do_cfg else 1)

scheduler = pipe.scheduler

# Latent shape is (B, in_channels, H, W) where in_channels matches DiT config
in_ch = pipe.transformer.config.in_channels
latents = randn_tensor(
    (batch_size, in_ch, latent_h, latent_w),
    generator=g, device=device, dtype=dtype
)
timesteps = set_sd3_timesteps(scheduler, STEPS, latents, pipe.transformer.config.patch_size, device)
t_to_unit = make_t_normalizer(scheduler)


# ---------------- Manual sampling loop ----------------
t_start = time.time()
with torch.inference_mode():
    for i, t in enumerate(timesteps):
        t_norm = t_to_unit(float(t.item() if isinstance(t, torch.Tensor) else t))
        current_t_norm["value"] = t_norm

        t = ensure_t_tensor(t, scheduler, device)

        latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        latent_model_input = maybe_scale_model_input(scheduler, latent_model_input, t)
        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred_uncond = fix_channels(noise_pred_uncond, latents)
            noise_pred_text = fix_channels(noise_pred_text, latents)
            noise_pred = noise_pred_uncond + GUIDANCE * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = fix_channels(noise_pred, latents)

        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, t[0] if t.numel() == 1 else t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)


# ---------------- Decode through VAE ----------------
img = decode_to_pil(pipe, latents)
OUT_NAME = resolve_output_path(OUT_NAME_RAW)
out_dir = os.path.dirname(OUT_NAME)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
img.save(OUT_NAME)
dt = time.time() - t_start
print(
    f"[OK] saved {OUT_NAME} | steps={STEPS} | impl={LOCAL_ATTN_IMPL} "
    f"| r={RADIUS} | intervals={LOCAL_INTERVALS} | time={dt:.2f}s"
)
