"""
Masked local attention modules built on torch SDPA.

These modules keep the original projection/output layers but route local
attention through torch.nn.functional.scaled_dot_product_attention with a
broadcastable boolean mask instead of explicitly gathering local windows.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.local_attention import current_t_norm


_GLOBAL_MASK_CACHE = {}


def _covers_full_grid(h: int, w: int, r: int) -> bool:
    return r >= max(h - 1, w - 1)


def _cache_key(prefix: str, h: int, w: int, r: int, device: torch.device, n_ctx: int = 0):
    return (prefix, h, w, r, str(device), n_ctx)


def _get_image_local_mask(h: int, w: int, r: int, device: torch.device) -> torch.Tensor:
    key = _cache_key("image", h, w, r, device)
    if key in _GLOBAL_MASK_CACHE:
        return _GLOBAL_MASK_CACHE[key]

    n_img = h * w
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
    dy = (coords[:, None, 0] - coords[None, :, 0]).abs()
    dx = (coords[:, None, 1] - coords[None, :, 1]).abs()
    mask = torch.maximum(dy, dx) <= r
    mask = mask.view(1, 1, n_img, n_img)
    _GLOBAL_MASK_CACHE[key] = mask
    return mask


def _get_joint_local_mask(h: int, w: int, r: int, n_ctx: int, device: torch.device) -> torch.Tensor:
    key = _cache_key("joint", h, w, r, device, n_ctx=n_ctx)
    if key in _GLOBAL_MASK_CACHE:
        return _GLOBAL_MASK_CACHE[key]

    image_mask = _get_image_local_mask(h, w, r, device).squeeze(0).squeeze(0)
    n_img = h * w
    n_total = n_img + n_ctx
    mask = torch.ones((n_total, n_total), dtype=torch.bool, device=device)
    mask[:n_img, :n_img] = image_mask
    mask = mask.view(1, 1, n_total, n_total)
    _GLOBAL_MASK_CACHE[key] = mask
    return mask


def _merge_attn_masks(
    local_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if local_mask is None:
        return attention_mask
    if attention_mask is None:
        return local_mask
    if attention_mask.dtype == torch.bool:
        return local_mask & attention_mask

    local_bias = torch.zeros(local_mask.shape, dtype=dtype, device=local_mask.device)
    local_bias.masked_fill_(~local_mask, torch.finfo(dtype).min)
    return local_bias + attention_mask


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )


class MaskedSDPALocalSelfAttention(nn.Module):
    """
    DiT self-attention replacement that applies a boolean local mask via SDPA.
    """

    def __init__(self, base_attn: nn.Module, h_tokens: int, w_tokens: int, r=2, local_intervals=None):
        super().__init__()
        self.base = base_attn
        self.h, self.w = h_tokens, w_tokens
        self.r = r

        if local_intervals is None:
            local_intervals = [[0.0, 1.0]]
        self.local_intervals = local_intervals

        required = ["to_q", "to_k", "to_v", "to_out", "heads"]
        for attr in required:
            if not hasattr(base_attn, attr):
                raise AttributeError(f"Base attention missing attr '{attr}'")
        self.to_q = base_attn.to_q
        self.to_k = base_attn.to_k
        self.to_v = base_attn.to_v
        self.to_out = base_attn.to_out
        self.heads = base_attn.heads

        self.head_dim = getattr(base_attn, "head_dim", None)
        if self.head_dim is None:
            inner_dim = base_attn.to_q.in_features
            self.head_dim = inner_dim // self.heads
        self.scale = getattr(base_attn, "scale", 1.0 / math.sqrt(self.head_dim))

    @staticmethod
    def _shape_to_heads(x, n_heads, head_dim):
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, n_heads, head_dim).transpose(1, 2).contiguous()

    def _is_local(self, t_norm: float) -> bool:
        for start, end in self.local_intervals:
            if start <= t_norm <= end:
                return True
        return False

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        t_norm: float = None,
        **kwargs,
    ):
        if t_norm is None:
            t_norm = current_t_norm.get("value", None)

        context = encoder_hidden_states if encoder_hidden_states is not None else None
        q = self.to_q(hidden_states)
        if context is None:
            k = self.to_k(hidden_states)
            v = self.to_v(hidden_states)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        bsz, n_tokens, _ = hidden_states.shape
        h_heads, d_head = self.heads, self.head_dim
        q = self._shape_to_heads(q, h_heads, d_head)
        k = self._shape_to_heads(k, h_heads, d_head)
        v = self._shape_to_heads(v, h_heads, d_head)

        do_local = (context is None) and (t_norm is not None) and self._is_local(float(t_norm)) and self.r >= 0
        if do_local and n_tokens != self.h * self.w:
            raise RuntimeError(
                f"Token count mismatch: got N={n_tokens} tokens but expected h*w={self.h * self.w}."
            )

        if do_local and not _covers_full_grid(self.h, self.w, self.r):
            local_mask = _get_image_local_mask(self.h, self.w, self.r, q.device)
            attn_mask = _merge_attn_masks(local_mask, attention_mask, q.dtype)
            out = _sdpa(q, k, v, attn_mask)
        else:
            out = _sdpa(q, k, v, attention_mask)

        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, h_heads * d_head)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class MaskedSDPALocalJointAttention(nn.Module):
    """
    SD3 joint-attention replacement that keeps text attention dense while masking
    only the image-to-image portion for image queries.
    """

    def __init__(self, base_attn: nn.Module, h_tokens: int, w_tokens: int, r=2, local_intervals=None):
        super().__init__()
        self.base = base_attn
        self.h, self.w = h_tokens, w_tokens
        self.r = r

        if local_intervals is None:
            local_intervals = [[0.0, 1.0]]
        self.local_intervals = local_intervals

        required = ["to_q", "to_k", "to_v", "to_out", "heads"]
        for attr in required:
            if not hasattr(base_attn, attr):
                raise AttributeError(f"Base attention missing attr '{attr}'")

        self.to_q = base_attn.to_q
        self.to_k = base_attn.to_k
        self.to_v = base_attn.to_v
        self.to_out = base_attn.to_out

        self.add_q_proj = getattr(base_attn, "add_q_proj", None)
        self.add_k_proj = getattr(base_attn, "add_k_proj", None)
        self.add_v_proj = getattr(base_attn, "add_v_proj", None)
        self.to_add_out = getattr(base_attn, "to_add_out", None)

        self.heads = base_attn.heads
        self.head_dim = getattr(base_attn, "head_dim", None)
        if self.head_dim is None:
            inner_dim = self.to_q.in_features
            self.head_dim = inner_dim // self.heads
        self.scale = getattr(base_attn, "scale", 1.0 / math.sqrt(self.head_dim))

        self.norm_q = getattr(base_attn, "norm_q", None)
        self.norm_k = getattr(base_attn, "norm_k", None)
        self.norm_added_q = getattr(base_attn, "norm_added_q", None)
        self.norm_added_k = getattr(base_attn, "norm_added_k", None)
        self.context_pre_only = getattr(base_attn, "context_pre_only", False)

    @staticmethod
    def _shape_to_heads(x, n_heads, head_dim):
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, n_heads, head_dim).transpose(1, 2).contiguous()

    def _is_local(self, t_norm: float) -> bool:
        for start, end in self.local_intervals:
            if start <= t_norm <= end:
                return True
        return False

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        t_norm: float = None,
        **kwargs,
    ):
        if t_norm is None:
            t_norm = current_t_norm.get("value", None)

        bsz, n_img, _ = hidden_states.shape
        h_heads, d_head = self.heads, self.head_dim

        q_img = self._shape_to_heads(self.to_q(hidden_states), h_heads, d_head)
        k_img = self._shape_to_heads(self.to_k(hidden_states), h_heads, d_head)
        v_img = self._shape_to_heads(self.to_v(hidden_states), h_heads, d_head)

        if self.norm_q is not None:
            q_img = self.norm_q(q_img)
        if self.norm_k is not None:
            k_img = self.norm_k(k_img)

        has_context = encoder_hidden_states is not None and self.add_q_proj is not None
        if has_context:
            q_ctx = self._shape_to_heads(self.add_q_proj(encoder_hidden_states), h_heads, d_head)
            k_ctx = self._shape_to_heads(self.add_k_proj(encoder_hidden_states), h_heads, d_head)
            v_ctx = self._shape_to_heads(self.add_v_proj(encoder_hidden_states), h_heads, d_head)
            if self.norm_added_q is not None:
                q_ctx = self.norm_added_q(q_ctx)
            if self.norm_added_k is not None:
                k_ctx = self.norm_added_k(k_ctx)
            n_ctx = q_ctx.shape[2]
        else:
            q_ctx = k_ctx = v_ctx = None
            n_ctx = 0

        do_local = (t_norm is not None) and self._is_local(float(t_norm)) and self.r >= 0
        if do_local and n_img != self.h * self.w:
            raise RuntimeError(
                f"Token count mismatch: got N={n_img} tokens but expected h*w={self.h * self.w}."
            )

        q_all = torch.cat([q_img, q_ctx], dim=2) if has_context else q_img
        k_all = torch.cat([k_img, k_ctx], dim=2) if has_context else k_img
        v_all = torch.cat([v_img, v_ctx], dim=2) if has_context else v_img

        if do_local and not _covers_full_grid(self.h, self.w, self.r):
            if has_context and n_ctx > 0:
                local_mask = _get_joint_local_mask(self.h, self.w, self.r, n_ctx, q_all.device)
            else:
                local_mask = _get_image_local_mask(self.h, self.w, self.r, q_all.device)
            attn_mask = _merge_attn_masks(local_mask, attention_mask, q_all.dtype)
            out_all = _sdpa(q_all, k_all, v_all, attn_mask)
        else:
            out_all = _sdpa(q_all, k_all, v_all, attention_mask)

        out_img = out_all[:, :, :n_img, :]
        out_ctx = out_all[:, :, n_img:, :] if has_context else None

        out_img = out_img.transpose(1, 2).contiguous().view(bsz, n_img, h_heads * d_head)
        out_img = self.to_out[0](out_img)
        out_img = self.to_out[1](out_img)

        if has_context and not self.context_pre_only:
            out_ctx = out_ctx.transpose(1, 2).contiguous().view(bsz, n_ctx, h_heads * d_head)
            if self.to_add_out is not None:
                out_ctx = self.to_add_out(out_ctx)
        else:
            out_ctx = None

        return out_img, out_ctx


def _looks_like_attn(module: nn.Module) -> bool:
    required = ["to_q", "to_k", "to_v", "to_out", "heads"]
    return all(hasattr(module, attr) for attr in required) and callable(getattr(module, "forward", None))


def swap_in_masked_sdpa_local_attn(
    transformer: nn.Module,
    h_tokens: int,
    w_tokens: int,
    r: int,
    local_intervals=None,
):
    """
    Replace attention blocks with SDPA-mask-based local attention variants.
    """
    replaced = 0
    modules_to_replace = []
    for name, module in transformer.named_modules():
        if _looks_like_attn(module) and "attn" in name.lower():
            if getattr(module, "is_cross_attention", False):
                continue
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            modules_to_replace.append((parent_name, attr_name, module))

    named_modules = dict(transformer.named_modules())
    for parent_name, attr_name, module in modules_to_replace:
        parent = named_modules[parent_name] if parent_name else transformer
        is_joint = getattr(module, "add_k_proj", None) is not None and getattr(module, "add_v_proj", None) is not None
        if is_joint:
            new_attn = MaskedSDPALocalJointAttention(module, h_tokens, w_tokens, r=r, local_intervals=local_intervals)
        else:
            new_attn = MaskedSDPALocalSelfAttention(module, h_tokens, w_tokens, r=r, local_intervals=local_intervals)
        setattr(parent, attr_name, new_attn)
        replaced += 1

    return replaced
