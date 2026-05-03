"""
Local attention module for DiT (Diffusion Transformer).

This module provides efficient local attention that automatically 
transitions to global attention when the window covers all tokens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global variable to communicate t_norm to attention modules
current_t_norm = {"value": None}


class EfficientLocalSelfAttention(nn.Module):
    """
    Drop-in module to replace a DiT self-attention block.

    - Uses local attention during specified time intervals (controlled by local_intervals).
    - Uses global (dense) attention outside those intervals.
    - Automatically becomes global when window covers all tokens.
    - Reuses the original module's projections and output layers.
    
    Args:
        local_intervals: List of [start, end] pairs defining when to use local attention.
                        e.g., [[0.0, 0.2], [0.7, 1.0]] means local when t_norm in [0,0.2] or [0.7,1.0]
                        where t_norm=1.0 is pure noise and t_norm=0.0 is clean image.
    """
    def __init__(self, base_attn: nn.Module, h_tokens: int, w_tokens: int, r=2, local_intervals=None):
        super().__init__()
        self.base = base_attn
        self.h, self.w = h_tokens, w_tokens
        self.r = r
        
        # local_intervals: list of [start, end] pairs where local attention is used
        # e.g., [[0.0, 0.2], [0.7, 1.0]] means local at t in [0.0,0.2] or [0.7,1.0]
        if local_intervals is None:
            local_intervals = [[0.0,1.0]]  # Default: always global (empty list = never local)
        self.local_intervals = local_intervals
        
        # Validate intervals
        for interval in self.local_intervals:
            assert len(interval) == 2, f"Each interval must be [start, end], got {interval}"
            assert 0.0 <= interval[0] <= interval[1] <= 1.0, f"Invalid interval {interval}"

        # Extract attributes we need from the base attention
        required = ["to_q", "to_k", "to_v", "to_out", "heads"]
        for attr in required:
            if not hasattr(base_attn, attr):
                raise AttributeError(f"Base attention missing attr '{attr}'")
        self.to_q   = base_attn.to_q
        self.to_k   = base_attn.to_k
        self.to_v   = base_attn.to_v
        self.to_out = base_attn.to_out
        self.heads  = base_attn.heads
        
        # These might not exist in all attention modules
        self.head_dim = getattr(base_attn, 'head_dim', None)
        if self.head_dim is None:
            # Infer from projection dimensions
            # to_q projects from inner_dim to (heads * head_dim)
            inner_dim = base_attn.to_q.in_features
            self.head_dim = inner_dim // self.heads
        
        self.scale = getattr(base_attn, 'scale', 1.0 / math.sqrt(self.head_dim))

    @staticmethod
    def _shape_to_heads(x, n_heads, head_dim):
        # [B,N,C] -> [B,H,N,D]
        b, n, _ = x.shape
        return x.view(b, n, n_heads, head_dim).transpose(1, 2).contiguous()

    def _dense_attn(self, q, k, v, attention_mask=None):
        # q,k,v: [B,H,N,D]
        if attention_mask is None:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,N,N]
        scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)  # [B,H,N,D]

    def _get_neighbor_indices(self, h: int, w: int, r: int, device: torch.device):
        """
        Build a padded neighbor index table for each token along with a validity mask.
        neighbors: [N, K] (K = max neighbors across tokens, <= N)
        mask: [N, K] with True where the index is valid.
        """
        cache_key = (h, w, r, device)
        if not hasattr(self, "_neighbor_cache"):
            self._neighbor_cache = {}
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]

        N = h * w
        neighbor_lists = []
        max_len = 0
        for i in range(N):
            y, x = divmod(i, w)
            y0, y1 = max(0, y - r), min(h - 1, y + r)
            x0, x1 = max(0, x - r), min(w - 1, x + r)
            idxs = []
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    idxs.append(yy * w + xx)
            neighbor_lists.append(idxs)
            max_len = max(max_len, len(idxs))

        neighbors = torch.zeros((N, max_len), dtype=torch.long, device=device)
        mask = torch.zeros((N, max_len), dtype=torch.bool, device=device)
        for i, idxs in enumerate(neighbor_lists):
            length = len(idxs)
            neighbors[i, :length] = torch.as_tensor(idxs, device=device, dtype=torch.long)
            mask[i, :length] = True

        self._neighbor_cache[cache_key] = (neighbors, mask)
        return neighbors, mask

    def _local_attn(self, q, k, v):
        """
        Efficient local attention that automatically becomes global when window covers all tokens.
        Gathers only in-window keys/values per token to avoid full [N,N] score matrices.
        q,k,v: [B,H,N,D]
        """
        B, Hh, N, Dh = q.shape
        h, w = self.h, self.w
        if N != h * w:
            raise RuntimeError(
                f"Token count mismatch: got N={N} tokens but expected h*w={h*w}. "
                "Local attention requires specifying matching Htok/Wtok during swap."
            )

        neighbors, neighbor_mask = self._get_neighbor_indices(h, w, self.r, q.device)
        K = neighbors.shape[1]

        q_flat = q.reshape(B * Hh, N, Dh)
        k_flat = k.reshape(B * Hh, N, Dh)
        v_flat = v.reshape(B * Hh, N, Dh)

        idx = neighbors.unsqueeze(0).expand(B * Hh, -1, -1)  # [BH,N,K]
        batch_idx = torch.arange(B * Hh, device=q.device).view(-1, 1, 1)

        k_neigh = k_flat[batch_idx, idx]  # [BH,N,K,D]
        v_neigh = v_flat[batch_idx, idx]  # [BH,N,K,D]

        scores = torch.einsum("bnd,bnkd->bnk", q_flat, k_neigh) * self.scale

        if neighbor_mask is not None:
            mask = neighbor_mask.unsqueeze(0).expand(B * Hh, -1, -1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        out_flat = torch.einsum("bnk,bnkd->bnd", probs, v_neigh)
        out = out_flat.view(B, Hh, N, Dh)
        return out

    def _get_local_mask(self, h, w, r, device):
        """Create a local attention mask [N, N] with True for allowed connections"""
        cache_key = (h, w, r, device)
        if not hasattr(self, '_mask_cache'):
            self._mask_cache = {}
        
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
            
        N = h * w
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        
        for i in range(N):
            # Convert flat index to 2D coordinates
            i_y, i_x = i // w, i % w
            
            for j in range(N):
                # Convert flat index to 2D coordinates  
                j_y, j_x = j // w, j % w
                
                # Check if j is within the local window of i
                dy = abs(i_y - j_y)
                dx = abs(i_x - j_x)
                
                if max(dy, dx) <= r:  # Chebyshev distance
                    mask[i, j] = True
        
        self._mask_cache[cache_key] = mask
        return mask

    def _is_local(self, t_norm: float) -> bool:
        """Check if t_norm falls within any local attention interval"""
        for start, end in self.local_intervals:
            if start <= t_norm <= end:
                return True
        return False

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, timestep=None, t_norm: float = None, **kwargs):
        if t_norm is None:
            global current_t_norm
            t_norm = current_t_norm.get("value", None)
        
        if encoder_hidden_states is not None:
            context = encoder_hidden_states
        else:
            context = None

        q = self.to_q(hidden_states)
        if context is None:
            k = self.to_k(hidden_states)
            v = self.to_v(hidden_states)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        B, N, _ = hidden_states.shape
        Hh, Dh = self.heads, self.head_dim
        q = self._shape_to_heads(q, Hh, Dh)
        k = self._shape_to_heads(k, Hh, Dh)
        v = self._shape_to_heads(v, Hh, Dh)

        do_local = (t_norm is not None) and self._is_local(float(t_norm))

        if (context is None) and do_local and self.r >= 0:
            out = self._local_attn(q, k, v)
        else:
            out = self._dense_attn(q, k, v, attention_mask)

        out = out.transpose(1, 2).contiguous().view(B, N, Hh * Dh)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class EfficientLocalJointAttention(nn.Module):
    """
    Joint attention variant for SD3 blocks.

    - Applies local attention to the image queries (hidden_states) over image keys/values.
    - Always keeps text/condition tokens dense so conditioning is preserved.
    - Returns a tuple (hidden_out, context_out) like JointAttnProcessor2_0.
    """

    def __init__(self, base_attn: nn.Module, h_tokens: int, w_tokens: int, r=2, local_intervals=None):
        super().__init__()
        self.base = base_attn
        self.h, self.w = h_tokens, w_tokens
        self.r = r

        if local_intervals is None:
            local_intervals = [[0.0, 1.0]]
        self.local_intervals = local_intervals

        for interval in self.local_intervals:
            assert len(interval) == 2, f"Each interval must be [start, end], got {interval}"
            assert 0.0 <= interval[0] <= interval[1] <= 1.0, f"Invalid interval {interval}"

        req = ["to_q", "to_k", "to_v", "to_out", "heads"]
        for attr in req:
            if not hasattr(base_attn, attr):
                raise AttributeError(f"Base attention missing attr '{attr}'")

        # Projections for the image branch
        self.to_q = base_attn.to_q
        self.to_k = base_attn.to_k
        self.to_v = base_attn.to_v
        self.to_out = base_attn.to_out

        # Projections for the context branch (text)
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

        # Optional norms used in SD3 attention
        self.norm_q = getattr(base_attn, "norm_q", None)
        self.norm_k = getattr(base_attn, "norm_k", None)
        self.norm_added_q = getattr(base_attn, "norm_added_q", None)
        self.norm_added_k = getattr(base_attn, "norm_added_k", None)

        self.context_pre_only = getattr(base_attn, "context_pre_only", False)

    @staticmethod
    def _shape_to_heads(x, n_heads, head_dim):
        b, n, _ = x.shape
        return x.view(b, n, n_heads, head_dim).transpose(1, 2).contiguous()

    def _dense(self, q_all, k_all, v_all, attention_mask=None):
        # q_all/k_all/v_all: [B,H,N,D]
        if attention_mask is None:
            return F.scaled_dot_product_attention(q_all, k_all, v_all, dropout_p=0.0, is_causal=False)

        scores = torch.matmul(q_all, k_all.transpose(-2, -1)) * self.scale
        scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v_all)  # [B,H,N,D]

    def _get_neighbors(self, h: int, w: int, r: int, device: torch.device):
        cache_key = (h, w, r, device)
        if not hasattr(self, "_neighbor_cache"):
            self._neighbor_cache = {}
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]

        N = h * w
        neighbor_lists = []
        max_len = 0
        for i in range(N):
            y, x = divmod(i, w)
            y0, y1 = max(0, y - r), min(h - 1, y + r)
            x0, x1 = max(0, x - r), min(w - 1, x + r)
            idxs = []
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    idxs.append(yy * w + xx)
            neighbor_lists.append(idxs)
            max_len = max(max_len, len(idxs))

        neighbors = torch.zeros((N, max_len), dtype=torch.long, device=device)
        mask = torch.zeros((N, max_len), dtype=torch.bool, device=device)
        for i, idxs in enumerate(neighbor_lists):
            length = len(idxs)
            neighbors[i, :length] = torch.as_tensor(idxs, device=device, dtype=torch.long)
            mask[i, :length] = True

        self._neighbor_cache[cache_key] = (neighbors, mask)
        return neighbors, mask

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
            global current_t_norm
            t_norm = current_t_norm.get("value", None)

        B, N_img, _ = hidden_states.shape
        Hh, Dh = self.heads, self.head_dim

        # Image projections
        q_img = self.to_q(hidden_states)
        k_img = self.to_k(hidden_states)
        v_img = self.to_v(hidden_states)

        q_img = self._shape_to_heads(q_img, Hh, Dh)
        k_img = self._shape_to_heads(k_img, Hh, Dh)
        v_img = self._shape_to_heads(v_img, Hh, Dh)

        if self.norm_q is not None:
            q_img = self.norm_q(q_img)
        if self.norm_k is not None:
            k_img = self.norm_k(k_img)

        has_context = encoder_hidden_states is not None and self.add_q_proj is not None
        if has_context:
            q_ctx = self.add_q_proj(encoder_hidden_states)
            k_ctx = self.add_k_proj(encoder_hidden_states)
            v_ctx = self.add_v_proj(encoder_hidden_states)

            q_ctx = self._shape_to_heads(q_ctx, Hh, Dh)
            k_ctx = self._shape_to_heads(k_ctx, Hh, Dh)
            v_ctx = self._shape_to_heads(v_ctx, Hh, Dh)

            if self.norm_added_q is not None:
                q_ctx = self.norm_added_q(q_ctx)
            if self.norm_added_k is not None:
                k_ctx = self.norm_added_k(k_ctx)

            N_ctx = q_ctx.shape[2]
        else:
            q_ctx = k_ctx = v_ctx = None
            N_ctx = 0

        do_local = (t_norm is not None) and self._is_local(float(t_norm)) and self.r >= 0
        N_expected = self.h * self.w
        if N_img != N_expected and do_local:
            raise RuntimeError(
                f"Token count mismatch: got N={N_img} tokens but expected h*w={N_expected}."
            )

        if do_local:
            neighbors, neighbor_mask = self._get_neighbors(self.h, self.w, self.r, hidden_states.device)
            K = neighbors.shape[1]

        if do_local:
            # Gather local image keys/values
            q_img_flat = q_img  # [B,H,N_img,D]
            k_img_flat = k_img
            v_img_flat = v_img

            idx = neighbors.unsqueeze(0).unsqueeze(0)  # [1,1,N_img,K]
            idx = idx.expand(B, Hh, -1, -1)  # [B,H,N_img,K]
            batch_idx = torch.arange(B, device=hidden_states.device).view(B, 1, 1, 1)
            head_idx = torch.arange(Hh, device=hidden_states.device).view(1, Hh, 1, 1)

            k_local = k_img_flat[batch_idx, head_idx, idx]  # [B,H,N_img,K,D]
            v_local = v_img_flat[batch_idx, head_idx, idx]

            scores_local = torch.einsum("bhid,bhild->bhil", q_img_flat, k_local) * self.scale
            scores_local = scores_local.masked_fill(
                ~neighbor_mask.unsqueeze(0).unsqueeze(0).expand(B, Hh, -1, -1),
                torch.finfo(scores_local.dtype).min,
            )

            if has_context and N_ctx > 0:
                scores_text = torch.einsum("bhid,bhjd->bhij", q_img_flat, k_ctx) * self.scale  # [B,H,N_img,N_ctx]
                scores = torch.cat([scores_local, scores_text], dim=-1)
                probs = torch.softmax(scores, dim=-1)
                probs_local, probs_text = probs[..., :K], probs[..., K:]
                out_img_local = torch.einsum("bhik,bhikd->bhid", probs_local, v_local)
                out_img_text = torch.einsum("bhij,bhjd->bhid", probs_text, v_ctx)
                out_img = out_img_local + out_img_text
            else:
                probs_local = torch.softmax(scores_local, dim=-1)
                out_img = torch.einsum("bhik,bhikd->bhid", probs_local, v_local)

            # Text queries stay dense over full key set
            if has_context and N_ctx > 0:
                k_all = torch.cat([k_img, k_ctx], dim=2)  # [B,H,N_img+N_ctx,D]
                v_all = torch.cat([v_img, v_ctx], dim=2)
                scores_ctx = torch.matmul(q_ctx, k_all.transpose(-2, -1)) * self.scale
                probs_ctx = torch.softmax(scores_ctx, dim=-1)
                out_ctx = torch.matmul(probs_ctx, v_all)
            else:
                out_ctx = None
        else:
            # Dense joint attention (matches JointAttnProcessor2_0 behavior)
            q_all = torch.cat([q_img, q_ctx], dim=2) if has_context else q_img
            k_all = torch.cat([k_img, k_ctx], dim=2) if has_context else k_img
            v_all = torch.cat([v_img, v_ctx], dim=2) if has_context else v_img

            out_all = self._dense(q_all, k_all, v_all, attention_mask)
            out_img = out_all[:, :, :N_img, :]
            out_ctx = out_all[:, :, N_img:, :] if has_context else None

        # Reshape back and project
        out_img = out_img.transpose(1, 2).contiguous().view(B, N_img, Hh * Dh)
        out_img = self.to_out[0](out_img)
        out_img = self.to_out[1](out_img)

        if has_context and not self.context_pre_only:
            out_ctx = out_ctx.transpose(1, 2).contiguous().view(B, N_ctx, Hh * Dh)
            if self.to_add_out is not None:
                out_ctx = self.to_add_out(out_ctx)
        else:
            out_ctx = None

        return out_img, out_ctx


# ---------------- Helper: replace attention submodules ----------------
def looks_like_attn(m: nn.Module) -> bool:
    need = ["to_q","to_k","to_v","to_out","heads"]
    return all(hasattr(m, a) for a in need) and callable(getattr(m, "forward", None))


def swap_in_efficient_local_attn(transformer: nn.Module, Htok: int, Wtok: int, r: int, local_intervals=None):
    """
    Replace attention modules in transformer with EfficientLocalSelfAttention.
    
    Args:
        transformer: The transformer model to modify
        Htok, Wtok: Token grid dimensions (height, width)
        r: Attention window radius
        local_intervals: List of [start, end] pairs defining when to use local attention
                        e.g., [[0.0, 0.2], [0.7, 1.0]] uses local at t_norm in [0,0.2] or [0.7,1.0]
                        None defaults to [] (always global attention)
    
    Returns:
        int: Number of attention modules replaced
    """
    replaced = 0
    modules_to_replace = []
    for name, module in transformer.named_modules():
        if looks_like_attn(module) and 'attn' in name.lower():
            # Skip cross-attention blocks that attend to encoder text/context tokens.
            if getattr(module, "is_cross_attention", False):
                continue
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            modules_to_replace.append((parent_name, attr_name, module))
    
    for parent_name, attr_name, module in modules_to_replace:
        if parent_name:
            parent = dict(transformer.named_modules())[parent_name]
        else:
            parent = transformer
        is_joint = getattr(module, "add_k_proj", None) is not None and getattr(module, "add_v_proj", None) is not None
        if is_joint:
            new_attn = EfficientLocalJointAttention(module, Htok, Wtok, r=r, local_intervals=local_intervals)
        else:
            new_attn = EfficientLocalSelfAttention(module, Htok, Wtok, r=r, local_intervals=local_intervals)
        setattr(parent, attr_name, new_attn)
        replaced += 1
    
    return replaced


# Alias for backward compatibility
LocalSelfAttention = EfficientLocalSelfAttention
