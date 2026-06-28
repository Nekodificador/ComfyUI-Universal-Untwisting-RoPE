"""Shared LTX-Video helpers for the consolidated Untwisting RoPE LTX node.

These functions were duplicated verbatim across David's three LTX packs (Basic / D-Structure /
LTX-zip). Hoisted here once; nodes/ltx.py is the single consolidated node that uses them.

LTX applies RoPE INSIDE CrossAttention.forward (before optimized_attention), so the classic
attn1_patch can't intercept it — we use patches_replace['dit'] per-block hooks instead. The PE
tuple is (cos_freq, sin_freq, split_mode_bool).
"""
from __future__ import annotations
import copy
from typing import Any, Tuple

import torch


def coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on', 'y', 't')
    return bool(value)


def get_diffusion_model(model: Any) -> Any:
    try:
        return model.get_model_object('diffusion_model')
    except Exception:
        pass
    try:
        return model.model.diffusion_model
    except Exception:
        pass
    return model


def sigma_from_timestep(timestep: Any) -> float:
    if torch.is_tensor(timestep):
        t = timestep.flatten()
        return float(t[0]) if t.numel() > 0 else 0.0
    return float(timestep)


def clone_model_options(opts: dict) -> dict:
    new_opts: dict = {}
    for k, v in opts.items():
        if k == 'transformer_options':
            new_opts[k] = dict(v)
        elif isinstance(v, dict):
            new_opts[k] = copy.copy(v)
        else:
            new_opts[k] = v
    return new_opts


# ─── LTX 3D RoPE frequency scaling ───────────────────────────────────────────

def _scale_pe_interleaved(cos_freq, sin_freq, high_scale, low_scale):
    """Scale spatial (h, w) components in interleaved-mode LTX PE.

    Last-dim pattern in groups of 6: [t_i, t_i, h_i, h_i, w_i, w_i], larger i = higher freq.
    Scales only h/w: upper-half i → high_scale (fine detail), lower-half → low_scale (coarse
    layout). Temporal (t) left untouched to preserve video coherence.
    """
    D = cos_freq.shape[-1]
    n_freq = D // 6
    if n_freq == 0:
        return cos_freq, sin_freq
    mid = max(1, n_freq // 2)
    cos_out = cos_freq.clone()
    sin_out = sin_freq.clone()
    for i in range(n_freq):
        scale = high_scale if i >= mid else low_scale
        if abs(scale - 1.0) < 1e-7:
            continue
        h0 = 6 * i + 2
        cos_out[..., h0:h0 + 2] = cos_freq[..., h0:h0 + 2] * scale
        sin_out[..., h0:h0 + 2] = sin_freq[..., h0:h0 + 2] * scale
        w0 = 6 * i + 4
        cos_out[..., w0:w0 + 2] = cos_freq[..., w0:w0 + 2] * scale
        sin_out[..., w0:w0 + 2] = sin_freq[..., w0:w0 + 2] * scale
    return cos_out, sin_out


def _scale_pe_split(cos_freq, sin_freq, high_scale, low_scale):
    """Split mode (B, H, T, head_dim//2): uniform low/high split on the last dim."""
    D = cos_freq.shape[-1]
    mid = max(1, D // 2)
    cos_out = cos_freq.clone()
    sin_out = sin_freq.clone()
    if abs(low_scale - 1.0) > 1e-7:
        cos_out[..., :mid] = cos_freq[..., :mid] * low_scale
        sin_out[..., :mid] = sin_freq[..., :mid] * low_scale
    if abs(high_scale - 1.0) > 1e-7:
        cos_out[..., mid:] = cos_freq[..., mid:] * high_scale
        sin_out[..., mid:] = sin_freq[..., mid:] * high_scale
    return cos_out, sin_out


def scale_ltx_pe(pe: tuple, high_scale: float, low_scale: float) -> tuple:
    """Dispatch to interleaved/split PE scaler based on the split_mode flag."""
    cos_freq, sin_freq = pe[0], pe[1]
    split_mode = pe[2] if len(pe) > 2 else False
    if split_mode:
        cos_out, sin_out = _scale_pe_split(cos_freq, sin_freq, high_scale, low_scale)
    else:
        cos_out, sin_out = _scale_pe_interleaved(cos_freq, sin_freq, high_scale, low_scale)
    return (cos_out, sin_out, split_mode)


def adain_full(target, ref_mean, ref_std, strength):
    """Full AdaIN with clamped std-ratio and linear blend by `strength` ∈ [0,1].

    transferred = (x - tgt_mean) * clamp(ref_std/tgt_std, 0.3, 3.0) + ref_mean
    out = (1-strength)*x + strength*transferred. Applied at ONE block (no /n_blocks).
    """
    tgt_mean = target.mean(dim=1, keepdim=True)
    tgt_std = target.std(dim=1, keepdim=True) + 1e-5
    std_ratio = torch.clamp(ref_std / tgt_std, 0.3, 3.0)
    transferred = (target - tgt_mean) * std_ratio + ref_mean
    s = max(0.0, min(1.0, float(strength)))
    return target * (1.0 - s) + transferred * s


def extend_pe(ref_pe_stored, tgt_pe, B: int, device, dtype):
    """Prepend ref PE to target PE along the sequence dim (handles 3D and 4D PE).
    Returns extended PE tuple, or None on failure (caller then skips token injection)."""
    if ref_pe_stored is None:
        return None
    if not (isinstance(ref_pe_stored, (list, tuple)) and len(ref_pe_stored) >= 2):
        return None
    r_cos_raw, r_sin_raw = ref_pe_stored[0], ref_pe_stored[1]
    if r_cos_raw is None or r_sin_raw is None:
        return None
    try:
        r_cos = r_cos_raw.to(device=device, dtype=dtype)
        r_sin = r_sin_raw.to(device=device, dtype=dtype)
        split_mode = ref_pe_stored[2] if len(ref_pe_stored) > 2 else False
        seq_dim = 2 if r_cos.dim() == 4 else 1
        expand_sz = [B] + [-1] * (r_cos.dim() - 1)
        r_cos = r_cos.expand(*expand_sz)
        r_sin = r_sin.expand(*expand_sz)
        ext_cos = torch.cat([r_cos, tgt_pe[0]], dim=seq_dim)
        ext_sin = torch.cat([r_sin, tgt_pe[1]], dim=seq_dim)
        return (ext_cos, ext_sin, split_mode)
    except Exception:
        return None
