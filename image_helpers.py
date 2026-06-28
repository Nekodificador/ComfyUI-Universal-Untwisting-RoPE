"""Self-contained reference handling.

# ponytail: this subset is COPIED from ComfyUI-NKD-Klein-Tools/helpers.py rather than
# imported, so this pack stays standalone. Coupling two repos for a couple of small
# functions would be worse than the duplication. If these drift, the Klein original is source.

These absorb the boilerplate that used to litter RoPE graphs as auxiliary nodes
(Scale-Image + VAEEncode + ReferenceLatent): resize the reference to the generation latent's
exact grid and VAE-encode it into a ComfyUI LATENT dict. The reference grid MUST match the
generation grid or RF inversion errors with a spatial mismatch.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def _resize(image: torch.Tensor, width: int, height: int, mode: str = "bilinear") -> torch.Tensor:
    if image.shape[1] == height and image.shape[2] == width:
        return image
    x = image.permute(0, 3, 1, 2)
    if mode == "area":
        x = F.interpolate(x, size=(height, width), mode="area")
    else:
        x = F.interpolate(x, size=(height, width), mode=mode, align_corners=False)
    if mode == "bicubic":
        x = x.clamp(0.0, 1.0)
    return x.permute(0, 2, 3, 1)


def _resize_auto(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """area for downscale, bicubic for upscale; no-op when already at size."""
    if image.shape[1] == height and image.shape[2] == width:
        return image
    if height * width < image.shape[1] * image.shape[2]:
        return _resize(image, width, height, mode="area")
    return _resize(image, width, height, mode="bicubic")


def _vae_spatial_ratio(vae) -> Optional[int]:
    """Integer spatial compression of the VAE (pixels per latent cell), or None for the
    lambda/tuple forms used by some video VAEs."""
    r = getattr(vae, "downscale_ratio", 8)
    return int(r) if isinstance(r, int) else None


def match_latent_grid(reference_latent: dict, target_latent: dict) -> dict:
    """Force a reference LATENT onto the target LATENT's exact spatial grid.

    RF inversion requires the reference's latent grid to match the generation latent EXACTLY
    (same h×w). Latent-space interpolation is fine here — the reference only seeds the RF trajectory.
    """
    rs = reference_latent.get("samples")
    ts = target_latent.get("samples") if isinstance(target_latent, dict) else None
    if not torch.is_tensor(rs) or not torch.is_tensor(ts):
        return reference_latent
    th, tw = int(ts.shape[-2]), int(ts.shape[-1])
    if rs.shape[-2:] == (th, tw):
        return reference_latent
    matched = F.interpolate(rs, size=(th, tw), mode="bilinear", align_corners=False)
    return {**reference_latent, "samples": matched}


def encode_reference_to_latent(ref_image: torch.Tensor, vae, target_latent: dict) -> dict:
    """Resize a reference IMAGE to the generation latent's exact pixel footprint, VAE-encode it,
    and guarantee the resulting latent lands on the same grid as ``target_latent``."""
    if not (isinstance(target_latent, dict) and torch.is_tensor(target_latent.get("samples"))):
        raise RuntimeError("encode_reference_to_latent: a valid target latent is required to size the reference.")
    ts = target_latent["samples"]
    th, tw = int(ts.shape[-2]), int(ts.shape[-1])
    ratio = _vae_spatial_ratio(vae)
    img = _resize_auto(ref_image, tw * ratio, th * ratio) if ratio else ref_image
    out = {"samples": vae.encode(img[:, :, :, :3])}
    return match_latent_grid(out, target_latent)  # final exact-grid guarantee
