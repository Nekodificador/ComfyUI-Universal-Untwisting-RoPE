"""FLUX.1 adapter for Untwisting RoPE.

FLUX.1 (`Flux` / `FluxInpaint` / `FluxSchnell`, incl. finetunes like FLUX.1-dev and the depth-
conditioned FLUX.1-Depth-dev) shares the EXACT block architecture with FLUX.2: both instantiate
ComfyUI's `comfy.ldm.flux.layers.DoubleStreamBlock` / `SingleStreamBlock` (same `modulation`,
`img_mod`, `yak_mlp`, `mlp_hidden_dim_first`, `pre_norm` … attributes; the patched forward already
branches on the `modulation`/`yak_mlp` flags). The flux2 adapter's patch is written against that
shared module, so FLUX.1 reuses it verbatim — only the model-identity match differs.

# ponytail: thin re-export of the proven flux2 patch instead of copying ~600 lines. The only
# FLUX.1-specific bit is matches_model (disjoint config set from "Flux2" → no collision). Additive:
# it can't affect any other model, and FLUX.1 errored ("no adapter") before this existed.

Depth note: FLUX.1-Depth-dev is depth-conditioned at the MODEL level (the depth map is fed through
its normal conditioning). This adapter only adds the RoPE style-transfer patch; depth structure
comes from the model. That reproduces the paper's Figure 16 (method on FLUX.1-Depth-dev).
"""
from __future__ import annotations

from typing import Any

from . import flux2 as _flux2

ARCHITECTURE = "flux1"
DISPLAY_NAME = "FLUX.1"

# FLUX.1 family config classes (NOT "Flux2" — kept disjoint so identify() never sees two matches).
SUPPORTED_MODEL_CONFIG_CLASSES: set[str] = {"Flux", "FluxInpaint", "FluxSchnell"}


def matches_model(model_info: dict[str, Any]) -> bool:
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES


def is_model_identity(model_info: dict[str, Any]) -> bool:
    return matches_model(model_info)


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    cfg = _flux2.default_runtime_cfg(dm)
    cfg["architecture"] = ARCHITECTURE
    return cfg


# Reuse flux2's proven implementations verbatim (shared comfy.ldm.flux block structure).
find_diffusion_model = _flux2.find_diffusion_model
axes_dims_from_dm = _flux2.axes_dims_from_dm
head_dim_from_dm = _flux2.head_dim_from_dm
is_attention_name = _flux2.is_attention_name
block_index_from_name = _flux2.block_index_from_name
is_joint_attention = _flux2.is_joint_attention
prepare_reference_conditioning = _flux2.prepare_reference_conditioning
patch_attention_modules = _flux2.patch_attention_modules
uses_reference_branch_kv = _flux2.uses_reference_branch_kv


def describe_match(model_info: dict[str, Any]) -> str:
    supported = ", ".join(sorted(SUPPORTED_MODEL_CONFIG_CLASSES))
    return (f"{DISPLAY_NAME}: model_config_class={model_info.get('model_config_class', '')!r}, "
            f"supported_classes={{{supported}}}")


__all__ = [
    "ARCHITECTURE", "DISPLAY_NAME", "SUPPORTED_MODEL_CONFIG_CLASSES",
    "matches_model", "is_model_identity", "find_diffusion_model", "default_runtime_cfg",
    "axes_dims_from_dm", "head_dim_from_dm", "is_attention_name", "block_index_from_name",
    "is_joint_attention", "prepare_reference_conditioning", "patch_attention_modules",
    "uses_reference_branch_kv", "describe_match",
]
