"""Krea 2 (K2) adapter for Untwisting RoPE.

Krea 2 is a single-stream MMDiT (comfy/ldm/krea2/model.py :: SingleStreamDiT): text and image
tokens are concatenated into ONE sequence ``[context | img]`` and run through ``blocks[i]``
(SingleStreamBlock), each with a single ``attn`` (class Attention) doing GQA + per-head QK-norm +
3-axis RoPE + sigmoid-gated output. The image tokens are the TAIL of the sequence.

This mirrors the Z-Image adapter (combined-sequence + GQA) but matches Krea2's attention layout:
separate wq/wk/wv, qknorm, ``apply_rope(q, k, freqs)``, and the ``wo(out * sigmoid(gate))`` output.

The only thing the attention module can't see is where the image tokens start (it only receives the
combined ``x``). We recover that by wrapping the DiT ``_forward`` to stash the image-token count in
cfg; the image range is then ``(seqlen - imglen, seqlen)``.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import comfy.ldm.common_dit
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked
from einops import rearrange


ARCHITECTURE = "krea2"
DISPLAY_NAME = "Krea 2"
CONFIG_KEY = "untwisting_rope"

SUPPORTED_MODEL_CONFIG_CLASSES = {"Krea2"}
DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


def matches_model(model_info: dict[str, Any]) -> bool:
    if str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES:
        return True
    # Fallback on the unet image_model tag in case a repack reports a different config class.
    unet = model_info.get("unet_config", {})
    image_model = unet.get("image_model", "") if isinstance(unet, dict) else model_info.get("image_model", "")
    return str(image_model) == "krea2"


def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None, False
        try:
            obj = getattr(obj, part)
        except Exception:
            return None, False
    return obj, True


def find_diffusion_model(model_patcher: Any) -> Any:
    for path in DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError("Could not find ComfyUI BaseModel.diffusion_model for Krea 2.")


def _first_attention(dm: Any) -> Any:
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        raise RuntimeError("Krea 2 metadata lookup failed: dm.blocks is missing.")
    attn = getattr(blocks[0], "attn", None)
    if attn is None:
        raise RuntimeError("Krea 2 metadata lookup failed: blocks[0].attn is missing.")
    return attn


def _head_dim(dm: Any) -> int:
    return int(getattr(_first_attention(dm), "headdim"))


def _axes_dims(dm: Any) -> list[int]:
    # Matches SingleStreamDiT: axes = [headdim - 12*(hd//16), 6*(hd//16), 6*(hd//16)].
    hd = _head_dim(dm)
    unit = hd // 16
    axes = [hd - 12 * unit, 6 * unit, 6 * unit]
    if sum(axes) != hd:
        return [hd]
    return axes


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"architecture": ARCHITECTURE}
    if dm is not None:
        cfg["head_dim"] = _head_dim(dm)
        cfg["axes_dims"] = _axes_dims(dm)
    return cfg


def is_krea2_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Krea 2 attention modules are named blocks.N.attn."""
    parts = str(name).split(".")
    if len(parts) != 3 or parts[0] != "blocks" or parts[2] != "attn":
        return False
    try:
        idx = int(parts[1])
    except Exception:
        return False
    return int(min_layer) <= idx <= int(max_layer)


def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    return is_krea2_attention_name(name, min_layer, max_layer)


def block_index_from_name(name: str) -> int:
    parts = str(name).split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except Exception:
            return -1
    return -1


def is_krea2_attention_module(module: Any) -> bool:
    required = ("wq", "wk", "wv", "gate", "wo", "qknorm", "heads", "kvheads", "headdim")
    return all(hasattr(module, a) for a in required) and callable(getattr(module, "forward", None))


def prepare_reference_conditioning(ref_conditioning, dm, device, dtype, stats=None, label="", helpers=None):
    return ref_conditioning, "not-applicable"


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _install_imglen_probe(dm: Any) -> None:
    """Wrap the DiT _forward so it records the image-token count into cfg.

    Krea 2's image tokens are the tail of the joint sequence; the attention modules only see the
    combined tensor, so they can't compute the image range alone. We compute it once per forward
    here (same patchify math as SingleStreamDiT._forward) and stash it in cfg.
    """
    if getattr(dm, "_untwist_krea2_forward_probe", False):
        return
    patch = int(getattr(dm, "patch", 2))
    orig_forward = dm._forward

    # ponytail: signature-agnostic on purpose — only `x` and transformer_options are ours to read.
    # ComfyUI keeps adding positional params to Krea2's _forward (ref_latents landed in c9602625);
    # pinning the full signature here means every upstream change breaks the probe.
    def probed_forward(self, x, *args, **kwargs):
        transformer_options = kwargs.get('transformer_options')
        if transformer_options is None:
            # it is always the last positional arg; the others are tensors/lists/None
            transformer_options = next((a for a in reversed(args) if isinstance(a, dict)), None)
        cfg = transformer_options.get(CONFIG_KEY) if isinstance(transformer_options, dict) else None
        if cfg and cfg.get("enabled"):
            try:
                xt = x
                if xt.ndim == 5:
                    b5, c5, t5, h5, w5 = xt.shape
                    xt = xt.reshape(b5 * t5, c5, h5, w5)
                xt = comfy.ldm.common_dit.pad_to_patch_size(xt, (patch, patch))
                H, W = int(xt.shape[-2]), int(xt.shape[-1])
                cfg["krea2_imglen"] = (H // patch) * (W // patch)
            except Exception:
                cfg.pop("krea2_imglen", None)
        return orig_forward(x, *args, **kwargs)

    dm._forward = types.MethodType(probed_forward, dm)
    dm._untwist_krea2_forward_probe = True


def patch_attention_modules(dm: Any, stats: Any, helpers: dict[str, Any] | None = None):
    helpers = helpers or {}

    required_helpers = ("lerp", "build_frequency_scale_vector", "apply_qkv_shared_effects",
                        "apply_attention_output_shared_effects")
    missing = [n for n in required_helpers if not callable(helpers.get(n))]
    if missing:
        raise RuntimeError(f"Krea 2 adapter missing required helper(s): {missing}")

    build_frequency_scale_vector = helpers["build_frequency_scale_vector"]
    apply_qkv_shared_effects = helpers["apply_qkv_shared_effects"]
    apply_attention_output_shared_effects = helpers["apply_attention_output_shared_effects"]

    _install_imglen_probe(dm)

    matched = installed = restored = 0
    patched_names: list[str] = []

    for name, module in dm.named_modules():
        if not is_krea2_attention_name(name, 0, 999):
            continue
        if not is_krea2_attention_module(module):
            continue

        matched += 1
        patched_names.append(name)

        if hasattr(module, "_untwist_orig_forward"):
            module.forward = module._untwist_orig_forward
            restored += 1
        else:
            module._untwist_orig_forward = module.forward
        original_forward = module._untwist_orig_forward

        def make_forward(orig, module_name):
            def patched_forward(self, x, freqs=None, mask=None, transformer_options={}):
                cfg = transformer_options.get(CONFIG_KEY) if isinstance(transformer_options, dict) else None
                if not cfg or not cfg.get("enabled"):
                    return orig(x, freqs=freqs, mask=mask, transformer_options=transformer_options)

                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                if target_bsz <= 0:
                    raise RuntimeError(f"Krea 2 Untwisting enabled in {module_name}, but cross_batch_target_batch={target_bsz}.")
                if not torch.is_tensor(x) or x.ndim != 3:
                    raise RuntimeError(f"Krea 2 Untwisting expected x as [B,S,C] in {module_name}; got ndim={getattr(x,'ndim',None)}.")

                bsz, seqlen, _ = x.shape
                if bsz < target_bsz * 2:
                    raise RuntimeError(f"Krea 2 Untwisting expected target+reference batches in {module_name}; bsz={bsz}, target_bsz={target_bsz}.")

                block_idx = int(transformer_options.get("block_index", block_index_from_name(module_name)))
                active_blocks = cfg.get("active_blocks", set())
                if active_blocks and block_idx not in active_blocks:
                    return orig(x, freqs=freqs, mask=mask, transformer_options=transformer_options)

                # Image tokens are the tail of the joint [context | img] sequence.
                imglen = int(cfg.get("krea2_imglen", 0))
                if imglen <= 0 or imglen > seqlen:
                    raise RuntimeError(
                        f"Krea 2 Untwisting could not resolve the image token range in {module_name}: "
                        f"imglen={imglen}, seqlen={seqlen} (DiT _forward probe did not populate cfg)."
                    )
                img_s, img_e = seqlen - imglen, seqlen

                if hasattr(stats, "attn_calls"):
                    stats.attn_calls += 1

                heads, kvheads, headdim = int(self.heads), int(self.kvheads), int(self.headdim)
                q = rearrange(self.wq(x), "B L (H D) -> B H L D", H=heads)
                k = rearrange(self.wk(x), "B L (H D) -> B H L D", H=kvheads)
                v = rearrange(self.wv(x), "B L (H D) -> B H L D", H=kvheads)
                gate = self.gate(x)
                q, k = self.qknorm(q, k)

                q, k, v = apply_qkv_shared_effects(
                    q, k, v, cfg, target_bsz, module_name,
                    layout="BHSD", token_ranges=[(img_s, img_e)],
                )

                q, k = apply_rope(q, k, freqs)

                progress = float(cfg.get("progress", 0.0))
                high_scale = _lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
                low_scale = _lerp(cfg["low_scale_start"], cfg["low_scale_end"], progress)
                beta = float(cfg.get("beta", 2.0))

                scale_vec = build_frequency_scale_vector(
                    headdim, cfg.get("axes_dims") or _axes_dims(dm),
                    high_scale, low_scale, beta, k.device, k.dtype, runtime_cfg=cfg,
                ).view(1, 1, 1, headdim)

                ref_k = k[target_bsz:target_bsz * 2, :, img_s:img_e, :] * scale_vec
                ref_v = v[target_bsz:target_bsz * 2, :, img_s:img_e, :]

                def expand_kv(kt, vt):
                    if kvheads <= 0 or heads % kvheads != 0:
                        raise RuntimeError(f"Krea 2 cannot expand KV heads in {module_name}: heads={heads}, kvheads={kvheads}.")
                    rep = heads // kvheads
                    if rep == 1:
                        return kt, vt
                    return kt.repeat_interleave(rep, dim=1), vt.repeat_interleave(rep, dim=1)

                # Target stream: attends to its joint sequence + paired reference image K/V.
                q_t = q[:target_bsz]
                k_t, v_t = expand_kv(torch.cat([k[:target_bsz], ref_k], dim=2),
                                     torch.cat([v[:target_bsz], ref_v], dim=2))
                out_t = optimized_attention_masked(q_t, k_t, v_t, heads, None, skip_reshape=True,
                                                   transformer_options=transformer_options)

                # Reference stream: evaluated normally so later blocks get valid reference activations.
                q_r = q[target_bsz:target_bsz * 2]
                k_r, v_r = expand_kv(k[target_bsz:target_bsz * 2], v[target_bsz:target_bsz * 2])
                out_r = optimized_attention_masked(q_r, k_r, v_r, heads, None, skip_reshape=True,
                                                   transformer_options=transformer_options)

                out_t, out_r = apply_attention_output_shared_effects(
                    out_t, out_r, cfg, target_bsz, module_name,
                    layout="BSD", token_ranges=[(img_s, img_e)],
                )

                outs = [out_t, out_r]
                if bsz > target_bsz * 2:
                    q_e = q[target_bsz * 2:]
                    k_e, v_e = expand_kv(k[target_bsz * 2:], v[target_bsz * 2:])
                    outs.append(optimized_attention_masked(q_e, k_e, v_e, heads, None, skip_reshape=True,
                                                           transformer_options=transformer_options))

                out = torch.cat(outs, dim=0)
                return self.wo(out * torch.sigmoid(gate))
            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, name), module)
        installed += 1

    if installed <= 0:
        raise RuntimeError("Krea 2 adapter patch failed: no compatible blocks.N.attn modules were installed.")
    return matched, installed, restored, patched_names


def uses_reference_branch_kv() -> bool:
    return False


__all__ = [
    "ARCHITECTURE", "DISPLAY_NAME", "CONFIG_KEY", "SUPPORTED_MODEL_CONFIG_CLASSES",
    "matches_model", "find_diffusion_model", "default_runtime_cfg",
    "is_attention_name", "is_krea2_attention_name", "block_index_from_name",
    "is_krea2_attention_module", "prepare_reference_conditioning",
    "patch_attention_modules", "uses_reference_branch_kv",
]
