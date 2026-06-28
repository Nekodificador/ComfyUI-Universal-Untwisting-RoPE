"""Universal Untwisting RoPE node.

One node for every supported image model. It auto-detects the model (via the engine's
adapter registry), runs RF inversion and the RoPE attention patch internally, and
absorbs the reference boilerplate (resize + VAE-encode) so the graph stays clean.

Design: this composes the engine's `RFInversion` + `UntwistingRoPE` nodes — exactly the
pattern David's KREA2 wrapper proved — generalized across models and made self-contained.

Friendly controls:
  - structure -> high-frequency RoPE scale (how much the reference's shape/composition carries).
  - style     -> low-frequency RoPE scale at end of trajectory (global feel / palette).
The remaining engine knobs (beta, scale endpoints, adain, blocks, RF solver) are hidden and
filled per-model by a profile table. Raw control is available via the optional `extensions` input
(the "Advanced Options" node). The base engine classes are not registered as standalone nodes.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

import torch

from .. import RFInversion, UntwistingRoPE
from .. import models as model_adapters
from ..image_helpers import encode_reference_to_latent, match_latent_grid


def _encode_text(clip, text: str):
    """Inline CLIP Text Encode (same as ComfyUI's CLIPTextEncode) so the reference prompt
    is handled inside the node instead of needing a separate encode node."""
    if clip is None:
        raise RuntimeError("UntwistingRoPEUniversal: clip is None — cannot encode the prompt.")
    tokens = clip.tokenize(text or "")
    return clip.encode_from_tokens_scheduled(tokens)


def _zero_out(conditioning):
    """Same as ComfyUI's ConditioningZeroOut — a neutral negative from a positive."""
    out = []
    for t in conditioning:
        d = t[1].copy()
        pooled = d.get("pooled_output")
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t[0]), d])
    return out


# Per-model hidden-knob profiles. structure -> high_scale_start, style -> low_scale_end
# are the two visible sliders; everything below is the model-specific scaffolding.
# ponytail: these are TUNING CONSTANTS, not law. krea2 reproduces David's known-good wrapper;
# the rest start from the engine defaults. Tweak per model as real outputs dictate.
_DEFAULT_PROFILE: Dict[str, Any] = dict(
    beta=50.0, high_scale_end=0.0, low_scale_start=1.0, adain_strength=0.5,
    blocks='0-999', rf_mode='rf_gamma', gamma=0.5, pmi_alpha=0.0,
    otip_strength=0.35, otip_clip_norm=20.0,
)

_PROFILES: Dict[str, Dict[str, Any]] = {
    # David's KREA2 wrapper values (flowturbo_pc solver, tuned scales/blocks for KREA2).
    'krea2': dict(
        beta=0.99, high_scale_end=0.0, low_scale_start=0.0, adain_strength=0.85,
        blocks='7-27', rf_mode='flowturbo_pc', gamma=0.0, pmi_alpha=0.0,
        otip_strength=0.0, otip_clip_norm=20.0,
    ),
    # flux2 / flux2-klein, zimage, qwen_image, anima fall back to _DEFAULT_PROFILE for now.
}


class UntwistingRoPEUniversal:
    CATEGORY = 'Universal Untwisting RoPE'
    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT', 'STRING')
    RETURN_NAMES = ('model', 'positive', 'negative', 'latent', 'info')
    FUNCTION = 'patch'
    DESCRIPTION = (
        'Universal training-free style transfer via Untwisting RoPE. Auto-detects the model '
        '(Flux.2/Klein, KREA2, Qwen-Image, Z-Image, Anima), runs RF inversion + the RoPE patch '
        'internally, encodes the prompt and resizes/encodes the reference for you. Presampling-style: '
        'outputs model + positive/negative conditioning + the passed-through latent, so it feeds the '
        'sampler directly. (Per the docs, the reference conditioning IS the target prompt — encoded once.)'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'prompt': ('STRING', {
                    'multiline': True, 'dynamicPrompts': True, 'default': '',
                    'tooltip': 'Reference prompt. Encoded internally with the connected clip '
                               '(no separate CLIP Text Encode node needed). Often the reference\'s '
                               'own description, or left empty.'
                }),
                'structure': ('FLOAT', {
                    'default': 1.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'High-frequency scale = structure. Higher pulls more of the '
                               'reference\'s shape/composition into the result.'
                }),
                'style': ('FLOAT', {
                    'default': 3.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'Low-frequency scale = style. Higher pulls more of the '
                               'reference\'s global feel/palette by the end of denoising.'
                }),
            },
            'optional': {
                'clip': ('CLIP', {
                    'tooltip': 'CLIP/text-encoder used to encode the prompt internally. '
                               'Not needed if you connect ref_conditioning directly.'
                }),
                'reference_image': ('IMAGE', {
                    'tooltip': 'Raw reference. Resized to the generation latent\'s grid and '
                               'VAE-encoded internally. Requires vae + latent. Ignored if '
                               'reference_latent is given.'
                }),
                'vae': ('VAE',),
                'latent': ('LATENT', {
                    'tooltip': 'The generation/empty latent. The reference is matched to its EXACT '
                               'grid (required so RF inversion does not error), and it is passed '
                               'through to the "latent" output — wire that straight to the sampler.'
                }),
                'reference_latent': ('LATENT', {
                    'tooltip': 'Pre-encoded reference. Takes precedence over reference_image+vae.'
                }),
                'ref_conditioning': ('CONDITIONING', {
                    'tooltip': 'Pre-encoded reference conditioning. Overrides clip+prompt '
                               '(use to plug into Klein/Krea Tools pipelines).'
                }),
                'strength_curve': ('FLOAT', {
                    'forceInput': True,
                    'tooltip': 'Per-step strength multiplier (FLOAT list, e.g. from NKD Sigmas Curve). '
                               'Scales the whole RoPE effect over the denoising trajectory; flat 1.0 = '
                               'neutral. Multiplies, does not overwrite, structure/style.'
                }),
                'extensions': ('UNTWISTING_ROPE_EXTENSIONS', {
                    'tooltip': 'Advanced Options pack (the "Universal Untwisting RoPE (Advanced Options)" node): '
                               'fine-tuning knobs + core overrides (looseness, tone_match, blocks).'
                }),
            },
        }

    def patch(
        self,
        model,
        prompt: str,
        structure: float,
        style: float,
        clip=None,
        reference_image=None,
        vae=None,
        latent: Optional[Dict[str, Any]] = None,
        reference_latent: Optional[Dict[str, Any]] = None,
        ref_conditioning=None,
        strength_curve=None,
        extensions: Optional[Dict[str, Any]] = None,
    ):
        verbose = True  # always on — debug info goes to console and the `info` output
        # ── Resolve the reference conditioning (encode the prompt internally) ─────
        if ref_conditioning is None:
            if clip is None:
                raise RuntimeError(
                    'UntwistingRoPEUniversal: connect a clip (the prompt is encoded internally) '
                    'or pass ref_conditioning directly.'
                )
            ref_conditioning = _encode_text(clip, prompt)

        # ── Resolve the reference latent (self-contained or pre-encoded) ──────────
        # The reference must land on the generation latent's exact grid, so `latent` is required
        # whenever we encode/resize the reference here.
        if reference_latent is None:
            if reference_image is None or vae is None:
                raise RuntimeError(
                    'UntwistingRoPEUniversal: provide either reference_latent, or reference_image + vae.'
                )
            if latent is None:
                raise RuntimeError(
                    'UntwistingRoPEUniversal: connect the generation latent (input "latent") so the '
                    'reference can be matched to its grid.'
                )
            reference_latent = encode_reference_to_latent(reference_image, vae, latent)
        elif latent is not None:
            # Pre-encoded reference still must land on the generation grid.
            reference_latent = match_latent_grid(reference_latent, latent)

        # ── Pick the per-model profile ───────────────────────────────────────────
        model_info = model_adapters.build_model_info(model)
        adapter = model_adapters.identify(model, model_info)
        arch = str(getattr(adapter, 'ARCHITECTURE', '') or '')
        prof = _PROFILES.get(arch, _DEFAULT_PROFILE)

        # ── Advanced overrides come from the Advanced Options node (sentinel -> model default) ──
        ext = extensions if isinstance(extensions, dict) else {}
        ov_beta = ext.get('override_beta', 0.0)
        ov_blocks = ext.get('override_blocks', '')
        ov_adain = ext.get('override_adain_strength', None)
        ov_high_end = ext.get('override_high_scale_end', None)
        ov_low_start = ext.get('override_low_scale_start', None)
        beta_eff = float(ov_beta) if float(ov_beta) > 0.0 else float(prof['beta'])
        blocks_eff = ov_blocks.strip() if isinstance(ov_blocks, str) and ov_blocks.strip() else str(prof['blocks'])
        adain_eff = float(ov_adain) if ov_adain is not None else float(prof['adain_strength'])
        # Schedule endpoints: present only when custom_schedule is on; else per-model default.
        high_end_eff = float(ov_high_end) if ov_high_end is not None else float(prof['high_scale_end'])
        low_start_eff = float(ov_low_start) if ov_low_start is not None else float(prof['low_scale_start'])

        # ── Step 1: build the RF inversion trajectory latent ─────────────────────
        model_rf, rf_latent = RFInversion().build(
            model=model,
            reference_latent=reference_latent,
            ref_conditioning=ref_conditioning,
            rf_mode=prof['rf_mode'],
            gamma=prof['gamma'],
            pmi_alpha=prof['pmi_alpha'],
            otip_strength=prof['otip_strength'],
            otip_clip_norm=prof['otip_clip_norm'],
            verbose=verbose,
        )

        # ── Step 2: apply the Untwisting RoPE attention patch ────────────────────
        kwargs = dict(
            model=model_rf,
            rf_inversion=rf_latent,
            beta=beta_eff,
            high_scale_start=float(structure),
            high_scale_end=high_end_eff,
            low_scale_start=low_start_eff,
            low_scale_end=float(style),
            adain_strength=adain_eff,
            blocks=blocks_eff,
            verbose=verbose,
        )
        # Always pass an extensions dict so the recommended axis-0 default applies even WITHOUT the
        # Advanced node connected: the engine defaults axis0 to 'default', which the engine docs warn
        # is disastrous for non-Flux.1 models (e.g. Z-Image) — 'match_axes' is the recommended general
        # default for good out-of-the-box results. The per-step strength curve lives on this node now.
        unofficial = dict(ext)
        unofficial.setdefault('axis0_rope_mode', 'match_axes')
        if strength_curve is not None:
            unofficial['strength_curve'] = strength_curve
        kwargs['unofficial_extensions'] = unofficial
        result = UntwistingRoPE().patch(**kwargs)
        patched_model = result[0] if isinstance(result, tuple) else result

        # ── Debug info ────────────────────────────────────────────────────────────
        try:
            rs = reference_latent.get('samples')
            ref_grid = f"{int(rs.shape[-2])}x{int(rs.shape[-1])}" if rs is not None else "?"
        except Exception:
            ref_grid = "?"
        has_curve = strength_curve is not None
        info = (
            f"adapter={arch or '?'}  model_config={model_info.get('model_config_class', '?')}\n"
            f"structure(high) {float(structure):.3g}->{high_end_eff:.3g}  "
            f"style(low) {low_start_eff:.3g}->{float(style):.3g}\n"
            f"looseness(beta)={beta_eff:.3g}  blocks={blocks_eff}  tone_match(adain)={adain_eff:.3g}\n"
            f"rf_mode={prof['rf_mode']}  ref_grid={ref_grid}  "
            f"extensions={'yes' if extensions is not None else 'no'}  "
            f"strength_curve={'yes' if has_curve else 'no'}"
        )
        if verbose:
            print(f"[UntwistingRoPE/Universal] {info}")

        # Export the conditionings (the RF-inversion conditioning == the target/positive, per the
        # docs) and pass the latent through — the node feeds the sampler directly, presampling-style.
        positive = ref_conditioning
        negative = _zero_out(positive)
        return (patched_model, positive, negative, latent, info)


NODE_CLASS_MAPPINGS = {
    'UntwistingRoPEUniversal': UntwistingRoPEUniversal,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'UntwistingRoPEUniversal': 'Universal Untwisting RoPE',
}
