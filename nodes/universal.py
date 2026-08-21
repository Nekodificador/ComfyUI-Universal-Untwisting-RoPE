"""Universal Untwisting RoPE node.

One node for every supported image model. It auto-detects the model (via the engine's
adapter registry), runs RF inversion and the RoPE attention patch internally, and
absorbs the reference boilerplate (resize + VAE-encode) so the graph stays clean.

Design: this composes the engine's `RFInversion` + `UntwistingRoPE` nodes — exactly the
pattern David's KREA2 wrapper proved — generalized across models and made self-contained.

Friendly controls:
  - structure_start/end -> high-frequency RoPE scale (reference shape/composition), start->end of denoise.
  - style_start/end     -> low-frequency RoPE scale (global feel / palette), start->end of denoise.
  - schedule_curve      -> easing preset shaping HOW structure/style ramp between start and end.
The remaining engine knobs (beta, adain, blocks, RF solver) are hidden and filled per-model by a
profile table. Raw control is available via the optional `extensions` input (the "Advanced Options"
node). The base engine classes are not registered as standalone nodes.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

import torch

from .. import RFInversion, UntwistingRoPE, SCHEDULE_CURVES
from .. import models as model_adapters
from ..image_helpers import encode_reference_to_latent


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
    beta=50.0, adain_strength=0.5,
    blocks='0-999', rf_mode='rf_gamma', gamma=0.5, pmi_alpha=0.0,
    otip_strength=0.35, otip_clip_norm=20.0,
)

_PROFILES: Dict[str, Dict[str, Any]] = {
    # David's KREA2 wrapper values (flowturbo_pc solver, tuned scales/blocks for KREA2).
    'krea2': dict(
        beta=0.99, adain_strength=0.85,
        blocks='7-27', rf_mode='flowturbo_pc', gamma=0.0, pmi_alpha=0.0,
        otip_strength=0.0, otip_clip_norm=20.0,
    ),
    # flux2 / flux2-klein, zimage, qwen_image, anima fall back to _DEFAULT_PROFILE for now.
}


class UntwistingRoPEUniversal:
    CATEGORY = 'Universal Untwisting RoPE'
    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT', 'VAE', 'STRING')
    RETURN_NAMES = ('model', 'positive', 'negative', 'latent', 'vae', 'info')
    FUNCTION = 'patch'
    DESCRIPTION = (
        'Universal training-free style transfer via Untwisting RoPE. Auto-detects the model '
        '(Flux.2/Klein, KREA2, Qwen-Image, Z-Image, Anima), runs RF inversion + the RoPE patch '
        'internally, and resizes + VAE-encodes the reference for you. Presampling-style: '
        'its slots mirror a presampling node output column (model/positive/negative/latent/image/vae), '
        'so the chain wires as short parallel cables; negative and vae pass straight through to the '
        'sampler directly. (Per the docs, the reference conditioning IS the target prompt — encoded once.)'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'positive': ('CONDITIONING', {
                    'tooltip': 'Target conditioning — this IS the reference conditioning '
                               '(per the docs). Feed it the "positive" output of a presampling '
                               'node (NKD Klein/Krea Tools), or a plain CLIP Text Encode.'
                }),
                'latent': ('LATENT', {
                    'tooltip': 'The generation/empty latent. The reference is matched to its EXACT '
                               'grid (required so RF inversion does not error), and it is passed '
                               'through to the "latent" output — wire that straight to the sampler.'
                }),
                'reference_image': ('IMAGE', {
                    'tooltip': 'The reference the style/structure is taken from. Resized to the '
                               'generation latent grid and VAE-encoded internally.'
                }),
                'vae': ('VAE', {
                    'tooltip': 'Encodes the reference. Also passed through to the "vae" output '
                               'so the decode after the sampler can take it from here.'
                }),
                'structure_start': ('FLOAT', {
                    'default': 1.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'High-frequency scale (structure) at the START of denoising. Higher '
                               'pulls more of the reference\'s shape/composition into the result.'
                               ' 0 = the reference contributes nothing in that band (off); '
                               'negatives INVERT the reference attention (anti-reference), '
                               'they do not reduce it further.'
                }),
                'structure_end': ('FLOAT', {
                    'default': 0.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'Structure (high-freq) at the END of denoising. Ramps from '
                               'structure_start to here along the trajectory (shaped by schedule_curve).'
                               ' 0 = the reference contributes nothing in that band (off); '
                               'negatives INVERT the reference attention (anti-reference), '
                               'they do not reduce it further.'
                }),
                'style_start': ('FLOAT', {
                    'default': 1.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'Low-frequency scale (style) at the START of denoising. Ramps to '
                               'style_end along the trajectory (shaped by schedule_curve).'
                               ' 0 = the reference contributes nothing in that band (off); '
                               'negatives INVERT the reference attention (anti-reference), '
                               'they do not reduce it further.'
                }),
                'style_end': ('FLOAT', {
                    'default': 3.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'Style (low-freq) at the END of denoising. Higher pulls more of the '
                               'reference\'s global feel/palette by the end.'
                               ' 0 = the reference contributes nothing in that band (off); '
                               'negatives INVERT the reference attention (anti-reference), '
                               'they do not reduce it further.'
                }),
                'schedule_curve': (list(SCHEDULE_CURVES), {
                    'default': 'linear',
                    'tooltip': 'How structure/style ramp from start to end over the trajectory. '
                               'linear = constant rate (original behavior). ease_in = slow then fast, '
                               'ease_out = fast then slow, ease_in_out/smoothstep = S-curve, '
                               'exponential = almost flat then sharp at the end, '
                               'logarithmic = the mirror: all the change up front, then flat.'
                }),
            },
            # Slot order mirrors the output column of presampling nodes (model / positive /
            # negative / latent / image / vae) so the whole chain wires as short parallel cables.
            'optional': {
                'negative': ('CONDITIONING', {
                    'tooltip': 'Passed straight through to the "negative" output. Connect the '
                               'presampling node\'s negative so it does not have to fly around this '
                               'node. If left empty, a neutral negative is derived from positive.'
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
        positive,
        latent: Dict[str, Any],
        reference_image,
        vae,
        structure_start: float,
        structure_end: float,
        style_start: float,
        style_end: float,
        schedule_curve: str = 'linear',
        negative=None,
        strength_curve=None,
        extensions: Optional[Dict[str, Any]] = None,
    ):
        verbose = True  # always on — debug info goes to console and the `info` output
        # The target conditioning IS the reference conditioning, per the docs.
        ref_conditioning = positive

        # ── Encode the reference onto the generation latent's EXACT grid ──────────
        # (mismatched grids make RF inversion error out)
        reference_latent = encode_reference_to_latent(reference_image, vae, latent)

        # ── Pick the per-model profile ───────────────────────────────────────────
        model_info = model_adapters.build_model_info(model)
        adapter = model_adapters.identify(model, model_info)
        arch = str(getattr(adapter, 'ARCHITECTURE', '') or '')
        prof = _PROFILES.get(arch, _DEFAULT_PROFILE)

        # ── Advanced overrides come from the Advanced Options node (sentinel -> model default) ──
        ext = extensions if isinstance(extensions, dict) else {}
        # looseness is a MULTIPLIER on the model's own beta — the absolute value differs ~50x
        # between models, so an absolute widget would mean opposite things per model.
        ov_beta_scale = ext.get('override_beta_scale', None)
        ov_blocks = ext.get('override_blocks', '')
        ov_adain = ext.get('override_adain_strength', None)
        beta_eff = float(prof['beta']) * (float(ov_beta_scale) if ov_beta_scale is not None else 1.0)
        blocks_eff = ov_blocks.strip() if isinstance(ov_blocks, str) and ov_blocks.strip() else str(prof['blocks'])
        adain_eff = float(ov_adain) if ov_adain is not None else float(prof['adain_strength'])

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
            high_scale_start=float(structure_start),
            high_scale_end=float(structure_end),
            low_scale_start=float(style_start),
            low_scale_end=float(style_end),
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
        unofficial['schedule_curve'] = schedule_curve
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
            f"structure(high) {float(structure_start):.3g}->{float(structure_end):.3g}  "
            f"style(low) {float(style_start):.3g}->{float(style_end):.3g}  curve={schedule_curve}\n"
            f"looseness(beta)={beta_eff:.3g}  blocks={blocks_eff}  tone_match(adain)={adain_eff:.3g}\n"
            f"rf_mode={prof['rf_mode']}  ref_grid={ref_grid}  "
            f"extensions={'yes' if extensions is not None else 'no'}  "
            f"strength_curve={'yes' if has_curve else 'no'}"
        )
        if verbose:
            print(f"[UntwistingRoPE/Universal] {info}")

        # Export the conditionings (the RF-inversion conditioning == the target/positive, per the
        # docs) and pass the latent through — the node feeds the sampler directly, presampling-style.
        out_positive = ref_conditioning
        # A connected negative is passed through untouched — presampling packs build a real one and
        # zeroing it out would throw their negative prompt away.
        out_negative = negative if negative is not None else _zero_out(out_positive)
        return (patched_model, out_positive, out_negative, latent, vae, info)


NODE_CLASS_MAPPINGS = {
    'UntwistingRoPEUniversal': UntwistingRoPEUniversal,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'UntwistingRoPEUniversal': 'Universal Untwisting RoPE',
}
