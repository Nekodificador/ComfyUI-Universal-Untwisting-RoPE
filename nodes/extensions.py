"""Advanced options pack for the universal Untwisting RoPE.

Bundles every advanced control into one optional node so the main node stays clean:
  - the engine's fine-tuning knobs (renamed to human terms: color/texture/bleeding/adherence),
  - the core overrides that otherwise use per-model defaults (looseness=beta, tone_match=adain,
    blocks).

Outputs an UNTWISTING_ROPE_EXTENSIONS dict. The universal node reads the override_* keys for the
core params and forwards the rest to the engine as unofficial extensions (which ignores the extras).
"""
from __future__ import annotations

from .. import UnofficialExtensions


class UntwistingRoPEExtensions:
    CATEGORY = 'Universal Untwisting RoPE'
    RETURN_TYPES = ('UNTWISTING_ROPE_EXTENSIONS',)
    RETURN_NAMES = ('extensions',)
    FUNCTION = 'build'
    DESCRIPTION = (
        'Advanced options for Untwisting RoPE — fine-tuning knobs plus core overrides '
        '(looseness, tone_match, blocks). Plug into the universal node\'s "extensions" input.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'color_transfer': ('FLOAT', {
                    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Match the result\'s color/tone to the reference (post-attention AdaIN).'
                }),
                'texture_transfer': ('FLOAT', {
                    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Transfer the reference\'s texture (variance-gated V AdaIN).'
                }),
                'bleeding_fix': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Inject the reference only where target and reference align '
                               '(cosine-gated V) — reduces style bleeding into unrelated areas.'
                }),
                'style_adherence': ('FLOAT', {
                    'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Project target keys toward the reference key subspace — stronger '
                               'adherence to the reference (can rigidify; keep low).'
                }),
            },
            'optional': {
                # ── Core overrides (sentinel = use the universal node\'s per-model default) ──
                'looseness': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 100.0, 'step': 0.01,
                    'tooltip': 'How literally the reference structure is copied (engine: beta). '
                               'Higher = looser/more creative; lower = more literal. Range is '
                               'technical, not 0-1. 0 = use the model default.'
                }),
                'tone_match': ('FLOAT', {
                    'default': 1.0, 'min': -1.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Match the reference\'s tone/contrast/color statistics (engine: '
                               'adain_strength). Default 1.0 = full (validated). '
                               '-1 = keep the per-model default · 0 = off · 0–1 = custom.'
                }),
                'blocks': ('STRING', {
                    'default': '',
                    'tooltip': 'Block ranges to patch, e.g. "7-27" or "0-8,28-37". '
                               'Empty = use the model default.'
                }),
                # ── Schedule endpoints. Off = model defaults; on = use the two values below. ──
                'custom_schedule': ('BOOLEAN', {
                    'default': False,
                    'tooltip': 'Use custom schedule endpoints below. Off = per-model defaults.'
                }),
                'structure_end': ('FLOAT', {
                    'default': 0.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'Where structure (high-freq) ends up by the end of denoising '
                               '(engine: high_scale_end). structure is its start. '
                               'Only used when custom_schedule is on.'
                }),
                'style_start': ('FLOAT', {
                    'default': 0.0, 'min': -4.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'Where style (low-freq) starts at the beginning of denoising '
                               '(engine: low_scale_start). style is its end. '
                               'Only used when custom_schedule is on.'
                }),
                'axis0_rope_mode': (['default', 'match_axes', 'constant'], {
                    'default': 'match_axes',
                    'tooltip': 'Axis-0 RoPE behavior. match_axes is a good default.'
                }),
                'axis0_rope_scale': ('FLOAT', {
                    'default': 1.0, 'min': 0.0, 'max': 8.0, 'step': 0.01,
                    'tooltip': 'Only used when axis0_rope_mode = constant.'
                }),
            },
        }

    def build(
        self,
        color_transfer: float = 1.0,
        texture_transfer: float = 1.0,
        bleeding_fix: float = 0.0,
        style_adherence: float = 0.35,
        looseness: float = 0.0,
        tone_match: float = 1.0,
        blocks: str = '',
        custom_schedule: bool = False,
        structure_end: float = 0.0,
        style_start: float = 0.0,
        axis0_rope_mode: str = 'match_axes',
        axis0_rope_scale: float = 1.0,
    ):
        data = UnofficialExtensions().build(
            post_attention_adain_strength=float(color_transfer),
            variance_gated_v_adain=float(texture_transfer),
            cosine_gated_v_injection=float(bleeding_fix),
            key_subspace_alignment=float(style_adherence),
            axis0_rope_mode=axis0_rope_mode,
            axis0_rope_scale=float(axis0_rope_scale),
        )[0]
        # Core overrides (universal node reads these; the engine ignores the override_* keys).
        data['override_beta'] = float(looseness)
        data['override_adain_strength'] = float(tone_match)
        data['override_blocks'] = str(blocks)
        # Schedule endpoints only override when explicitly enabled — no magic sentinel value.
        if custom_schedule:
            data['override_high_scale_end'] = float(structure_end)
            data['override_low_scale_start'] = float(style_start)
        return (data,)


class UntwistingRoPEAdvancedSettingsSimple:
    """Minimal "Advanced Options" — David's preferred 4-slider layout.

    Same backend keys as ``UntwistingRoPEExtensions`` but:
      • Only the 4 artistic knobs are exposed (no core overrides, no
        custom_schedule, no axis-0 RoPE controls).
      • All defaults are 0.0 — start neutral and dial things in by hand,
        instead of inheriting active engine defaults.
      • ``bleeding_factor_fix`` and ``color_transfer`` allow slider values up
        to 2 (the engine still clamps to [0, 1]; the wider slider gives David
        the visual UX he prefers).
      • ``advanced_mixer`` is a fine-grained 0-0.01 slider that maps to
        upstream 0-1 via ×100 — designed for subtle key-subspace tuning.

    Drop-in replacement for ``UntwistingRoPEExtensions`` for users who want a
    minimalist surface.
    """

    CATEGORY     = 'Universal Untwisting RoPE'
    RETURN_TYPES = ('UNTWISTING_ROPE_EXTENSIONS',)
    RETURN_NAMES = ('extensions',)
    FUNCTION     = 'build'
    DESCRIPTION  = (
        'Minimalist Advanced Options — 4 artistic sliders, all defaults at 0. '
        'No core engine overrides. Same UNTWISTING_ROPE_EXTENSIONS output type, '
        'plug-compatible with the universal node\'s "extensions" input.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'color_transfer': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 2.0, 'step': 0.01,
                    'display': 'slider',
                }),
                'bleeding_factor_fix': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 2.0, 'step': 0.01,
                    'display': 'slider',
                }),
                'texture_transfer': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'display': 'slider',
                }),
                'advanced_mixer': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 0.01, 'step': 0.0001,
                    'display': 'slider',
                }),
            },
        }

    def build(self,
              color_transfer:      float = 0.0,
              bleeding_factor_fix: float = 0.0,
              texture_transfer:    float = 0.0,
              advanced_mixer:      float = 0.0):

        # advanced_mixer: slider 0-0.01 → upstream key_subspace_alignment 0-1
        internal_mixer = float(advanced_mixer) * 100.0

        data = UnofficialExtensions().build(
            post_attention_adain_strength = float(color_transfer),
            variance_gated_v_adain        = float(texture_transfer),
            cosine_gated_v_injection      = float(bleeding_factor_fix),
            key_subspace_alignment        = internal_mixer,
            axis0_rope_mode               = 'default',   # hidden hardcoded
            axis0_rope_scale              = 0.0,         # hidden hardcoded
        )[0]
        return (data,)


NODE_CLASS_MAPPINGS = {
    'UntwistingRoPEExtensions':                 UntwistingRoPEExtensions,
    'UntwistingRoPEAdvancedSettingsSimple':     UntwistingRoPEAdvancedSettingsSimple,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'UntwistingRoPEExtensions':                 'Universal Untwisting RoPE (Advanced Options)',
    'UntwistingRoPEAdvancedSettingsSimple':     'Universal Untwisting RoPE (Advanced Settings — Simple)',
}
