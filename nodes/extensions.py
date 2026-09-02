"""Advanced options pack for the universal Untwisting RoPE.

One optional node holding the fine-tuning knobs, so the main node stays clean. Every widget here
means exactly its value — no sentinels, no "0 = ignore me". The engine internals that an artist
should never have to reason about (axis-0 RoPE behavior) are fixed to the recommended value inside
`build`; `blocks` stays only as a debug knob for tuning per-model profiles.

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
        'Optional fine-tuning for Untwisting RoPE — you do NOT need this node to get a result; '
        'start with the main node alone and add this only when something specific is off. The '
        'defaults here are the validated preset, not a neutral setting, so change ONE slider at a '
        'time. Each tooltip starts with the symptom that slider fixes. '
        'Plug into the universal node\'s "extensions" input.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'color_transfer': ('FLOAT', {
                    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Symptom: "the colors/tone of the reference did not come across" '
                               '(raise) — or "it stole colors I wanted to keep" (lower). '
                               'Matches the result\'s color and tone to the reference.'
                }),
                'texture_transfer': ('FLOAT', {
                    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Symptom: "it took the colors but not the brushwork/grain/material" '
                               '(raise) — or "the surface got too busy" (lower). '
                               'Transfers the reference\'s texture.'
                }),
                'bleeding_fix': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Symptom: "the style is leaking into areas where it does not belong" '
                               '(raise). Applies the reference only where target and reference '
                               'actually correspond. 0 = off.'
                }),
                'style_adherence': ('FLOAT', {
                    'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Symptom: "it still does not look enough like the reference" (raise, '
                               'in small steps). Pushes the result toward the reference\'s look. '
                               'Too high goes rigid and lifeless — this is the first slider to '
                               'lower if the image looks stiff.'
                }),
                'looseness_scale': ('FLOAT', {
                    'default': 1.0, 'min': 0.25, 'max': 4.0, 'step': 0.05,
                    'tooltip': 'Symptom: "it copies the reference\'s shapes too literally" (raise) / '
                               '"it ignored the reference\'s composition" (lower). '
                               'MULTIPLIER on the model\'s own value — 1.0 = exactly what the model '
                               'defaults to, 2.0 = twice as loose, 0.5 = twice as literal. '
                               '(The absolute value differs ~50x between models, so it is relative.)'
                }),
                'tone_match': ('FLOAT', {
                    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Symptom: "the contrast/brightness does not match the reference" '
                               '(raise) — or "the result came out flat/washed out" (lower). '
                               'Matches the reference\'s overall tone statistics. 0 = off, '
                               '1 = full (the validated default).'
                }),
            },
            'optional': {
                'blocks': ('STRING', {
                    'default': '',
                    'tooltip': 'DEBUG — for tuning per-model profiles, not for artwork. Which '
                               'transformer blocks get patched, e.g. "7-27" or "0-8,28-37". '
                               'Empty = the model\'s own range. Wrong ranges silently produce '
                               'garbage or no effect at all.'
                }),
            },
        }

    def build(
        self,
        color_transfer: float = 1.0,
        texture_transfer: float = 1.0,
        bleeding_fix: float = 0.0,
        style_adherence: float = 0.35,
        looseness_scale: float = 1.0,
        tone_match: float = 1.0,
        blocks: str = '',
    ):
        data = UnofficialExtensions().build(
            post_attention_adain_strength=float(color_transfer),
            variance_gated_v_adain=float(texture_transfer),
            cosine_gated_v_injection=float(bleeding_fix),
            key_subspace_alignment=float(style_adherence),
            # Fixed to the recommended value: the engine's own 'default' is documented as
            # disastrous for non-Flux.1 models, so there is no artist-facing choice to make here.
            axis0_rope_mode='match_axes',
            axis0_rope_scale=1.0,
        )[0]
        # Core overrides (universal node reads these; the engine ignores the override_* keys).
        data['override_beta_scale'] = float(looseness_scale)
        data['override_adain_strength'] = float(tone_match)
        data['override_blocks'] = str(blocks)
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
