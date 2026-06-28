"""Consolidated Untwisting RoPE for LTX-Video.

Merges David's three LTX packs into one node:
  - Basic (per-step reference, AdaIN only)              -> reference_mode='per_step', structure_strength=0
  - D-Structure (per-step + token injection + AV)       -> reference_mode='per_step', structure_strength>0
  - LTX-zip (single pre-pass, sigma-gated AdaIN)         -> reference_mode='single_pass'

Mechanism: scale the spatial (h/w) RoPE frequencies per block, optionally prepend reference tokens
into self-attention (StyleAligned, structure_strength), and AdaIN the reference's tone at block 0.
The reference is run through the model to capture features — either once (single_pass) or at each
step's sigma (per_step).

# ponytail: kept faithful to David's tested behavior (param names/mappings, σ=0.15 single-pass,
# sigma-gated AdaIN). It targets LTX-Video, which we can't test here — left for David to validate
# and polish. The win is the dedup (3 files -> 1 node + ltx_helpers) and the single mode selector.
"""
from __future__ import annotations
from typing import Any, Dict

import torch
import comfy.patcher_extension

from .ltx_helpers import (
    coerce_bool, get_diffusion_model, sigma_from_timestep, clone_model_options,
    scale_ltx_pe, adain_full, extend_pe,
)

_PREFIX = '[UntwistingRoPE-LTX]'
_START_BLOCK = 0
_END_BLOCK = 999
_SINGLE_PASS_SIGMA = 0.15  # fixed noising level for the one-shot reference pass


def _make_block_patch(block_idx, high_scale, low_scale, adain_strength, structure_strength,
                      state, start_block, end_block, adain_target_block, sigma_gate_adain,
                      block_module=None):
    """patches_replace hook for one transformer block. Handles PE scaling, token injection
    (pure-video at block input, AV via attn1 hooks), and AdaIN at the target block."""
    is_av_module = block_module is not None and hasattr(block_module, 'audio_to_video_attn')
    attn1_module = getattr(block_module, 'attn1', None) if is_av_module else None

    def patch(args: dict, extra: dict) -> dict:
        original_block = extra['original_block']
        ltx_opts = args.get('transformer_options', {}).get('untwisting_rope_ltx', {})
        is_ref = ltx_opts.get('is_ref_pass', False)
        in_range = start_block <= block_idx <= end_block

        raw_img_in = args.get('img')
        is_list_in = isinstance(raw_img_in, (list, tuple))
        img_in = raw_img_in[0] if is_list_in else raw_img_in

        # PE scaling (target pass, in range). Video PE is 'pe' (LTXV) or 'v_pe' (LTXAV).
        if in_range and not is_ref:
            if 'v_pe' in args and args['v_pe'] is not None:
                args = {**args, 'v_pe': scale_ltx_pe(args['v_pe'], high_scale, low_scale)}
            elif 'pe' in args and args['pe'] is not None:
                args = {**args, 'pe': scale_ltx_pe(args['pe'], high_scale, low_scale)}

        # ── Token injection — pure-video blocks: prepend ref tokens at block input ──
        T_ref_injected = 0
        if not is_av_module and in_range and structure_strength > 0.0:
            if is_ref:
                state['ref_hidden_inputs'][block_idx] = img_in.detach().cpu()
                for pe_key in ('v_pe', 'pe'):
                    pe_val = args.get(pe_key)
                    if pe_val is not None:
                        state['ref_pe_inputs'][block_idx] = (
                            pe_val[0].detach().cpu(), pe_val[1].detach().cpu(),
                            pe_val[2] if len(pe_val) > 2 else False,
                        )
                        break
            else:
                ref_hidden = state.get('ref_hidden_inputs', {}).get(block_idx)
                if ref_hidden is not None:
                    B = img_in.shape[0]
                    ref_h = (ref_hidden.to(device=img_in.device, dtype=img_in.dtype)
                             .expand(B, -1, -1) * structure_strength)
                    ref_pe_data = state.get('ref_pe_inputs', {}).get(block_idx)
                    ext_pe = None
                    pe_key_found = None
                    for pe_key in ('v_pe', 'pe'):
                        tgt_pe = args.get(pe_key)
                        if tgt_pe is not None:
                            ext_pe = extend_pe(ref_pe_data, tgt_pe, B, img_in.device, img_in.dtype)
                            pe_key_found = pe_key
                            break
                    if ext_pe is not None:
                        args = {**args, 'img': torch.cat([ref_h, img_in], dim=1), pe_key_found: ext_pe}
                        T_ref_injected = ref_h.shape[1]

        # ── Token injection — AV blocks: inject via attn1 forward hooks only ──
        hook_handles = []
        attn1_captures = []
        if is_av_module and attn1_module is not None and in_range and structure_strength > 0.0:
            if is_ref:
                def _capture_pre(module, h_args, h_kwargs):
                    attn1_captures.append((h_args[0].detach().cpu() if h_args else None,
                                           h_kwargs.get('pe', None)))
                    return None
                try:
                    hook_handles.append(attn1_module.register_forward_pre_hook(_capture_pre, with_kwargs=True))
                except TypeError:
                    def _capture_pre_nkw(module, h_args):
                        attn1_captures.append((h_args[0].detach().cpu() if h_args else None, None))
                    hook_handles.append(attn1_module.register_forward_pre_hook(_capture_pre_nkw))
            else:
                ref_attn_data = state.get('ref_attn_inputs', {}).get(block_idx)
                if ref_attn_data is not None:
                    ref_x, ref_pe_stored = ref_attn_data
                    B = img_in.shape[0]
                    ref_tokens = (ref_x.to(device=img_in.device, dtype=img_in.dtype)
                                  .expand(B, -1, -1) * structure_strength)
                    T_ref_for_hook = ref_tokens.shape[1]
                    injection_done = [False]

                    def _inject_pre(module, h_args, h_kwargs):
                        x = h_args[0]
                        tgt_pe = h_kwargs.get('pe')
                        new_pe = None
                        if tgt_pe is not None:
                            new_pe = extend_pe(ref_pe_stored, tgt_pe, B, x.device, x.dtype)
                            if new_pe is None:
                                return None
                        x_ext = torch.cat([ref_tokens, x], dim=1)
                        new_kw = {**h_kwargs, 'pe': new_pe} if new_pe is not None else h_kwargs
                        injection_done[0] = True
                        return (x_ext,) + h_args[1:], new_kw

                    def _trim_post(module, h_args, output):
                        if (injection_done[0] and isinstance(output, torch.Tensor)
                                and output.shape[1] >= T_ref_for_hook):
                            return output[:, T_ref_for_hook:, :]
                        return output
                    try:
                        hook_handles.append(attn1_module.register_forward_pre_hook(_inject_pre, with_kwargs=True))
                        hook_handles.append(attn1_module.register_forward_hook(_trim_post))
                    except TypeError:
                        pass

        # ── Run block ──
        try:
            result = original_block(args)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass

        if is_ref and is_av_module and attn1_captures:
            norm_vx_cap, pe_cap = attn1_captures[0]
            stored_pe = None
            if pe_cap is not None and isinstance(pe_cap, (list, tuple)) and len(pe_cap) >= 2:
                pc0, pc1 = pe_cap[0], pe_cap[1]
                if pc0 is not None and pc1 is not None:
                    stored_pe = (pc0.detach().cpu() if isinstance(pc0, torch.Tensor) else pc0,
                                 pc1.detach().cpu() if isinstance(pc1, torch.Tensor) else pc1,
                                 pe_cap[2] if len(pe_cap) > 2 else False)
            state['ref_attn_inputs'][block_idx] = (norm_vx_cap, stored_pe)

        if not in_range:
            return result

        raw_out = result['img']
        is_list_out = isinstance(raw_out, (list, tuple))
        out_img = raw_out[0] if is_list_out else raw_out

        if T_ref_injected > 0:
            out_img = out_img[:, T_ref_injected:, :]

        if is_ref:
            state['ref_features_current'][block_idx] = (
                out_img.mean(dim=1, keepdim=True).detach().cpu().float(),
                (out_img.std(dim=1, keepdim=True) + 1e-5).detach().cpu().float(),
            )
        elif block_idx == adain_target_block and adain_strength > 0.0:
            ref_data = state.get('ref_features_current', {}).get(block_idx)
            if ref_data is not None:
                # single_pass captured the ref at a fixed sigma → gate AdaIN by the step's sigma so
                # style is strong early and fades as structure solidifies. per_step matches sigmas
                # already (std-ratio ≈ 1) → use full strength.
                if sigma_gate_adain:
                    eff = float(adain_strength) * float(ltx_opts.get('current_sigma', 1.0))
                else:
                    eff = float(adain_strength)
                if eff > 0.0:
                    ref_mean, ref_std = ref_data
                    ref_mean = ref_mean.to(device=out_img.device, dtype=out_img.dtype)
                    ref_std = ref_std.to(device=out_img.device, dtype=out_img.dtype)
                    out_img = adain_full(out_img, ref_mean, ref_std, eff)

        if is_list_out:
            result = {**result, 'img': [out_img] + list(raw_out[1:])}
        else:
            result = {**result, 'img': out_img}
        return result

    return patch


def _reset_state(state):
    state['ref_features_current'] = {}
    state['ref_hidden_inputs'] = {}
    state['ref_pe_inputs'] = {}
    state['ref_attn_inputs'] = {}


def _run_reference_pass(model_clone, dm, state, ref_cpu, c, input_x, sigma_for_ref, verbose):
    """One no_grad forward of the reference latent at `sigma_for_ref`, capturing per-block
    features/inputs into `state`. Shared by both reference modes."""
    state['is_ref_pass'] = True
    _reset_state(state)
    try:
        ref = ref_cpu.to(device=input_x.device, dtype=input_x.dtype)
        try:
            ref_scaled = model_clone.model.latent_format.process_in(ref)
        except Exception:
            ref_scaled = ref
        if ref_scaled.dim() == 4:
            ref_scaled = ref_scaled.unsqueeze(2)  # [B,C,H,W] -> [B,C,1,H,W]
        try:
            model_dtype = next(dm.parameters()).dtype
            ref_scaled = ref_scaled.to(dtype=model_dtype)
        except StopIteration:
            model_dtype = ref_scaled.dtype

        if state.get('ref_noise') is None or state['ref_noise'].shape != ref_scaled.shape:
            state['ref_noise'] = torch.randn_like(ref_scaled)
        ref_noise = state['ref_noise'].to(device=ref_scaled.device, dtype=ref_scaled.dtype)
        ref_at_sigma = ref_scaled * (1.0 - sigma_for_ref) + ref_noise * sigma_for_ref
        ref_ts = torch.tensor([sigma_for_ref], device=ref_at_sigma.device, dtype=ref_at_sigma.dtype)

        def _unwrap(v):
            return v.cond if hasattr(v, 'cond') else v
        context = _unwrap(c.get('c_crossattn', None))
        attn_mask = _unwrap(c.get('attention_mask', None))
        frame_rate = _unwrap(c.get('frame_rate', 25))
        if isinstance(frame_rate, torch.Tensor):
            frame_rate = float(frame_rate.flatten()[0])
        try:
            if context is not None:
                context = context.to(dtype=model_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=model_dtype)
            ref_ts = ref_ts.to(dtype=model_dtype)
        except Exception:
            pass

        ref_to = {'untwisting_rope_ltx': {'is_ref_pass': True, 'state': state}}
        src_to = model_clone.model_options.get('transformer_options', {})
        if 'patches_replace' in src_to:
            ref_to['patches_replace'] = src_to['patches_replace']

        is_ltxav = hasattr(dm, 'separate_audio_and_video_latents')
        ref_input = [ref_at_sigma] if is_ltxav else ref_at_sigma
        extra_fwd = {'audio_length': 0} if is_ltxav else {}

        with torch.no_grad():
            dm.forward(ref_input, ref_ts, context=context, attention_mask=attn_mask,
                       frame_rate=frame_rate, transformer_options=ref_to, **extra_fwd)
        if verbose:
            print(f'{_PREFIX} ref pass σ={sigma_for_ref:.3f}  '
                  f'adain-blocks={len(state["ref_features_current"])}  '
                  f'attn-blocks={len(state["ref_attn_inputs"])}')
    except Exception as exc:
        import traceback
        print(f'{_PREFIX} ⚠ reference pass failed σ={sigma_for_ref:.3f}: {exc}')
        traceback.print_exc()
    state['is_ref_pass'] = False


class UntwistingRoPELTX:
    CATEGORY = 'DAIVID/RoPE'
    RETURN_TYPES = ('MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'patch'
    DESCRIPTION = (
        'Untwisting RoPE for LTX-Video (consolidates David\'s Basic / D-Structure / LTX-zip packs). '
        'Scales spatial RoPE frequencies, optionally injects reference tokens (structure_strength), '
        'and AdaIN-matches tone. reference_mode picks per-step (precise) vs single pre-pass (fast).'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'reference_latent': ('LATENT',),
                'attenuation': ('FLOAT', {
                    'default': 0.1, 'min': -5.0, 'max': 5.0, 'step': 0.01,
                    'tooltip': 'High-freq spatial (structural) attenuation. 0 = neutral. Clamped ≥0.'
                }),
                'semantic': ('FLOAT', {
                    'default': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Low-freq spatial (semantic) suppression. 0 = neutral, 1 = fully suppressed.'
                }),
                'adain_strength': ('FLOAT', {
                    'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Color/tone style via AdaIN at block 0. 0 = off.'
                }),
                'structure_strength': ('FLOAT', {
                    'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                    'tooltip': 'Structural style via token injection (StyleAligned): ref tokens are '
                               'prepended to self-attention so it copies shapes/composition. 0 = off '
                               '(= the Basic/LTX-zip behavior).'
                }),
                'reference_mode': (['per_step', 'single_pass'], {
                    'default': 'per_step',
                    'tooltip': 'per_step: run the reference at each step\'s sigma (precise, AdaIN at '
                               'full strength; one extra forward per step). single_pass: run the '
                               'reference once at sigma=0.15 (fast, AdaIN gated by sigma).'
                }),
                'verbose': ('BOOLEAN', {'default': False}),
            },
        }

    def patch(self, model, reference_latent, attenuation=0.1, semantic=0.1, adain_strength=0.3,
              structure_strength=0.0, reference_mode='per_step', verbose=False):
        verbose = coerce_bool(verbose)
        single_pass = (reference_mode == 'single_pass')

        dm = get_diffusion_model(model)
        if not hasattr(dm, 'transformer_blocks'):
            raise RuntimeError(f'{_PREFIX} Requires an LTX-Video model. Found: {type(dm).__name__}.')
        n_blocks = len(dm.transformer_blocks)

        if not isinstance(reference_latent, dict) or 'samples' not in reference_latent:
            raise RuntimeError(f'{_PREFIX} reference_latent must be a ComfyUI LATENT dict.')
        ref_raw = reference_latent['samples'].detach().clone()
        try:
            ref_samples = model.model.process_latent_in(ref_raw)
        except Exception:
            ref_samples = ref_raw
        ref_cpu = ref_samples.to('cpu')

        # David's suppression mappings (kept faithful): 0 = neutral.
        high_scale = max(0.0, 1.0 - float(attenuation) * 0.1)
        low_scale = max(0.0, 1.0 - float(semantic))

        if verbose:
            n_av = sum(1 for b in dm.transformer_blocks if hasattr(b, 'audio_to_video_attn'))
            print(f'{_PREFIX} blocks={n_blocks} (video={n_blocks - n_av}, AV={n_av})  '
                  f'mode={reference_mode}  att={attenuation:.3f}->{high_scale:.3f}  '
                  f'sem={semantic:.3f}->{low_scale:.3f}  adain={adain_strength:.3f}  '
                  f'structure={structure_strength:.3f}')

        state: Dict[str, Any] = {
            'ref_features_current': {}, 'ref_hidden_inputs': {}, 'ref_pe_inputs': {},
            'ref_attn_inputs': {}, 'ref_noise': None, 'ref_pass_done': False, 'is_ref_pass': False,
        }

        model_clone = model.clone()
        model_clone.model_options = clone_model_options(model_clone.model_options)
        to = model_clone.model_options.setdefault('transformer_options', {})
        pr = to.setdefault('patches_replace', {}).setdefault('dit', {})
        for i in range(n_blocks):
            pr[('double_block', i)] = _make_block_patch(
                i, high_scale, low_scale, float(adain_strength), float(structure_strength),
                state, _START_BLOCK, _END_BLOCK, adain_target_block=0,
                sigma_gate_adain=single_pass, block_module=dm.transformer_blocks[i],
            )

        def sampler_sample_wrapper(executor, model_wrap, sigmas, extra_args, callback,
                                   noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            _reset_state(state)
            state['ref_noise'] = None
            state['ref_pass_done'] = False
            state['is_ref_pass'] = False
            return executor(model_wrap, sigmas, extra_args, callback, noise,
                            latent_image, denoise_mask, disable_pbar)

        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, sampler_sample_wrapper,
            model_clone.model_options, is_model_options=True,
        )

        old_wrapper = model_clone.model_options.get('model_function_wrapper', None)
        active = (adain_strength > 0.0 or structure_strength > 0.0)

        def model_function_wrapper(apply_model, args):
            input_x = args['input']
            timestep = args['timestep']
            c = args['c'].copy()
            current_sigma = float(sigma_from_timestep(timestep))
            dm_local = get_diffusion_model(model_clone)

            if active:
                if single_pass:
                    if not state['ref_pass_done']:
                        state['ref_pass_done'] = True
                        _run_reference_pass(model_clone, dm_local, state, ref_cpu, c, input_x,
                                            _SINGLE_PASS_SIGMA, verbose)
                else:
                    _run_reference_pass(model_clone, dm_local, state, ref_cpu, c, input_x,
                                        current_sigma, verbose)

            to_main = dict(c.get('transformer_options', {}))
            to_main['untwisting_rope_ltx'] = {
                'is_ref_pass': False, 'state': state, 'current_sigma': current_sigma,
            }
            c['transformer_options'] = to_main

            if old_wrapper is not None:
                return old_wrapper(apply_model, {**args, 'c': c})
            return apply_model(input_x, timestep, **c)

        model_clone.model_options = clone_model_options(model_clone.model_options)
        model_clone.set_model_unet_function_wrapper(model_function_wrapper)
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    'UntwistingRoPELTX': UntwistingRoPELTX,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'UntwistingRoPELTX': 'Untwisting RoPE (LTX-Video)',
}
