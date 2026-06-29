# DAIVID Untwisting RoPE (Universal)

Training-free style transfer for DiT image models via **Untwisting RoPE** — frequency control for
shared attention. This pack **unifies** previously separate per-model efforts into a single
universal node:

- **Engine:** [BigStationW/ComfyUi-Untwisting-RoPE](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE) (MIT) — the original technique and the per-model adapter system.
- **Model adaptations:** David / ld2worksai-create — FLUX.2 Klein, KREA2.
- **Unification & universal-node redesign:** Nekodificador.

Paper: [Untwisting RoPE: Frequency Control for Shared Attention in DiTs](https://arxiv.org/abs/2602.05013) · https://untwisting-rope.github.io/

## What changed vs the originals

- **One repo, no external dependency.** KREA2 used to require BigStationW's pack installed
  separately — now the engine is vendored, so everything works standalone.
- **One universal node** for image models that auto-detects the model and dispatches to the right
  adapter (no more one-node-per-model).
- **Self-contained reference handling.** The node resizes the reference to the target resolution
  and VAE-encodes it internally — no more auxiliary Scale-Image / VAEEncode / ReferenceLatent
  nodes cluttering the graph. You can still pass a pre-encoded `reference_latent` (e.g. from NKD
  Klein/Krea Tools) to avoid double-encoding.
- **Friendly controls** (`attenuation` / `semantic`) with per-model defaults; raw parameters
  available via an optional extensions node.

## Supported models

| Family | Node | Source |
|---|---|---|
| Z-Image / Z-Image Turbo | Universal | engine |
| Anima | Universal | engine |
| Flux.2 family (incl. FLUX.2 Klein) | Universal | engine + David |
| Flux.1 family (Flux / Schnell / Inpaint; incl. **FLUX.1-Depth-dev**) | Universal | Nekodificador |
| Qwen-Image / Edit family | Universal | engine |
| KREA2 | Universal | David |

## Installation

Clone into `ComfyUI/custom_nodes` and restart ComfyUI. No extra custom nodes required.

## Credits

This is a derivative work. The core algorithm and adapter architecture are © BigStationW (MIT);
model adaptations © David (ld2worksai-create); unification © Nekodificador. See `LICENSE`.
