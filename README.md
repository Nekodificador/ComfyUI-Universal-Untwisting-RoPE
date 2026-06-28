# DAIVID Untwisting RoPE (Universal)

Training-free style transfer for DiT models via **Untwisting RoPE** — frequency control for
shared attention. This pack **unifies** three previously separate efforts into one repository
with a single universal node (image models) plus a consolidated LTX-Video node:

- **Engine:** [BigStationW/ComfyUi-Untwisting-RoPE](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE) (MIT) — the original technique and the per-model adapter system.
- **Model adaptations:** David / ld2worksai-create — FLUX.2 Klein, KREA2, LTX-Video.
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
- **LTX consolidated** — the three LTX variants merged into one node with a mode selector.

## Supported models

| Family | Node | Source |
|---|---|---|
| Z-Image / Z-Image Turbo | Universal | engine |
| Anima | Universal | engine |
| Flux.2 family (incl. FLUX.2 Klein) | Universal | engine + David |
| Qwen-Image / Edit family | Universal | engine |
| KREA2 | Universal | David |
| LTX-Video | LTX node | David |

## Installation

Clone into `ComfyUI/custom_nodes` and restart ComfyUI. No extra custom nodes required.

## Notes for collaborators

- **LTX-Video node is not yet tested.** It consolidates David's three LTX packs (Basic /
  D-Structure / LTX-zip) into one node (`reference_mode` per_step/single_pass + `structure_strength`
  for token injection, shared code in `nodes/ltx_helpers.py`). The merge is faithful to David's
  tested behavior but has not been run end-to-end here — **validate and tune on real LTX-Video**.
  Its params keep David's original names/mappings (`attenuation`/`semantic` as suppression), which
  differ from the image node's `structure`/`style` (direct scales) — reconcile if desired.
- **LoRAAugmenter intentionally omitted.** David's FLUX.2 pack had a `LoRAAugmenter` (load LoRA +
  pick a random image from a folder + apply RoPE). It mixes three concerns that ComfyUI already
  covers separately; left out to avoid duplication. If wanted, the clean form is a minimal
  "random image from folder" node chained with LoraLoader + the universal node.

## Credits

This is a derivative work. The core algorithm and adapter architecture are © BigStationW (MIT);
model adaptations © David (ld2worksai-create); unification © Nekodificador. See `LICENSE`.
