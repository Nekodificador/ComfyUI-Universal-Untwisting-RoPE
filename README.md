# Universal Untwisting RoPE

Training-free style transfer for DiT models via Untwisting RoPE, frequency control for shared
attention. This pack unifies three previously separate efforts into one repository around a
single universal node that auto-detects the loaded image model.
Clone into `ComfyUI/custom_nodes` and restart ComfyUI. No extra custom nodes required.

- **Engine:** [BigStationW/ComfyUi-Untwisting-RoPE](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE) (MIT), the original technique and the per-model adapter system.
- **Model adaptations:** David / ld2worksai-create, FLUX.2 Klein, KREA2.
- **Unification and universal-node redesign:** Nekodificador.

Paper: [Untwisting RoPE: Frequency Control for Shared Attention in DiTs](https://arxiv.org/abs/2602.05013) · https://untwisting-rope.github.io/

## Image models

One node for every supported image model: Z-Image, Anima, Flux.1, Flux.2/Klein, Qwen-Image and
KREA2. It auto-detects the model, runs RF inversion and the RoPE attention patch internally, and
resizes + VAE-encodes the reference for you, so no auxiliary Scale-Image / VAEEncode /
ReferenceLatent nodes clutter the graph. An optional second node adds fine-tuning knobs on top.

| Node | What it does |
|---|---|
| [Universal Untwisting RoPE](docs/universal.md) | Auto-detects the model and runs training-free style transfer from a reference image. |
| [Universal Untwisting RoPE (Advanced Options)](docs/universal.md) | Optional fine-tuning knobs for the main node: color/texture transfer, bleeding fix, style adherence, looseness, tone match. |


## Examples
### Krea2
<img width="1502" height="333" alt="image" src="https://github.com/user-attachments/assets/02a82f03-f8da-4cef-8ff0-2f2ae7f50455" />

> Workflow included in 📂example_workflows, so you can call it from the ComfyUI Templates menu

### Flux Klein
<img width="1660" height="819" alt="image" src="https://github.com/user-attachments/assets/15ee44b9-b9a2-47ba-9914-f2ef97a75b03" />
>> Workflow included in 📂example_workflows, so you can call it from the ComfyUI Templates menu

## Credits

This is a derivative work.

- **Core algorithm and per-model adapter architecture:** © BigStationW, MIT licensed. Vendored
  into this repo so KREA2 no longer needs the original pack installed separately.
- **Model adaptations (FLUX.2 Klein, KREA2):** © David / ld2worksai-create.
- **Flux.1 family adapter:** Nekodificador.
- **Unification, universal node, and this repository:** Nekodificador.

Techniques implemented here, beyond the base paper: RF-Solver / RF-Edit
([arXiv:2411.04746](https://arxiv.org/abs/2411.04746)), FireFlow
([arXiv:2412.07517](https://arxiv.org/abs/2412.07517)), FlowTurbo
([arXiv:2409.18128](https://arxiv.org/abs/2409.18128)), PMI
([arXiv:2602.11850](https://arxiv.org/abs/2602.11850)), OTIP
([arXiv:2508.02363](https://arxiv.org/abs/2508.02363)), AdaIN
([arXiv:1703.06868](https://arxiv.org/abs/1703.06868)), the ConsiStory feature-injection idea
([arXiv:2402.03286](https://arxiv.org/abs/2402.03286)), and CACTIF's similarity-filtered attention
([arXiv:2505.16360](https://arxiv.org/abs/2505.16360)).

See `LICENSE` for the full license text.
