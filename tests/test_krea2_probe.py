"""One check: the Krea2 imglen probe must survive ComfyUI adding positional params to _forward.

Run: python tests/test_krea2_probe.py   (from the pack root, with ComfyUI's venv)
"""
import sys, os, importlib.util
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # ComfyUI root

spec = importlib.util.spec_from_file_location(
    "krea2_adapter", os.path.join(os.path.dirname(__file__), "..", "models", "krea2.py"))
krea2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(krea2)

import torch


class FakeDiT:
    """Stands in for SingleStreamDiT with whatever signature upstream feels like today."""
    patch = 2

    def __init__(self, signature):
        self.seen = None
        if signature == "old":       # pre-c9602625
            def _forward(x, timesteps, context, attention_mask=None, transformer_options={}, **kw):
                self.seen = ("old", transformer_options); return x
        else:                        # ref_latents added as a 5th positional
            def _forward(x, timesteps, context, attention_mask=None, ref_latents=None,
                         transformer_options={}, **kw):
                self.seen = ("new", transformer_options); return x
        self._forward = _forward


def run(signature, call):
    dm = FakeDiT(signature)
    krea2._install_imglen_probe(dm)
    topts = {krea2.CONFIG_KEY: {"enabled": True}}
    x = torch.zeros(1, 4, 64, 64)          # 32x32 patches of 2 -> 32*32 = 1024 image tokens
    call(dm, x, topts)
    cfg = topts[krea2.CONFIG_KEY]
    assert dm.seen[1] is topts, f"{signature}: transformer_options not forwarded ({dm.seen})"
    assert cfg.get("krea2_imglen") == 1024, f"{signature}: imglen={cfg.get('krea2_imglen')}"


# new signature, all positional — this is exactly how comfy/ldm/krea2/model.py calls it
run("new", lambda dm, x, t: dm._forward(x, None, None, None, None, t))
# old signature, all positional — the pre-c9602625 shape
run("old", lambda dm, x, t: dm._forward(x, None, None, None, t))
# transformer_options passed by keyword instead
run("new", lambda dm, x, t: dm._forward(x, None, None, transformer_options=t))
# no cfg present at all -> must not raise, must still call through
dm = FakeDiT("new"); krea2._install_imglen_probe(dm)
dm._forward(torch.zeros(1, 4, 8, 8), None, None, None, None, {})
assert dm.seen[0] == "new"

print("ok - krea2 imglen probe survives old/new signatures")
