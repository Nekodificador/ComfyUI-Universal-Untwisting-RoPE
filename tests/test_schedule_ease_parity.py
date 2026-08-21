"""One check: the preview's easing must match the backend's, or the chart lies about the run.

Both sides are extracted from their real source files (not re-implemented here) and compared over
a grid. Needs node on PATH for the JS half; skips that half with a message if it is missing.

Run: python tests/test_schedule_ease_parity.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRID = [i / 40 for i in range(41)]
CURVES = ["linear", "ease_in", "ease_out", "ease_in_out", "smoothstep", "exponential", "bogus"]


def python_ease():
    """exec just the easing function out of __init__.py (importing it would pull in comfy)."""
    src = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
    m = re.search(r"^def _ease_schedule_progress\(.*?(?=^def )", src, re.S | re.M)
    assert m, "could not find _ease_schedule_progress in __init__.py"
    ns = {"Any": object}
    exec(m.group(0), ns)
    return ns["_ease_schedule_progress"]


def js_ease_values():
    src = open(os.path.join(ROOT, "js", "schedule_preview.js"), encoding="utf-8").read()
    m = re.search(r"^function ease\(curve, p\) \{.*?^\}", src, re.S | re.M)
    assert m, "could not find ease() in js/schedule_preview.js"
    script = "%s\nconsole.log(JSON.stringify(%s.map(([c,p]) => ease(c,p))));" % (
        m.group(0),
        json.dumps([[c, p] for c in CURVES for p in GRID]),
    )
    out = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


ease = python_ease()
expected = [ease(c, p) for c in CURVES for p in GRID]

# the Python side alone must stay sane whatever the widget sends
assert ease("linear", 0.37) == 0.37
assert ease("bogus", 0.37) == 0.37, "unknown curve must fall back to linear"
assert ease("ease_in", -5) == 0.0 and ease("ease_in", 5) == 1.0, "progress must be clamped"

if shutil.which("node") is None:
    print("ok - python easing sane (node not found, JS parity not checked)")
    sys.exit(0)

got = js_ease_values()
assert len(got) == len(expected), "grid size mismatch"
worst = max(abs(a - b) for a, b in zip(got, expected))
assert worst < 1e-12, "preview easing drifted from the backend: max diff %g" % worst
print("ok - preview easing matches the backend on %d samples (max diff %g)" % (len(got), worst))
