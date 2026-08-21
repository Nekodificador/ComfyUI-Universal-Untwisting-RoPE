/**
 * Read-only schedule preview for the Universal Untwisting RoPE node.
 *
 * Draws the two curves the node actually runs — structure (high-freq) and style (low-freq) —
 * across the denoise trajectory, so you can see where they cross before sampling.
 *
 * The maths mirrors the backend one-for-one (see _ease_schedule_progress and the schedule block
 * in __init__.py): value(p) = lerp(start, end, ease(curve, p)). Keep the two in lock-step.
 *
 * Vanilla on purpose: this pack has no Vue/Vite build and a static chart does not need one.
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "UntwistingRoPEUniversal";
const EXT_NAME = "UntwistingRoPE.SchedulePreview";
const REV = "2026-08-21a";

const CANVAS_AR = 0.46; // chart height = node width * this
const MIN_W = 240;

const C = {
  bg: "#111318",
  grid: "rgba(255,255,255,0.06)",
  gridBorder: "rgba(255,255,255,0.16)",
  zero: "rgba(255,255,255,0.28)",
  structure: "#4ab4ff",
  style: "#ffd166",
  label: "rgba(255,255,255,0.40)",
  labelDim: "rgba(255,255,255,0.22)",
};

const WATCHED = ["structure_start", "structure_end", "style_start", "style_end", "schedule_curve"];

// Mirror of _ease_schedule_progress in __init__.py — keep both sides identical.
function ease(curve, p) {
  p = Math.max(0, Math.min(1, p));
  switch (String(curve || "linear").trim().toLowerCase()) {
    case "ease_in":
      return p * p;
    case "ease_out":
      return 1 - (1 - p) ** 2;
    case "ease_in_out":
      return p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2;
    case "smoothstep":
      return p * p * (3 - 2 * p);
    case "exponential":
      return p <= 0 ? 0 : 2 ** (10 * p - 10);
    default:
      return p;
  }
}

/**
 * Pin the DOM widget back to the node's logical width.
 * The classic renderer mis-sizes DOM widgets on selection / re-layout; node.size[0] never lies.
 * Copied from the NKD packs — do not "improve" without re-measuring.
 */
function keepDomWidgetSized(node, container) {
  const MAX_MARGIN = 40;
  let enforcingW = false;
  let goodMargin = 15;
  const vueMode = () => !!window.LiteGraph?.vueNodesMode;
  const clamp = () => {
    if (enforcingW) return;
    if (vueMode()) {
      if (container.style.width) container.style.width = "";
      return;
    }
    const nodeW = node.size?.[0];
    if (!nodeW) return;
    const host = container.parentElement;
    const hostW = host ? host.clientWidth : 0;
    const broken = hostW > 0 && (hostW > nodeW * 1.2 || hostW < nodeW * 0.7);
    if (!broken) {
      if (container.style.width) {
        enforcingW = true;
        container.style.width = "";
        requestAnimationFrame(() => {
          enforcingW = false;
        });
      }
      const cw = container.clientWidth;
      if (cw > 0 && cw <= nodeW && cw >= nodeW - MAX_MARGIN) goodMargin = nodeW - cw;
      return;
    }
    const ref = Math.round(nodeW - goodMargin);
    if (ref > 0 && Math.abs(container.clientWidth - ref) > 2) {
      enforcingW = true;
      container.style.boxSizing = "border-box";
      container.style.width = ref + "px";
      requestAnimationFrame(() => {
        enforcingW = false;
      });
    }
  };
  clamp();
  const ro = new ResizeObserver(clamp);
  ro.observe(container);
  const origResize = node.onResize;
  node.onResize = function () {
    origResize?.apply(this, arguments);
    clamp();
  };
  const iv = window.setInterval(clamp, 250);
  return {
    margin: () => goodMargin,
    dispose: () => {
      ro.disconnect();
      clearInterval(iv);
    },
  };
}

function makeChart(container) {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.aspectRatio = String(1 / CANVAS_AR);
  canvas.style.display = "block";
  canvas.style.borderRadius = "4px";
  container.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  let vals = {
    structure_start: 1,
    structure_end: 0,
    style_start: 1,
    style_end: 3,
    schedule_curve: "linear",
  };

  function sync() {
    const r = canvas.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const s = Math.max(window.devicePixelRatio || 1, 2);
    const w = Math.round(r.width * s);
    const h = Math.round(r.height * s);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.setTransform(s, 0, 0, s, 0, 0);
    return true;
  }

  function draw() {
    const r = canvas.getBoundingClientRect();
    const W = r.width;
    const H = r.height;
    if (W < 1 || H < 1) return;
    const PAD = { l: 28, r: 6, t: 8, b: 14 };
    const IW = Math.max(1, W - PAD.l - PAD.r);
    const IH = Math.max(1, H - PAD.t - PAD.b);

    const a = Number(vals.structure_start) || 0;
    const b = Number(vals.structure_end) || 0;
    const c = Number(vals.style_start) || 0;
    const d = Number(vals.style_end) || 0;
    let lo = Math.min(a, b, c, d, 0);
    let hi = Math.max(a, b, c, d, 0);
    if (hi - lo < 1e-6) hi = lo + 1;
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;

    const X = (p) => PAD.l + p * IW;
    const Y = (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * IH;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const x = X(i / 4);
      ctx.beginPath();
      ctx.moveTo(x, PAD.t);
      ctx.lineTo(x, PAD.t + IH);
      ctx.stroke();
    }
    ctx.strokeStyle = C.gridBorder;
    ctx.strokeRect(PAD.l, PAD.t, IW, IH);

    // Zero line: below it the reference is inverted, not just weaker — worth seeing.
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = C.zero;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(PAD.l, Y(0));
      ctx.lineTo(PAD.l + IW, Y(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.font = "9px sans-serif";
    ctx.fillStyle = C.label;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(hi.toFixed(1), PAD.l - 4, PAD.t + 4);
    ctx.fillText(lo.toFixed(1), PAD.l - 4, PAD.t + IH - 4);
    ctx.fillStyle = C.labelDim;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("start", PAD.l, H - 3);
    ctx.textAlign = "right";
    ctx.fillText("end", PAD.l + IW, H - 3);

    const drawCurve = (s, e, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const N = 64;
      for (let i = 0; i <= N; i++) {
        const p = i / N;
        const v = s + (e - s) * ease(vals.schedule_curve, p);
        const x = X(p);
        const y = Y(v);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      }
      ctx.stroke();
    };
    drawCurve(a, b, C.structure);
    drawCurve(c, d, C.style);

    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillStyle = C.structure;
    ctx.fillText("structure", PAD.l + 4, PAD.t + 3);
    ctx.fillStyle = C.style;
    ctx.fillText("style", PAD.l + 4, PAD.t + 14);
  }

  const ro = new ResizeObserver(() => {
    if (sync()) draw();
  });
  ro.observe(canvas);

  return {
    setValues(v) {
      vals = { ...vals, ...v };
      if (sync()) draw();
    },
    forceResize() {
      const ok = sync();
      if (ok) draw();
      return ok;
    },
    dispose() {
      ro.disconnect();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    // "Refresh node definitions" re-runs this on the SAME prototype; without the guard the
    // onNodeCreated wraps stack and each node grows another chart per refresh.
    if (nodeType.prototype.__ropeSchedulePreview) return;
    nodeType.prototype.__ropeSchedulePreview = true;

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = origCreated?.apply(this, arguments);
      const node = this;

      const container = document.createElement("div");
      container.style.width = "100%";
      container.style.minWidth = MIN_W + "px";
      container.style.boxSizing = "border-box";

      const chart = makeChart(container);
      const push = () => {
        const out = {};
        for (const name of WATCHED) {
          const w = node.widgets?.find((x) => x.name === name);
          if (w) out[name] = w.value;
        }
        chart.setValues(out);
      };

      const height = () => Math.round((node.size?.[0] || MIN_W) * CANVAS_AR);
      node.addDOMWidget("schedule_preview", "ROPE_SCHEDULE_PREVIEW", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: height,
        getMaxHeight: height,
        getHeight: height,
      });

      const sizer = keepDomWidgetSized(node, container);
      const minNodeWidth = () => MIN_W + sizer.margin();

      const origComputeSize = node.computeSize.bind(node);
      node.computeSize = function () {
        const sz = origComputeSize();
        if (sz[0] < minNodeWidth()) sz[0] = minNodeWidth();
        // never report less height than the chart needs, whatever the layout thinks
        const needed = Math.round((sz[0] || MIN_W) * CANVAS_AR);
        if (sz[1] < needed) sz[1] = needed;
        return sz;
      };
      const origResize = node.onResize;
      node.onResize = function (size) {
        origResize?.apply(this, arguments);
        if (size[0] < minNodeWidth()) size[0] = minNodeWidth();
      };

      // widget.callback is the only edit hook that fires in BOTH renderers.
      for (const name of WATCHED) {
        const w = node.widgets?.find((x) => x.name === name);
        if (!w || w.__ropePreviewCb) continue;
        const orig = w.callback;
        w.callback = function () {
          const r = orig?.apply(this, arguments);
          push();
          return r;
        };
        w.__ropePreviewCb = true;
      }
      // Backstop for programmatic changes (undo, workflow load, other extensions).
      const poll = window.setInterval(push, 250);

      requestAnimationFrame(() => {
        chart.forceResize();
        push();
        node.setSize([Math.max(node.size[0], minNodeWidth()), node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      });

      const origConfigure = node.onConfigure;
      node.onConfigure = function () {
        const r = origConfigure?.apply(this, arguments);
        requestAnimationFrame(() => {
          chart.forceResize();
          push();
        });
        return r;
      };
      const origRemoved = node.onRemoved;
      node.onRemoved = function () {
        clearInterval(poll);
        sizer.dispose();
        chart.dispose();
        return origRemoved?.apply(this, arguments);
      };

      return result;
    };
  },
});

console.log("[Universal Untwisting RoPE] schedule preview rev " + REV);
