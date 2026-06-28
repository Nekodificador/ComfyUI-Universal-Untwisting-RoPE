import { app } from "../../scripts/app.js";

// Conditional widget visibility for the Advanced Options node.
// Helpers are the proven pattern from NKD Klein Tools — works in BOTH the classic
// canvas renderer (1.0) and the Vue Nodes renderer (2.0). See the reasoning in each.

function hideWidget(widget) {
    widget.hidden = true;                               // classic canvas (1.0)
    if (widget.options) widget.options.hidden = true;   // Vue layout (2.0)
    widget._dv_hidden = true;
    widget.computeSize = () => [0, -4];                 // collapse the row on canvas (1.0)
}

function showWidget(widget) {
    widget.hidden = false;
    if (widget.options) widget.options.hidden = false;
    widget._dv_hidden = false;
    delete widget.computeSize;
}

// Push visibility to both renderers. Re-assign node.widgets to invalidate the 2.0
// reactive snapshot, re-emit a no-op property change so Vue re-reads it, and resize
// the node so collapsed rows disappear on the canvas.
function refreshNode(node) {
    if (Array.isArray(node.widgets)) {
        node.widgets = [...node.widgets];
    }
    node.graph?.trigger?.("node:property:changed", {
        type: "node:property:changed",
        nodeId: node.id,
        property: "bgcolor",
        oldValue: node.bgcolor,
        newValue: node.bgcolor,
    });
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

function getWidgetValue(node, name) {
    return node.widgets?.find(w => w.name === name)?.value;
}

function setWidgetVisible(node, name, visible) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    if (visible) showWidget(w); else hideWidget(w);
}

// Fires in both renderers: canvas calls widget.callback on edit, Vue routes edits
// through createWidgetUpdateHandler → widget.callback. Idempotent per widget.
function wrapWidgetCallback(node, name, handler) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w || w._dv_cb_wrapped) return;
    const orig = w.callback;
    w.callback = function () {
        const r = orig?.apply(this, arguments);
        handler(node);
        return r;
    };
    w._dv_cb_wrapped = true;
}

function updateScheduleWidgets(node) {
    const custom = getWidgetValue(node, "custom_schedule") === true;
    setWidgetVisible(node, "structure_end", custom);
    setWidgetVisible(node, "style_start", custom);
    refreshNode(node);
}

app.registerExtension({
    name: "DAIVID.UntwistingRoPE.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "UntwistingRoPEExtensions") return;

        const origOnCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnCreated?.apply(this, arguments);
            wrapWidgetCallback(this, "custom_schedule", updateScheduleWidgets);
            requestAnimationFrame(() => updateScheduleWidgets(this));
            return r;
        };

        // Re-apply after a saved workflow restores widget values.
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure?.apply(this, arguments);
            requestAnimationFrame(() => updateScheduleWidgets(this));
            return r;
        };
    },
});
