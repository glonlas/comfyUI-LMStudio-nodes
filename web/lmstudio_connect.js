import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CONNECT_NODE_IDS = new Set(["LMStudio_Connect", "LMStudio - Connect"]);
const PROMPT_NODE_IDS = new Set([
  "LMStudio_TextGen",
  "LMStudio - Text Gen",
  "LMStudio_ImageToText",
  "LMStudio - Image To Text",
]);
const MODEL_PLACEHOLDER = "<refresh models>";

function isConnectNodeDefinition(nodeData) {
  return (
    CONNECT_NODE_IDS.has(nodeData?.name) ||
    CONNECT_NODE_IDS.has(nodeData?.display_name) ||
    CONNECT_NODE_IDS.has(nodeData?.node_id)
  );
}

function isPromptNodeDefinition(nodeData) {
  return (
    PROMPT_NODE_IDS.has(nodeData?.name) ||
    PROMPT_NODE_IDS.has(nodeData?.display_name) ||
    PROMPT_NODE_IDS.has(nodeData?.node_id)
  );
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget?.name === name);
}

function getModelWidget(node) {
  // The backend now ships `model` as a native combo, so there is exactly one
  // model widget — no injected duplicate. Match by id first, display name second.
  return getWidget(node, "model") || getWidget(node, "Model");
}

function normalizeModelList(models) {
  if (!Array.isArray(models)) {
    return [];
  }
  const out = [];
  const seen = new Set();
  for (const model of models) {
    const value = String(model ?? "").trim();
    if (!value || seen.has(value)) {
      continue;
    }
    seen.add(value);
    out.push(value);
  }
  return out;
}

// Keep the currently-selected model visible even before a refresh: a workflow
// reloaded from disk restores the saved value, but the combo only knows the
// placeholder until the server is queried again.
function ensureSelectionInOptions(node) {
  const widget = getModelWidget(node);
  if (!widget?.options) {
    return;
  }
  const current = String(widget.value ?? "").trim();
  const options = normalizeModelList(widget.options.values || []);
  if (options.length === 0) {
    options.push(MODEL_PLACEHOLDER);
  }
  if (current && current !== MODEL_PLACEHOLDER && !options.includes(current)) {
    options.unshift(current);
  }
  widget.options.values = options;
}

function applyModelOptions(node, models) {
  const widget = getModelWidget(node);
  if (!widget?.options) {
    return;
  }

  const current = String(widget.value ?? "").trim();
  let nextValues = normalizeModelList(models);
  if (current && current !== MODEL_PLACEHOLDER && !nextValues.includes(current)) {
    nextValues.unshift(current);
  }
  if (nextValues.length === 0) {
    nextValues = [MODEL_PLACEHOLDER];
  }

  widget.options.values = nextValues;
  widget.value = current && nextValues.includes(current) ? current : nextValues[0];
  widget.callback?.(widget.value);

  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

function buildQuery(node) {
  const serverWidget = getWidget(node, "server_url");
  const tokenWidget = getWidget(node, "api_token");
  const timeoutWidget = getWidget(node, "timeout_seconds");

  const serverUrl = String(serverWidget?.value ?? "").trim();
  if (!serverUrl) {
    throw new Error("Server URL is required.");
  }

  const params = new URLSearchParams();
  params.set("server_url", serverUrl);
  params.set("api_token", String(tokenWidget?.value ?? "-").trim() || "-");
  if (timeoutWidget?.value != null) {
    params.set("timeout_seconds", String(timeoutWidget.value));
  }
  return params;
}

async function fetchJson(path) {
  const response = await api.fetchApi(path, { method: "GET" });
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }

  if (!response.ok || payload?.ok === false) {
    const reason = payload?.error || `Request failed (${response.status})`;
    throw new Error(reason);
  }

  return payload;
}

function notify(severity, message) {
  const summary = severity === "error" ? "LMStudio — Error" : "LMStudio";
  if (typeof app.extensionManager?.toast?.add === "function") {
    app.extensionManager.toast.add({
      severity,
      summary,
      detail: message,
      life: severity === "error" ? 6000 : 3500,
    });
  } else if (severity === "error") {
    console.error(`[LMStudio] ${message}`);
  } else {
    console.info(`[LMStudio] ${message}`);
  }
}

// Run an async button handler with a "…" progress label and unified error toasts,
// so the user gets feedback on the node instead of a blocking window.alert.
async function runWithButtonFeedback(button, busyLabel, action, failPrefix) {
  const originalLabel = button.name;
  button.name = busyLabel;
  button.disabled = true;
  try {
    await action();
  } catch (error) {
    const message = error?.message || String(error);
    console.error(`[LMStudio] ${failPrefix}`, error);
    notify("error", `${failPrefix}: ${message}`);
  } finally {
    button.name = originalLabel;
    button.disabled = false;
  }
}

function fitNodeToWidgets(node) {
  if (typeof node.computeSize !== "function" || typeof node.setSize !== "function") {
    return;
  }

  const computed = node.computeSize();
  if (!Array.isArray(computed) || computed.length < 2) {
    return;
  }

  const width = Math.max(node.size?.[0] ?? 0, computed[0] ?? 0);
  const height = Math.max(node.size?.[1] ?? 0, computed[1] ?? 0);
  node.setSize([width, height]);
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

function makeTextareaResizable(node, widgetName) {
  const widget = getWidget(node, widgetName);
  if (!widget || widget.__lmstudioResizableApplied) {
    return;
  }

  const textarea = widget.inputEl;
  if (!textarea || textarea.tagName !== "TEXTAREA") {
    return;
  }

  textarea.style.resize = "vertical";
  textarea.style.overflowY = "auto";
  if (!textarea.style.minHeight) {
    textarea.style.minHeight = "92px";
  }

  if (typeof widget.computeSize === "function" && !widget.__lmstudioWrappedComputeSize) {
    const baseComputeSize = widget.computeSize.bind(widget);
    widget.computeSize = (width) => {
      const size = baseComputeSize(width);
      const fallback = Array.isArray(size) ? size : [0, 0];
      const dynamicHeight = Math.max(fallback[1] || 0, (textarea.offsetHeight || 0) + 10);
      return [fallback[0] || 0, dynamicHeight];
    };
    widget.__lmstudioWrappedComputeSize = true;
  }

  const onResizeOrInput = () => fitNodeToWidgets(node);
  textarea.addEventListener("input", onResizeOrInput);
  textarea.addEventListener("mouseup", onResizeOrInput);

  widget.__lmstudioResizableApplied = true;
}

function attachResizablePromptWidgets(node) {
  if (node.__lmstudioPromptResizeAttached) {
    return;
  }

  // Widgets may not be fully mounted at the first tick.
  const tryAttach = () => {
    makeTextareaResizable(node, "system_prompt");
    makeTextareaResizable(node, "user_prompt");
    fitNodeToWidgets(node);
  };

  tryAttach();
  setTimeout(tryAttach, 0);
  setTimeout(tryAttach, 120);
  setTimeout(tryAttach, 300);

  node.__lmstudioPromptResizeAttached = true;
}

function attachButtons(node) {
  if (node.__lmstudioButtonsAttached) {
    return;
  }

  ensureSelectionInOptions(node);

  const refreshModels = async () => {
    const query = buildQuery(node);
    const payload = await fetchJson(`/lmstudio/models?${query.toString()}`);
    const models = payload.models || [];
    applyModelOptions(node, models);
    notify("success", `Loaded ${models.length} model(s). Pick one from the Model dropdown.`);
  };

  const testConnection = async () => {
    const query = buildQuery(node);
    const payload = await fetchJson(`/lmstudio/test?${query.toString()}`);
    const models = payload.models || [];
    applyModelOptions(node, models);
    notify("success", payload.message || `Connected. ${models.length} model(s) available.`);
  };

  const refreshButton = node.addWidget("button", "🔄  Refresh Models", null, () =>
    runWithButtonFeedback(
      refreshButton,
      "Refreshing…",
      refreshModels,
      "Model refresh failed"
    )
  );

  const testButton = node.addWidget("button", "🔌  Test Connection", null, () =>
    runWithButtonFeedback(
      testButton,
      "Testing…",
      testConnection,
      "Connectivity test failed"
    )
  );

  node.__lmstudioButtonsAttached = true;
}

app.registerExtension({
  name: "lmstudio.connect.node",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (isConnectNodeDefinition(nodeData)) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        attachButtons(this);
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        ensureSelectionInOptions(this);
      };
    }

    if (isPromptNodeDefinition(nodeData)) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        attachResizablePromptWidgets(this);
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        attachResizablePromptWidgets(this);
      };
    }
  },
});
