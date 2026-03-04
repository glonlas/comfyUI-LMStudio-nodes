import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CONNECT_NODE_IDS = new Set(["LMStudio_Connect", "LMStudio - Connect"]);
const MODEL_PLACEHOLDER = "<refresh models>";

function isConnectNodeDefinition(nodeData) {
  return (
    CONNECT_NODE_IDS.has(nodeData?.name) ||
    CONNECT_NODE_IDS.has(nodeData?.display_name) ||
    CONNECT_NODE_IDS.has(nodeData?.node_id)
  );
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget?.name === name);
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

function ensureModelDropdown(node) {
  if (node.__lmstudioModelDropdown && node.__lmstudioModelTextWidget) {
    return {
      comboWidget: node.__lmstudioModelDropdown,
      textWidget: node.__lmstudioModelTextWidget,
    };
  }

  const modelTextWidget = getWidget(node, "model");
  if (!modelTextWidget) {
    return null;
  }

  modelTextWidget.hidden = true;
  const currentModel = String(modelTextWidget.value ?? "").trim();
  const initialValues = currentModel && currentModel !== MODEL_PLACEHOLDER
    ? [currentModel]
    : [MODEL_PLACEHOLDER];

  const comboWidget = node.addWidget(
    "combo",
    "Model",
    initialValues[0],
    (value) => {
      modelTextWidget.value = String(value ?? "").trim();
    },
    { values: initialValues }
  );

  node.__lmstudioModelDropdown = comboWidget;
  node.__lmstudioModelTextWidget = modelTextWidget;

  return { comboWidget, textWidget: modelTextWidget };
}

function syncDropdownFromText(node) {
  const refs = ensureModelDropdown(node);
  if (!refs) {
    return;
  }

  const { comboWidget, textWidget } = refs;
  const currentModel = String(textWidget.value ?? "").trim();
  const options = normalizeModelList(comboWidget?.options?.values || []);

  if (!currentModel) {
    return;
  }

  if (!options.includes(currentModel)) {
    options.unshift(currentModel);
    comboWidget.options.values = options;
  }
  comboWidget.value = currentModel;
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

function applyModelOptions(node, models) {
  const refs = ensureModelDropdown(node);
  if (!refs) {
    return;
  }

  const { comboWidget, textWidget } = refs;
  const currentModel = String(textWidget.value ?? "").trim();

  let nextValues = normalizeModelList(models);
  if (currentModel && currentModel !== MODEL_PLACEHOLDER && !nextValues.includes(currentModel)) {
    nextValues.unshift(currentModel);
  }
  if (nextValues.length === 0) {
    nextValues = [MODEL_PLACEHOLDER];
  }

  comboWidget.options.values = nextValues;

  const selected = currentModel && nextValues.includes(currentModel)
    ? currentModel
    : nextValues[0];

  comboWidget.value = selected;
  textWidget.value = selected;

  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

function notifySuccess(message) {
  if (typeof app.extensionManager?.toast?.add === "function") {
    app.extensionManager.toast.add({
      severity: "success",
      summary: "LMStudio",
      detail: message,
      life: 3500,
    });
  } else {
    console.info(`[LMStudio] ${message}`);
  }
}

function attachButtons(node) {
  if (node.__lmstudioButtonsAttached) {
    return;
  }

  ensureModelDropdown(node);
  syncDropdownFromText(node);

  const refreshModels = async () => {
    const query = buildQuery(node);
    const payload = await fetchJson(`/lmstudio/models?${query.toString()}`);
    applyModelOptions(node, payload.models || []);
  };

  const testConnection = async () => {
    const query = buildQuery(node);
    const payload = await fetchJson(`/lmstudio/test?${query.toString()}`);
    const models = payload.models || [];
    applyModelOptions(node, models);

    const message = payload.message || `Connected. ${models.length} model(s) available.`;
    notifySuccess(message);
  };

  node.addWidget("button", "Refresh Models", null, async () => {
    try {
      await refreshModels();
    } catch (error) {
      const message = error?.message || String(error);
      console.error("[LMStudio] Model refresh failed", error);
      window.alert(`Model refresh failed: ${message}`);
    }
  });

  node.addWidget("button", "Test Connection", null, async () => {
    try {
      await testConnection();
    } catch (error) {
      const message = error?.message || String(error);
      console.error("[LMStudio] Connectivity test failed", error);
      window.alert(`Connectivity test failed: ${message}`);
    }
  });

  node.__lmstudioButtonsAttached = true;
}

app.registerExtension({
  name: "lmstudio.connect.node",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isConnectNodeDefinition(nodeData)) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      attachButtons(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      ensureModelDropdown(this);
      syncDropdownFromText(this);
    };
  },
});
