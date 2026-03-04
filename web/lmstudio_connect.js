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
  const modelWidget = getWidget(node, "model");
  if (!modelWidget) {
    return;
  }

  const nextValues = Array.isArray(models) && models.length > 0 ? models : [MODEL_PLACEHOLDER];
  if (!modelWidget.options) {
    modelWidget.options = {};
  }
  modelWidget.options.values = nextValues;

  if (!nextValues.includes(modelWidget.value)) {
    modelWidget.value = nextValues[0];
  }

  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

function attachButtons(node) {
  if (node.__lmstudioButtonsAttached) {
    return;
  }

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
  };

  node.addWidget("button", "Refresh Models", null, async () => {
    try {
      await refreshModels();
    } catch (error) {
      console.error("[LMStudio] Model refresh failed", error);
      window.alert(`Model refresh failed: ${error.message}`);
    }
  });

  node.addWidget("button", "Test Connection", null, async () => {
    try {
      await testConnection();
    } catch (error) {
      console.error("[LMStudio] Connectivity test failed", error);
      window.alert(`Connectivity test failed: ${error.message}`);
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
  },
});
