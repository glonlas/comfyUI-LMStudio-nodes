from __future__ import annotations

from typing import Any

from comfy_api.latest import io, ui

from .client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    MODEL_PLACEHOLDER,
    get_server_models,
    normalize_api_key,
    normalize_server_url,
)
from .iotypes import ParamConnection
from .models import LMStudioConnectionPayload


class LMStudioConnect(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LMStudio_Connect",
            display_name="LMStudio - Connect",
            category="LMStudio",
            description=(
                "Creates a reusable LMStudio connection for downstream nodes. "
                "Use the refresh/test buttons on this node to fetch models from the remote server."
            ),
            inputs=[
                io.String.Input(
                    id="server_url",
                    display_name="Server URL",
                    default="http://127.0.0.1:1234",
                    placeholder="http://192.0.2.10:1234",
                    tooltip="LMStudio server URL (without /v1).",
                ),
                io.String.Input(
                    id="api_token",
                    display_name="API Token",
                    default="-",
                    placeholder="Leave '-' for lm-studio",
                    tooltip="Bearer token for LMStudio OpenAI-compatible API.",
                ),
                io.String.Input(
                    id="model",
                    display_name="Model",
                    default=MODEL_PLACEHOLDER,
                    tooltip=(
                        "Model id. The web extension renders this field as a dropdown and syncs the selected model value."
                    ),
                ),
                io.Boolean.Input(
                    id="reasoning_enabled",
                    display_name="Enable Reasoning",
                    default=False,
                    tooltip="Enable model reasoning hints when using the responses endpoint.",
                ),
                io.Boolean.Input(
                    id="use_tooling_mcp",
                    display_name="Use Tooling / MCP",
                    default=False,
                    tooltip=(
                        "Expose intent to use MCP tooling. Useful only when your target model/session is configured "
                        "for tool-enabled responses."
                    ),
                ),
                io.Int.Input(
                    id="max_tokens",
                    display_name="Max Token",
                    default=DEFAULT_MAX_TOKENS,
                    min=1,
                    max=1_000_000,
                    tooltip="Max output tokens for downstream generation nodes.",
                    advanced=True,
                ),
                io.Float.Input(
                    id="temperature",
                    display_name="Temperature",
                    default=DEFAULT_TEMPERATURE,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Sampling temperature for downstream generation nodes.",
                    advanced=True,
                ),
                io.Int.Input(
                    id="timeout_seconds",
                    display_name="Connection Timeout (seconds)",
                    default=DEFAULT_TIMEOUT_SECONDS,
                    min=1,
                    max=3600,
                    tooltip="HTTP request timeout for all LMStudio calls.",
                    advanced=True,
                ),
                io.Boolean.Input(
                    id="test_connectivity",
                    display_name="Test Connectivity On Execute",
                    default=True,
                    tooltip="When enabled, validate server reachability and model availability during execution.",
                    advanced=True,
                ),
            ],
            outputs=[
                ParamConnection.Output(
                    id="connection",
                    display_name="Connection",
                    tooltip="Reusable LMStudio connection payload for text/image nodes.",
                ),
                io.String.Output(
                    id="status",
                    display_name="Status",
                    tooltip="Connection status summary.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        server_url: str,
        timeout_seconds: int,
        max_tokens: int,
        **_: Any,
    ) -> bool | str:
        try:
            normalize_server_url(server_url)
        except ValueError as exc:
            return str(exc)

        if timeout_seconds < 1:
            return "timeout_seconds must be >= 1"
        if max_tokens < 1:
            return "max_tokens must be >= 1"
        return True

    @classmethod
    def execute(
        cls,
        server_url: str,
        api_token: str | None,
        model: str,
        reasoning_enabled: bool,
        test_connectivity: bool,
        max_tokens: int,
        temperature: float,
        timeout_seconds: int,
        use_tooling_mcp: bool,
    ) -> io.NodeOutput:
        normalized_server_url = normalize_server_url(server_url)
        model_name = (model or "").strip()
        models: list[str] = []

        should_probe_models = test_connectivity or model_name in {"", MODEL_PLACEHOLDER}
        if should_probe_models:
            models = get_server_models(
                server_url=normalized_server_url,
                api_key=api_token,
                timeout_seconds=timeout_seconds,
            )

        if model_name in {"", MODEL_PLACEHOLDER}:
            if models:
                model_name = models[0]
            else:
                raise ValueError(
                    "No model selected and no loaded model was discovered on the LMStudio server."
                )

        if models and model_name not in models:
            raise ValueError(
                f"Selected model '{model_name}' is not available on the server. "
                "Use the refresh button and pick a loaded model."
            )

        payload = LMStudioConnectionPayload(
            server_url=normalized_server_url,
            base_url=f"{normalized_server_url}/v1",
            api_key=normalize_api_key(api_token),
            model=model_name,
            reasoning_enabled=reasoning_enabled,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            use_tooling_mcp=use_tooling_mcp,
        )

        model_count = len(models)
        if should_probe_models:
            status = (
                f"Connected to {normalized_server_url}. "
                f"Found {model_count} model(s). Using '{model_name}'."
            )
        else:
            status = f"Connection prepared for {normalized_server_url}. Using '{model_name}'."

        return io.NodeOutput(
            payload,
            status,
            ui=ui.PreviewText(status),
        )
