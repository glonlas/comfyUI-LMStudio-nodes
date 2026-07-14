from __future__ import annotations

from comfy_api.latest import io, ui

from .client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING,
    DEFAULT_TIMEOUT_SECONDS,
    MODEL_PLACEHOLDER,
    THINK_MODE_DEFAULT,
    THINK_MODE_FORCE,
    THINK_MODE_SUPPRESS,
    THINKING_OPTIONS,
    get_server_models,
    normalize_api_key,
    normalize_server_url,
    resolve_thinking_mode,
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
                "Set the Server URL, then click 'Refresh Models' to load and pick a model. "
                "'Test Connection' verifies the server is reachable."
            ),
            inputs=[
                io.String.Input(
                    id="server_url",
                    display_name="Server URL",
                    default="http://127.0.0.1:1234",
                    placeholder="http://10.168.168.7:1234",
                    tooltip="LMStudio server address, e.g. http://127.0.0.1:1234 (no /v1).",
                ),
                io.Combo.Input(
                    id="model",
                    display_name="Model",
                    options=[MODEL_PLACEHOLDER],
                    default=MODEL_PLACEHOLDER,
                    tooltip=(
                        "Loaded model to use. Click 'Refresh Models' to populate this "
                        "dropdown from the server, then pick one."
                    ),
                ),
                io.Combo.Input(
                    id="thinking",
                    display_name="Thinking / Reasoning",
                    options=list(THINKING_OPTIONS),
                    default=DEFAULT_THINKING,
                    tooltip=(
                        "Controls the model's reasoning phase.\n"
                        "• auto — let the model decide; Gemma-family models are nudged on "
                        "(they stay silent otherwise).\n"
                        "• on — force reasoning on (enable_thinking:true).\n"
                        "• off — suppress reasoning so the model answers directly, saving tokens."
                    ),
                ),
                io.String.Input(
                    id="api_token",
                    display_name="API Token",
                    default="-",
                    placeholder="Leave '-' for lm-studio",
                    tooltip="Bearer token for the LMStudio OpenAI-compatible API. Leave '-' for local servers.",
                    advanced=True,
                ),
                io.Boolean.Input(
                    id="use_tooling_mcp",
                    display_name="Use Tooling / MCP",
                    default=False,
                    tooltip=(
                        "Signal intent to use MCP tooling. Only useful when the target "
                        "model/session is configured for tool-enabled responses."
                    ),
                    advanced=True,
                ),
                io.Int.Input(
                    id="max_tokens",
                    display_name="Max Tokens",
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
        model: str | None = None,
    ) -> bool | str:
        # `model` is accepted here purely so ComfyUI skips its built-in combo
        # "value not in list" check: the real option list is only known at
        # runtime (fetched from the server), so the schema ships a placeholder.
        # Actual model availability is validated in execute().
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
        thinking: str,
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
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            use_tooling_mcp=use_tooling_mcp,
        )

        think_mode = resolve_thinking_mode(thinking, model_name)
        think_label = {
            THINK_MODE_FORCE: "on",
            THINK_MODE_SUPPRESS: "off",
            THINK_MODE_DEFAULT: "model default",
        }[think_mode]
        thinking_note = f" Thinking: {thinking} ({think_label})."

        model_count = len(models)
        if should_probe_models:
            status = (
                f"Connected to {normalized_server_url}. "
                f"Found {model_count} model(s). Using '{model_name}'.{thinking_note}"
            )
        else:
            status = (
                f"Connection prepared for {normalized_server_url}. "
                f"Using '{model_name}'.{thinking_note}"
            )

        return io.NodeOutput(
            payload,
            status,
            ui=ui.PreviewText(status),
        )
