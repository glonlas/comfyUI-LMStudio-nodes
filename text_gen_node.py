from __future__ import annotations

from typing import Any

from comfy_api.latest import io, ui

from .client import (
    build_chat_messages,
    build_responses_input_text,
    create_openai_client,
    dump_openai_response,
    extract_chat_completion_text,
    extract_responses_text,
    resolve_request_seed,
)
from .iotypes import ParamConnection
from .models import LMStudioConnectionPayload


class LMStudioTextGen(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LMStudio_TextGen",
            display_name="LMStudio - Text Gen",
            category="LMStudio",
            description="Generate text from an LMStudio connection with system and user prompts.",
            inputs=[
                ParamConnection.Input(
                    id="connection",
                    display_name="Connection",
                    tooltip="Connection output from LMStudio - Connect node.",
                ),
                io.String.Input(
                    id="system_prompt",
                    display_name="System Prompt (LM Souls)",
                    multiline=True,
                    default="You are LM Souls, an expert prompt engineer and creative assistant.",
                    tooltip="System instructions that define assistant behavior.",
                ),
                io.String.Input(
                    id="user_prompt",
                    display_name="User Prompt",
                    multiline=True,
                    default="Write a concise prompt for a cinematic portrait scene.",
                    tooltip="User input to transform into a final response.",
                ),
                io.Int.Input(
                    id="seed",
                    display_name="Seed",
                    default=-1,
                    min=-1,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.randomize,
                    tooltip="-1 uses a secure random seed.",
                ),
            ],
            outputs=[
                io.String.Output(
                    id="response_text",
                    display_name="Response",
                    tooltip="Generated text output.",
                ),
                io.Int.Output(
                    id="used_seed",
                    display_name="Used Seed",
                    tooltip="Resolved seed used for the request.",
                ),
                io.String.Output(
                    id="raw_response",
                    display_name="Raw Response",
                    tooltip="Raw model response serialized as JSON/text for debugging.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, user_prompt: str | None = None) -> bool | str:
        # user_prompt is often linked from another node; defer strict checks to execute().
        return True

    @classmethod
    def _responses_kwargs(
        cls,
        *,
        connection: LMStudioConnectionPayload,
        system_prompt: str,
        user_prompt: str,
        seed: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": connection.model,
            "input": build_responses_input_text(user_prompt),
            "temperature": connection.temperature,
            "max_output_tokens": connection.max_tokens,
            "seed": seed,
        }
        if system_prompt.strip():
            kwargs["instructions"] = system_prompt
        if connection.reasoning_enabled:
            kwargs["reasoning"] = {"effort": "medium"}

        # MCP/tooling requires additional tool definitions and server details.
        # Keep an explicit metadata signal for future extension without sending invalid tool payloads.
        if connection.use_tooling_mcp:
            kwargs["metadata"] = {"lmstudio_tooling_mcp_requested": "true"}

        return kwargs

    @classmethod
    def execute(
        cls,
        connection: LMStudioConnectionPayload,
        system_prompt: str,
        user_prompt: str,
        seed: int,
    ) -> io.NodeOutput:
        system_prompt = system_prompt or ""
        user_prompt = user_prompt or ""
        if not user_prompt.strip():
            raise ValueError("user_prompt must not be empty")

        resolved_seed = resolve_request_seed(seed)
        client = create_openai_client(
            server_url=connection.server_url,
            api_key=connection.api_key,
            timeout_seconds=connection.timeout_seconds,
        )

        via_endpoint = "responses"
        fallback_reason: str | None = None

        try:
            response = client.responses.create(
                **cls._responses_kwargs(
                    connection=connection,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    seed=resolved_seed,
                )
            )
            text = extract_responses_text(response).strip()
            if not text:
                raise ValueError("responses endpoint returned no text output")
            raw_response = dump_openai_response(response)
        except Exception as responses_error:
            via_endpoint = "chat.completions"
            fallback_reason = str(responses_error)

            completion = client.chat.completions.create(
                model=connection.model,
                messages=build_chat_messages(system_prompt, user_prompt),
                seed=resolved_seed,
                temperature=connection.temperature,
                max_tokens=connection.max_tokens,
                n=1,
            )
            text = extract_chat_completion_text(completion).strip()
            if not text:
                raise ValueError("chat.completions fallback returned no text output")
            raw_response = dump_openai_response(completion)

        status = f"Model '{connection.model}' via {via_endpoint} (seed={resolved_seed})."
        if fallback_reason:
            status += f" Fallback reason: {fallback_reason}"

        return io.NodeOutput(
            text,
            resolved_seed,
            raw_response,
            ui=ui.PreviewText(status),
        )
