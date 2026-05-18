from __future__ import annotations

from typing import Any

from comfy_api.latest import io, ui

from .client import (
    build_chat_messages_with_image,
    comfy_image_to_base64_png_url,
    create_openai_client,
    extract_chat_completion_text,
    extract_responses_text,
    resolve_request_seed,
    strip_think_content,
)
from .iotypes import ParamConnection
from .models import LMStudioConnectionPayload


class LMStudioImageToText(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LMStudio_ImageToText",
            display_name="LMStudio - Image To Text",
            category="LMStudio",
            description="Generate text from image(s) and prompts using an LMStudio vision-capable model.",
            inputs=[
                ParamConnection.Input(
                    id="connection",
                    display_name="Connection",
                    tooltip="Connection output from LMStudio - Connect node.",
                ),
                io.Image.Input(
                    id="image",
                    display_name="Image",
                    tooltip="Input image tensor.",
                ),
                io.String.Input(
                    id="system_prompt",
                    display_name="System Prompt (LM Souls)",
                    multiline=True,
                    default="You are LM Souls, an expert image analyst and prompt writer.",
                    tooltip="System instructions that define assistant behavior.",
                ),
                io.String.Input(
                    id="user_prompt",
                    display_name="User Prompt",
                    multiline=True,
                    default="Describe this image and provide an optimized generation prompt.",
                    tooltip="User instruction for image understanding.",
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
        image,
        system_prompt: str,
        user_prompt: str,
        seed: int,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]

        if image.ndim == 4:
            for idx in range(image.shape[0]):
                content.append(
                    {
                        "type": "input_image",
                        "image_url": comfy_image_to_base64_png_url(image[idx]),
                    }
                )
        else:
            content.append(
                {
                    "type": "input_image",
                    "image_url": comfy_image_to_base64_png_url(image),
                }
            )

        kwargs: dict[str, Any] = {
            "model": connection.model,
            "input": [{"role": "user", "content": content}],
            "temperature": connection.temperature,
            "max_output_tokens": connection.max_tokens,
            "seed": seed,
        }
        if system_prompt.strip():
            kwargs["instructions"] = system_prompt
        if connection.reasoning_enabled:
            kwargs["reasoning"] = {"effort": "medium"}
        if connection.use_tooling_mcp:
            kwargs["metadata"] = {"lmstudio_tooling_mcp_requested": "true"}

        return kwargs

    @classmethod
    def execute(
        cls,
        connection: LMStudioConnectionPayload,
        image,
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
                    image=image,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    seed=resolved_seed,
                )
            )
            text = extract_responses_text(response).strip()
            if not text:
                raise ValueError("responses endpoint returned no text output")
        except Exception as responses_error:
            via_endpoint = "chat.completions"
            fallback_reason = str(responses_error)

            is_batch = getattr(image, "ndim", 0) == 4
            batch_size = image.shape[0] if is_batch else 1
            first_image = image[0] if is_batch else image
            image_data_url = comfy_image_to_base64_png_url(first_image)

            try:
                completion = client.chat.completions.create(
                    model=connection.model,
                    messages=build_chat_messages_with_image(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_data_url=image_data_url,
                    ),
                    seed=resolved_seed,
                    temperature=connection.temperature,
                    max_tokens=connection.max_tokens,
                    n=1,
                )
                text = extract_chat_completion_text(completion).strip()
                if not text:
                    raise ValueError("chat.completions fallback returned no text output")
            except Exception as chat_error:
                raise RuntimeError(
                    f"Both endpoints failed. responses error: {fallback_reason}. "
                    f"chat.completions error: {chat_error}"
                ) from chat_error

            if batch_size > 1:
                fallback_reason += (
                    f" Note: chat.completions fallback only supports one image; "
                    f"{batch_size - 1} additional frame(s) were dropped."
                )

        text = strip_think_content(text)

        status = f"Model '{connection.model}' via {via_endpoint} (seed={resolved_seed})."
        if fallback_reason:
            status += f" Fallback reason: {fallback_reason}"

        return io.NodeOutput(
            text,
            ui=ui.PreviewText(status),
        )
