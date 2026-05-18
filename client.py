from __future__ import annotations

import base64
import math
import re
import secrets
from io import BytesIO
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from openai import OpenAI


DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_API_KEY_PLACEHOLDER = "-"
DEFAULT_API_KEY_FALLBACK = "lm-studio"
MODEL_PLACEHOLDER = "<refresh models>"


def normalize_server_url(server_url: str) -> str:
    value = (server_url or "").strip()
    if not value:
        raise ValueError("server_url must not be empty")

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("server_url must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("server_url must include host and optional port")

    # Ensure no trailing slash and no duplicate /v1.
    value = value.rstrip("/")
    if value.lower().endswith("/v1"):
        value = value[:-3]
    return value


def to_openai_base_url(server_url: str) -> str:
    normalized = normalize_server_url(server_url)
    return f"{normalized}/v1"


def normalize_api_key(api_key: str | None) -> str:
    value = (api_key or "").strip()
    if not value or value == DEFAULT_API_KEY_PLACEHOLDER:
        return DEFAULT_API_KEY_FALLBACK
    return value


def create_openai_client(
    *,
    server_url: str,
    api_key: str | None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = 1,
):
    # Local import to keep module importable in environments without openai installed.
    from openai import OpenAI

    base_url = to_openai_base_url(server_url)
    token = normalize_api_key(api_key)
    return OpenAI(
        api_key=token,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def list_models(client: OpenAI) -> list[str]:
    response = client.models.list()
    ids: list[str] = []
    for model in getattr(response, "data", []):
        model_id = getattr(model, "id", None)
        if isinstance(model_id, str) and model_id:
            ids.append(model_id)
    # Stable ordering for deterministic dropdown behavior.
    return sorted(set(ids))


def get_server_models(
    *,
    server_url: str,
    api_key: str | None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[str]:
    client = create_openai_client(
        server_url=server_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    return list_models(client)


def coerce_seed(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "seed provided as list/tuple must contain exactly one element"
            )
        return coerce_seed(value[0])

    # Support numpy scalar-like values.
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("seed float must be finite")
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception as exc:
            raise ValueError("seed bytes must be utf-8 decodable") from exc
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped, 10)
        except ValueError as exc:
            raise ValueError(f"seed string must be an integer, got: {value!r}") from exc

    raise ValueError(
        "Unsupported seed type. Allowed: int, float, str, bytes, bytearray, "
        "None, or single-item list/tuple."
    )


def resolve_request_seed(value: Any) -> int:
    seed = coerce_seed(value)
    if seed is None or seed == -1:
        return secrets.randbelow(2**63 - 1) + 1
    if seed < -1:
        raise ValueError("seed must be -1 or >= 0")
    return seed


def comfy_image_to_base64_png_url(image_tensor) -> str:
    image_np = image_tensor.cpu().numpy()
    if image_np.ndim == 4:
        # Comfy image batches are [B, H, W, C]; for single-image prompts use first frame.
        image_np = image_np[0]
    i = np.multiply(255.0, image_np)
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64_png = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_png}"


def build_responses_input_text(user_prompt: str | None) -> list[dict[str, Any]]:
    prompt_text = user_prompt or ""
    return [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": prompt_text}],
        }
    ]


def extract_chat_completion_text(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(part, dict):
                dict_text = part.get("text")
                if isinstance(dict_text, str):
                    parts.append(dict_text)
        return "\n".join(p for p in parts if p)
    return ""


def extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                # Common response shape: {"content": [{"type": "output_text", "text": "..."}]}
                content = item.get("content", [])
                if isinstance(content, list):
                    for entry in content:
                        if isinstance(entry, dict):
                            text = entry.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                continue

            content = getattr(item, "content", None)
            if isinstance(content, list):
                for entry in content:
                    text = getattr(entry, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)

        if parts:
            return "\n".join(parts)

    return ""


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_THINK_START_RE = re.compile(r"<think>.*$", flags=re.IGNORECASE | re.DOTALL)


def strip_think_content(text: str | None) -> str:
    """
    Remove reasoning blocks enclosed in <think>...</think>.
    Also removes dangling <think> blocks without a closing tag.
    """
    if not text:
        return ""

    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_START_RE.sub("", cleaned)
    return cleaned.strip()


def build_chat_messages(system_prompt: str | None, user_prompt: str | None) -> list[dict[str, Any]]:
    system_text = system_prompt or ""
    user_text = user_prompt or ""
    messages: list[dict[str, Any]] = []
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return messages


def build_chat_messages_with_image(
    system_prompt: str | None,
    user_prompt: str | None,
    image_data_url: str,
) -> list[dict[str, Any]]:
    system_text = system_prompt or ""
    user_text = user_prompt or ""
    messages: list[dict[str, Any]] = []
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    )
    return messages
