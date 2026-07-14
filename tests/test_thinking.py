from __future__ import annotations

import pytest

from helpers.imports import import_repo_module


def _client():
    return import_repo_module("client", force_reload=True)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("google/gemma-3-12b", True),
        ("Gemma2-9B", True),
        ("gemma", True),
        ("qwen3-8b", False),
        ("llama-3.1-8b", False),
        ("", False),
        (None, False),
    ],
)
def test_model_thinks_off_by_default(model, expected) -> None:
    client = _client()
    assert client.model_thinks_off_by_default(model) is expected


@pytest.mark.parametrize(
    "thinking, model, expected",
    [
        # Explicit off/on always win regardless of family.
        ("off", "qwen3-8b", "suppress"),
        ("off", "gemma-3", "suppress"),
        ("on", "gemma-3", "force"),
        ("on", "qwen3-8b", "force"),
        # auto defers for thinking-by-default families, forces Gemma.
        ("auto", "qwen3-8b", "default"),
        ("auto", "google/gemma-3-12b", "force"),
        # Normalization: case/whitespace and unknown/blank fall back to auto.
        ("  OFF  ", "qwen", "suppress"),
        ("", "qwen", "default"),
        (None, "gemma-3", "force"),
    ],
)
def test_resolve_thinking_mode(thinking, model, expected) -> None:
    client = _client()
    assert client.resolve_thinking_mode(thinking, model) == expected


def test_thinking_chat_template_kwargs() -> None:
    client = _client()
    assert client.thinking_chat_template_kwargs(client.THINK_MODE_SUPPRESS) == {
        "enable_thinking": False
    }
    assert client.thinking_chat_template_kwargs(client.THINK_MODE_FORCE) == {
        "enable_thinking": True
    }
    assert client.thinking_chat_template_kwargs(client.THINK_MODE_DEFAULT) is None


def test_apply_thinking_prefill_only_on_suppress() -> None:
    client = _client()
    messages = [{"role": "user", "content": "hi"}]

    suppressed = client.apply_thinking_prefill(messages, client.THINK_MODE_SUPPRESS)
    assert suppressed[-1] == {"role": "assistant", "content": "<think></think>"}
    # Original list is not mutated.
    assert messages == [{"role": "user", "content": "hi"}]

    for mode in (client.THINK_MODE_FORCE, client.THINK_MODE_DEFAULT):
        assert client.apply_thinking_prefill(messages, mode) == messages
