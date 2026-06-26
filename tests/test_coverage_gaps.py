"""
Tests targeting the remaining uncovered branches in client.py and image_to_text_node.py.

Coverage gaps addressed (as reported by pytest-cov on the main branch):
  client.py         : lines 117-118, 142, 207, 222->228, 224->223, 226->223, 234->232, 291->293
  image_to_text_node: lines 112->114, 157, 181, 183->189
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import client
from helpers.fakes import FakeTensor, make_chat_completion, ns
from helpers.imports import import_repo_module


# ---------------------------------------------------------------------------
# client.py – coerce_seed: .item() call raises, falls back to plain value
# ---------------------------------------------------------------------------

class _BadItem:
    """Numpy-scalar-like object whose .item() always raises."""

    def item(self):
        raise RuntimeError("item() failed")


def test_coerce_seed_item_raises_falls_back_to_unsupported_type() -> None:
    """
    When value has .item() but the call raises, coerce_seed catches the exception
    (lines 117-118) and falls through to the type-check below. Since _BadItem is
    not int/float/str/bytes, it must then hit the final raise (line 142).
    """
    with pytest.raises(ValueError, match="Unsupported seed type"):
        client.coerce_seed(_BadItem())


def test_coerce_seed_unsupported_type_raises() -> None:
    """
    A plain object with no recognised numeric/string interface must raise the
    'Unsupported seed type' error (line 142).
    """
    with pytest.raises(ValueError, match="Unsupported seed type"):
        client.coerce_seed(object())

    with pytest.raises(ValueError, match="Unsupported seed type"):
        client.coerce_seed({"a": 1})


# ---------------------------------------------------------------------------
# client.py – extract_chat_completion_text: content=[] returns ""
# ---------------------------------------------------------------------------

def test_extract_chat_completion_text_empty_list_content() -> None:
    """
    When the message.content is an empty list the function should return "".
    This exercises the branch at line 207 where content is a list but parts
    ends up empty so "\n".join returns "".
    """
    completion = make_chat_completion([])
    assert client.extract_chat_completion_text(completion) == ""


def test_extract_chat_completion_text_list_with_no_usable_text() -> None:
    """
    List content items where getattr(part, 'text', None) is None and the item
    is also not a dict – they are silently skipped, exercising the branches
    at lines 222->228 / 224->223 / 226->223.
    """
    content = [
        ns(other="no-text"),          # has no .text, not a dict – skipped
        ns(text=None),                # text attr is None – skipped
        {"other": "ignored"},         # dict without 'text' key – skipped
        ns(text="kept"),              # text attr is str – included
    ]
    completion = make_chat_completion(content)
    assert client.extract_chat_completion_text(completion) == "kept"


def test_extract_chat_completion_text_non_list_non_str_content() -> None:
    """
    When message.content is neither a str nor a list (e.g. an int or None),
    the function should return "" via the final branch at line 207.
    """
    assert client.extract_chat_completion_text(make_chat_completion(42)) == ""
    assert client.extract_chat_completion_text(make_chat_completion(None)) == ""


# ---------------------------------------------------------------------------
# client.py – extract_responses_text: output list with no content at all
# ---------------------------------------------------------------------------

def test_extract_responses_text_output_list_empty_items() -> None:
    """
    output is a list but its items carry neither usable 'content' lists nor
    a 'text' attribute, so parts stays empty and the function returns "".
    This exercises the branch at line 234->232.
    """
    response = ns(output_text="", output=[ns(content=None), ns()])
    assert client.extract_responses_text(response) == ""


def test_extract_responses_text_output_none() -> None:
    """output attribute is None – falls through to return "" at the end."""
    assert client.extract_responses_text(ns(output_text="", output=None)) == ""


# ---------------------------------------------------------------------------
# client.py – build_chat_messages_with_image: no system prompt (line 291->293)
# ---------------------------------------------------------------------------

def test_build_chat_messages_with_image_no_system_prompt() -> None:
    """
    When system_prompt is blank/empty the system message must be omitted so
    only the user message is present.  This covers branch 291->293.
    """
    messages = client.build_chat_messages_with_image(
        system_prompt="  ",
        user_prompt="describe it",
        image_data_url="data:image/png;base64,abc==",
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    user_content = messages[0]["content"]
    assert user_content[0] == {"type": "text", "text": "describe it"}
    assert user_content[1]["type"] == "image_url"

    # None system_prompt is treated the same way.
    messages_none = client.build_chat_messages_with_image(
        system_prompt=None,
        user_prompt="hello",
        image_data_url="data:image/png;base64,xyz==",
    )
    assert len(messages_none) == 1


# ---------------------------------------------------------------------------
# image_to_text_node.py – _responses_kwargs: empty system_prompt (line 112->114)
# ---------------------------------------------------------------------------

def _image_module():
    return import_repo_module("image_to_text_node", force_reload=True)


def _connection_payload(*, reasoning_enabled: bool = False, use_tooling_mcp: bool = False):
    models = import_repo_module("models", force_reload=True)
    return models.LMStudioConnectionPayload(
        server_url="http://127.0.0.1:1234",
        base_url="http://127.0.0.1:1234/v1",
        api_key="token",
        model="vision-model",
        reasoning_enabled=reasoning_enabled,
        max_tokens=128,
        temperature=0.3,
        timeout_seconds=30,
        use_tooling_mcp=use_tooling_mcp,
    )


def _single_image() -> FakeTensor:
    return FakeTensor(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))


def test_responses_kwargs_no_system_prompt_omits_instructions() -> None:
    """
    When system_prompt is blank the 'instructions' key must NOT appear in
    the returned kwargs.  This covers the branch at line 112->114.
    """
    image_node = _image_module()
    kwargs = image_node.LMStudioImageToText._responses_kwargs(
        connection=_connection_payload(),
        image=_single_image(),
        system_prompt="",
        user_prompt="describe",
        seed=1,
    )
    assert "instructions" not in kwargs

    kwargs_whitespace = image_node.LMStudioImageToText._responses_kwargs(
        connection=_connection_payload(),
        image=_single_image(),
        system_prompt="   ",
        user_prompt="describe",
        seed=1,
    )
    assert "instructions" not in kwargs_whitespace


# ---------------------------------------------------------------------------
# image_to_text_node.py – execute: responses returns empty text (line 157)
# ---------------------------------------------------------------------------

def test_execute_responses_empty_text_falls_back_to_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the responses endpoint returns an empty string the node must raise
    a ValueError internally and fall back to chat.completions.
    This covers line 157 ('raise ValueError("responses endpoint returned no text output")').
    """
    image_node = _image_module()

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="chat fallback text"))]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: SimpleNamespace(output_text="")),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)),
    )
    monkeypatch.setattr(image_node, "resolve_request_seed", lambda _: 5)
    monkeypatch.setattr(image_node, "create_openai_client", lambda **_: fake_client)

    output = image_node.LMStudioImageToText.execute(
        connection=_connection_payload(),
        image=_single_image(),
        system_prompt="sys",
        user_prompt="describe",
        seed=-1,
    )

    assert output[0] == "chat fallback text"
    assert "via chat.completions" in output.ui.text
    assert "responses endpoint returned no text output" in output.ui.text


# ---------------------------------------------------------------------------
# image_to_text_node.py – execute: chat fallback also empty (line 181)
# ---------------------------------------------------------------------------

def test_execute_both_endpoints_empty_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When both the responses endpoint and the chat.completions fallback return
    empty text, a ValueError must propagate to the caller.
    This covers line 181 ('raise ValueError("chat.completions fallback returned no text output")').
    """
    image_node = _image_module()

    empty_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: SimpleNamespace(output_text="")),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: empty_completion)),
    )
    monkeypatch.setattr(image_node, "resolve_request_seed", lambda _: 3)
    monkeypatch.setattr(image_node, "create_openai_client", lambda **_: fake_client)

    with pytest.raises(ValueError, match="chat.completions fallback returned no text output"):
        image_node.LMStudioImageToText.execute(
            connection=_connection_payload(),
            image=_single_image(),
            system_prompt="sys",
            user_prompt="describe",
            seed=0,
        )


# ---------------------------------------------------------------------------
# image_to_text_node.py – execute: chat fallback single image, no dropped-frames note (line 183->189)
# ---------------------------------------------------------------------------

def test_execute_chat_fallback_single_image_no_dropped_frames_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the responses endpoint fails and the image is a *single* frame (not a
    batch), the fallback status message must NOT mention dropped frames.
    This covers the branch at line 183->189 where batch_size == 1.
    """
    image_node = _image_module()

    def raise_responses(**kwargs):
        raise RuntimeError("responses unavailable")

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="single fallback"))]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=raise_responses),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)),
    )
    monkeypatch.setattr(image_node, "resolve_request_seed", lambda _: 7)
    monkeypatch.setattr(image_node, "create_openai_client", lambda **_: fake_client)

    output = image_node.LMStudioImageToText.execute(
        connection=_connection_payload(),
        image=_single_image(),
        system_prompt="sys",
        user_prompt="describe",
        seed=0,
    )

    assert output[0] == "single fallback"
    assert "via chat.completions" in output.ui.text
    assert "dropped" not in output.ui.text
