from __future__ import annotations

import base64
from io import BytesIO
import sys
import types

import numpy as np
import pytest
from PIL import Image

import client
from helpers.fakes import FakeTensor, make_chat_completion, ns


def _decode_data_url_png(data_url: str) -> Image.Image:
    assert data_url.startswith("data:image/png;base64,")
    encoded = data_url.split(",", maxsplit=1)[1]
    raw = base64.b64decode(encoded)
    return Image.open(BytesIO(raw))


def test_normalize_server_url_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        client.normalize_server_url("")
    with pytest.raises(ValueError, match="http:// or https://"):
        client.normalize_server_url("ftp://127.0.0.1:1234")
    with pytest.raises(ValueError, match="include host"):
        client.normalize_server_url("http://")


def test_to_openai_base_url_appends_v1() -> None:
    assert client.to_openai_base_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234/v1"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "lm-studio"),
        ("", "lm-studio"),
        ("  ", "lm-studio"),
        ("-", "lm-studio"),
        ("  token-1  ", "token-1"),
    ],
)
def test_normalize_api_key(value: str | None, expected: str) -> None:
    assert client.normalize_api_key(value) == expected


def test_create_openai_client_passes_expected_args(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    created = client.create_openai_client(
        server_url="http://localhost:1234",
        api_key="  api  ",
        timeout_seconds=30,
        max_retries=4,
    )

    assert isinstance(created, FakeOpenAI)
    assert captured == {
        "api_key": "api",
        "base_url": "http://localhost:1234/v1",
        "timeout": 30,
        "max_retries": 4,
    }


def test_list_models_deduplicates_and_sorts() -> None:
    model_data = [
        ns(id="model-b"),
        ns(id="model-a"),
        ns(id="model-b"),
        ns(id=""),
        ns(id=None),
        ns(other="ignored"),
    ]
    fake_client = ns(models=ns(list=lambda: ns(data=model_data)))
    assert client.list_models(fake_client) == ["model-a", "model-b"]


def test_get_server_models_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = object()
    monkeypatch.setattr(client, "create_openai_client", lambda **_: fake_client)
    monkeypatch.setattr(client, "list_models", lambda _c: ["x", "y"])

    assert (
        client.get_server_models(
            server_url="http://localhost:1234",
            api_key="-",
            timeout_seconds=5,
        )
        == ["x", "y"]
    )


def test_coerce_seed_numpy_bool_and_bytes_error() -> None:
    assert client.coerce_seed(np.int64(12)) == 12
    assert client.coerce_seed(True) == 1
    assert client.coerce_seed(False) == 0

    with pytest.raises(ValueError, match="utf-8"):
        client.coerce_seed(b"\xff")


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_coerce_seed_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        client.coerce_seed(value)


def test_resolve_request_seed_handles_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client.secrets, "randbelow", lambda _: 42)
    assert client.resolve_request_seed(None) == 43
    assert client.resolve_request_seed(-1) == 43
    assert client.resolve_request_seed(11) == 11

    with pytest.raises(ValueError, match=">= 0"):
        client.resolve_request_seed(-2)


def test_comfy_image_to_base64_png_url_single_and_batch() -> None:
    single = FakeTensor(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
    url_single = client.comfy_image_to_base64_png_url(single)
    image_single = _decode_data_url_png(url_single)
    assert image_single.size == (1, 1)
    assert image_single.getpixel((0, 0))[:3] == (255, 0, 0)

    batch = FakeTensor(
        np.array(
            [
                [[[1.0, 0.0, 0.0]]],  # red
                [[[0.0, 0.0, 1.0]]],  # blue
            ],
            dtype=np.float32,
        )
    )
    url_batch = client.comfy_image_to_base64_png_url(batch)
    image_batch = _decode_data_url_png(url_batch)
    assert image_batch.getpixel((0, 0))[:3] == (255, 0, 0)


def test_build_responses_input_text_handles_none() -> None:
    payload = client.build_responses_input_text(None)
    assert payload == [{"role": "user", "content": [{"type": "input_text", "text": ""}]}]


def test_extract_chat_completion_text_variants() -> None:
    assert client.extract_chat_completion_text(ns(choices=[])) == ""
    assert client.extract_chat_completion_text(ns(choices=[ns(message=None)])) == ""
    assert client.extract_chat_completion_text(make_chat_completion("plain")) == "plain"

    content = [
        "line1",
        ns(text="line2"),
        {"text": "line3"},
        ns(other="ignored"),
        {"other": "ignored"},
    ]
    assert client.extract_chat_completion_text(make_chat_completion(content)) == "line1\nline2\nline3"


def test_extract_responses_text_variants() -> None:
    assert client.extract_responses_text(ns(output_text="from-output-text")) == "from-output-text"

    response_from_dict = ns(
        output=[{"content": [{"type": "output_text", "text": "from-dict"}]}]
    )
    assert client.extract_responses_text(response_from_dict) == "from-dict"

    response_from_object = ns(output=[ns(content=[ns(text="from-object-content")]), ns(text="from-object")])
    assert client.extract_responses_text(response_from_object) == "from-object-content\nfrom-object"

    assert client.extract_responses_text(ns(output_text="", output=[])) == ""


def test_extract_responses_text_no_duplicate_when_item_has_content_and_text() -> None:
    # An output item that has BOTH a content list and a top-level text attribute
    # should only yield text from the content list; the item-level text field
    # must NOT be appended a second time (regression guard for the duplicate-text bug).
    item_with_both = ns(content=[ns(text="the-answer")], text="the-answer")
    response = ns(output=[item_with_both])
    result = client.extract_responses_text(response)
    assert result == "the-answer", (
        f"Expected 'the-answer' but got {result!r} — "
        "item.text was appended even though content was already processed"
    )



def test_extract_responses_text_dict_item_non_list_content_is_skipped() -> None:
    # Branch 221->227: dict item where "content" is not a list.
    # The function should skip it (continue) and return empty string — not crash.
    response = ns(output=[{"content": "not-a-list"}, {"content": {"nested": "dict"}}])
    assert client.extract_responses_text(response) == ""


def test_extract_responses_text_dict_item_content_list_with_non_dict_entries() -> None:
    # Branch 223->222: content list contains non-dict entries (e.g. plain strings).
    # Those entries must be silently ignored; only dict entries with a str "text" key count.
    response = ns(output=[{"content": ["string-entry", 42, None, {"text": "kept"}]}])
    assert client.extract_responses_text(response) == "kept"


def test_extract_responses_text_dict_item_content_entry_text_non_string() -> None:
    # Branch 225->222: content entry is a dict but "text" value is not a str.
    # These entries must be silently ignored.
    response = ns(output=[{"content": [{"text": 42}, {"text": None}, {"text": "good"}]}])
    assert client.extract_responses_text(response) == "good"


def test_extract_responses_text_object_item_content_entry_text_non_string() -> None:
    # Branch 233->231: object item with content list where getattr(entry, "text") is not a str.
    # The entry should be skipped; other valid entries should still be collected.
    bad_entry = ns(text=99)
    good_entry = ns(text="valid")
    response = ns(output=[ns(content=[bad_entry, good_entry])])
    assert client.extract_responses_text(response) == "valid"


def test_strip_think_content_case_insensitive_and_multiple() -> None:
    text = "A\n<THINK>hidden-1</THINK>\nB\n<think>hidden-2</think>\nC"
    assert client.strip_think_content(text) == "A\n\nB\n\nC"
    assert client.strip_think_content("Visible\n<think>unterminated") == "Visible"
    assert client.strip_think_content(None) == ""


def test_build_chat_messages_with_and_without_system_prompt() -> None:
    with_system = client.build_chat_messages("System", "User")
    assert with_system[0] == {"role": "system", "content": "System"}
    assert with_system[1] == {"role": "user", "content": "User"}

    no_system = client.build_chat_messages("   ", "User")
    assert no_system == [{"role": "user", "content": "User"}]


def test_build_chat_messages_with_image_shape() -> None:
    messages = client.build_chat_messages_with_image(
        system_prompt="sys",
        user_prompt="describe",
        image_data_url="data:image/png;base64,abc",
    )
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0] == {"type": "text", "text": "describe"}
    assert messages[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }
