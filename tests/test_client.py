from __future__ import annotations

import pytest

import client


def test_normalize_server_url_trims_and_strips_v1() -> None:
    assert client.normalize_server_url(" http://10.168.168.7:1234/v1/ ") == "http://10.168.168.7:1234"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (42, 42),
        (42.9, 42),
        ("17", 17),
        (b"99", 99),
        ([123], 123),
        ((456,), 456),
    ],
)
def test_coerce_seed_supported_values(value, expected) -> None:
    assert client.coerce_seed(value) == expected


@pytest.mark.parametrize("value", ["", [], [1, 2], "abc", float("nan")])
def test_coerce_seed_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        client.coerce_seed(value)


def test_resolve_request_seed_negative_one_generates_positive_seed() -> None:
    resolved = client.resolve_request_seed(-1)
    assert isinstance(resolved, int)
    assert resolved > 0


def test_build_responses_input_text_shape() -> None:
    payload = client.build_responses_input_text("hello")
    assert payload[0]["role"] == "user"
    assert payload[0]["content"][0]["type"] == "input_text"
    assert payload[0]["content"][0]["text"] == "hello"


def test_build_chat_messages_handles_none_prompts() -> None:
    messages = client.build_chat_messages(None, None)
    assert messages == [{"role": "user", "content": ""}]
