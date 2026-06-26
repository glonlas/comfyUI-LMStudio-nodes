"""
Tests for behavioral contracts that 100% line/branch coverage leaves unverified.

Each test is annotated with the mutation it would catch (i.e. what regression
would slip through if the test didn't exist).
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import client
from helpers.imports import import_repo_module


# ---------------------------------------------------------------------------
# client.py – normalize_server_url
# ---------------------------------------------------------------------------


def test_normalize_server_url_whitespace_only_raises() -> None:
    """
    Whitespace-only input must raise 'must not be empty'.
    A mutation that strips before the emptiness check would silently accept
    whitespace and try to parse it, producing a confusing error or wrong URL.
    """
    with pytest.raises(ValueError, match="must not be empty"):
        client.normalize_server_url("   ")


def test_normalize_server_url_trailing_slash_no_v1() -> None:
    """
    A URL with a trailing slash but no /v1 suffix must have the slash stripped.
    If the strip was removed, to_openai_base_url would produce 'http://host//v1'.
    """
    assert client.normalize_server_url("http://127.0.0.1:1234/") == "http://127.0.0.1:1234"


def test_normalize_server_url_https_accepted() -> None:
    """
    https:// scheme must be accepted.
    A mutation that only allows http:// would silently break TLS setups.
    """
    assert client.normalize_server_url("https://example.com:5000") == "https://example.com:5000"


# ---------------------------------------------------------------------------
# client.py – coerce_seed: whitespace-only string
# ---------------------------------------------------------------------------


def test_coerce_seed_whitespace_only_string_returns_none() -> None:
    """
    A whitespace-only seed string (e.g. "   ") must return None (treated as
    'no seed provided'), exactly like an empty string.
    A mutation that compared `value == ""` instead of `not stripped` would
    return None for "" but raise ValueError for "   ", breaking nodes that
    receive padding from UI widgets.
    """
    assert client.coerce_seed("   ") is None
    assert client.coerce_seed("\t") is None


# ---------------------------------------------------------------------------
# client.py – resolve_request_seed: seed=0 is valid (boundary)
# ---------------------------------------------------------------------------


def test_resolve_request_seed_zero_is_valid() -> None:
    """
    seed=0 is explicitly >= 0 and not -1, so it must be returned as-is.
    A mutation of the seed == -1 check to seed <= 0 would incorrectly
    randomise seed=0 instead of honouring the deterministic value.
    """
    assert client.resolve_request_seed(0) == 0


def test_resolve_request_seed_large_positive_passes_through() -> None:
    """
    A large positive seed must pass through unchanged.
    Ensures no accidental clipping or modulo is applied.
    """
    big = 2**62
    assert client.resolve_request_seed(big) == big


# ---------------------------------------------------------------------------
# client.py – build_responses_input_text: content structure
# ---------------------------------------------------------------------------


def test_build_responses_input_text_content_type_is_input_text() -> None:
    """
    The 'type' field in the content entry MUST be 'input_text', not 'text'.
    A mutation swapping the type string would break LMStudio's Responses API
    but would not be caught by shape-only assertions.
    """
    payload = client.build_responses_input_text("hello world")
    entry = payload[0]["content"][0]
    assert entry["type"] == "input_text"
    assert entry["text"] == "hello world"
    # Exactly one content entry; no extra fields.
    assert len(payload[0]["content"]) == 1


# ---------------------------------------------------------------------------
# client.py – build_chat_messages: system prompt skipping edge
# ---------------------------------------------------------------------------


def test_build_chat_messages_whitespace_system_skipped() -> None:
    """
    A system_prompt that is only whitespace must be omitted (same as empty),
    so that only the user message appears.
    A mutation removing the .strip() call would include a whitespace-only
    system message, poisoning the model's context.
    """
    messages = client.build_chat_messages("   \t  ", "ask something")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "ask something"


def test_build_chat_messages_empty_user_prompt_included() -> None:
    """
    Even an empty user prompt must produce a user message (the slot must be
    present). A mutation that skipped the user message when content is empty
    would cause the API call to fail with a missing-message error.
    """
    messages = client.build_chat_messages("System", "")
    # system + user
    assert len(messages) == 2
    assert messages[1] == {"role": "user", "content": ""}


# ---------------------------------------------------------------------------
# client.py – list_models: ordering / dedup contract
# ---------------------------------------------------------------------------


def test_list_models_returns_sorted_unique_ids() -> None:
    """
    list_models must return a sorted, deduplicated list.
    A mutation removing sorted() would produce an arbitrary order, breaking
    deterministic dropdown selection in the UI.
    A mutation removing set() dedup would expose duplicate model entries.
    """
    from types import SimpleNamespace as NS

    model_data = [
        NS(id="z-model"),
        NS(id="a-model"),
        NS(id="m-model"),
        NS(id="a-model"),  # duplicate
        NS(id=""),         # empty – filtered
    ]
    fake_client = NS(models=NS(list=lambda: NS(data=model_data)))
    result = client.list_models(fake_client)
    assert result == ["a-model", "m-model", "z-model"]
    # Idempotent: second call produces same order.
    assert client.list_models(fake_client) == result


# ---------------------------------------------------------------------------
# client.py – strip_think_content: multiple blocks, ordering preserved
# ---------------------------------------------------------------------------


def test_strip_think_content_preserves_surrounding_text_order() -> None:
    """
    Text before and after think blocks must appear in the correct order.
    A mutation reversing the substitution or collapsing surrounding text
    would reorder the visible output.
    """
    result = client.strip_think_content("A <think>x</think> B <think>y</think> C")
    assert result == "A  B  C"


def test_strip_think_content_only_think_block_returns_empty() -> None:
    """
    A string consisting entirely of a think block must return "".
    A mutation that stripped surrounding text instead of the block would
    produce the block content rather than the empty string.
    """
    assert client.strip_think_content("<think>only reasoning</think>") == ""


# ---------------------------------------------------------------------------
# routes.py – _query_int: negative integer rejected
# ---------------------------------------------------------------------------


def _routes_module():
    return import_repo_module("routes", force_reload=True)


def test_query_int_negative_value_raises() -> None:
    """
    Negative integers (e.g. -1, -5) must be rejected with '>= 1'.
    The check is `value < 1`, which covers both 0 (already tested upstream)
    and negative numbers.  A mutation changing < 1 to == 0 would silently
    accept -1 as a timeout, causing immediate connection failures.
    """
    routes = _routes_module()
    for raw in ["-1", "-100"]:
        with pytest.raises(ValueError, match="must be >= 1"):
            routes._query_int(
                SimpleNamespace(query={"timeout_seconds": raw}),
                "timeout_seconds",
                15,
            )


def test_query_int_empty_string_uses_default() -> None:
    """
    An empty-string query value must fall back to the default, not raise.
    A mutation that only checked `raw is None` (not `raw == ""`) would try
    to parse "" and raise ValueError instead of returning the default.
    """
    routes = _routes_module()
    request = SimpleNamespace(query={"timeout_seconds": ""})
    assert routes._query_int(request, "timeout_seconds", 15) == 15


# ---------------------------------------------------------------------------
# routes.py – _models_handler: empty models list produces correct default_model
# ---------------------------------------------------------------------------


def test_models_handler_empty_models_default_model_is_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the server returns no models, default_model must be "" (empty string),
    not raise an IndexError.
    A mutation changing `models[0] if models else ""` to just `models[0]`
    would crash on empty lists.
    """
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://x:1234")
    monkeypatch.setattr(routes, "get_server_models", lambda **kwargs: [])
    request = SimpleNamespace(query={"server_url": "http://x:1234"})
    response = asyncio.run(routes._models_handler(request))
    payload = json.loads(response.text)
    assert response.status == 200
    assert payload["ok"] is True
    assert payload["default_model"] == ""
    assert payload["models"] == []


# ---------------------------------------------------------------------------
# routes.py – _test_handler: model_count matches actual list length
# ---------------------------------------------------------------------------


def test_test_handler_model_count_matches_models_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    model_count in the response must equal len(models).
    The existing test only checks model_count==1; this one checks that the
    count is derived from the list, not hardcoded.
    A mutation replacing len(models) with 1 would pass the existing test
    but fail here.
    """
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://x:1234")
    monkeypatch.setattr(routes, "get_server_models", lambda **kwargs: ["a", "b", "c"])
    request = SimpleNamespace(query={"server_url": "http://x:1234"})
    response = asyncio.run(routes._test_handler(request))
    payload = json.loads(response.text)
    assert payload["model_count"] == 3
    assert payload["models"] == ["a", "b", "c"]
    assert "3 model(s)" in payload["message"]


def test_test_handler_error_prefix_is_connection_test_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The 502 error message for _test_handler must use the prefix
    'Connection test failed', NOT 'Failed to list models' (which is the
    _models_handler prefix).  A copy-paste mutation that shared the same
    error string between the two handlers would confuse operator debugging.
    """
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://x:1234")
    monkeypatch.setattr(
        routes,
        "get_server_models",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("timeout")),
    )
    request = SimpleNamespace(query={"server_url": "http://x:1234"})
    response = asyncio.run(routes._test_handler(request))
    payload = json.loads(response.text)
    assert response.status == 502
    assert "Connection test failed" in payload["error"]
    assert "Failed to list models" not in payload["error"]


def test_models_handler_error_prefix_is_failed_to_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Symmetric guard: the 502 error from _models_handler must use
    'Failed to list models', not 'Connection test failed'.
    """
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://x:1234")
    monkeypatch.setattr(
        routes,
        "get_server_models",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    request = SimpleNamespace(query={"server_url": "http://x:1234"})
    response = asyncio.run(routes._models_handler(request))
    payload = json.loads(response.text)
    assert response.status == 502
    assert "Failed to list models" in payload["error"]
    assert "Connection test failed" not in payload["error"]


# ---------------------------------------------------------------------------
# connect_node.py – auto-discover model when model="" and test_connectivity=False
# ---------------------------------------------------------------------------


def _connect_module():
    return import_repo_module("connect_node", force_reload=True)


def test_execute_discovers_model_when_empty_and_connectivity_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When model="" (blank) and test_connectivity=False, the node must still
    probe the server (because no model is selected) and use the first result.
    should_probe_models is `test_connectivity OR model in {"", MODEL_PLACEHOLDER}`.
    A mutation changing `or` to `and` would skip the probe and raise ValueError
    ('No model selected') instead of auto-selecting.
    """
    connect_node = _connect_module()
    monkeypatch.setattr(connect_node, "get_server_models", lambda **_: ["auto-model"])

    output = connect_node.LMStudioConnect.execute(
        server_url="http://127.0.0.1:1234",
        api_token="-",
        model="",
        reasoning_enabled=False,
        test_connectivity=False,  # disabled, but model is blank → must still probe
        max_tokens=128,
        temperature=0.5,
        timeout_seconds=10,
        use_tooling_mcp=False,
    )
    payload = output[0]
    assert payload.model == "auto-model"


def test_execute_status_no_probe_does_not_say_found_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When test_connectivity=False and a model is explicitly set, the status
    message must say 'Connection prepared' and NOT say 'Found N model(s)'.
    A mutation that always used the probe-status template would expose
    incorrect model-count information.
    """
    connect_node = _connect_module()
    monkeypatch.setattr(
        connect_node,
        "get_server_models",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    output = connect_node.LMStudioConnect.execute(
        server_url="http://127.0.0.1:1234",
        api_token="-",
        model="chosen-model",
        reasoning_enabled=False,
        test_connectivity=False,
        max_tokens=128,
        temperature=0.5,
        timeout_seconds=10,
        use_tooling_mcp=False,
    )
    status = output[1]
    assert "Connection prepared" in status
    assert "Found" not in status


def test_execute_status_with_probe_includes_model_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the probe runs, the status must include 'Found N model(s)' with the
    actual count.
    A mutation hardcoding the count as 0 or 1 would pass the existing test
    (which uses 2 models) but would be exposed by checking a different count.
    """
    connect_node = _connect_module()
    monkeypatch.setattr(
        connect_node,
        "get_server_models",
        lambda **_: ["x", "y", "z"],
    )
    output = connect_node.LMStudioConnect.execute(
        server_url="http://127.0.0.1:1234",
        api_token="-",
        model="x",
        reasoning_enabled=False,
        test_connectivity=True,
        max_tokens=128,
        temperature=0.5,
        timeout_seconds=10,
        use_tooling_mcp=False,
    )
    status = output[1]
    assert "Found 3 model(s)" in status


# ---------------------------------------------------------------------------
# text_gen_node.py – _responses_kwargs: 'input' field is always present
# ---------------------------------------------------------------------------


def _text_module():
    return import_repo_module("text_gen_node", force_reload=True)


def _connection_payload(**overrides):
    models = import_repo_module("models", force_reload=True)
    base = dict(
        server_url="http://127.0.0.1:1234",
        base_url="http://127.0.0.1:1234/v1",
        api_key="token",
        model="model-a",
        reasoning_enabled=False,
        max_tokens=111,
        temperature=0.4,
        timeout_seconds=30,
        use_tooling_mcp=False,
    )
    base.update(overrides)
    return models.LMStudioConnectionPayload(**base)


def test_responses_kwargs_input_field_is_present_and_has_user_role() -> None:
    """
    The 'input' key in _responses_kwargs must be set to a list containing
    a user-role message with the prompt text.
    A mutation that removed or misspelled the 'input' key would send an
    incomplete payload that LMStudio's Responses API would reject.
    """
    text_gen_node = _text_module()
    kwargs = text_gen_node.LMStudioTextGen._responses_kwargs(
        connection=_connection_payload(),
        system_prompt="",
        user_prompt="describe the image",
        seed=1,
    )
    assert "input" in kwargs
    assert isinstance(kwargs["input"], list)
    assert len(kwargs["input"]) == 1
    assert kwargs["input"][0]["role"] == "user"
    # The content must carry the user_prompt text.
    content = kwargs["input"][0]["content"]
    assert any(
        (isinstance(entry, dict) and entry.get("text") == "describe the image")
        for entry in content
    ), f"user_prompt not found in content: {content!r}"


def test_responses_kwargs_temperature_and_max_tokens_forwarded() -> None:
    """
    temperature and max_output_tokens in _responses_kwargs must mirror the
    connection payload values exactly.
    A mutation swapping temperature and max_output_tokens, or using a
    hardcoded default, would break generation quality silently.
    """
    text_gen_node = _text_module()
    conn = _connection_payload(temperature=1.5, max_tokens=512)
    kwargs = text_gen_node.LMStudioTextGen._responses_kwargs(
        connection=conn,
        system_prompt="",
        user_prompt="hello",
        seed=7,
    )
    assert kwargs["temperature"] == 1.5
    assert kwargs["max_output_tokens"] == 512


# ---------------------------------------------------------------------------
# text_gen_node.py – execute: seed appears in status for fallback path too
# ---------------------------------------------------------------------------


def test_execute_fallback_status_includes_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The status message must include the resolved seed even when falling back
    to chat.completions.  The existing fallback test checks for the fallback
    reason, but NOT for the seed, so a mutation zeroing the seed in the
    status template would go undetected.
    """
    text_gen_node = _text_module()

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: (_ for _ in ()).throw(RuntimeError("responses off"))
        ),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)),
    )
    monkeypatch.setattr(text_gen_node, "resolve_request_seed", lambda _: 42)
    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)

    output = text_gen_node.LMStudioTextGen.execute(
        connection=_connection_payload(),
        system_prompt="sys",
        user_prompt="hello",
        seed=0,
    )
    assert output[0] == "Answer"
    assert "(seed=42)" in output.ui.text
    assert "via chat.completions" in output.ui.text


def test_execute_user_prompt_none_treated_as_empty_raises() -> None:
    """
    user_prompt=None must be treated as empty and raise ValueError.
    A mutation that skipped the `user_prompt = user_prompt or ""` coercion
    would cause a NoneType AttributeError instead of a clean ValueError.
    """
    text_gen_node = _text_module()
    with pytest.raises(ValueError, match="user_prompt must not be empty"):
        text_gen_node.LMStudioTextGen.execute(
            connection=_connection_payload(),
            system_prompt="sys",
            user_prompt=None,
            seed=1,
        )
