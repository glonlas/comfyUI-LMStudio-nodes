from __future__ import annotations

from types import SimpleNamespace

import pytest

from helpers.imports import import_repo_module


def _text_module():
    return import_repo_module("text_gen_node", force_reload=True)


def _connection_payload():
    models = import_repo_module("models", force_reload=True)
    return models.LMStudioConnectionPayload(
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


def test_define_schema_and_validate_inputs() -> None:
    text_gen_node = _text_module()
    schema = text_gen_node.LMStudioTextGen.define_schema()
    assert schema.node_id == "LMStudio_TextGen"
    assert [entry.id for entry in schema.inputs] == [
        "connection",
        "system_prompt",
        "user_prompt",
        "seed",
    ]
    assert text_gen_node.LMStudioTextGen.validate_inputs(None) is True


def test_responses_kwargs_include_optional_fields() -> None:
    text_gen_node = _text_module()
    connection = _connection_payload()
    connection = connection.__class__(**{**connection.__dict__, "reasoning_enabled": True, "use_tooling_mcp": True})

    kwargs = text_gen_node.LMStudioTextGen._responses_kwargs(
        connection=connection,
        system_prompt="sys",
        user_prompt="user",
        seed=12,
    )
    assert kwargs["model"] == "model-a"
    assert kwargs["seed"] == 12
    assert kwargs["instructions"] == "sys"
    assert kwargs["reasoning"] == {"effort": "medium"}
    assert kwargs["metadata"] == {"lmstudio_tooling_mcp_requested": "true"}


def test_responses_kwargs_omit_optional_fields_when_disabled() -> None:
    text_gen_node = _text_module()
    connection = _connection_payload()
    kwargs = text_gen_node.LMStudioTextGen._responses_kwargs(
        connection=connection,
        system_prompt="   ",
        user_prompt="user",
        seed=10,
    )
    assert "instructions" not in kwargs
    assert "reasoning" not in kwargs
    assert "metadata" not in kwargs


def test_execute_rejects_empty_user_prompt() -> None:
    text_gen_node = _text_module()
    with pytest.raises(ValueError, match="user_prompt must not be empty"):
        text_gen_node.LMStudioTextGen.execute(
            connection=_connection_payload(),
            system_prompt="sys",
            user_prompt="   ",
            seed=1,
        )


def test_execute_uses_responses_endpoint_and_strips_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_gen_node = _text_module()
    captured_kwargs: dict[str, object] = {}

    def create_response(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(output_text="Answer <think>internal</think> final")

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=create_response),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run"))
            )
        ),
    )

    monkeypatch.setattr(text_gen_node, "resolve_request_seed", lambda _: 77)
    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)

    output = text_gen_node.LMStudioTextGen.execute(
        connection=_connection_payload(),
        system_prompt="sys",
        user_prompt="hello",
        seed=-1,
    )

    assert output[0] == "Answer  final"
    assert "via responses" in output.ui.text
    assert "(seed=77)" in output.ui.text
    assert captured_kwargs["seed"] == 77


def test_execute_falls_back_to_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    text_gen_node = _text_module()

    def raise_responses(**kwargs):
        raise RuntimeError("responses disabled")

    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Fallback answer"))])
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=raise_responses),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: completion)),
    )

    monkeypatch.setattr(text_gen_node, "resolve_request_seed", lambda _: 99)
    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)

    output = text_gen_node.LMStudioTextGen.execute(
        connection=_connection_payload(),
        system_prompt="sys",
        user_prompt="hello",
        seed=0,
    )

    assert output[0] == "Fallback answer"
    assert "via chat.completions" in output.ui.text
    assert "responses disabled" in output.ui.text


def test_execute_raises_if_both_endpoints_return_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_gen_node = _text_module()

    def empty_responses(**kwargs):
        return SimpleNamespace(output_text="")

    empty_completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=empty_responses),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: empty_completion)),
    )

    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)
    monkeypatch.setattr(text_gen_node, "resolve_request_seed", lambda _: 55)

    with pytest.raises(RuntimeError) as exc_info:
        text_gen_node.LMStudioTextGen.execute(
            connection=_connection_payload(),
            system_prompt="sys",
            user_prompt="hello",
            seed=1,
        )

    message = str(exc_info.value)
    assert "Both endpoints failed" in message
    assert "responses endpoint returned no text output" in message
    assert "chat.completions fallback returned no text output" in message


def test_execute_chains_both_errors_when_chat_completions_also_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_gen_node = _text_module()

    def raise_responses(**kwargs):
        raise RuntimeError("responses boom")

    def raise_chat(**kwargs):
        raise RuntimeError("chat boom")

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=raise_responses),
        chat=SimpleNamespace(completions=SimpleNamespace(create=raise_chat)),
    )

    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)
    monkeypatch.setattr(text_gen_node, "resolve_request_seed", lambda _: 55)

    with pytest.raises(RuntimeError) as exc_info:
        text_gen_node.LMStudioTextGen.execute(
            connection=_connection_payload(),
            system_prompt="sys",
            user_prompt="hello",
            seed=1,
        )

    message = str(exc_info.value)
    assert "Both endpoints failed" in message
    assert "responses boom" in message
    assert "chat boom" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "chat boom"
