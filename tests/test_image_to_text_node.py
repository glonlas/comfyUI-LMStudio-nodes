from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from helpers.fakes import FakeTensor
from helpers.imports import import_repo_module


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


def _batch_image() -> FakeTensor:
    return FakeTensor(
        np.array(
            [
                [[[1.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 1.0]]],
            ],
            dtype=np.float32,
        )
    )


def test_define_schema_and_validate_inputs() -> None:
    image_node = _image_module()
    schema = image_node.LMStudioImageToText.define_schema()
    assert schema.node_id == "LMStudio_ImageToText"
    assert [entry.id for entry in schema.inputs] == [
        "connection",
        "image",
        "system_prompt",
        "user_prompt",
        "seed",
    ]
    assert image_node.LMStudioImageToText.validate_inputs(None) is True


def test_responses_kwargs_single_and_batch() -> None:
    image_node = _image_module()
    kwargs_single = image_node.LMStudioImageToText._responses_kwargs(
        connection=_connection_payload(),
        image=_single_image(),
        system_prompt="sys",
        user_prompt="describe",
        seed=7,
    )
    assert kwargs_single["instructions"] == "sys"
    single_content = kwargs_single["input"][0]["content"]
    assert len(single_content) == 2
    assert single_content[1]["type"] == "input_image"

    kwargs_batch = image_node.LMStudioImageToText._responses_kwargs(
        connection=_connection_payload(reasoning_enabled=True, use_tooling_mcp=True),
        image=_batch_image(),
        system_prompt="sys",
        user_prompt="describe",
        seed=7,
    )
    batch_content = kwargs_batch["input"][0]["content"]
    assert len(batch_content) == 3
    assert kwargs_batch["reasoning"] == {"effort": "medium"}
    assert kwargs_batch["metadata"] == {"lmstudio_tooling_mcp_requested": "true"}


def test_execute_rejects_empty_user_prompt() -> None:
    image_node = _image_module()
    with pytest.raises(ValueError, match="user_prompt must not be empty"):
        image_node.LMStudioImageToText.execute(
            connection=_connection_payload(),
            image=_single_image(),
            system_prompt="sys",
            user_prompt="   ",
            seed=1,
        )


def test_execute_responses_success_and_think_strip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_node = _image_module()

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text="See this <think>internal</think> result")
        ),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run"))
            )
        ),
    )
    monkeypatch.setattr(image_node, "resolve_request_seed", lambda _: 17)
    monkeypatch.setattr(image_node, "create_openai_client", lambda **_: fake_client)

    output = image_node.LMStudioImageToText.execute(
        connection=_connection_payload(),
        image=_single_image(),
        system_prompt="sys",
        user_prompt="describe",
        seed=-1,
    )

    assert output[0] == "See this  result"
    assert "via responses" in output.ui.text
    assert "(seed=17)" in output.ui.text


def test_execute_chat_fallback_reports_dropped_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_node = _image_module()

    def raise_responses(**kwargs):
        raise RuntimeError("responses unavailable")

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="fallback vision result"))]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=raise_responses),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: completion)),
    )
    monkeypatch.setattr(image_node, "resolve_request_seed", lambda _: 99)
    monkeypatch.setattr(image_node, "create_openai_client", lambda **_: fake_client)

    output = image_node.LMStudioImageToText.execute(
        connection=_connection_payload(),
        image=_batch_image(),
        system_prompt="sys",
        user_prompt="describe",
        seed=0,
    )

    assert output[0] == "fallback vision result"
    assert "via chat.completions" in output.ui.text
    assert "responses unavailable" in output.ui.text
    assert "additional frame(s) were dropped" in output.ui.text
