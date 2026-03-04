from __future__ import annotations

import pytest

from helpers.imports import import_repo_module


def _connect_module():
    return import_repo_module("connect_node", force_reload=True)


def test_define_schema_has_expected_order_and_defaults() -> None:
    connect_node = _connect_module()
    schema = connect_node.LMStudioConnect.define_schema()

    assert schema.node_id == "LMStudio_Connect"
    input_ids = [entry.id for entry in schema.inputs]
    assert input_ids == [
        "server_url",
        "api_token",
        "model",
        "reasoning_enabled",
        "use_tooling_mcp",
        "max_tokens",
        "temperature",
        "timeout_seconds",
        "test_connectivity",
    ]

    by_id = {entry.id: entry for entry in schema.inputs}
    assert by_id["server_url"].default == "http://127.0.0.1:1234"
    assert by_id["api_token"].default == "-"
    assert by_id["max_tokens"].default == 1024
    assert by_id["temperature"].default == 0.7
    assert by_id["timeout_seconds"].default == 600
    assert by_id["max_tokens"].advanced is True
    assert by_id["temperature"].advanced is True
    assert by_id["timeout_seconds"].advanced is True
    assert by_id["test_connectivity"].advanced is True


def test_validate_inputs() -> None:
    connect_node = _connect_module()
    cls = connect_node.LMStudioConnect

    assert cls.validate_inputs("http://127.0.0.1:1234", 10, 128) is True
    assert "http://" in str(cls.validate_inputs("localhost:1234", 10, 128))
    assert cls.validate_inputs("http://127.0.0.1:1234", 0, 128) == "timeout_seconds must be >= 1"
    assert cls.validate_inputs("http://127.0.0.1:1234", 10, 0) == "max_tokens must be >= 1"


def test_execute_with_placeholder_model_uses_discovered_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_node = _connect_module()
    monkeypatch.setattr(connect_node, "get_server_models", lambda **_: ["m1", "m2"])

    output = connect_node.LMStudioConnect.execute(
        server_url=" http://10.0.0.1:1234/v1/ ",
        api_token="-",
        model="<refresh models>",
        reasoning_enabled=True,
        test_connectivity=True,
        max_tokens=321,
        temperature=0.6,
        timeout_seconds=22,
        use_tooling_mcp=True,
    )

    payload = output[0]
    status = output[1]
    assert payload.server_url == "http://10.0.0.1:1234"
    assert payload.base_url == "http://10.0.0.1:1234/v1"
    assert payload.model == "m1"
    assert payload.reasoning_enabled is True
    assert payload.use_tooling_mcp is True
    assert payload.max_tokens == 321
    assert payload.temperature == 0.6
    assert payload.timeout_seconds == 22
    assert "Found 2 model(s)" in status
    assert "Using 'm1'" in status


def test_execute_without_probe_keeps_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_node = _connect_module()

    def _should_not_run(**kwargs):
        raise AssertionError("get_server_models should not run when test_connectivity=False")

    monkeypatch.setattr(connect_node, "get_server_models", _should_not_run)

    output = connect_node.LMStudioConnect.execute(
        server_url="http://127.0.0.1:1234",
        api_token="token",
        model="manually-selected-model",
        reasoning_enabled=False,
        test_connectivity=False,
        max_tokens=128,
        temperature=0.5,
        timeout_seconds=10,
        use_tooling_mcp=False,
    )
    payload = output[0]
    status = output[1]
    assert payload.model == "manually-selected-model"
    assert "Connection prepared" in status
    assert "Using 'manually-selected-model'" in status


def test_execute_raises_when_selected_model_not_on_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_node = _connect_module()
    monkeypatch.setattr(connect_node, "get_server_models", lambda **_: ["other-model"])

    with pytest.raises(ValueError, match="not available on the server"):
        connect_node.LMStudioConnect.execute(
            server_url="http://127.0.0.1:1234",
            api_token="-",
            model="requested-model",
            reasoning_enabled=False,
            test_connectivity=True,
            max_tokens=128,
            temperature=0.5,
            timeout_seconds=10,
            use_tooling_mcp=False,
        )


def test_execute_raises_when_no_model_selected_and_none_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_node = _connect_module()
    monkeypatch.setattr(connect_node, "get_server_models", lambda **_: [])

    with pytest.raises(ValueError, match="No model selected"):
        connect_node.LMStudioConnect.execute(
            server_url="http://127.0.0.1:1234",
            api_token="-",
            model="<refresh models>",
            reasoning_enabled=False,
            test_connectivity=True,
            max_tokens=128,
            temperature=0.5,
            timeout_seconds=10,
            use_tooling_mcp=False,
        )
