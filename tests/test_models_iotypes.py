from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from helpers.imports import import_repo_module


def test_connection_payload_is_frozen_dataclass() -> None:
    models = import_repo_module("models", force_reload=True)
    payload = models.LMStudioConnectionPayload(
        server_url="http://127.0.0.1:1234",
        base_url="http://127.0.0.1:1234/v1",
        api_key="-",
        model="m1",
        reasoning_enabled=False,
        max_tokens=64,
        temperature=0.5,
        timeout_seconds=10,
        use_tooling_mcp=False,
    )

    assert payload.model == "m1"
    with pytest.raises(FrozenInstanceError):
        payload.model = "m2"


def test_param_connection_custom_type_smoke() -> None:
    iotypes = import_repo_module("iotypes", force_reload=True)
    assert hasattr(iotypes.ParamConnection, "Input")
    assert hasattr(iotypes.ParamConnection, "Output")

    input_port = iotypes.ParamConnection.Input(id="connection")
    output_port = iotypes.ParamConnection.Output(id="connection")
    assert input_port.id == "connection"
    assert output_port.id == "connection"
