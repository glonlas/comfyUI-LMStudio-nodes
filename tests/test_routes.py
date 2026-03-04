from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from helpers.imports import import_repo_module


def _routes_module():
    return import_repo_module("routes", force_reload=True)


def test_query_int_parsing() -> None:
    routes = _routes_module()
    request = SimpleNamespace(query={})
    assert routes._query_int(request, "timeout_seconds", 15) == 15

    request = SimpleNamespace(query={"timeout_seconds": "12"})
    assert routes._query_int(request, "timeout_seconds", 15) == 12

    with pytest.raises(ValueError, match="must be an integer"):
        routes._query_int(SimpleNamespace(query={"timeout_seconds": "abc"}), "timeout_seconds", 15)
    with pytest.raises(ValueError, match="must be >= 1"):
        routes._query_int(SimpleNamespace(query={"timeout_seconds": "0"}), "timeout_seconds", 15)


def test_models_handler_success(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://10.0.0.1:1234")
    monkeypatch.setattr(routes, "get_server_models", lambda **kwargs: ["m1", "m2"])

    request = SimpleNamespace(query={"server_url": " http://10.0.0.1:1234 ", "api_token": "-", "timeout_seconds": "9"})
    response = asyncio.run(routes._models_handler(request))

    payload = json.loads(response.text)
    assert response.status == 200
    assert payload["ok"] is True
    assert payload["server_url"] == "http://10.0.0.1:1234"
    assert payload["models"] == ["m1", "m2"]
    assert payload["default_model"] == "m1"


def test_models_handler_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: (_ for _ in ()).throw(ValueError("bad url")))
    request = SimpleNamespace(query={"server_url": "bad"})
    response = asyncio.run(routes._models_handler(request))
    payload = json.loads(response.text)
    assert response.status == 400
    assert payload["ok"] is False
    assert payload["error"] == "bad url"


def test_models_handler_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://x:1234")
    monkeypatch.setattr(
        routes,
        "get_server_models",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("backend down")),
    )
    request = SimpleNamespace(query={"server_url": "http://x:1234"})
    response = asyncio.run(routes._models_handler(request))
    payload = json.loads(response.text)
    assert response.status == 502
    assert payload["ok"] is False
    assert "Failed to list models" in payload["error"]


def test_test_handler_success(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://x:1234")
    monkeypatch.setattr(routes, "get_server_models", lambda **kwargs: ["m1"])
    request = SimpleNamespace(query={"server_url": "http://x:1234", "timeout_seconds": "9"})
    response = asyncio.run(routes._test_handler(request))
    payload = json.loads(response.text)
    assert response.status == 200
    assert payload["ok"] is True
    assert payload["model_count"] == 1
    assert payload["models"] == ["m1"]
    assert "Connected to http://x:1234" in payload["message"]


def test_test_handler_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda _: (_ for _ in ()).throw(ValueError("invalid")))
    request = SimpleNamespace(query={"server_url": "bad"})
    response = asyncio.run(routes._test_handler(request))
    payload = json.loads(response.text)
    assert response.status == 400
    assert payload["ok"] is False
    assert payload["error"] == "invalid"


def test_test_handler_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _routes_module()
    monkeypatch.setattr(routes, "normalize_server_url", lambda url: "http://x:1234")
    monkeypatch.setattr(
        routes,
        "get_server_models",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    request = SimpleNamespace(query={"server_url": "http://x:1234"})
    response = asyncio.run(routes._test_handler(request))
    payload = json.loads(response.text)
    assert response.status == 502
    assert payload["ok"] is False
    assert "Connection test failed" in payload["error"]


def test_register_routes_is_idempotent(prompt_server_routes) -> None:
    routes = _routes_module()
    routes.register_routes()
    routes.register_routes()

    paths = [path for path, _handler in prompt_server_routes.handlers]
    assert paths == ["/lmstudio/models", "/lmstudio/test"]
