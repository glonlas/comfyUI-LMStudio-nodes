from __future__ import annotations

import asyncio
import errno
import json
import socket
import sys
from types import SimpleNamespace

import pytest

from helpers.imports import import_repo_module


def _network():
    return import_repo_module("network")


def _blocked_error() -> Exception:
    """The exception chain the OpenAI client produces when macOS denies a LAN connection."""
    try:
        try:
            raise OSError(errno.EHOSTUNREACH, "No route to host")
        except OSError as os_error:
            raise ConnectionError("Connection error.") from os_error
    except ConnectionError as exc:
        return exc


# --- find_errno -------------------------------------------------------------

def test_find_errno_reads_oserror_cause_and_context() -> None:
    network = _network()
    assert network.find_errno(OSError(errno.ECONNREFUSED, "refused")) == errno.ECONNREFUSED
    assert network.find_errno(_blocked_error()) == errno.EHOSTUNREACH

    try:
        try:
            raise OSError(errno.EHOSTUNREACH, "No route to host")
        except OSError:
            raise RuntimeError("wrapped without from")
    except RuntimeError as exc:
        assert network.find_errno(exc) == errno.EHOSTUNREACH


def test_find_errno_falls_back_to_message_text() -> None:
    network = _network()
    assert network.find_errno(RuntimeError("httpcore: [Errno 65] No route to host")) == 65
    # An OSError without an errno still gets its text checked.
    assert network.find_errno(OSError("[Errno 61] Connection refused")) == 61


def test_find_errno_none_and_cycles() -> None:
    network = _network()
    assert network.find_errno(ValueError("nothing here")) is None

    first, second = RuntimeError("a"), RuntimeError("b")
    first.__cause__, second.__cause__ = second, first
    assert network.find_errno(first) is None


# --- is_local_network_host ---------------------------------------------------

@pytest.mark.parametrize(
    "host",
    ["10.0.0.5", "172.16.3.4", "192.168.1.10", "fe80::1%en0", "fd00::7",
     "lmstudio.local", "lmstudio", "studio-box."],
)
def test_local_network_hosts(host: str) -> None:
    assert _network().is_local_network_host(host) is True


@pytest.mark.parametrize("host", [None, "", "127.0.0.1", "::1", "8.8.8.8", "localhost"])
def test_not_local_network_hosts(host) -> None:
    assert _network().is_local_network_host(host) is False


def test_hostnames_are_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    network = _network()

    def fake_getaddrinfo(host, port):
        table = {"lan.example.com": "192.168.1.10", "public.example.com": "93.184.216.34"}
        if host not in table:
            raise socket.gaierror("unknown host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], 0))]

    monkeypatch.setattr(network.socket, "getaddrinfo", fake_getaddrinfo)
    assert network.is_local_network_host("lan.example.com") is True
    assert network.is_local_network_host("public.example.com") is False
    assert network.is_local_network_host("missing.example.com") is False


# --- local_network_hint -----------------------------------------------------

def test_hint_only_on_macos_for_blocked_lan_connections() -> None:
    network = _network()
    blocked = _blocked_error()
    url = "http://192.168.1.10:1234"

    assert network.local_network_hint(blocked, url, platform="linux") is None
    refused = OSError(errno.ECONNREFUSED, "Connection refused")
    assert network.local_network_hint(refused, url, platform="darwin") is None
    assert network.local_network_hint(blocked, "http://8.8.8.8:1234", platform="darwin") is None
    assert network.local_network_hint(blocked, url, platform="darwin") is not None


def test_hint_is_built_from_the_users_own_server_url() -> None:
    network = _network()
    hint = network.local_network_hint(
        _blocked_error(), "http://192.168.50.23:4321", platform="darwin"
    )

    assert "Local Network" in hint
    assert "http://192.168.50.23:4321" in hint
    assert "--target 192.168.50.23:4321" in hint
    assert "server_url to http://127.0.0.1:4321" in hint
    assert str(network.FORWARDER_SCRIPT) in hint
    assert network.FORWARDER_SCRIPT.is_file()
    assert "certificate" not in hint


def test_hint_default_ports_ipv6_and_https() -> None:
    network = _network()
    blocked = _blocked_error()

    http_hint = network.local_network_hint(blocked, "http://studio.local", platform="darwin")
    assert "--target studio.local:80" in http_hint

    https_hint = network.local_network_hint(blocked, "https://[fd00::7]", platform="darwin")
    assert "--target [fd00::7]:443" in https_hint
    assert "server_url to https://127.0.0.1:443" in https_hint
    assert "certificate" in https_hint


def test_hint_defaults_to_the_running_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    network = _network()
    monkeypatch.setattr(network, "sys", SimpleNamespace(platform="darwin"))
    assert network.local_network_hint(_blocked_error(), "http://10.0.0.2:1234") is not None
    monkeypatch.setattr(network, "sys", SimpleNamespace(platform="win32"))
    assert network.local_network_hint(_blocked_error(), "http://10.0.0.2:1234") is None


# --- explain_network_errors -------------------------------------------------

def test_explain_network_errors_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    network = _network()
    monkeypatch.setattr(network, "sys", SimpleNamespace(platform="darwin"))

    with network.explain_network_errors("http://10.0.0.2:1234"):
        pass

    with pytest.raises(ValueError, match="unrelated"):
        with network.explain_network_errors("http://10.0.0.2:1234"):
            raise ValueError("unrelated")


def test_explain_network_errors_rewrites_blocked_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    network = _network()
    monkeypatch.setattr(network, "sys", SimpleNamespace(platform="darwin"))
    original = _blocked_error()

    with pytest.raises(network.LocalNetworkBlockedError) as caught:
        with network.explain_network_errors("http://10.0.0.2:1234"):
            raise original

    assert caught.value.__cause__ is original
    assert isinstance(caught.value, ConnectionError)
    assert "--target 10.0.0.2:1234" in str(caught.value)


def test_real_openai_client_error_chain_is_recognised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact shape of the failure reported by users: openai -> httpx -> OSError(65)."""
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    network = _network()
    monkeypatch.setattr(network, "sys", SimpleNamespace(platform="darwin"))

    def handler(request):
        try:
            raise OSError(errno.EHOSTUNREACH, "No route to host")
        except OSError as exc:
            raise httpx.ConnectError("[Errno 65] No route to host", request=request) from exc

    client = openai.OpenAI(
        api_key="lm-studio",
        base_url="http://192.168.1.10:1234/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(network.LocalNetworkBlockedError) as caught:
        with network.explain_network_errors("http://192.168.1.10:1234"):
            client.models.list()
    assert isinstance(caught.value.__cause__, openai.APIConnectionError)


# --- wired into the nodes and routes ----------------------------------------

def _darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_network(), "sys", SimpleNamespace(platform="darwin"))


def _raise_blocked(**_kwargs):
    raise _blocked_error()


def test_connect_node_explains_blocked_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_node = import_repo_module("connect_node", force_reload=True)
    _darwin(monkeypatch)
    monkeypatch.setattr(connect_node, "get_server_models", _raise_blocked)

    with pytest.raises(ConnectionError, match="Local Network"):
        connect_node.LMStudioConnect.execute(
            server_url="http://192.168.1.10:1234", api_token="-", model="m",
            thinking="auto", test_connectivity=True, max_tokens=64,
            temperature=0.5, timeout_seconds=5, use_tooling_mcp=False,
        )


def test_test_route_explains_blocked_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = import_repo_module("routes", force_reload=True)
    _darwin(monkeypatch)
    monkeypatch.setattr(routes, "get_server_models", _raise_blocked)
    request = SimpleNamespace(query={"server_url": "http://192.168.1.10:1234"})

    response = asyncio.run(routes._test_handler(request))

    assert response.status == 502
    error = json.loads(response.text)["error"]
    assert "Local Network" in error
    assert "--target 192.168.1.10:1234" in error


def test_text_gen_explains_blocked_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    text_gen_node = import_repo_module("text_gen_node", force_reload=True)
    models = import_repo_module("models")
    _darwin(monkeypatch)

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(_blocked_error())),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(_blocked_error()))
        ),
    )
    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)
    connection = models.LMStudioConnectionPayload(
        server_url="http://192.168.1.10:1234", base_url="http://192.168.1.10:1234/v1",
        api_key="lm-studio", model="m", thinking="off", max_tokens=64,
        temperature=0.5, timeout_seconds=5, use_tooling_mcp=False,
    )

    with pytest.raises(ConnectionError, match="Local Network"):
        text_gen_node.LMStudioTextGen.execute(
            connection=connection, system_prompt="", user_prompt="hi", seed=1,
        )


def test_other_platforms_keep_the_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_node = import_repo_module("connect_node", force_reload=True)
    monkeypatch.setattr(_network(), "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(connect_node, "get_server_models", _raise_blocked)

    with pytest.raises(ConnectionError) as caught:
        connect_node.LMStudioConnect.execute(
            server_url="http://192.168.1.10:1234", api_token="-", model="m",
            thinking="auto", test_connectivity=True, max_tokens=64,
            temperature=0.5, timeout_seconds=5, use_tooling_mcp=False,
        )
    assert str(caught.value) == "Connection error."
    assert sys.platform  # the real platform is untouched
