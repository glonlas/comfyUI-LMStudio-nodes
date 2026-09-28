from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import importlib.util
import logging
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lan_forward.py"


def _load():
    spec = importlib.util.spec_from_file_location("lan_forward_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fwd = _load()


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _echo_server():
    async def echo(reader, writer):
        data = await reader.read()  # until the client half-closes
        writer.write(b"echo:" + data)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


# --- parse_endpoint ---------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "default_host", "expected"),
    [
        ("192.168.1.10:1234", None, ("192.168.1.10", 1234)),
        (" studio.local:8080 ", None, ("studio.local", 8080)),
        ("[fd00::7]:1234", None, ("fd00::7", 1234)),
        ("http://192.168.1.10:1234/v1", None, ("192.168.1.10", 1234)),
        ("http://studio.local", None, ("studio.local", 80)),
        ("https://studio.local/v1", None, ("studio.local", 443)),
        ("11234", "127.0.0.1", ("127.0.0.1", 11234)),
        (":11234", "127.0.0.1", ("127.0.0.1", 11234)),
    ],
)
def test_parse_endpoint(value, default_host, expected) -> None:
    assert fwd.parse_endpoint(value, default_host=default_host) == expected


@pytest.mark.parametrize(
    ("value", "default_host", "message"),
    [
        ("studio.local", None, "expected host:port"),
        ("1234", None, "expected host:port"),
        ("", None, "expected host:port"),
        ("studio.local:abc", None, "invalid port"),
        ("studio.local:0", None, "out of range"),
        ("0", "127.0.0.1", "out of range"),
    ],
)
def test_parse_endpoint_rejects(value, default_host, message) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        fwd.parse_endpoint(value, default_host=default_host)


def test_is_loopback() -> None:
    assert fwd.is_loopback("localhost")
    assert fwd.is_loopback("127.0.0.1")
    assert fwd.is_loopback("::1")
    assert not fwd.is_loopback("192.168.1.10")
    assert not fwd.is_loopback("studio.local")


# --- describe_connect_error -------------------------------------------------

def test_describe_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "timed out" in fwd.describe_connect_error(asyncio.TimeoutError())
    refused = OSError(errno.ECONNREFUSED, "refused")
    assert "connection refused" in fwd.describe_connect_error(refused)

    unreachable = OSError(errno.EHOSTUNREACH, "No route to host")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "Local Network" in fwd.describe_connect_error(unreachable)
    monkeypatch.setattr(sys, "platform", "linux")
    assert fwd.describe_connect_error(unreachable) == str(unreachable)

    assert fwd.describe_connect_error(RuntimeError()) == "RuntimeError"


# --- relaying ---------------------------------------------------------------

def test_relays_both_directions_with_half_close() -> None:
    async def scenario():
        upstream, upstream_port = await _echo_server()
        relay = await fwd.start(("127.0.0.1", 0), ("127.0.0.1", upstream_port), 2)
        port = relay.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            payload = b"x" * (fwd.BUFFER_SIZE * 3)  # more than one read
            writer.write(payload)
            await writer.drain()
            writer.write_eof()
            reply = await asyncio.wait_for(reader.read(), 5)
            writer.close()
            return reply, payload
        finally:
            relay.close()
            upstream.close()
            await relay.wait_closed()
            await upstream.wait_closed()

    reply, payload = asyncio.run(scenario())
    assert reply == b"echo:" + payload


def test_unreachable_target_closes_the_client(caplog: pytest.LogCaptureFixture) -> None:
    dead_port = _free_port()

    async def scenario():
        relay = await fwd.start(("127.0.0.1", 0), ("127.0.0.1", dead_port), 2)
        port = relay.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            data = await asyncio.wait_for(reader.read(), 5)
            writer.close()
            return data
        finally:
            relay.close()
            await relay.wait_closed()

    with caplog.at_level(logging.INFO, logger="lan_forward"):
        assert asyncio.run(scenario()) == b""
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "is not reachable yet" in messages
    assert "cannot reach" in messages


def test_probe_reports_reachable_and_timeout(monkeypatch: pytest.MonkeyPatch,
                                            caplog: pytest.LogCaptureFixture) -> None:
    async def reachable():
        upstream, upstream_port = await _echo_server()
        relay = await fwd.start(("127.0.0.1", 0), ("127.0.0.1", upstream_port), 2)
        relay.close()
        upstream.close()
        await relay.wait_closed()
        await upstream.wait_closed()

    with caplog.at_level(logging.INFO, logger="lan_forward"):
        asyncio.run(reachable())
    assert any("is reachable" in r.getMessage() for r in caplog.records)

    async def slow(self):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(fwd.Forwarder, "open_upstream", slow)
    assert "timed out" in asyncio.run(fwd.Forwarder(("10.0.0.1", 1), 0.01).probe())


class _Writer:
    def __init__(self, *, can_eof=True, eof_error=None, close_error=None):
        self.data = b""
        self.can_eof = can_eof
        self.eof_error = eof_error
        self.close_error = close_error
        self.eof = False
        self.closed = False

    def write(self, data):
        self.data += data

    async def drain(self):
        return None

    def can_write_eof(self):
        return self.can_eof

    def write_eof(self):
        if self.eof_error:
            raise self.eof_error
        self.eof = True

    def close(self):
        self.closed = True

    async def wait_closed(self):
        if self.close_error:
            raise self.close_error


class _Reader:
    def __init__(self, *chunks, error=None):
        self.chunks = list(chunks)
        self.error = error

    async def read(self, _size=-1):
        if self.chunks:
            return self.chunks.pop(0)
        if self.error:
            raise self.error
        return b""


def test_pipe_edge_cases() -> None:
    writer = _Writer()
    asyncio.run(fwd._pipe(_Reader(b"a", b"b", error=ConnectionResetError()), writer))
    assert writer.data == b"ab" and writer.eof

    no_eof = _Writer(can_eof=False)
    asyncio.run(fwd._pipe(_Reader(b"a"), no_eof))
    assert not no_eof.eof

    failing_eof = _Writer(eof_error=RuntimeError("closed"))
    asyncio.run(fwd._pipe(_Reader(), failing_eof))


def test_close_swallows_connection_errors() -> None:
    writer = _Writer(close_error=ConnectionResetError())
    asyncio.run(fwd._close(writer))
    assert writer.closed


def test_serve_runs_until_cancelled(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario():
        task = asyncio.ensure_future(fwd.serve(("127.0.0.1", 0), ("127.0.0.1", _free_port()), 1))
        await asyncio.sleep(0.2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.INFO, logger="lan_forward"):
        asyncio.run(scenario())
    assert any("server_url to http://127.0.0.1:0" in r.getMessage() for r in caplog.records)


# --- command line -----------------------------------------------------------

def _fake_run(exc):
    def run(coro):
        coro.close()
        if exc:
            raise exc

    return run


def test_main_rejects_bad_arguments() -> None:
    with pytest.raises(SystemExit):
        fwd.main([])
    with pytest.raises(SystemExit):
        fwd.main(["--target", "no-port"])
    with pytest.raises(SystemExit):
        fwd.main(["--target", "127.0.0.1:1234"])  # would forward to itself


def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(fwd.asyncio, "run", _fake_run(KeyboardInterrupt()))
    assert fwd.main(["--target", "192.168.1.10:1234", "--quiet"]) == 0

    monkeypatch.setattr(fwd.asyncio, "run", _fake_run(OSError(errno.EADDRINUSE, "in use")))
    with caplog.at_level(logging.INFO, logger="lan_forward"):
        assert fwd.main(["--target", "192.168.1.10:1234", "--listen", "0.0.0.0:11234"]) == 1
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "already in use" in messages
    assert "exposes your LM Studio server" in messages

    monkeypatch.setattr(fwd.asyncio, "run", _fake_run(OSError(errno.EACCES, "denied")))
    with caplog.at_level(logging.INFO, logger="lan_forward"):
        assert fwd.main(["--target", "192.168.1.10:1234", "--listen", "80"]) == 1
    assert any("cannot listen" in r.getMessage() for r in caplog.records)
