#!/usr/bin/env python3
"""Relay a local port to an LM Studio server elsewhere on your network.

Why this exists: on macOS, an app without Local Network permission gets
"No route to host" for every address on the local network. Some apps that
launch ComfyUI (Comfy Desktop, for one) never ask for that permission, so it
cannot be granted in System Settings. Connections to this Mac itself (loopback)
are always allowed, so run this relay from Terminal, which can be granted the
permission, and point the LMStudio - Connect node at http://127.0.0.1:<port>.

    python3 lan_forward.py --target 192.168.1.10:1234
    python3 lan_forward.py --target http://lmstudio.local:1234 --listen 11234

Standard library only; works with the Python 3.9 that ships with macOS.
Stop it with Ctrl+C.
"""
from __future__ import annotations

import argparse
import asyncio
import errno
import ipaddress
import logging
import sys
from typing import Optional, Sequence, Tuple
from urllib.parse import urlsplit

Endpoint = Tuple[str, int]

log = logging.getLogger("lan_forward")

DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_CONNECT_TIMEOUT = 10.0
BUFFER_SIZE = 64 * 1024


def parse_endpoint(value: str, default_host: Optional[str] = None) -> Endpoint:
    """Parse "host:port", "[v6]:port", a bare "port", or a URL such as http://host:1234/v1."""
    raw = (value or "").strip()
    if raw.isdigit() and default_host is not None:
        host, port = default_host, int(raw)
    else:
        parts = urlsplit(raw if "://" in raw else f"//{raw}")
        try:
            port = parts.port
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid port in {value!r}") from exc
        if port is None and parts.scheme in ("http", "https"):
            port = 443 if parts.scheme == "https" else 80
        host = parts.hostname or default_host
        if not host or port is None:
            raise argparse.ArgumentTypeError(
                f"expected host:port (for example 192.168.1.10:1234), got {value!r}"
            )
    if not 0 < port < 65536:
        raise argparse.ArgumentTypeError(f"port out of range in {value!r}")
    return host, port


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def describe_connect_error(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "timed out (is the server running and is its port open?)"
    if isinstance(exc, OSError) and exc.errno == errno.EHOSTUNREACH and sys.platform == "darwin":
        return (
            "No route to host. On macOS this usually means the app running this script "
            "has no Local Network permission either: run it from Terminal or iTerm and "
            "allow that app under System Settings > Privacy & Security > Local Network."
        )
    if isinstance(exc, OSError) and exc.errno == errno.ECONNREFUSED:
        return "connection refused (is LM Studio's server started, and on that port?)"
    return str(exc) or type(exc).__name__


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy one direction until EOF, then half-close so the other direction can finish."""
    try:
        while True:
            data = await reader.read(BUFFER_SIZE)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            if writer.can_write_eof():
                writer.write_eof()
        except (OSError, RuntimeError):
            pass


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass


class Forwarder:
    def __init__(self, target: Endpoint, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT):
        self.target = target
        self.connect_timeout = connect_timeout

    async def open_upstream(self) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.wait_for(
            asyncio.open_connection(*self.target), timeout=self.connect_timeout
        )

    async def handle(self, client_reader: asyncio.StreamReader,
                     client_writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await self.open_upstream()
        except (OSError, asyncio.TimeoutError) as exc:
            log.warning("cannot reach %s:%s: %s", *self.target, describe_connect_error(exc))
            await _close(client_writer)
            return
        try:
            await asyncio.gather(
                _pipe(client_reader, upstream_writer),
                _pipe(upstream_reader, client_writer),
            )
        finally:
            await _close(upstream_writer)
            await _close(client_writer)

    async def probe(self) -> Optional[str]:
        """None when the target accepts a connection, otherwise why it did not."""
        try:
            _, writer = await self.open_upstream()
        except (OSError, asyncio.TimeoutError) as exc:
            return describe_connect_error(exc)
        await _close(writer)
        return None


async def start(listen: Endpoint, target: Endpoint,
                connect_timeout: float = DEFAULT_CONNECT_TIMEOUT) -> asyncio.AbstractServer:
    forwarder = Forwarder(target, connect_timeout)
    server = await asyncio.start_server(forwarder.handle, *listen)
    log.info("forwarding %s:%s -> %s:%s", *listen, *target)
    problem = await forwarder.probe()
    if problem:
        log.warning("target %s:%s is not reachable yet: %s", *target, problem)
    else:
        log.info("target %s:%s is reachable", *target)
    return server


async def serve(listen: Endpoint, target: Endpoint, connect_timeout: float) -> None:
    server = await start(listen, target, connect_timeout)
    scheme_hint = f"http://{listen[0]}:{listen[1]}"
    log.info("set the LMStudio - Connect node's server_url to %s (Ctrl+C to stop)", scheme_hint)
    async with server:
        await server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relay a local port to an LM Studio server on your network "
                    "(works around macOS Local Network privacy).",
    )
    parser.add_argument(
        "--target", required=True,
        help="LM Studio server as host:port or URL, e.g. 192.168.1.10:1234",
    )
    parser.add_argument(
        "--listen", default=None,
        help="local [host:]port to accept connections on "
             f"(default: {DEFAULT_LISTEN_HOST} and the target's port)",
    )
    parser.add_argument(
        "--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT,
        help=f"seconds to wait for the target on each connection (default: {DEFAULT_CONNECT_TIMEOUT:g})",
    )
    parser.add_argument("--quiet", action="store_true", help="only log warnings")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target = parse_endpoint(args.target)
        listen = (parse_endpoint(args.listen, default_host=DEFAULT_LISTEN_HOST)
                  if args.listen else (DEFAULT_LISTEN_HOST, target[1]))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if is_loopback(target[0]) and target[1] == listen[1]:
        parser.error("target and listen address are the same; this would forward to itself")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not is_loopback(listen[0]):
        log.warning("listening on %s exposes your LM Studio server to other machines", listen[0])

    try:
        asyncio.run(serve(listen, target, args.connect_timeout))
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            log.error(
                "port %s is already in use on %s. Another forwarder or a local LM Studio "
                "may be running; pick another one with --listen.", listen[1], listen[0],
            )
        else:
            log.error("cannot listen on %s:%s: %s", *listen, exc)
        return 1
    return 0  # pragma: no cover - serve() only returns by raising


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
