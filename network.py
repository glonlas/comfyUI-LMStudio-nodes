"""Turn opaque connection failures into actionable messages.

macOS Local Network privacy blocks apps without the permission from reaching
any host on the local network, and the only symptom is a generic
"[Errno 65] No route to host" deep inside the OpenAI client. Some apps that
launch ComfyUI (Comfy Desktop, for one) never ask for the permission, so they
never appear in System Settings and the user has nothing to switch on.
Loopback is always allowed, which is what the bundled forwarder relies on.
"""
from __future__ import annotations

import errno
import ipaddress
import re
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


FORWARDER_SCRIPT = Path(__file__).resolve().parent / "scripts" / "lan_forward.py"

# What macOS reports when Local Network privacy denies a connection.
_BLOCKED_ERRNOS = frozenset({errno.EHOSTUNREACH})
_ERRNO_IN_TEXT_RE = re.compile(r"\[Errno (\d+)\]")


class LocalNetworkBlockedError(ConnectionError):
    """macOS refused to let this process reach a host on the local network."""


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def find_errno(exc: BaseException) -> int | None:
    """First OS error number anywhere in the exception chain."""
    for item in _exception_chain(exc):
        if isinstance(item, OSError) and item.errno:
            return item.errno
        match = _ERRNO_IN_TEXT_RE.search(str(item))
        if match:
            return int(match.group(1))
    return None


def _is_lan_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    return (ip.is_private or ip.is_link_local) and not ip.is_loopback


def is_local_network_host(host: str | None) -> bool:
    """True when `host` is (or resolves to) a non-loopback local network address."""
    if not host:
        return False
    if _is_lan_ip(host):
        return True
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return False  # a literal IP that is loopback or public
    except ValueError:
        pass
    if host.endswith(".local") or "." not in host.rstrip("."):
        return host.lower() != "localhost"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return any(_is_lan_ip(str(info[4][0])) for info in infos)


def _format_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def local_network_hint(
    exc: BaseException, server_url: str, platform: str | None = None
) -> str | None:
    """Explain a macOS Local Network privacy denial, or None if that is not what happened."""
    if (platform or sys.platform) != "darwin":
        return None
    if find_errno(exc) not in _BLOCKED_ERRNOS:
        return None

    parsed = urlparse(server_url)
    host = parsed.hostname
    if not is_local_network_host(host):
        return None

    scheme = parsed.scheme or "http"
    port = parsed.port or (443 if scheme == "https" else 80)
    target = f"{_format_host(host)}:{port}"
    lines = [
        f"macOS blocked ComfyUI from reaching the LM Studio server at {server_url} "
        '("No route to host").',
        "",
        "This is macOS Local Network privacy, not a network fault: the app that runs "
        f"ComfyUI is not allowed to talk to devices on your local network, even though "
        f"other apps can reach {host}.",
        "",
        "Fix it one of these ways:",
        "  1. System Settings > Privacy & Security > Local Network: switch on the app "
        "that runs ComfyUI (Terminal, iTerm, Comfy Desktop...), then quit it fully "
        "(Cmd+Q) and relaunch.",
        "  2. If that app is not in the list (some apps, such as Comfy Desktop, never "
        "ask for the permission), run the bundled forwarder in a Terminal window and "
        "leave it open:",
        f'       python3 "{FORWARDER_SCRIPT}" --target {target}',
        f"     then set server_url to {scheme}://127.0.0.1:{port}",
    ]
    if scheme == "https":
        lines.append(
            "     (with https, the server certificate must also be valid for 127.0.0.1)"
        )
    return "\n".join(lines)


@contextmanager
def explain_network_errors(server_url: str) -> Iterator[None]:
    """Re-raise a macOS Local Network denial as LocalNetworkBlockedError with a fix."""
    try:
        yield
    except Exception as exc:
        hint = local_network_hint(exc, server_url)
        if hint is None:
            raise
        raise LocalNetworkBlockedError(hint) from exc
