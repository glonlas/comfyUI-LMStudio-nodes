from __future__ import annotations

from aiohttp import web
from server import PromptServer

from .client import get_server_models, normalize_server_url


_ROUTES_REGISTERED = False


def _query_int(request: web.Request, key: str, default: int) -> int:
    raw = request.query.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{key} must be >= 1")
    return value


async def _models_handler(request: web.Request) -> web.Response:
    server_url = (request.query.get("server_url") or "").strip()
    api_token = (request.query.get("api_token") or "-").strip() or "-"

    try:
        timeout_seconds = _query_int(request, "timeout_seconds", 15)
        normalized_url = normalize_server_url(server_url)
        models = get_server_models(
            server_url=normalized_url,
            api_key=api_token,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response(
            {
                "ok": False,
                "error": f"Failed to list models from LMStudio: {exc}",
            },
            status=502,
        )

    return web.json_response(
        {
            "ok": True,
            "server_url": normalized_url,
            "models": models,
            "default_model": models[0] if models else "",
        }
    )


async def _test_handler(request: web.Request) -> web.Response:
    server_url = (request.query.get("server_url") or "").strip()
    api_token = (request.query.get("api_token") or "-").strip() or "-"

    try:
        # Use the same short default as _models_handler; the JS always sends the
        # widget value anyway, so this only matters when the endpoint is called directly.
        timeout_seconds = _query_int(request, "timeout_seconds", 15)
        normalized_url = normalize_server_url(server_url)
        models = get_server_models(
            server_url=normalized_url,
            api_key=api_token,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response(
            {
                "ok": False,
                "error": f"Connection test failed: {exc}",
            },
            status=502,
        )

    return web.json_response(
        {
            "ok": True,
            "server_url": normalized_url,
            "model_count": len(models),
            "models": models,
            "message": f"Connected to {normalized_url} with {len(models)} model(s).",
        }
    )


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    routes = PromptServer.instance.routes
    routes.get("/lmstudio/models")(_models_handler)
    routes.get("/lmstudio/test")(_test_handler)

    _ROUTES_REGISTERED = True
