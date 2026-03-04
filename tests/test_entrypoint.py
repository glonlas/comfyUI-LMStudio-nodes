from __future__ import annotations

import asyncio

from helpers.imports import load_repo_package


def test_comfy_entrypoint_and_node_list(monkeypatch) -> None:
    pkg = load_repo_package(force_reload=True)

    called = {"count": 0}
    monkeypatch.setattr(pkg, "register_routes", lambda: called.__setitem__("count", called["count"] + 1))

    extension = asyncio.run(pkg.comfy_entrypoint())
    assert isinstance(extension, pkg.LMStudioExtension)

    asyncio.run(extension.on_load())
    assert called["count"] == 1

    nodes = asyncio.run(extension.get_node_list())
    node_names = [node.__name__ for node in nodes]
    assert node_names == [
        "LMStudioConnect",
        "LMStudioImageToText",
        "LMStudioTextGen",
    ]
