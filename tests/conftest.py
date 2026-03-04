from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Field:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FieldType:
    class Input(_Field):
        pass

    class Output(_Field):
        pass


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *values, ui=None):
        self.values = values
        self.ui = ui

    def __getitem__(self, index: int):
        return self.values[index]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


class _ControlAfterGenerate:
    randomize = "randomize"


class _ComfyNode:
    pass


class _ComfyExtension:
    async def on_load(self) -> None:
        return None

    async def get_node_list(self) -> list[type]:
        return []


def _custom_type(_name: str):
    class _Custom:
        class Input(_Field):
            pass

        class Output(_Field):
            pass

    return _Custom


class _RoutesRegistry:
    def __init__(self):
        self.handlers: list[tuple[str, object]] = []

    def get(self, path: str):
        def decorator(handler):
            self.handlers.append((path, handler))
            return handler

        return decorator


class _PromptServer:
    instance = types.SimpleNamespace(routes=_RoutesRegistry())


def _install_comfy_stubs() -> None:
    io_module = types.SimpleNamespace(
        Schema=_Schema,
        ComfyNode=_ComfyNode,
        NodeOutput=_NodeOutput,
        ControlAfterGenerate=_ControlAfterGenerate,
        String=_FieldType,
        Boolean=_FieldType,
        Int=_FieldType,
        Float=_FieldType,
        Image=_FieldType,
        Custom=_custom_type,
    )
    ui_module = types.SimpleNamespace(
        PreviewText=lambda text: types.SimpleNamespace(text=text)
    )

    latest_module = types.ModuleType("comfy_api.latest")
    latest_module.io = io_module
    latest_module.ui = ui_module
    latest_module.ComfyExtension = _ComfyExtension

    comfy_module = types.ModuleType("comfy_api")
    comfy_module.latest = latest_module

    sys.modules["comfy_api"] = comfy_module
    sys.modules["comfy_api.latest"] = latest_module


def _install_server_stub() -> None:
    server_module = types.ModuleType("server")
    server_module.PromptServer = _PromptServer
    sys.modules["server"] = server_module


_install_comfy_stubs()
_install_server_stub()

# Pytest may import repository __init__.py as a top-level module named "__init__",
# which breaks relative imports in that file. Pre-seed a shim module for test collection.
if "__init__" not in sys.modules:
    init_shim = types.ModuleType("__init__")
    init_shim.__file__ = str(REPO_ROOT / "__init__.py")
    sys.modules["__init__"] = init_shim


@pytest.fixture
def prompt_server_routes():
    routes = _RoutesRegistry()
    _PromptServer.instance = types.SimpleNamespace(routes=routes)
    return routes
