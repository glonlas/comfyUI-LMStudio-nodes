from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


PACKAGE_NAME = "lmstudio_node_pkg"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_repo_package(*, force_reload: bool = False):
    if force_reload:
        for name in list(sys.modules):
            if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
                sys.modules.pop(name, None)

    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]

    init_file = REPO_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        init_file,
        submodule_search_locations=[str(REPO_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create module spec for repository package")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def import_repo_module(module_name: str, *, force_reload: bool = False):
    load_repo_package(force_reload=force_reload)
    return importlib.import_module(f"{PACKAGE_NAME}.{module_name}")
