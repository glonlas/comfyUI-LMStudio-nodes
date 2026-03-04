from __future__ import annotations

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .connect_node import LMStudioConnect
from .image_to_text_node import LMStudioImageToText
from .text_gen_node import LMStudioTextGen
from .routes import register_routes

WEB_DIRECTORY = "./web"


class LMStudioExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        register_routes()

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LMStudioConnect,
            LMStudioImageToText,
            LMStudioTextGen,
        ]


async def comfy_entrypoint() -> LMStudioExtension:
    return LMStudioExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
