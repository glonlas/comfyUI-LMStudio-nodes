"""
Behavioral-contract tests for image encoding and node status messages.

Each test targets a specific mutation that existing coverage does not catch:

1.  comfy_image_to_base64_png_url — image_url field key in _responses_kwargs content.
2.  comfy_image_to_base64_png_url — RGBA (4-channel) tensors round-trip correctly.
3.  comfy_image_to_base64_png_url — non-1x1 image preserves correct dimensions.
4.  image_to_text_node.execute — status message includes the model name.
5.  image_to_text_node._responses_kwargs — every image content entry has valid data URL.
6.  text_gen_node.execute — status message includes the model name.
"""
from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import client
from helpers.fakes import FakeTensor
from helpers.imports import import_repo_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_data_url_png(data_url: str) -> Image.Image:
    assert data_url.startswith("data:image/png;base64,"), (
        f"Expected data:image/png;base64, prefix, got: {data_url[:40]!r}"
    )
    encoded = data_url.split(",", maxsplit=1)[1]
    raw = base64.b64decode(encoded)
    return Image.open(BytesIO(raw))


def _image_module():
    return import_repo_module("image_to_text_node", force_reload=True)


def _text_module():
    return import_repo_module("text_gen_node", force_reload=True)


def _connection_payload(model: str = "vision-model"):
    models = import_repo_module("models", force_reload=True)
    return models.LMStudioConnectionPayload(
        server_url="http://127.0.0.1:1234",
        base_url="http://127.0.0.1:1234/v1",
        api_key="token",
        model=model,
        reasoning_enabled=False,
        max_tokens=128,
        temperature=0.3,
        timeout_seconds=30,
        use_tooling_mcp=False,
    )


def _single_image() -> FakeTensor:
    return FakeTensor(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))


def _batch_image() -> FakeTensor:
    return FakeTensor(
        np.array(
            [
                [[[1.0, 0.0, 0.0]]],  # red
                [[[0.0, 0.0, 1.0]]],  # blue
            ],
            dtype=np.float32,
        )
    )


# ---------------------------------------------------------------------------
# client.comfy_image_to_base64_png_url — RGBA (4-channel) round-trip
# ---------------------------------------------------------------------------


def test_comfy_image_to_base64_png_url_rgba_round_trip() -> None:
    """
    A 4-channel (RGBA) float32 tensor must survive the encode-decode round-trip.
    The encoded PNG must be decodable and preserve the RGB channels.
    A mutation that accidentally dropped channels (e.g. slicing to [:,:,:3]) would
    change the pixel values and break this assertion.
    """
    # 1x1 RGBA image: R=1.0, G=0.5, B=0.0, A=0.8
    rgba = FakeTensor(np.array([[[1.0, 0.5, 0.0, 0.8]]], dtype=np.float32))
    url = client.comfy_image_to_base64_png_url(rgba)
    img = _decode_data_url_png(url)
    # PIL will create an RGBA image for 4-channel input.
    r, g, b, a = img.getpixel((0, 0))
    assert r == 255, f"Red channel: expected 255, got {r}"
    assert g == 127 or g == 128, f"Green channel: expected ~127, got {g}"  # float rounding
    assert b == 0, f"Blue channel: expected 0, got {b}"
    assert a == 204 or a == 203, f"Alpha channel: expected ~204, got {a}"  # 0.8*255


def test_comfy_image_to_base64_png_url_correct_dimensions() -> None:
    """
    The encoded PNG must preserve the exact width and height of the input tensor.
    A mutation that transposed H and W (e.g. using .T) would swap width/height
    and break this assertion.
    """
    # 3 rows, 5 columns, 3 channels
    h, w = 3, 5
    arr = np.zeros((h, w, 3), dtype=np.float32)
    tensor = FakeTensor(arr)
    url = client.comfy_image_to_base64_png_url(tensor)
    img = _decode_data_url_png(url)
    assert img.size == (w, h), f"Expected ({w}, {h}), got {img.size}"


def test_comfy_image_to_base64_png_url_batch_uses_first_frame_exactly() -> None:
    """
    For a batch tensor (ndim==4), only frame 0 must be encoded.
    Frame 1 (blue) must NOT appear in the output.
    A mutation that encoded frame 1 instead of frame 0 would change the pixel
    from red to blue.
    """
    batch = FakeTensor(
        np.array(
            [
                [[[1.0, 0.0, 0.0]]],  # frame 0: red
                [[[0.0, 0.0, 1.0]]],  # frame 1: blue
            ],
            dtype=np.float32,
        )
    )
    url = client.comfy_image_to_base64_png_url(batch)
    img = _decode_data_url_png(url)
    r, g, b = img.getpixel((0, 0))[:3]
    assert r == 255 and g == 0 and b == 0, (
        f"Expected red (255,0,0) from frame 0, got ({r},{g},{b})"
    )


# ---------------------------------------------------------------------------
# image_to_text_node._responses_kwargs — image_url field key correctness
# ---------------------------------------------------------------------------


def test_responses_kwargs_image_content_field_key_is_image_url() -> None:
    """
    Each image entry in the content list must use the key 'image_url', not 'url'
    or any other variant.  A mutation renaming the field would break LMStudio's
    Responses API contract silently.
    This test covers both the single (ndim==3) and batch (ndim==4) paths.
    """
    image_node = _image_module()

    # Single image (3D tensor from __getitem__ on batch)
    single = FakeTensor(np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32))
    kwargs_single = image_node.LMStudioImageToText._responses_kwargs(
        connection=_connection_payload(),
        image=single,
        system_prompt="sys",
        user_prompt="describe",
        seed=1,
    )
    single_content = kwargs_single["input"][0]["content"]
    image_entry = single_content[1]
    assert image_entry["type"] == "input_image"
    assert "image_url" in image_entry, (
        f"Expected 'image_url' key, found keys: {list(image_entry.keys())}"
    )
    assert image_entry["image_url"].startswith("data:image/png;base64,")

    # Batch (4D tensor) — every image entry must have valid image_url
    batch = _batch_image()
    kwargs_batch = image_node.LMStudioImageToText._responses_kwargs(
        connection=_connection_payload(),
        image=batch,
        system_prompt="sys",
        user_prompt="describe",
        seed=2,
    )
    batch_content = kwargs_batch["input"][0]["content"]
    image_entries = [e for e in batch_content if e.get("type") == "input_image"]
    assert len(image_entries) == 2, f"Expected 2 image entries, got {len(image_entries)}"
    for i, entry in enumerate(image_entries):
        assert "image_url" in entry, (
            f"Image entry {i} missing 'image_url' key; keys: {list(entry.keys())}"
        )
        assert entry["image_url"].startswith("data:image/png;base64,"), (
            f"Image entry {i} 'image_url' lacks data URL prefix: {entry['image_url'][:40]!r}"
        )


def test_responses_kwargs_batch_images_are_distinct_data_urls() -> None:
    """
    In a multi-frame batch, the data URLs for different frames must differ.
    A mutation that passed the same frame index for all images would produce
    identical URLs and break this assertion.
    """
    image_node = _image_module()
    batch = FakeTensor(
        np.array(
            [
                [[[1.0, 0.0, 0.0]]],  # red
                [[[0.0, 0.0, 1.0]]],  # blue
            ],
            dtype=np.float32,
        )
    )
    kwargs = image_node.LMStudioImageToText._responses_kwargs(
        connection=_connection_payload(),
        image=batch,
        system_prompt="",
        user_prompt="describe",
        seed=3,
    )
    content = kwargs["input"][0]["content"]
    image_urls = [e["image_url"] for e in content if e.get("type") == "input_image"]
    assert len(image_urls) == 2
    assert image_urls[0] != image_urls[1], (
        "Both batch frames produced the same data URL — frame indexing is broken"
    )


# ---------------------------------------------------------------------------
# image_to_text_node.execute — status message includes the model name
# ---------------------------------------------------------------------------


def test_execute_status_includes_model_name_responses_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The status message must embed the actual model name from the connection payload.
    A mutation hardcoding a model name (e.g. always "vision-model") would pass
    tests that don't check the name, but fail here when a different model is used.
    """
    image_node = _image_module()
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: SimpleNamespace(output_text="some result")
        ),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: (_ for _ in ()).throw(AssertionError("no fallback"))
            )
        ),
    )
    monkeypatch.setattr(image_node, "resolve_request_seed", lambda _: 11)
    monkeypatch.setattr(image_node, "create_openai_client", lambda **_: fake_client)

    output = image_node.LMStudioImageToText.execute(
        connection=_connection_payload(model="my-custom-vision-model"),
        image=_single_image(),
        system_prompt="",
        user_prompt="describe this",
        seed=5,
    )
    assert "my-custom-vision-model" in output.ui.text, (
        f"Model name not found in status: {output.ui.text!r}"
    )


def test_execute_status_includes_model_name_fallback_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The model name must also appear in the status when chat.completions fallback is used.
    """
    image_node = _image_module()
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="fallback result"))]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: (_ for _ in ()).throw(RuntimeError("responses off"))
        ),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)),
    )
    monkeypatch.setattr(image_node, "resolve_request_seed", lambda _: 22)
    monkeypatch.setattr(image_node, "create_openai_client", lambda **_: fake_client)

    output = image_node.LMStudioImageToText.execute(
        connection=_connection_payload(model="fallback-model-xyz"),
        image=_single_image(),
        system_prompt="",
        user_prompt="what is this",
        seed=0,
    )
    assert "fallback-model-xyz" in output.ui.text, (
        f"Model name not found in fallback status: {output.ui.text!r}"
    )


# ---------------------------------------------------------------------------
# text_gen_node.execute — status message includes the model name
# ---------------------------------------------------------------------------


def _text_connection_payload(model: str = "text-model"):
    models = import_repo_module("models", force_reload=True)
    return models.LMStudioConnectionPayload(
        server_url="http://127.0.0.1:1234",
        base_url="http://127.0.0.1:1234/v1",
        api_key="token",
        model=model,
        reasoning_enabled=False,
        max_tokens=256,
        temperature=0.7,
        timeout_seconds=30,
        use_tooling_mcp=False,
    )


def test_text_gen_execute_status_includes_model_name_responses_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The status message in LMStudioTextGen.execute must include the actual model name.
    A mutation hardcoding the model name in the status template would break this.
    """
    text_gen_node = _text_module()
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: SimpleNamespace(output_text="text output")
        ),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: (_ for _ in ()).throw(AssertionError("no fallback"))
            )
        ),
    )
    monkeypatch.setattr(text_gen_node, "resolve_request_seed", lambda _: 33)
    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)

    output = text_gen_node.LMStudioTextGen.execute(
        connection=_text_connection_payload(model="unique-text-model-abc"),
        system_prompt="",
        user_prompt="generate something",
        seed=5,
    )
    assert "unique-text-model-abc" in output.ui.text, (
        f"Model name not found in text_gen status: {output.ui.text!r}"
    )


def test_text_gen_execute_status_includes_model_name_fallback_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The model name must appear in the status even when chat.completions fallback is used.
    """
    text_gen_node = _text_module()
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="text fallback answer"))]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: (_ for _ in ()).throw(RuntimeError("no responses"))
        ),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)),
    )
    monkeypatch.setattr(text_gen_node, "resolve_request_seed", lambda _: 44)
    monkeypatch.setattr(text_gen_node, "create_openai_client", lambda **_: fake_client)

    output = text_gen_node.LMStudioTextGen.execute(
        connection=_text_connection_payload(model="special-fallback-model"),
        system_prompt="",
        user_prompt="ask something",
        seed=0,
    )
    assert "special-fallback-model" in output.ui.text, (
        f"Model name not found in fallback status: {output.ui.text!r}"
    )
