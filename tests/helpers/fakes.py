from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


def ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def make_chat_completion(content: Any) -> SimpleNamespace:
    return ns(choices=[ns(message=ns(content=content))])


def make_response_with_output_text(text: str) -> SimpleNamespace:
    return ns(output_text=text)


def make_response_with_output_items(*items: Any) -> SimpleNamespace:
    return ns(output=list(items))


class FakeTensor:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array, dtype=np.float32)

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __getitem__(self, index: int) -> "FakeTensor":
        return FakeTensor(self._array[index])
