from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


def ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def make_chat_completion(content: Any) -> SimpleNamespace:
    return ns(choices=[ns(message=ns(content=content))])


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
