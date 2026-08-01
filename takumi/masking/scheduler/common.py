from __future__ import annotations

import multiprocessing as mp
from typing import List


def validate_lengths(*lists: List) -> None:
    if len(lists) < 2:
        raise ValueError("At least two lists must be provided for validation.")
    if not all(len(lst) == len(lists[0]) for lst in lists):
        lengths = [len(lst) for lst in lists]
        raise ValueError(
            f"All lists must have the same length. Received {lengths}."
        )


class SharedCounter:
    def __init__(self, initial_value: int, final_value: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self._value = mp.Value("i", initial_value)

    def step(self):
        with self._value.get_lock():
            current_value = self._value.value
            new_value = min(current_value + 1, self.final_value)
            self._value.value = new_value

    def get(self):
        return self._value.value

    def reset(self):
        with self._value.get_lock():
            self._value.value = self.initial_value

    def __repr__(self):
        return (
            f"SharedCounter(value={self.value}, "
            f"initial_value={self.initial_value}, "
            f"final_value={self.final_value})"
        )

    @property
    def value(self):
        return self.get()
