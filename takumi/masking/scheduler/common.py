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
    """A step counter shared across processes.

    Deliberately built on the *spawn* context rather than the default one.
    The lock inside a ``multiprocessing.Value`` is bound to whichever context
    created it, and a fork-context lock cannot be handed to a spawn or
    forkserver child — Python raises "A SemLock created in a fork context is
    being shared with a process in a spawn context". A spawn-context lock, by
    contrast, works with all three start methods.

    Since the default start method varies by platform *and* Python version
    (macOS: spawn; Linux: fork through 3.13, forkserver from 3.14), using the
    default here would make the counter portable on some machines and broken
    on others. Pinning it to spawn makes it portable everywhere, including
    ``DataLoader(multiprocessing_context="spawn")`` on Linux. forkserver would
    work equally well but does not exist on Windows.

    Do not "simplify" this back to ``mp.Value(...)``; see
    tests/test_scheduler_multiprocessing.py.
    """

    def __init__(self, initial_value: int, final_value: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self._value = mp.get_context("spawn").Value("i", initial_value)

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
