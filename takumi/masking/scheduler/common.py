from __future__ import annotations

import multiprocessing as mp
from typing import List


def validate_lengths(*lists: List) -> None:
    """Require two or more lists, all of the same length.

    Args:
        *lists (Sequence): The lists to compare.

    Returns:
        None

    Raises:
        ValueError: If fewer than two lists are given or their lengths
            differ.

    Example:
        >>> validate_lengths([1, 2], [3, 4])
        >>> validate_lengths([1, 2], [3])
        Traceback (most recent call last):
        ...
        ValueError: All lists must have the same length. Received [2, 1].
    """
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

    Args:
        initial_value (int): Starting value, restored by ``reset``.
        final_value (int): Ceiling; ``step`` never goes past it.

    Example:
        >>> counter = SharedCounter(initial_value=0, final_value=2)
        >>> for _ in range(5):
        ...     counter.step()
        >>> counter.value  # clamped at final_value
        2
    """

    def __init__(self, initial_value: int, final_value: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self._value = mp.get_context("spawn").Value("i", initial_value)

    def step(self):
        """Increment by one under the lock, stopping at ``final_value``.

        Returns:
            None

        Example:
            >>> counter = SharedCounter(0, 10)
            >>> counter.step()
            >>> counter.get()
            1
        """
        with self._value.get_lock():
            current_value = self._value.value
            new_value = min(current_value + 1, self.final_value)
            self._value.value = new_value

    def get(self):
        """Current value.

        Returns:
            int: The shared value.

        Example:
            >>> SharedCounter(3, 10).get()
            3
        """
        return self._value.value

    def reset(self):
        """Set the value back to ``initial_value`` under the lock.

        Returns:
            None

        Example:
            >>> counter = SharedCounter(0, 10)
            >>> counter.step()
            >>> counter.reset()
            >>> counter.get()
            0
        """
        with self._value.get_lock():
            self._value.value = self.initial_value

    def set(self, value: int):
        """Set the value under the lock, for example when resuming a run.

        Every process sharing the counter sees the new value, just as it sees
        ``step``.

        Args:
            value (int): New value, in ``[initial_value, final_value]``.

        Returns:
            None

        Raises:
            ValueError: If ``value`` is outside the counter's range.

        Example:
            >>> counter = SharedCounter(0, 10)
            >>> counter.set(7)
            >>> counter.get()
            7
        """
        value = int(value)
        if not (self.initial_value <= value <= self.final_value):
            raise ValueError(
                f"value must be in [{self.initial_value}, "
                f"{self.final_value}], got {value}"
            )
        with self._value.get_lock():
            self._value.value = value

    def __repr__(self):
        """Show the current value and both bounds.

        Returns:
            str: For example ``SharedCounter(value=1, initial_value=0,
            final_value=5)``.

        Example:
            >>> counter = SharedCounter(0, 5)
            >>> counter.step()
            >>> counter
            SharedCounter(value=1, initial_value=0, final_value=5)
        """
        return (
            f"SharedCounter(value={self.value}, "
            f"initial_value={self.initial_value}, "
            f"final_value={self.final_value})"
        )

    @property
    def value(self):
        """Current value; same as ``get()``.

        Returns:
            int: The shared value.

        Example:
            >>> SharedCounter(4, 10).value
            4
        """
        return self.get()
