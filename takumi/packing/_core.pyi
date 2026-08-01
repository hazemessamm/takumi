"""Type stubs for the compiled takumi.packing._core extension.

Keep this in sync with the nanobind bindings in src/packing/packing.cpp.
The compiled module has no inline signatures Python tooling can read, so this
stub is what gives editors and mypy autocomplete/type-checking.

Parameter *names* matter here as much as types: nanobind exposes them as
keyword arguments, so a stub that renames one type-checks clean and then
fails at runtime.

All five functions raise ``ValueError`` (from C++ ``std::invalid_argument``)
on inputs they cannot pack: a ``tolerance`` below 1, or any length greater
than ``max_length``.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

def pack_first_fit(
    lengths: Sequence[int],
    max_length: int,
    tolerance: int = ...,
    seed: Optional[int] = ...,
) -> List[List[int]]: ...


def pack_worst_fit(
    lengths: Sequence[int],
    max_length: int,
    tolerance: int = ...,
    seed: Optional[int] = ...,
) -> List[List[int]]: ...


def group_by_size(
    lengths: Sequence[int],
    max_length: Optional[int] = ...,
) -> Tuple[Dict[int, int], Dict[int, List[int]]]: ...


def pack_histogram(
    counts: Mapping[int, int],
    max_length: int,
) -> Dict[Tuple[int, ...], int]: ...


def pack_histogram_batched(
    counts: Mapping[int, int],
    max_length: int,
) -> Dict[Tuple[int, ...], int]: ...
