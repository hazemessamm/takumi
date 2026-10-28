"""Type stubs for the compiled takumi.packing._core extension.

Keep this in sync with the nanobind bindings in src/packing/packing.cpp.
The compiled module has no inline signatures Python tooling can read, so this
stub is what gives editors and mypy autocomplete/type-checking.

Parameter *names* matter here as much as types: nanobind exposes them as
keyword arguments, so a stub that renames one type-checks clean and then
fails at runtime.

The docstrings below mirror the ``R"doc"`` strings in packing.cpp (what
``help()`` shows at runtime); change both together.

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
) -> List[List[int]]:
    """First-Fit-Decreasing bin packing.

    Sorts sequences longest first, then places each into the first (oldest) open
    bin that still has room, opening a fresh bin when none fits.

    Args:
        lengths (Sequence[int], length N): Sequence lengths, each at most
            ``max_length``.
        max_length (int): Bin capacity in tokens.
        tolerance (int): Bucket width for the longest-first sort, ``>= 1``;
            lengths in the same bucket count as ties.
        seed (int | None): Shuffles ties reproducibly on a given platform;
            any integer in ``[0, 2**64)``, like NumPy's. ``None`` breaks ties
            by original index.

    Returns:
        list[list[int]]: Bins, each a list of indices into ``lengths``.

    Raises:
        ValueError: if ``tolerance`` is below 1, or any length exceeds
            ``max_length``.

    Example:
        >>> pack_first_fit([7, 6, 5, 4, 3, 2], max_length=10)
        [[0, 4], [1, 3], [2, 5]]
        >>> a = pack_first_fit([7, 6, 5, 4, 3, 2], 10, tolerance=2, seed=42)
        >>> a == pack_first_fit([7, 6, 5, 4, 3, 2], 10, tolerance=2, seed=42)
        True
    """
    ...


def pack_worst_fit(
    lengths: Sequence[int],
    max_length: int,
    tolerance: int = ...,
    seed: Optional[int] = ...,
) -> List[List[int]]:
    """Heap-based Worst-Fit-Decreasing bin packing.

    Sorts sequences longest first, then places each into the *loosest* open bin
    that still has room — the bin with the most remaining capacity — rather than
    the first one that fits. A max-heap keyed on remaining capacity makes that
    choice O(log n) per sequence.

    Args:
        lengths (Sequence[int], length N): Sequence lengths, each at most
            ``max_length``.
        max_length (int): Bin capacity in tokens.
        tolerance (int): Bucket width for the longest-first sort, ``>= 1``;
            lengths in the same bucket count as ties.
        seed (int | None): Shuffles ties reproducibly on a given platform;
            any integer in ``[0, 2**64)``, like NumPy's. ``None`` breaks ties
            by original index.

    Returns:
        list[list[int]]: Bins, each a list of indices into ``lengths``.

    Raises:
        ValueError: if ``tolerance`` is below 1, or any length exceeds
            ``max_length``.

    Example:
        >>> pack_worst_fit([7, 6, 5, 4, 3, 2], max_length=10)
        [[0, 5], [1, 4], [2, 3]]
    """
    ...


def group_by_size(
    lengths: Sequence[int],
    max_length: Optional[int] = ...,
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Compute a histogram of sequence lengths and a mapping from lengths to indices.

    Args:
        lengths (Sequence[int], length N): Sequence lengths.
        max_length (int | None): If given, lengths above it are clipped to it.

    Returns:
        tuple[dict[int, int], dict[int, list[int]]]: ``(counts,
        length_to_indices)``: how many sequences have each length, and their
        indices in ascending order. Both dicts list lengths in ascending order
        on every platform.

    Example:
        >>> counts, pools = group_by_size([3, 5, 3, 9], max_length=8)
        >>> counts  # 9 was clipped to 8
        {3: 2, 5: 1, 8: 1}
        >>> pools
        {3: [0, 2], 5: [1], 8: [3]}
    """
    ...


def pack_histogram(
    counts: Mapping[int, int],
    max_length: int,
) -> Dict[Tuple[int, ...], int]:
    """Pack a histogram of sequence lengths into bins.

    Worst-fit-decreasing over lengths rather than individual sequences, so the
    result is a reusable blueprint of bin layouts.

    Args:
        counts (Mapping[int, int]): Length to how many sequences have it; each
            length at most ``max_length``.
        max_length (int): Bin capacity in tokens.

    Returns:
        dict[tuple[int, ...], int]: Bin layout (the lengths sharing a bin,
        longest first) to how many bins use it, in sorted key order.

    Raises:
        ValueError: if any length exceeds ``max_length``.

    Example:
        >>> pack_histogram({7: 1, 6: 1, 5: 1, 4: 1, 3: 1, 2: 1}, max_length=10)
        {(5, 4): 1, (6, 3): 1, (7, 2): 1}
    """
    ...


def pack_histogram_batched(
    counts: Mapping[int, int],
    max_length: int,
) -> Dict[Tuple[int, ...], int]:
    """Batched worst-fit-decreasing packing of a length histogram.

    Same inputs and output as pack_histogram, but fills a whole run of equal-length
    items per heap operation instead of one at a time, which is faster when many
    items fit per bin. The packing is quality-equivalent to pack_histogram (bin
    count matches on realistic inputs, differs by at most one bin on adversarial
    ones); its templates tend to group more identical lengths per bin.

    Args:
        counts (Mapping[int, int]): Length to how many sequences have it; each
            length at most ``max_length``.
        max_length (int): Bin capacity in tokens.

    Returns:
        dict[tuple[int, ...], int]: Bin layout (the lengths sharing a bin,
        longest first) to how many bins use it, in sorted key order.

    Raises:
        ValueError: if any length exceeds ``max_length``.

    Example:
        >>> pack_histogram_batched({2: 10, 3: 4}, max_length=8)
        {(2, 2, 2, 2): 2, (3, 3, 2): 2}
    """
    ...
