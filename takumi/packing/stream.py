"""Lazily materialize the concrete bins of a histogram packing, one at a time.

``stream_packed_bins`` takes a histogram blueprint — the ``{layout: count}``
templates and per-length index pools produced by the histogram packers — and
yields its bins for a single epoch, drawing dataset indices from the pools
without replacement so every sequence appears exactly once. All randomness
flows through numpy's ``default_rng(seed)``, so a given seed reproduces the
same epoch on every platform and numpy version — the reason this stays in
Python rather than the C++ core, whose ``std::`` distributions are not
portably reproducible.

Most callers reach this through ``PackingPlan.iter_bins``, which carries the
two mappings around as one object; this module works on the raw data and knows
nothing about the plan type.

This is an independent re-implementation; it deliberately shares no naming or
structure with any reference packer's epoch materializer.
"""

from __future__ import annotations

from typing import Iterator, List, Mapping, Sequence, Tuple

import numpy as np

BinLayout = Tuple[int, ...]


def stream_packed_bins(
    templates: Mapping[BinLayout, int],
    index_pools: Mapping[int, Sequence[int]],
    seed: int | None = None,
    interleave: bool = False,
) -> Iterator[List[int]]:
    """Yield one concrete bin at a time for a single epoch of a blueprint.

    Args:
        templates (Mapping[tuple[int, ...], int]): Bin layouts to emit,
            mapped to how many bins use each; a layout is a tuple of the
            sequence lengths sharing a bin.
        index_pools (Mapping[int, Sequence[int]]): For each length, the
            dataset indices that have it.
        seed (int | None): Seed for both the per-length draw order and the
            layout emission order. The same seed reproduces the same epoch
            exactly.
        interleave (bool): When ``False`` (default) every bin of one layout is emitted
            back-to-back in a seed-shuffled layout order — cheap, only
            ``O(distinct layouts)`` of state. When ``True`` the individual bins
            are shuffled together so consecutive bins tend to differ in shape,
            at the cost of ``O(total bins)`` of state.

    Yields:
        list[int]: Each bin as a list of dataset indices.

    Raises:
        ValueError: if a layout asks for more indices of some length than the
            matching pool holds (inconsistent templates / index pools).

    Example:
        >>> bins = stream_packed_bins(
        ...     templates={(6, 4): 1, (5, 5): 1},
        ...     index_pools={6: [0], 4: [1], 5: [2, 3]},
        ...     seed=0,
        ... )
        >>> list(bins)
        [[0, 1], [2, 3]]
    """
    rng = np.random.default_rng(seed)

    # One exhaustible, pre-shuffled stream of indices per length. Consuming it
    # with ``next()`` keeps the "already used" bookkeeping implicit inside the
    # iterator instead of in a parallel table of offsets.
    draw: dict[int, Iterator] = {
        length: iter(rng.permutation(np.asarray(members)))
        for length, members in index_pools.items()
    }

    for layout in _emission_order(templates, rng, interleave):
        row: List[int] = []
        for length in layout:
            stream = draw.get(length)
            picked = next(stream, None) if stream is not None else None
            if picked is None:
                raise ValueError(
                    f"index pool for length {length} ran dry; templates and "
                    f"index pools disagree on how many sequences exist"
                )
            row.append(int(picked))
        yield row


def _emission_order(
    layouts: Mapping[BinLayout, int],
    rng: np.random.Generator,
    interleave: bool,
) -> Iterator[BinLayout]:
    """Yield the layout of each bin to emit, in the order it should appear.

    Grouped (``interleave=False``): shuffle the distinct layouts, then repeat
    each ``count`` times. Interleaved: expand to one entry per bin and shuffle
    the whole run. Both orderings are produced by ``rng.permutation`` over an
    index range, so the sequence is a pure function of the seed.

    Args:
        layouts (Mapping[tuple[int, ...], int]): Bin layout to bin count.
        rng (np.random.Generator): Source of the shuffle.
        interleave (bool): Shuffle individual bins instead of layouts.

    Yields:
        tuple[int, ...]: The layout of each bin, ``sum(layouts.values())``
        in total.

    Example:
        >>> import numpy as np
        >>> layouts = {(6, 4): 2, (5, 5): 1}
        >>> list(_emission_order(layouts, np.random.default_rng(0), interleave=False))
        [(6, 4), (6, 4), (5, 5)]
        >>> list(_emission_order(layouts, np.random.default_rng(0), interleave=True))
        [(5, 5), (6, 4), (6, 4)]
    """
    distinct = list(layouts.keys())

    if interleave:
        expanded = [
            layout for layout in distinct for _ in range(layouts[layout])
        ]
        for i in rng.permutation(len(expanded)):
            yield expanded[i]
    else:
        for i in rng.permutation(len(distinct)):
            layout = distinct[i]
            for _ in range(layouts[layout]):
                yield layout
