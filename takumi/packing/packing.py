"""Packing algorithms behind one entry point: ``pack``.

One dataset can be packed two ways whose products are fundamentally different:

* the concrete packers (``first_fit`` / ``worst_fit``) assign every sequence to
  a bin up front, so the result is a plain ``list[list[int]]`` of dataset
  indices — there is nothing more to wrap;
* the histogram packers (``histogram`` / ``histogram_batched``) produce a
  reusable *blueprint* — ``{layout: count}`` plus per-length index pools — that
  becomes concrete bins one epoch at a time, with a different index draw per
  seed. ``PackingPlan`` bundles that blueprint with its ``max_length`` so the
  pieces travel together into streaming and reporting.

``pack(lengths, max_length, algorithm=...)`` dispatches by name through the
``_PACKERS`` registry. Adding an algorithm means adding one registry entry
(plus an ``@overload`` line if it introduces new keyword options); nothing
else in the package changes.
"""

from __future__ import annotations

from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from takumi.packing._core import group_by_size
from takumi.packing._core import pack_first_fit
from takumi.packing._core import pack_histogram
from takumi.packing._core import pack_histogram_batched
from takumi.packing._core import pack_worst_fit
from takumi.packing.packing_report import PackingReport
from takumi.packing.stream import stream_packed_bins


BinLayout = Tuple[int, ...]
Bins = List[List[int]]


class PackingPlan:
    """A reusable histogram-packing blueprint.

    ``templates`` maps each bin layout (a tuple of the sequence lengths sharing
    a bin) to how many bins use it; ``index_pools`` maps each length to the
    dataset indices that have it. The plan is epoch-agnostic: ``iter_bins``
    draws one epoch's concrete bins, and different seeds place different
    indices into the same layouts.

    Usually built by ``pack(..., algorithm="histogram")`` or
    ``"histogram_batched"``.

    Args:
        templates (Mapping[tuple[int, ...], int]): Bin layout to bin count.
        index_pools (Mapping[int, Sequence[int]]): Length to the dataset
            indices with that length.
        max_length (int): Bin capacity in tokens.

    Example:
        >>> plan = PackingPlan(
        ...     templates={(6, 4): 1, (5, 5): 1},
        ...     index_pools={6: [0], 4: [1], 5: [2, 3]},
        ...     max_length=10,
        ... )
        >>> sorted(sorted(b) for b in plan.iter_bins(seed=0))
        [[0, 1], [2, 3]]
    """

    def __init__(
        self,
        templates: Mapping[BinLayout, int],
        index_pools: Mapping[int, Sequence[int]],
        max_length: int,
    ):
        self.templates: Dict[BinLayout, int] = dict(templates)
        self.index_pools: Dict[int, Sequence[int]] = dict(index_pools)
        self.max_length = int(max_length)

    def iter_bins(
        self,
        seed: Optional[int] = None,
        interleave: bool = False,
    ) -> Iterator[List[int]]:
        """Stream one epoch's concrete bins; see ``stream_packed_bins``.

        Args:
            seed (int | None): Picks which indices fill the layouts and in
                what order bins come out; the same seed gives the same epoch.
            interleave (bool): Shuffle individual bins together instead of
                emitting each layout's bins back to back.

        Returns:
            Iterator[list[int]]: Each bin as a list of dataset indices; every
            index appears exactly once per epoch.

        Example:
            >>> from takumi.packing import pack
            >>> plan = pack([7, 6, 5, 4, 3, 2], max_length=10, algorithm="histogram")
            >>> list(plan.iter_bins(seed=0))
            [[0, 5], [2, 3], [1, 4]]
            >>> list(plan.iter_bins(seed=1, interleave=True))
            [[2, 3], [1, 4], [0, 5]]
        """
        return stream_packed_bins(
            self.templates, self.index_pools, seed=seed, interleave=interleave
        )

    def report(self) -> PackingReport:
        """Packing-quality summary of the blueprint.

        Returns:
            PackingReport: Bin count, token efficiency, and fullness
            percentiles.

        Example:
            >>> from takumi.packing import pack
            >>> plan = pack([7, 6, 5, 4, 3, 2], max_length=10, algorithm="histogram")
            >>> report = plan.report()
            >>> report.n_bins, report.efficiency
            (3, 0.9)
        """
        return PackingReport.from_templates(self.templates, self.max_length)

    def __repr__(self) -> str:
        """Summarize layouts, bins per epoch, and capacity.

        Returns:
            str: For example ``PackingPlan(3 distinct layouts, 3 bins/epoch,
            max_length=10)``.

        Example:
            >>> from takumi.packing import pack
            >>> pack([7, 6, 5, 4, 3, 2], max_length=10, algorithm="histogram")
            PackingPlan(3 distinct layouts, 3 bins/epoch, max_length=10)
        """
        n_bins = sum(self.templates.values())
        return (
            f"PackingPlan({len(self.templates)} distinct layouts, "
            f"{n_bins} bins/epoch, max_length={self.max_length})"
        )


def _histogram(lengths: Sequence[int], max_length: int) -> PackingPlan:
    """``pack``'s ``"histogram"`` algorithm: group by length, then pack.

    Args:
        lengths (Sequence[int], length N): Sequence lengths.
        max_length (int): Bin capacity in tokens.

    Returns:
        PackingPlan: Blueprint from `pack_histogram`.

    Example:
        >>> _histogram([7, 6, 5, 4, 3, 2], 10).templates
        {(5, 4): 1, (6, 3): 1, (7, 2): 1}
    """
    counts, index_pools = group_by_size(lengths, max_length)
    return PackingPlan(pack_histogram(counts, max_length), index_pools, max_length)


def _histogram_batched(lengths: Sequence[int], max_length: int) -> PackingPlan:
    """``pack``'s ``"histogram_batched"`` algorithm: group by length, then pack.

    Args:
        lengths (Sequence[int], length N): Sequence lengths.
        max_length (int): Bin capacity in tokens.

    Returns:
        PackingPlan: Blueprint from `pack_histogram_batched`.

    Example:
        >>> _histogram_batched([2] * 10 + [3] * 4, 8).templates
        {(2, 2, 2, 2): 2, (3, 3, 2): 2}
    """
    counts, index_pools = group_by_size(lengths, max_length)
    return PackingPlan(
        pack_histogram_batched(counts, max_length), index_pools, max_length
    )


_PACKERS = {
    "first_fit": pack_first_fit,
    "worst_fit": pack_worst_fit,
    "histogram": _histogram,
    "histogram_batched": _histogram_batched,
}


@overload
def pack(
    lengths: Sequence[int],
    max_length: int,
    algorithm: Literal["first_fit", "worst_fit"] = ...,
    *,
    tolerance: int = ...,
    seed: Optional[int] = ...,
) -> Bins: ...


@overload
def pack(
    lengths: Sequence[int],
    max_length: int,
    algorithm: Literal["histogram", "histogram_batched"],
) -> PackingPlan: ...


def pack(
    lengths: Sequence[int],
    max_length: int,
    algorithm: str = "first_fit",
    **kwargs,
) -> Union[Bins, PackingPlan]:
    """Pack ``lengths`` into bins of capacity ``max_length`` using ``algorithm``.

    Args:
        lengths (Sequence[int], length N): Sequence lengths; bin entries are
            indices into this sequence. Each must be ``<= max_length``.
        max_length (int): Bin capacity in tokens.
        algorithm (str): One of ``"first_fit"``, ``"worst_fit"`` (concrete),
            ``"histogram"``, ``"histogram_batched"`` (blueprint).
        **kwargs: Forwarded to the packer. The concrete packers accept
            ``tolerance`` (int, ``>= 1``) and ``seed`` (int | None); the
            histogram packers take none.

    Returns:
        list[list[int]] | PackingPlan: Bins of indices into ``lengths`` for
        the concrete algorithms, or a ``PackingPlan`` for the histogram
        algorithms.

    Raises:
        ValueError: if ``algorithm`` is not a registered packer name, or the
            packer rejects its inputs.

    Example:
        >>> lengths = [7, 6, 5, 4, 3, 2]
        >>> pack(lengths, max_length=10)
        [[0, 4], [1, 3], [2, 5]]
        >>> pack(lengths, max_length=10, algorithm="worst_fit")
        [[0, 5], [1, 4], [2, 3]]
        >>> pack(lengths, max_length=10, algorithm="histogram")
        PackingPlan(3 distinct layouts, 3 bins/epoch, max_length=10)
    """
    try:
        packer = _PACKERS[algorithm]
    except KeyError:
        raise ValueError(
            f"unknown packing algorithm {algorithm!r}; "
            f"expected one of {sorted(_PACKERS)}"
        ) from None
    return packer(lengths, max_length, **kwargs)
