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
        """Stream one epoch's concrete bins; see ``stream_packed_bins``."""
        return stream_packed_bins(
            self.templates, self.index_pools, seed=seed, interleave=interleave
        )

    def report(self) -> PackingReport:
        """Packing-quality summary of the blueprint."""
        return PackingReport.from_templates(self.templates, self.max_length)

    def __repr__(self) -> str:
        n_bins = sum(self.templates.values())
        return (
            f"PackingPlan({len(self.templates)} distinct layouts, "
            f"{n_bins} bins/epoch, max_length={self.max_length})"
        )


def _histogram(lengths: Sequence[int], max_length: int) -> PackingPlan:
    counts, index_pools = group_by_size(lengths, max_length)
    return PackingPlan(pack_histogram(counts, max_length), index_pools, max_length)


def _histogram_batched(lengths: Sequence[int], max_length: int) -> PackingPlan:
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
        lengths: Sequence lengths; bin entries are indices into this sequence.
        max_length: Bin capacity in tokens.
        algorithm: One of ``"first_fit"``, ``"worst_fit"`` (concrete),
            ``"histogram"``, ``"histogram_batched"`` (blueprint).
        **kwargs: Forwarded to the packer. The concrete packers accept
            ``tolerance`` and ``seed``; the histogram packers take none.

    Returns:
        ``list[list[int]]`` of bins for the concrete algorithms, or a
        ``PackingPlan`` for the histogram algorithms.

    Raises:
        ValueError: if ``algorithm`` is not a registered packer name.
    """
    try:
        packer = _PACKERS[algorithm]
    except KeyError:
        raise ValueError(
            f"unknown packing algorithm {algorithm!r}; "
            f"expected one of {sorted(_PACKERS)}"
        ) from None
    return packer(lengths, max_length, **kwargs)
