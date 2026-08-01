"""Packing-quality statistics (pure Python/numpy).

``PackingReport`` summarizes how fully a packing used its bins. It accepts either
packer output shape — a list of index bins (``from_bins``) or a
template-to-count mapping (``from_templates``) — and reduces both to the same
``(bin token-sum, multiplicity)`` distribution before computing one shared set
of metrics, so results from the two paths are directly comparable.
"""

from dataclasses import dataclass
from typing import List, Mapping, Tuple

import numpy as np

BinLayout = Tuple[int, ...]

# Fullness percentiles reported by every PackingReport.
_PERCENTILES = (50, 90, 99)


@dataclass(repr=False)
class PackingReport:
    n_bins: int
    n_sequences: int
    total_tokens: int
    total_capacity: int
    padding_tokens: int
    efficiency: float
    padding_ratio: float
    bin_fullness_p50: float
    bin_fullness_p90: float
    bin_fullness_p99: float
    max_length: int

    @classmethod
    def from_bins(
        cls,
        bins: List[List[int]],
        lengths: List[int],
        max_length: int,
    ) -> "PackingReport":
        """Summarize a ``list[list[int]]`` packing (``pack_first_fit`` /
        ``pack_worst_fit``), where each bin lists indices into ``lengths``."""
        lengths_arr = np.asarray(lengths, dtype=np.int64)
        bin_sums = np.fromiter(
            (int(lengths_arr[bin_].sum()) for bin_ in bins),
            dtype=np.int64,
            count=len(bins),
        )
        # Every bin is a distinct entry, so each carries weight one.
        multiplicity = np.ones(len(bins), dtype=np.int64)
        n_sequences = sum(len(bin_) for bin_ in bins)
        return cls._reduce(bin_sums, multiplicity, n_sequences, max_length)

    @classmethod
    def from_templates(
        cls,
        templates: Mapping[BinLayout, int],
        max_length: int,
    ) -> "PackingReport":
        """Summarize a ``{template: count}`` packing (``pack_histogram``), where
        a template is a tuple of the lengths sharing a bin."""
        items = list(templates.items())
        bin_sums = np.fromiter(
            (sum(template) for template, _ in items),
            dtype=np.int64,
            count=len(items),
        )
        multiplicity = np.fromiter(
            (count for _, count in items),
            dtype=np.int64,
            count=len(items),
        )
        n_sequences = sum(len(template) * count for template, count in items)
        return cls._reduce(bin_sums, multiplicity, n_sequences, max_length)

    @classmethod
    def _reduce(
        cls,
        bin_sums: np.ndarray,
        multiplicity: np.ndarray,
        n_sequences: int,
        max_length: int,
    ) -> "PackingReport":
        """Assemble the stats from a weighted distribution of per-bin token
        sums. Both public constructors funnel through here."""
        n_bins = int(multiplicity.sum())
        total_tokens = int((bin_sums * multiplicity).sum())
        total_capacity = n_bins * max_length
        used = (total_tokens / total_capacity) if total_capacity else 0.0

        p50, p90, p99 = _fullness_percentiles(bin_sums, multiplicity, max_length)

        return cls(
            n_bins=n_bins,
            n_sequences=n_sequences,
            total_tokens=total_tokens,
            total_capacity=total_capacity,
            padding_tokens=total_capacity - total_tokens,
            efficiency=used,
            padding_ratio=(1.0 - used) if total_capacity else 0.0,
            bin_fullness_p50=p50,
            bin_fullness_p90=p90,
            bin_fullness_p99=p99,
            max_length=max_length,
        )

    def __repr__(self) -> str:
        avg_per_bin = self.n_sequences / self.n_bins if self.n_bins else 0.0
        return (
            f"PackingReport: {self.n_bins:,} bins, {self.n_sequences:,} sequences "
            f"({avg_per_bin:.2f} avg/bin)\n"
            f"  tokens     {self.total_tokens:,}/{self.total_capacity:,} used "
            f"(efficiency {self.efficiency:.2%}, padding {self.padding_ratio:.2%})\n"
            f"  fullness   p50 {self.bin_fullness_p50:.1%} | "
            f"p90 {self.bin_fullness_p90:.1%} | p99 {self.bin_fullness_p99:.1%}\n"
            f"  max_length {self.max_length:,}"
        )


def _fullness_percentiles(
    bin_sums: np.ndarray,
    multiplicity: np.ndarray,
    max_length: int,
) -> Tuple[float, float, float]:
    """Nearest-rank percentiles of per-bin fullness (``token_sum / max_length``),
    each bin weighted by ``multiplicity``, returned as fractions in ``[0, 1]``.

    Operates on the distinct ``(sum, count)`` pairs via a cumulative-weight
    search, so the template path never expands to one entry per bin.
    """
    if bin_sums.size == 0 or max_length == 0:
        return (0.0, 0.0, 0.0)

    order = np.argsort(bin_sums, kind="stable")
    sorted_sums = bin_sums[order]
    cumulative = np.cumsum(multiplicity[order])
    total = int(cumulative[-1])

    ranks = np.ceil(np.array(_PERCENTILES) / 100.0 * total)
    picks = np.searchsorted(cumulative, ranks, side="left")
    picks = np.clip(picks, 0, sorted_sums.size - 1)

    fullness = sorted_sums[picks] / max_length
    return (float(fullness[0]), float(fullness[1]), float(fullness[2]))
