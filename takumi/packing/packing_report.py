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
    """How fully a packing used its bins.

    Build one with ``from_bins`` (concrete packers) or ``from_templates``
    (histogram packers); both give directly comparable numbers.

    Attributes:
        n_bins (int): Number of bins.
        n_sequences (int): Number of sequences across all bins.
        total_tokens (int): Sum of all sequence lengths.
        total_capacity (int): ``n_bins * max_length``.
        padding_tokens (int): ``total_capacity - total_tokens``.
        efficiency (float): ``total_tokens / total_capacity``, in ``[0, 1]``.
        padding_ratio (float): ``1 - efficiency``.
        bin_fullness_p50 (float): Median bin fullness, as a fraction.
        bin_fullness_p90 (float): 90th-percentile bin fullness.
        bin_fullness_p99 (float): 99th-percentile bin fullness.
        max_length (int): Bin capacity in tokens.

    Example:
        >>> from takumi.packing import pack
        >>> lengths = [7, 6, 5, 4, 3, 2]
        >>> report = PackingReport.from_bins(pack(lengths, 10), lengths, 10)
        >>> report.n_bins, report.padding_tokens, report.efficiency
        (3, 3, 0.9)
    """

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
        ``pack_worst_fit``), where each bin lists indices into ``lengths``.

        Args:
            bins (list[list[int]]): Bins of indices into ``lengths``.
            lengths (list[int], length N): Sequence lengths.
            max_length (int): Bin capacity in tokens.

        Returns:
            PackingReport: The packing's statistics.

        Example:
            >>> report = PackingReport.from_bins([[0, 1], [2]], [6, 4, 5], 10)
            >>> report.n_bins, report.total_tokens, report.bin_fullness_p50
            (2, 15, 0.5)
        """
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
        a template is a tuple of the lengths sharing a bin.

        Args:
            templates (Mapping[tuple[int, ...], int]): Bin layout to bin
                count.
            max_length (int): Bin capacity in tokens.

        Returns:
            PackingReport: The packing's statistics, computed without
            expanding to one entry per bin.

        Example:
            >>> report = PackingReport.from_templates({(6, 4): 3, (5,): 1}, 10)
            >>> report.n_bins, report.n_sequences, report.efficiency
            (4, 7, 0.875)
        """
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
        sums. Both public constructors funnel through here.

        Args:
            bin_sums (np.ndarray of int, shape (K,)): Token sum of each
                distinct bin.
            multiplicity (np.ndarray of int, shape (K,)): How many bins have
                each sum.
            n_sequences (int): Total sequences across all bins.
            max_length (int): Bin capacity in tokens.

        Returns:
            PackingReport: The assembled statistics.

        Example:
            >>> import numpy as np
            >>> report = PackingReport._reduce(np.array([10, 5]), np.array([3, 1]), 7, 10)
            >>> report.total_capacity, report.padding_tokens
            (40, 5)
        """
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
        """Four-line human-readable summary.

        Returns:
            str: Bin and sequence counts, token efficiency, fullness
            percentiles, and capacity.

        Example:
            >>> from takumi.packing import pack
            >>> print(pack([7, 6, 5, 4, 3, 2], 10, algorithm="histogram").report())
            PackingReport: 3 bins, 6 sequences (2.00 avg/bin)
              tokens     27/30 used (efficiency 90.00%, padding 10.00%)
              fullness   p50 90.0% | p90 90.0% | p99 90.0%
              max_length 10
        """
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

    Args:
        bin_sums (np.ndarray of int, shape (K,)): Token sum of each distinct
            bin.
        multiplicity (np.ndarray of int, shape (K,)): How many bins have each
            sum.
        max_length (int): Bin capacity in tokens.

    Returns:
        tuple[float, float, float]: Fullness at p50, p90, and p99; all zero
        for an empty packing.

    Example:
        >>> import numpy as np
        >>> _fullness_percentiles(np.array([5, 10]), np.array([3, 1]), 10)
        (0.5, 1.0, 1.0)
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
