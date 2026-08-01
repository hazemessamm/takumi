"""Tests for the C++/nanobind batched histogram packer (pack_histogram_batched).

The batched packer collapses runs of equal-length items into one heap step. We
pin its behavior against a pure-Python port of the reference algorithm, plus the
invariants a correct histogram packing must satisfy, plus quality parity with
the per-item pack_histogram.
"""

import heapq
import random
from collections import Counter

import pytest

from takumi.packing import pack_histogram, pack_histogram_batched


def reference_batched(counts, max_seq_len):
    """Pure-Python worst-fit-decreasing *batched* packer, mirroring seqpack.

    Uses a min-heap over (-remaining, bin_idx). The C++ port tie-breaks equal
    remaining toward the larger bin_idx instead. Two bins with equal remaining
    can hold different compositions, so the exact {template: count} mapping can
    differ -- but filling either yields the same resulting remaining, so bin
    count and the multiset of per-bin sums are preserved. Those are the quality
    invariants the differential test below locks in (same approach as the
    worst-fit packer's tests).
    """
    for length in counts:
        if length > max_seq_len:
            raise ValueError(f"length {length} exceeds max_seq_len {max_seq_len}")

    heap = []  # (-remaining, bin_idx)
    bin_compositions = []
    for length in sorted(counts, reverse=True):
        remaining_count = counts[length]

        while remaining_count > 0 and heap and -heap[0][0] >= length:
            neg_remaining, bin_idx = heapq.heappop(heap)
            bin_remaining = -neg_remaining
            k = min(remaining_count, bin_remaining // length)
            bin_compositions[bin_idx].extend([length] * k)
            bin_remaining -= k * length
            heapq.heappush(heap, (-bin_remaining, bin_idx))
            remaining_count -= k

        items_per_new_bin = max_seq_len // length
        while remaining_count > 0:
            k = min(remaining_count, items_per_new_bin)
            bin_idx = len(bin_compositions)
            bin_compositions.append([length] * k)
            heapq.heappush(heap, (-(max_seq_len - k * length), bin_idx))
            remaining_count -= k

    return Counter(tuple(c) for c in bin_compositions)


def total_items(templates):
    return sum(len(t) * c for t, c in templates.items())


def length_demand(templates):
    demand = Counter()
    for template, count in templates.items():
        for length in template:
            demand[length] += count
    return demand


def bin_sums(templates):
    """Sorted multiset of per-bin token sums (one entry per bin)."""
    sums = []
    for template, count in templates.items():
        sums.extend([sum(template)] * count)
    return sorted(sums)


def test_worked_example():
    # Four length-3 items into capacity 8: max-fill puts two per bin.
    result = pack_histogram_batched({3: 4}, max_length=8)
    assert result == {(3, 3): 2}


def test_no_bin_exceeds_capacity_and_covers_demand():
    counts = {5: 3, 4: 5, 3: 2, 1: 7}
    cap = 8
    result = pack_histogram_batched(counts, cap)
    for template in result:
        assert sum(template) <= cap
    # Every input item is placed exactly once.
    assert length_demand(result) == Counter(counts)
    assert total_items(result) == sum(counts.values())


def test_empty_input():
    assert pack_histogram_batched({}, max_length=10) == {}


def test_single_length_fills_bins():
    # 10 items of length 2 into capacity 8 -> 4 per bin -> two full + one of two.
    result = pack_histogram_batched({2: 10}, max_length=8)
    assert result == {(2, 2, 2, 2): 2, (2, 2): 1}


def test_raises_when_length_exceeds_capacity():
    with pytest.raises(ValueError):
        pack_histogram_batched({12: 1}, max_length=8)


def test_is_deterministic():
    counts = {7: 4, 5: 6, 3: 9, 2: 11}
    assert pack_histogram_batched(counts, 16) == pack_histogram_batched(counts, 16)


def test_quality_matches_python_reference_randomized():
    # Exact templates can differ by the documented tie-break; bin count, the
    # per-bin sum multiset, and the length demand must match exactly.
    rng = random.Random(0)
    for _ in range(300):
        cap = rng.randint(4, 40)
        counts = {
            length: rng.randint(1, 20)
            for length in rng.sample(range(1, cap + 1), rng.randint(1, cap))
        }
        got = pack_histogram_batched(counts, cap)
        ref = dict(reference_batched(counts, cap))

        assert sum(got.values()) == sum(ref.values())      # bin count
        assert bin_sums(got) == bin_sums(ref)               # per-bin fill
        assert length_demand(got) == length_demand(ref)     # every item placed
        assert all(sum(t) <= cap for t in got)              # valid


def test_quality_parity_with_per_item_packer():
    # Docstring promise: same total items, and bin count within one bin of the
    # per-item pack_histogram on realistic inputs.
    rng = random.Random(1)
    for _ in range(300):
        cap = rng.randint(8, 40)
        counts = {
            length: rng.randint(1, 15)
            for length in rng.sample(range(1, cap + 1), rng.randint(1, cap))
        }
        batched = pack_histogram_batched(counts, cap)
        per_item = pack_histogram(counts, cap)

        assert total_items(batched) == total_items(per_item) == sum(counts.values())
        n_batched = sum(batched.values())
        n_per_item = sum(per_item.values())
        assert abs(n_batched - n_per_item) <= 1
