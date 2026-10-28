"""Tests for the C++/nanobind heap (worst-fit-decreasing) packer.

Unlike greedy, we do NOT assert exact equality with a Python reference: the C++
max-heap tie-breaks toward the larger bin_idx while Python's heapq tie-breaks
toward the smaller one (see the fidelity note in packing.cpp). Both are valid
WFD. So the reference here locks in the *quality* invariants that DO hold —
validity, and bin-count / bin-sum parity — not the exact assignment.
"""

import heapq
import random

from takumi.packing import pack_worst_fit


def reference_heap(lengths, max_seq_len, tolerance=1):
    """Pure-Python worst-fit-decreasing, mirroring the original seqpack heapq
    version. Min-heap over (-remaining, bin_idx) = max-heap on remaining."""
    order = sorted(range(len(lengths)), key=lambda i: -(lengths[i] // tolerance))
    bins = []
    heap = []  # (-remaining, bin_idx)
    for idx in order:
        seq_len = lengths[idx]
        if heap and -heap[0][0] >= seq_len:
            neg_remaining, bin_idx = heapq.heappop(heap)
            bins[bin_idx].append(idx)
            heapq.heappush(heap, (-(-neg_remaining - seq_len), bin_idx))
        else:
            bin_idx = len(bins)
            bins.append([idx])
            heapq.heappush(heap, (-(max_seq_len - seq_len), bin_idx))
    return bins


def test_no_bin_exceeds_capacity():
    lengths = [7, 6, 5, 4, 3, 2]
    for bin_ in pack_worst_fit(lengths, max_length=10):
        assert sum(lengths[i] for i in bin_) <= 10


def test_all_indices_packed_exactly_once():
    lengths = [7, 6, 5, 4, 3, 2]
    bins = pack_worst_fit(lengths, max_length=10)
    flat = [i for bin_ in bins for i in bin_]
    assert sorted(flat) == list(range(len(lengths)))


def test_worked_example_from_docs():
    # PACKING.md WFD: [7,6,5,4,3,2] cap 10 -> [[7,2],[6,3],[5,4]].
    # No ties in this input, so C++ and Python agree exactly.
    bins = pack_worst_fit([7, 6, 5, 4, 3, 2], max_length=10)
    assert bins == [[0, 5], [1, 4], [2, 3]]


def test_empty_input():
    assert pack_worst_fit([], max_length=10) == []


def test_quality_matches_reference_randomized():
    # Deterministic seed -> this test is reproducible. We assert the invariants
    # that survive the tie-break difference: validity, same bin count, and the
    # same multiset of per-bin token sums.
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(0, 60)
        cap = rng.randint(4, 40)
        lengths = [rng.randint(1, cap) for _ in range(n)]

        bins = pack_worst_fit(lengths, cap)
        ref = reference_heap(lengths, cap)

        # validity
        flat = sorted(i for b in bins for i in b)
        assert flat == list(range(n))
        assert all(sum(lengths[i] for i in b) <= cap for b in bins)

        # quality parity (not exact assignment)
        assert len(bins) == len(ref)
        cpp_sums = sorted(sum(lengths[i] for i in b) for b in bins)
        ref_sums = sorted(sum(lengths[i] for i in b) for b in ref)
        assert cpp_sums == ref_sums


def test_seed_is_reproducible():
    lengths = [5, 5, 5, 4, 4, 3, 2, 2, 1]
    a = pack_worst_fit(lengths, 10, seed=42)
    b = pack_worst_fit(lengths, 10, seed=42)
    assert a == b


def test_golden_seeded_packing():
    # Same seed, same packing on Linux, macOS and Windows; see the matching
    # test in test_packing.py.
    lengths = [5, 3, 5, 2, 3, 5, 2, 3, 4, 1]
    assert pack_worst_fit(lengths, 8, seed=0) == [
        [2, 3], [5, 1], [0, 4], [8, 7], [6, 9],
    ]
