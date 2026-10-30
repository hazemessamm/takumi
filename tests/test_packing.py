"""Tests for the C++/nanobind greedy packer.

These mirror the original seqpack test suite and add a differential test against
a small pure-Python reference so we know the port is faithful, not just plausible.
"""

import random

import pytest

from takumi.packing import pack_first_fit


def reference_greedy(lengths, max_seq_len, tolerance=1):
    """Pure-Python First-Fit-Decreasing with a *stable* tie sort.

    The C++ core uses std::stable_sort, so this reference breaks length-ties the
    same way (by original index). NOTE: the original numpy seqpack uses an
    unstable np.argsort, so it can disagree on tie order — but only on *which*
    equal-length item lands in which bin; bin count and per-bin sums match.
    """
    order = sorted(range(len(lengths)), key=lambda i: -(lengths[i] // tolerance))
    bins, remaining = [], []
    for idx in order:
        seq_len = lengths[idx]
        for b in range(len(bins)):
            if remaining[b] >= seq_len:
                bins[b].append(idx)
                remaining[b] -= seq_len
                break
        else:
            bins.append([idx])
            remaining.append(max_seq_len - seq_len)
    return bins


def test_no_bin_exceeds_capacity():
    lengths = [7, 6, 5, 4, 3, 2]
    for bin_ in pack_first_fit(lengths, max_length=10):
        assert sum(lengths[i] for i in bin_) <= 10


def test_all_indices_packed_exactly_once():
    lengths = [7, 6, 5, 4, 3, 2]
    bins = pack_first_fit(lengths, max_length=10)
    flat = [i for bin_ in bins for i in bin_]
    assert sorted(flat) == list(range(len(lengths)))


def test_worked_example_from_docs():
    # PACKING.md: lengths [7,6,5,4,3,2], cap 10 -> [[7,3],[6,4],[5,2]]
    bins = pack_first_fit([7, 6, 5, 4, 3, 2], max_length=10)
    assert bins == [[0, 4], [1, 3], [2, 5]]


def test_consecutive_lengths_fill_every_bin():
    # Lengths 1..999 (length k + 1 at index k) into bins of 999: 999 goes
    # alone, then each length k pairs with 999 - k, so all 500 bins are full.
    # Every sequence fits exactly one open bin here, so this pins the answer
    # at scale but cannot tell First-Fit from another fit rule; the reference
    # tests below do that.
    lengths = list(range(1, 1000))
    expected = [[998]] + [[998 - k, k - 1] for k in range(1, 500)]
    assert pack_first_fit(lengths, max_length=999) == expected


def test_matches_python_reference_randomized():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 50)
        cap = rng.randint(4, 32)
        lengths = [rng.randint(1, cap) for _ in range(n)]
        assert pack_first_fit(lengths, cap) == reference_greedy(lengths, cap)


# Thousands of sequences, each length repeated many times: First-Fit often has
# several open bins to choose from, and equal lengths must break ties by
# position exactly as the reference does. "long tail" keeps most sequences
# short, so bins hold many of them.
@pytest.mark.parametrize(
    ("distribution", "cap"),
    [("uniform", 64), ("uniform", 1024), ("long tail", 256)],
)
def test_matches_python_reference_at_scale(distribution, cap):
    rng = random.Random(cap)
    if distribution == "uniform":
        lengths = [rng.randint(1, cap) for _ in range(3000)]
    else:
        lengths = [min(cap, 1 + int(rng.expovariate(8 / cap))) for _ in range(3000)]
    assert pack_first_fit(lengths, cap) == reference_greedy(lengths, cap)


def test_empty_input():
    assert pack_first_fit([], max_length=10) == []


def test_seed_is_reproducible():
    # The defining property of a seed: same seed -> identical output, every time.
    lengths = [5, 5, 5, 4, 4, 3, 2, 2, 1]
    a = pack_first_fit(lengths, 10, seed=42)
    b = pack_first_fit(lengths, 10, seed=42)
    assert a == b


# 14904199615042885880 is seed_from_components(42, 0, 7): that helper returns
# seeds in [0, 2**64), half of them >= 2**63. Written out so packing tests
# don't import takumi.masking, which needs torch.
@pytest.mark.parametrize("seed", [2**63, 2**64 - 1, 14904199615042885880])
def test_seed_accepts_the_full_64_bit_range(seed):
    lengths = [5, 5, 5, 4, 4, 3, 2, 2, 1]
    assert pack_first_fit(lengths, 10, seed=seed) == pack_first_fit(
        lengths, 10, seed=seed
    )


def test_golden_seeded_packing():
    # Pins the seeded tie order. CI runs this on Linux, macOS and Windows
    # (libstdc++, libc++, MSVC), so passing everywhere is what backs "same
    # seed, same packing on every platform": the tie-breakers are raw
    # mt19937_64 output, which the C++ standard fixes bit for bit.
    lengths = [5, 3, 5, 2, 3, 5, 2, 3, 4, 1]
    assert pack_first_fit(lengths, 8, seed=0) == [
        [2, 7], [5, 4], [0, 1], [8, 3, 6], [9],
    ]
    assert pack_first_fit(lengths, 8, seed=2**64 - 1) == [
        [0, 7], [2, 1], [5, 4], [8, 6, 3], [9],
    ]
    assert pack_first_fit(lengths, 8, tolerance=2, seed=7) == [
        [5, 4], [2, 6, 9], [8, 3], [0, 7], [1],
    ]


def test_seed_bits_above_32_change_the_packing():
    # std::mt19937 kept only the low 32 bits, so these two seeds collided.
    lengths = [5] * 40  # all tied: the seed alone orders them
    assert pack_first_fit(lengths, 10, seed=1) != pack_first_fit(
        lengths, 10, seed=1 + 2**32
    )


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_seed_outside_64_bit_range_is_rejected(seed):
    with pytest.raises(TypeError):
        pack_first_fit([5, 4, 3], 10, seed=seed)


def test_seed_produces_valid_packing():
    lengths = [5, 5, 5, 4, 4, 3, 2, 2, 1]
    bins = pack_first_fit(lengths, 10, seed=7)
    flat = sorted(i for bin_ in bins for i in bin_)
    assert flat == list(range(len(lengths)))                     # every index once
    assert all(sum(lengths[i] for i in b) <= 10 for b in bins)   # no bin over cap


def test_seed_affects_ordering_across_many_seeds():
    # Robust "seed has an effect": over many seeds we must see >1 ordering.
    # Not a coin flip -- with fully-tied items, all-identical is astronomically
    # unlikely, so this won't flake the way a single seed-pair comparison would.
    lengths = [5] * 12  # all tied -> maximal shuffle freedom
    orderings = {
        tuple(pack_first_fit(lengths, 10, seed=s)[0]) for s in range(20)
    }
    assert len(orderings) > 1

