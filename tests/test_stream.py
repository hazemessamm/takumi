"""Tests for stream_packed_bins — the one-pass bin emitter.

Exercises the histogram-packing streaming path on raw blueprints (templates +
index pools): full coverage, layout fidelity, seed reproducibility, and the
grouped vs interleaved ordering contract. The ``PackingPlan`` wrapper that
carries these mappings around is covered in test_pack.py.
"""

import random
from collections import Counter
from itertools import groupby

import pytest

from takumi.packing import (
    group_by_size,
    pack_histogram,
    stream_packed_bins,
)


def make_blueprint(lengths, max_seq_len):
    """(layouts, index_pools) for `lengths`, via the real pipeline."""
    counts, index_pools = group_by_size(lengths, max_seq_len)
    return pack_histogram(counts, max_seq_len), index_pools


def test_returns_a_lazy_iterator():
    layouts, pools = make_blueprint([3, 2, 1], max_seq_len=4)
    stream = stream_packed_bins(layouts, pools)
    # A generator is its own iterator and is not a materialized list.
    assert iter(stream) is stream
    assert not isinstance(stream, list)


def test_every_index_emitted_exactly_once():
    lengths = [5, 5, 4, 4, 3, 2, 2, 1]
    layouts, pools = make_blueprint(lengths, max_seq_len=8)
    for interleave in (False, True):
        bins = list(stream_packed_bins(layouts, pools, seed=0, interleave=interleave))
        flat = sorted(i for b in bins for i in b)
        assert flat == list(range(len(lengths)))


def test_emitted_bins_match_layout_multiset():
    lengths = [6, 5, 4, 4, 3, 3, 2, 1]
    layouts, pools = make_blueprint(lengths, max_seq_len=8)
    bins = list(stream_packed_bins(layouts, pools, seed=1))
    # Each emitted bin's actual lengths must reconstruct one layout, and the
    # multiset of emitted layouts must equal what pack_histogram asked for.
    emitted = Counter(tuple(lengths[i] for i in b) for b in bins)
    assert emitted == Counter(dict(layouts))


def test_same_seed_is_reproducible():
    lengths = [5, 5, 5, 4, 4, 3, 2, 2, 1]
    layouts, pools = make_blueprint(lengths, max_seq_len=10)
    a = list(stream_packed_bins(layouts, pools, seed=42))
    b = list(stream_packed_bins(layouts, pools, seed=42))
    assert a == b


def test_epoch_ignores_dict_and_index_order():
    # Same contents, different listing order: C++ hash maps hand back keys in
    # an order that differs across standard libraries, and callers may build
    # pools by hand, so the epoch must depend on contents alone.
    lengths = [5, 3, 5, 2, 3, 5, 2, 3, 4, 1, 3, 2]
    layouts, pools = make_blueprint(lengths, max_seq_len=8)
    rng = random.Random(0)
    for _ in range(5):
        shuffled_layouts = dict(rng.sample(list(layouts.items()), len(layouts)))
        shuffled_pools = {
            length: rng.sample(members, len(members))
            for length, members in rng.sample(list(pools.items()), len(pools))
        }
        for seed in (0, 7):
            for interleave in (False, True):
                assert list(
                    stream_packed_bins(shuffled_layouts, shuffled_pools,
                                       seed=seed, interleave=interleave)
                ) == list(
                    stream_packed_bins(layouts, pools, seed=seed,
                                       interleave=interleave)
                )


def test_group_by_size_returns_ascending_keys():
    counts, pools = group_by_size([3, 5, 3, 8, 3, 5, 3, 1])
    assert list(counts) == [1, 3, 5, 8]
    assert list(pools) == [1, 3, 5, 8]


def test_golden_epoch():
    # Pins one epoch end to end (group_by_size -> pack_histogram -> stream).
    # CI runs this on Linux, macOS and Windows, so passing everywhere is what
    # backs "same seed, same epoch on every platform". A failure after a NumPy
    # upgrade means its Generator changed the shuffle (NEP 19 allows that).
    lengths = [5, 3, 5, 2, 3, 5, 2, 3, 4, 1]
    layouts, pools = make_blueprint(lengths, max_seq_len=8)
    assert list(stream_packed_bins(layouts, pools, seed=0)) == [
        [3, 9], [5, 7], [2, 1], [0, 6], [8, 4],
    ]
    assert list(stream_packed_bins(layouts, pools, seed=0, interleave=True)) == [
        [5, 7], [8, 1], [2, 3], [6, 9], [0, 4],
    ]


def test_seed_changes_the_stream():
    # Many identical-length sequences -> lots of shuffle freedom, so distinct
    # seeds must yield distinct emissions with overwhelming probability.
    lengths = [2] * 16
    layouts, pools = make_blueprint(lengths, max_seq_len=8)
    seen = {
        tuple(tuple(b) for b in stream_packed_bins(layouts, pools, seed=s))
        for s in range(20)
    }
    assert len(seen) > 1


def test_grouped_mode_keeps_each_layout_contiguous():
    # Two distinct layouts, distinguishable by bin size (1 vs 2 indices).
    layouts = {(3,): 3, (2, 1): 3}
    pools = {3: [10, 11, 12], 2: [20, 21, 22], 1: [30, 31, 32]}
    for seed in range(10):
        sizes = [len(b) for b in stream_packed_bins(layouts, pools, seed=seed)]
        # Grouped => at most one contiguous run per distinct layout shape.
        runs = [k for k, _ in groupby(sizes)]
        assert len(runs) == len(set(sizes))


def test_interleave_can_break_contiguity():
    layouts = {(3,): 4, (2, 1): 4}
    pools = {3: [10, 11, 12, 13], 2: [20, 21, 22, 23], 1: [30, 31, 32, 33]}
    # Over several seeds at least one ordering must interleave the two shapes.
    saw_interleaving = False
    for seed in range(15):
        sizes = [
            len(b)
            for b in stream_packed_bins(layouts, pools, seed=seed, interleave=True)
        ]
        if len([k for k, _ in groupby(sizes)]) > len(set(sizes)):
            saw_interleaving = True
            break
    assert saw_interleaving


def test_empty_layouts_yields_nothing():
    assert list(stream_packed_bins({}, {}, seed=0)) == []


def test_exhausted_pool_raises():
    # Layout wants two length-5 indices but only one exists.
    with pytest.raises(ValueError, match="ran dry"):
        list(stream_packed_bins({(5, 5): 1}, {5: [0]}, seed=0))


def test_missing_length_raises():
    with pytest.raises(ValueError, match="ran dry"):
        list(stream_packed_bins({(9,): 1}, {5: [0]}, seed=0))


def test_matches_random_pipeline():
    # Property test: for random datasets, a full pass always covers every index
    # once and never violates the layout multiset.
    rng = random.Random(0)
    for _ in range(100):
        n = rng.randint(1, 60)
        cap = rng.randint(4, 24)
        lengths = [rng.randint(1, cap) for _ in range(n)]
        layouts, pools = make_blueprint(lengths, cap)
        bins = list(stream_packed_bins(layouts, pools, seed=rng.randint(0, 10_000)))
        assert sorted(i for b in bins for i in b) == list(range(n))
        assert Counter(tuple(lengths[i] for i in b) for b in bins) == Counter(dict(layouts))
