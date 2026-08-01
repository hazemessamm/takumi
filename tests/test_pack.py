"""Tests for the ``pack`` entry point and the ``PackingPlan`` blueprint.

``pack`` dispatches by algorithm name: the concrete algorithms return plain
``list[list[int]]`` bins, the histogram algorithms return a ``PackingPlan``
whose ``iter_bins(seed=...)`` draws one epoch's bins. These tests pin the
dispatch contract and the plan's behavior; the streaming mechanics themselves
live in test_stream.py.
"""

import pytest

from takumi.packing import (
    PackingPlan,
    PackingReport,
    group_by_size,
    pack,
    pack_first_fit,
    pack_histogram,
    pack_histogram_batched,
    pack_worst_fit,
    stream_packed_bins,
)

LENGTHS = [7, 6, 5, 4, 3, 2]
CAP = 10


# --- dispatch ---------------------------------------------------------------

def test_default_algorithm_is_first_fit():
    assert pack(LENGTHS, CAP) == pack_first_fit(LENGTHS, CAP)


def test_dispatches_by_name():
    assert pack(LENGTHS, CAP, "first_fit") == pack_first_fit(LENGTHS, CAP)
    assert pack(LENGTHS, CAP, "worst_fit") == pack_worst_fit(LENGTHS, CAP)


def test_concrete_kwargs_pass_through():
    lengths = [5, 5, 5, 4, 4, 3, 2, 2, 1]
    got = pack(lengths, 10, "first_fit", tolerance=1, seed=42)
    assert got == pack_first_fit(lengths, 10, 1, 42)


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="unknown packing algorithm"):
        pack(LENGTHS, CAP, "frist_fit")


def test_histogram_returns_a_plan():
    assert isinstance(pack(LENGTHS, CAP, "histogram"), PackingPlan)
    assert isinstance(pack(LENGTHS, CAP, "histogram_batched"), PackingPlan)


def test_histogram_plan_matches_underlying_packers():
    counts, pools = group_by_size(LENGTHS, CAP)
    plan = pack(LENGTHS, CAP, "histogram")
    assert plan.templates == pack_histogram(counts, CAP)
    assert plan.index_pools == pools
    assert plan.max_length == CAP

    batched = pack(LENGTHS, CAP, "histogram_batched")
    assert batched.templates == pack_histogram_batched(counts, CAP)


# --- PackingPlan ------------------------------------------------------------

def test_iter_bins_covers_every_index():
    lengths = [5, 5, 4, 4, 3, 2, 2, 1]
    plan = pack(lengths, 8, "histogram")
    bins = list(plan.iter_bins(seed=0))
    assert sorted(i for b in bins for i in b) == list(range(len(lengths)))


def test_iter_bins_matches_free_function():
    plan = pack(LENGTHS, CAP, "histogram")
    assert list(plan.iter_bins(seed=3)) == list(
        stream_packed_bins(plan.templates, plan.index_pools, seed=3)
    )


def test_plan_direct_construction():
    plan = PackingPlan({(3,): 2}, {3: [0, 1]}, max_length=3)
    assert sorted(i for b in plan.iter_bins(seed=0) for i in b) == [0, 1]


def test_plan_report_matches_from_templates():
    plan = pack(LENGTHS, CAP, "histogram")
    assert plan.report() == PackingReport.from_templates(plan.templates, CAP)


def test_concrete_bins_feed_report_directly():
    bins = pack(LENGTHS, CAP, "first_fit")
    report = PackingReport.from_bins(bins, LENGTHS, CAP)
    assert report.n_sequences == len(LENGTHS)
    assert report.n_bins == len(bins)


def test_plan_repr():
    plan = pack(LENGTHS, CAP, "histogram")
    assert "PackingPlan" in repr(plan)
    assert f"max_length={CAP}" in repr(plan)
