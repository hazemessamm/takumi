"""Behavioral tests for the composable masking module.

These exercise the public contract of ``takumi.masking``: reproducibility,
special-token protection, the three corruption schemes, focus biasing, span
selection, the outcome masks, and preset equivalence — independent of any
particular tokenizer.
"""

from __future__ import annotations

import sys
import types

# The package root eagerly imports the compiled packing extension. When it
# has not been built (pure-Python checkout), stub it so the masking tests can
# still run on their own; when it is built this block is a no-op.
try:  # pragma: no cover - environment dependent
    import takumi.packing._core  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _core = types.ModuleType("takumi.packing._core")
    for _name in (
        "pack_first_fit",
        "pack_worst_fit",
        "group_by_size",
        "pack_histogram",
        "pack_histogram_batched",
    ):
        setattr(_core, _name, lambda *a, **k: None)
    sys.modules["takumi.packing._core"] = _core

import numpy as np
import pytest

from takumi.masking import backend
from takumi.masking import BetaMasker
from takumi.masking import BetaRate
from takumi.masking import Bernoulli
from takumi.masking import CallContext
from takumi.masking import Corruptor
from takumi.masking import DiffusionMasker
from takumi.masking import FixedMasker
from takumi.masking import FixedRate
from takumi.masking import FocusBiased
from takumi.masking import HybridSpanMasker
from takumi.masking import MaskOnly
from takumi.masking import Mixed
from takumi.masking import ScheduleRate
from takumi.masking import Spans
from takumi.masking import focus_from_reference

VOCAB = list(range(10, 60))
SPECIALS = [0]
MASK = 99

# The backend is chosen once at import time from TAKUMI_MASKING_BACKEND, not
# from what happens to be installed — so "is torch importable?" is the wrong
# question for a test that asserts torch tensors come back out.
ACTIVE_BACKEND = backend.framework(None)


def _batch(rows=8, length=32, low=10, high=60):
    rng = np.random.default_rng(0)
    return rng.integers(low, high, size=(rows, length))


# --------------------------------------------------------------------------
# reproducibility & I/O contract
# --------------------------------------------------------------------------

def test_seed_is_bit_exact():
    m = BetaMasker(0.3, 12.0, 0.8, 0.1, MASK, VOCAB, SPECIALS)
    seq = _batch()
    a = m(seq, seed=123)
    b = m(seq, seed=123)
    assert np.array_equal(a.sequences, b.sequences)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.masked, b.masked)


def test_different_seeds_differ():
    m = BetaMasker(0.3, 12.0, 0.8, 0.1, MASK, VOCAB, SPECIALS)
    seq = _batch()
    assert not np.array_equal(m(seq, seed=1).sequences, m(seq, seed=2).sequences)


def test_input_not_mutated():
    m = FixedMasker(0.5, 0.8, 0.1, MASK, VOCAB, SPECIALS)
    seq = _batch()
    before = seq.copy()
    m(seq, seed=7)
    assert np.array_equal(seq, before)


def test_labels_are_original_at_selected_and_ignore_elsewhere():
    m = FixedMasker(0.5, 0.8, 0.1, MASK, VOCAB, SPECIALS)
    seq = _batch()
    res = m(seq, seed=5)
    selected = res.labels != -100
    assert np.array_equal(res.labels[selected], seq[selected])
    assert np.all(res.labels[~selected] == -100)


def test_result_is_positional_and_named():
    m = FixedMasker(0.5, 0.8, 0.1, MASK, VOCAB, SPECIALS)
    res = m(_batch(), seed=1)
    seqs, labels, masked, mutated, kept = res  # positional 5-tuple unpack
    assert seqs is res.sequences and labels is res.labels
    assert masked is res.masked and mutated is res.mutated and kept is res.kept


@pytest.mark.skipif(
    ACTIVE_BACKEND != "torch",
    reason=f"needs the torch backend; TAKUMI_MASKING_BACKEND selected "
    f"{ACTIVE_BACKEND!r}, so corruptors return {ACTIVE_BACKEND} arrays",
)
def test_torch_roundtrip_matches_numpy():
    torch = pytest.importorskip("torch")
    m = BetaMasker(0.4, 20.0, 0.8, 0.1, MASK, VOCAB, SPECIALS)
    seq = _batch()
    n = m(seq, seed=99)
    t = m(torch.as_tensor(seq), seed=99)
    for field in ("sequences", "labels", "masked", "mutated", "kept"):
        assert isinstance(getattr(t, field), torch.Tensor)
        assert np.array_equal(getattr(t, field).numpy(), getattr(n, field))
    assert t.masked.dtype == torch.bool


# --------------------------------------------------------------------------
# special-token protection
# --------------------------------------------------------------------------

def test_specials_never_selected():
    seq = np.zeros((4, 20), dtype=np.int64)
    seq[:, ::2] = 11  # even positions eligible, odd positions are special 0
    m = FixedMasker(0.9, 1.0, 0.0, MASK, VOCAB, SPECIALS)
    res = m(seq, seed=3)
    special = seq == 0
    assert np.all(res.sequences[special] == 0)
    assert np.all(res.labels[special] == -100)


def test_excluded_ids_override():
    seq = _batch()
    m = FixedMasker(0.9, 1.0, 0.0, MASK, VOCAB, SPECIALS)
    res = m(seq, seed=3, excluded_ids=[11, 12, 13])
    for tok in (11, 12, 13):
        assert np.all(res.labels[seq == tok] == -100)


# --------------------------------------------------------------------------
# corruption schemes & outcome masks
# --------------------------------------------------------------------------

def test_mask_only_replaces_all_selected():
    corr = Corruptor(FixedRate(0.6), Bernoulli(), MaskOnly(MASK), SPECIALS)
    res = corr(_batch(), seed=8)
    selected = res.labels != -100
    assert np.all(res.sequences[selected] == MASK)
    assert np.array_equal(res.masked, selected)
    assert not res.mutated.any()
    assert not res.kept.any()


def test_split_masks_partition_selected():
    corr = Corruptor(
        FixedRate(0.6), Bernoulli(), Mixed(0.8, 0.1, MASK, VOCAB), SPECIALS
    )
    seq = np.full((100, 100), 25, dtype=np.int64)  # single eligible token
    res = corr(seq, seed=13)
    selected = res.labels != -100
    # mutually exclusive
    assert not np.any(res.masked & res.mutated)
    assert not np.any(res.masked & res.kept)
    assert not np.any(res.mutated & res.kept)
    # union is exactly the selected set
    assert np.array_equal(res.masked | res.mutated | res.kept, selected)
    # each mask agrees with what actually happened to the tokens
    assert np.all(res.sequences[res.masked] == MASK)
    assert np.all(np.isin(res.sequences[res.mutated], VOCAB))
    assert np.all(res.sequences[res.kept] == 25)


def test_mixed_80_10_10_distribution():
    corr = Corruptor(
        FixedRate(0.5), Bernoulli(), Mixed(0.8, 0.1, MASK, VOCAB), SPECIALS
    )
    seq = np.full((200, 200), 25, dtype=np.int64)
    res = corr(seq, seed=11)
    n = int((res.labels != -100).sum())
    assert abs(int(res.masked.sum()) / n - 0.8) < 0.02
    assert abs(int(res.mutated.sum()) / n - 0.1) < 0.02
    assert abs(int(res.kept.sum()) / n - 0.1) < 0.02


# --------------------------------------------------------------------------
# rate policies
# --------------------------------------------------------------------------

def test_beta_rate_mean():
    policy = BetaRate(0.3, 50.0)
    rng = np.random.default_rng(0)
    draws = [policy(CallContext(rng=rng)) for _ in range(5000)]
    assert abs(np.mean(draws) - 0.3) < 0.02


def test_schedule_rate_endpoints():
    policy = ScheduleRate(10, "cosine")
    assert policy.rate_at(0) == pytest.approx(0.0)
    assert policy.rate_at(10) == pytest.approx(1.0)


def test_schedule_rate_requires_timestep():
    corr = Corruptor(ScheduleRate(10), Bernoulli(), MaskOnly(MASK), SPECIALS)
    with pytest.raises(ValueError):
        corr(_batch(), seed=1)  # no timestep


def test_diffusion_endpoints():
    m = DiffusionMasker(total_timesteps=10, mask_token_id=MASK,
                        special_token_ids=SPECIALS)
    seq = _batch()
    lab0 = m(seq, seed=1, timestep=0).labels
    lab_t = m(seq, seed=1, timestep=10).labels
    assert np.all(lab0 == -100)                 # nothing masked at t=0
    assert np.all(lab_t != -100)                # every eligible masked at t=T


# --------------------------------------------------------------------------
# focus biasing
# --------------------------------------------------------------------------

def test_force_include_selects_focus():
    corr = Corruptor(
        FixedRate(1e-9), FocusBiased("force_include"),
        Mixed(0.8, 0.1, MASK, VOCAB), SPECIALS,
    )
    seq = _batch()
    focus = np.zeros_like(seq, dtype=bool)
    focus[:, 5] = True
    res = corr(seq, seed=4, focus=focus)
    assert np.all(res.labels[:, 5] != -100)


def test_multiplicative_boosts_focus_rate():
    corr = Corruptor(
        FixedRate(0.1), FocusBiased("multiplicative", 5.0),
        Mixed(0.8, 0.1, MASK, VOCAB), SPECIALS,
    )
    seq = np.full((400, 40), 25, dtype=np.int64)
    focus = np.zeros_like(seq, dtype=bool)
    focus[:, :20] = True
    sel = corr(seq, seed=6, focus=focus).labels != -100
    assert sel[:, :20].mean() > sel[:, 20:].mean() * 2


def test_weighted_focus_strength_zero_does_not_crash():
    corr = Corruptor(
        FixedRate(0.5), FocusBiased("weighted", 0.0),
        Mixed(0.8, 0.1, MASK, VOCAB), SPECIALS,
    )
    seq = _batch()
    focus = np.ones_like(seq, dtype=bool)  # all focus -> all weight 0
    res = corr(seq, seed=9, focus=focus)  # must not raise
    assert res.sequences.shape == seq.shape


def test_weighted_preserves_budget():
    corr = Corruptor(
        FixedRate(0.5), FocusBiased("weighted", 3.0),
        Mixed(0.8, 0.1, MASK, VOCAB), SPECIALS,
    )
    seq = np.full((1, 100), 25, dtype=np.int64)
    focus = np.zeros_like(seq, dtype=bool)
    focus[:, :50] = True
    res = corr(seq, seed=1, focus=focus)
    assert int((res.labels != -100).sum()) == 50  # round(0.5 * 100)


# --------------------------------------------------------------------------
# span selection
# --------------------------------------------------------------------------

def test_spans_respect_specials_and_produce_runs():
    seq = _batch(rows=6, length=60)
    seq[:, 0] = 0  # a special somewhere
    corr = Corruptor(
        FixedRate(0.5),
        Spans(length_p=0.3, max_length=8, span_fraction=1.0),
        Mixed(0.8, 0.1, MASK, VOCAB),
        SPECIALS,
    )
    res = corr(seq, seed=2)
    assert np.all(res.labels[seq == 0] == -100)
    sel = res.labels != -100
    assert np.any(sel[:, :-1] & sel[:, 1:])  # at least one run of length >= 2


def test_hybrid_span_supports_focus():
    m = HybridSpanMasker(
        mlm_prob=0.5, mlm_concentration=20.0,
        span_fraction_mean=0.3, span_fraction_concentration=10.0,
        span_length_p=0.4, max_span_length=5,
        masking_prob=0.8, mutation_prob=0.1,
        mask_token_id=MASK, valid_token_ids=VOCAB, special_token_ids=SPECIALS,
        focus_strength=5.0,
    )
    seq = np.full((200, 40), 25, dtype=np.int64)
    focus = np.zeros_like(seq, dtype=bool)
    focus[:, :20] = True
    sel = m(seq, seed=3, focus=focus).labels != -100
    assert sel[:, :20].mean() > sel[:, 20:].mean()


# --------------------------------------------------------------------------
# preset == explicit composition
# --------------------------------------------------------------------------

def test_beta_preset_matches_explicit_composition():
    seq = _batch()
    preset = BetaMasker(0.3, 12.0, 0.8, 0.1, MASK, VOCAB, SPECIALS)
    explicit = Corruptor(
        BetaRate(0.3, 12.0),
        FocusBiased("multiplicative", 2.0),
        Mixed(0.8, 0.1, MASK, VOCAB),
        SPECIALS,
    )
    p = preset(seq, seed=42)
    e = explicit(seq, seed=42)
    assert np.array_equal(p.sequences, e.sequences)
    assert np.array_equal(p.labels, e.labels)


# --------------------------------------------------------------------------
# focus_from_reference helper
# --------------------------------------------------------------------------

def test_focus_from_reference_marks_differences():
    seq = np.array([[1, 2, 3, 4]])
    ref = np.array([1, 9, 3, 9])
    focus = focus_from_reference(seq, ref)
    assert np.array_equal(focus, np.array([[False, True, False, True]]))


def test_focus_from_reference_shape_mismatch_raises():
    with pytest.raises(ValueError):
        focus_from_reference(np.zeros((2, 4)), np.zeros((2, 5)))
