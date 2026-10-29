"""Behavioral tests for the composable masking module.

These exercise the public contract of ``takumi.masking``: reproducibility,
special-token protection, the three corruption schemes, focus biasing, span
selection, per-row rates, the outcome masks, preset equivalence, and MaskGIT
training and decoding — independent of any particular tokenizer.
"""

from __future__ import annotations

import json
import math
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
from takumi.masking import CosineSchedule
from takumi.masking import Corruptor
from takumi.masking import DiffusionMasker
from takumi.masking import elbo_loss
from takumi.masking import EMAScheduler
from takumi.masking import ExactCount
from takumi.masking import FixedMasker
from takumi.masking import FixedRate
from takumi.masking import FocusBiased
from takumi.masking import HybridSpanMasker
from takumi.masking import LinearSchedule
from takumi.masking import LinearScheduler
from takumi.masking import MaskGITDecoder
from takumi.masking import MaskGITMasker
from takumi.masking import MaskOnly
from takumi.masking import MaskSchedule
from takumi.masking import Mixed
from takumi.masking import MLMProbabilitySampler
from takumi.masking import ScheduleRate
from takumi.masking import Spans
from takumi.masking import SqrtSchedule
from takumi.masking import StagedLinearScheduler
from takumi.masking import UniformTimestepRate
from takumi.masking import focus_from_reference
from takumi.masking.rates import RateDraw
from takumi.masking.rates import RatePolicy
from takumi.masking.schedules import resolve_schedule

VOCAB = list(range(10, 60))
SPECIALS = [0]
MASK = 99

# The backend is chosen once at import time from TAKUMI_MASKING_BACKEND, not
# from what happens to be installed — so "is torch importable?" is the wrong
# question for a test that asserts torch tensors come back out.
ACTIVE_BACKEND = backend.framework()


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
    seqs, labels, masked, mutated, kept, timesteps, weights = res
    assert seqs is res.sequences and labels is res.labels
    assert masked is res.masked and mutated is res.mutated and kept is res.kept
    assert timesteps is res.timesteps and weights is res.weights


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
    for field in ("sequences", "labels", "masked", "mutated", "kept", "weights"):
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
    draws = [policy(CallContext(rng=rng)).rate for _ in range(5000)]
    assert abs(np.mean(draws) - 0.3) < 0.02


def test_schedule_rate_endpoints():
    policy = ScheduleRate(10, "cosine", time="discrete")
    assert policy.rate_at(0) == pytest.approx(0.0)
    assert policy.rate_at(10) == pytest.approx(1.0)


def test_schedule_rate_requires_timestep():
    corr = Corruptor(ScheduleRate(10, time="discrete"), Bernoulli(), MaskOnly(MASK),
                     SPECIALS)
    with pytest.raises(ValueError):
        corr(_batch(), seed=1)  # no timestep


def test_diffusion_endpoints():
    m = DiffusionMasker(total_timesteps=10, mask_token_id=MASK,
                        special_token_ids=SPECIALS, time="discrete")
    seq = _batch()
    lab0 = m(seq, seed=1, timestep=0).labels
    lab_t = m(seq, seed=1, timestep=10).labels
    assert np.all(lab0 == -100)                 # nothing masked at t=0
    assert np.all(lab_t != -100)                # every eligible masked at t=T


def test_cosine_schedule_matches_maskgit():
    # MaskGIT's gamma(r) = cos(r * pi / 2) at progress r = 1 - t / T. The
    # endpoints alone cannot tell it from its mirror image 1 - cos(t*pi/2T),
    # so check the midpoint and the mean too.
    policy = ScheduleRate(100, "cosine", time="continuous")
    assert policy.rate_at(50) == pytest.approx(math.sqrt(0.5))
    mean = np.mean([policy.rate_at(t) for t in np.linspace(0, 100, 10001)])
    assert mean == pytest.approx(2 / math.pi, abs=1e-3)


def test_schedule_rate_per_row_timesteps():
    corr = Corruptor(ScheduleRate(10, time="discrete"), Bernoulli(), MaskOnly(MASK),
                     SPECIALS)
    labels = corr(_batch(rows=2), seed=1, timestep=np.array([0, 10])).labels
    assert np.all(labels[0] == -100)
    assert np.all(labels[1] != -100)


def test_per_row_rate_must_match_batch():
    corr = Corruptor(ScheduleRate(10, time="discrete"), Bernoulli(), MaskOnly(MASK),
                     SPECIALS)
    with pytest.raises(ValueError):
        corr(_batch(rows=2), seed=1, timestep=np.array([0, 5, 10]))


def test_uniform_timestep_rate_draws_per_row():
    policy = UniformTimestepRate("cosine")
    ctx = CallContext(rng=np.random.default_rng(0), batch_size=20000)
    rates = policy(ctx).rate
    assert rates.shape == (20000,)
    assert np.all((rates > 0.0) & (rates <= 1.0))
    assert abs(rates.mean() - 2 / math.pi) < 0.01
    assert rates.std() > 0.2  # rows really differ


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


def test_focus_accepts_nested_lists():
    corr = Corruptor(
        FixedRate(1e-9), FocusBiased("force_include"), MaskOnly(MASK), SPECIALS
    )
    res = corr(_batch(rows=1, length=4), seed=0, focus=[[0, 1, 0, 1]])
    assert (res.labels[0] != -100).tolist() == [False, True, False, True]


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
# per-row rates & exact-count selection
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "select",
    [
        Bernoulli(),
        FocusBiased("multiplicative"),
        FocusBiased("weighted"),
        Spans(length_p=0.3, max_length=8),
        ExactCount(),
    ],
    ids=["bernoulli", "multiplicative", "weighted", "spans", "exact"],
)
def test_selectors_honor_per_row_rates(select):
    seq = _batch(rows=2)
    seq[:, 0] = 0
    eligible = seq != 0
    focus = np.zeros_like(seq, dtype=bool)
    focus[:, :10] = True
    rates = np.array([0.0, 1.0])
    sel = select(np.random.default_rng(0), rates, eligible, focus)
    assert not sel[0].any()
    assert np.array_equal(sel[1], eligible[1])


def test_exact_count_selects_ceil_of_rate():
    eligible = np.ones((3, 10), dtype=bool)
    eligible[2, 5:] = False
    rates = np.array([0.3, 0.01, 0.5])
    sel = ExactCount()(np.random.default_rng(0), rates, eligible, None)
    # 0.3 * 10 is 3 despite float error; ceil(0.1) = 1; ceil(0.5 * 5) = 3
    assert sel.sum(axis=1).tolist() == [3, 1, 3]
    assert not np.any(sel & ~eligible)


def test_exact_count_is_uniform_over_positions():
    eligible = np.ones((20000, 10), dtype=bool)
    sel = ExactCount()(np.random.default_rng(0), 0.3, eligible, None)
    assert np.all(sel.sum(axis=1) == 3)
    assert np.all(np.abs(sel.mean(axis=0) - 0.3) < 0.02)


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


def test_maskgit_preset_matches_explicit_composition():
    seq = _batch()
    preset = MaskGITMasker(MASK, SPECIALS)
    explicit = Corruptor(
        UniformTimestepRate("cosine"), ExactCount(), MaskOnly(MASK), SPECIALS
    )
    p = preset(seq, seed=42)
    e = explicit(seq, seed=42)
    assert np.array_equal(p.sequences, e.sequences)
    assert np.array_equal(p.labels, e.labels)


# --------------------------------------------------------------------------
# MaskGIT training & decoding
# --------------------------------------------------------------------------

def test_maskgit_masker_masks_each_row_at_its_own_rate():
    seq = _batch(rows=2000, length=50)
    seq[:, 0] = 0
    res = MaskGITMasker(MASK, SPECIALS)(seq, seed=0)
    selected = res.labels != -100
    assert np.array_equal(res.masked, selected)
    assert np.all(res.sequences[selected] == MASK)
    assert not selected[:, 0].any()
    per_row = selected.sum(axis=1)
    assert per_row.min() >= 1
    assert len(np.unique(per_row)) > 10
    # ceil(cos schedule * 49 eligible): the mean sits just above 2 / pi
    assert abs(per_row.mean() / 49 - 2 / math.pi) < 0.03


def test_maskgit_decoder_unmasks_on_schedule():
    steps, rows, length = 8, 4, 64
    decoder = MaskGITDecoder(MASK, total_steps=steps, seed=1)
    rng = np.random.default_rng(0)
    tokens = np.full((rows, length), MASK)
    tokens[:, 0] = 0  # known prompt token
    num_unknown = length - 1
    unknown = num_unknown
    for step in range(steps):
        sampled = rng.integers(10, 60, size=(rows, length))
        probs = rng.random((rows, length))
        tokens = decoder(tokens, sampled, probs, step, num_unknown)
        assert np.all(tokens[:, 0] == 0)
        if step < steps - 1:
            ratio = math.cos(math.pi / 2 * (step + 1) / steps)
            unknown = max(1, min(unknown - 1, int(num_unknown * ratio)))
        else:
            unknown = 0
        assert np.all((tokens == MASK).sum(axis=1) == unknown)


def test_maskgit_decoder_remasks_least_confident():
    decoder = MaskGITDecoder(MASK, total_steps=2, temperature=0.0)
    tokens = np.array([[42, MASK, MASK, MASK, MASK, MASK]])
    sampled = np.array([[11, 12, 13, 14, 15, 16]])
    probs = np.array([[0.0, 0.9, 0.1, 0.8, 0.2, 0.7]])
    out = decoder(tokens, sampled, probs, step=0, num_unknown=5)
    # floor(5 * cos(pi / 4)) = 3 re-masked: the three lowest probabilities
    # among unknowns. The known token keeps its value despite prob 0.
    assert out.tolist() == [[42, 12, MASK, 14, MASK, MASK]]


def _decode_step_inputs(rng, shape, vocab=16):
    logits = rng.normal(size=shape + (vocab,)).astype(np.float32)
    sampled, probs = backend.sample(logits, mask_token_id=3, rng=rng)
    return logits, backend.to_numpy(sampled), backend.to_numpy(probs)


def test_remask_outside_window_matches_plain_decoding():
    # Same seed, so both decoders see the same noise.
    plain = MaskGITDecoder(3, total_steps=4, seed=1)
    # progress is 0.25, 0.5, 0.75, 1.0: never inside (0.9, 0.95)
    remask = MaskGITDecoder(3, total_steps=4, remask_prob=1.0,
                            remask_window=(0.9, 0.95), seed=1)
    rng = np.random.default_rng(0)
    a = b = np.full((2, 16), 3)
    generated = a == 3
    for step in range(4):
        _, sampled, probs = _decode_step_inputs(rng, (2, 16))
        a = plain(a, sampled, probs, step, 16)
        b = remask(b, sampled, probs, step, 16, generated=generated)
        assert np.array_equal(a, b)


def test_remask_sends_committed_tokens_back():
    decoder = MaskGITDecoder(3, total_steps=4, remask_prob=1.0,
                             remask_window=(0.5, 0.5), seed=1)  # step 1 only
    rng = np.random.default_rng(0)
    tokens = np.full((2, 16), 3)
    generated = tokens == 3
    _, sampled, probs = _decode_step_inputs(rng, (2, 16))
    tokens = decoder(tokens, sampled, probs, 0, 16, generated=generated)
    committed = tokens != 3
    assert committed.any()
    _, sampled, probs = _decode_step_inputs(rng, (2, 16))
    tokens = decoder(tokens, sampled, probs, 1, 16, generated=generated)
    assert np.all(tokens[committed] == 3)


def test_remask_never_touches_prompt_tokens():
    decoder = MaskGITDecoder(3, total_steps=6, remask_prob=1.0,
                             remask_window=(0.0, 1.0), seed=1)
    rng = np.random.default_rng(0)
    tokens = np.full((2, 16), 3)
    tokens[:, :4] = 42  # prompt
    generated = tokens == 3
    for step in range(6):
        _, sampled, probs = _decode_step_inputs(rng, (2, 16))
        tokens = decoder(tokens, sampled, probs, step, 12,
                         generated=generated)
        assert np.all(tokens[:, :4] == 42)
    assert not np.any(tokens == 3)  # the last step still reveals everything


def test_remask_with_current_probs_picks_least_confident():
    decoder = MaskGITDecoder(3, total_steps=4, remask_prob=0.5,
                             remask_window=(0.5, 0.5), seed=1)
    rng = np.random.default_rng(0)
    tokens = np.full((8, 32), 3)
    generated = tokens == 3
    _, sampled, probs = _decode_step_inputs(rng, (8, 32))
    tokens = decoder(tokens, sampled, probs, 0, 32, generated=generated)
    committed = tokens != 3
    current = rng.random((8, 32))
    _, sampled, probs = _decode_step_inputs(rng, (8, 32))
    out = decoder(tokens, sampled, probs, 1, 32, generated=generated,
                  current_probs=current)
    remasked = committed & (out == 3)
    kept = committed & (out != 3)
    assert remasked.any()
    for b in range(8):
        if remasked[b].any() and kept[b].any():
            assert current[b][remasked[b]].max() <= current[b][kept[b]].min()


def test_remask_requires_generated():
    decoder = MaskGITDecoder(3, total_steps=4, remask_prob=0.2)
    tokens = np.full((1, 8), 3)
    with pytest.raises(ValueError):
        decoder(tokens, tokens, np.ones((1, 8)), 0, 8)


@pytest.mark.parametrize(
    "kwargs",
    [{"remask_prob": 1.5}, {"remask_prob": -0.1},
     {"remask_window": (0.8, 0.2)}, {"remask_window": (0.5, 1.2)}],
)
def test_remask_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        MaskGITDecoder(3, total_steps=4, **kwargs)


def test_backend_token_probs_matches_softmax_and_zeroes_out_of_vocab():
    logits = np.random.default_rng(0).normal(size=(2, 5, 4)).astype(np.float32)
    tokens = np.array([[0, 1, 2, 3, 4], [3, 2, 1, 0, 4]])  # 4 is outside V=4
    probs = backend.to_numpy(backend.token_probs(logits, tokens))
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    expected = np.take_along_axis(
        softmax, np.minimum(tokens, 3)[..., None], axis=-1
    )[..., 0]
    expected[tokens >= 4] = 0.0
    assert probs.shape == (2, 5)
    assert np.allclose(probs, expected, atol=1e-6)


def test_backend_sample_accepts_mask_id_outside_vocab():
    logits = np.zeros((1, 4, 8), dtype=np.float32)
    sampled, _ = backend.sample(logits, mask_token_id=8, rng=0)  # id == V
    assert backend.to_numpy(sampled).shape == (1, 4)


def test_backend_sample_greedy_returns_untempered_probs():
    logits = np.array([[[0.0, 5.0, 1.0], [9.0, 0.0, 0.0]]], dtype=np.float32)
    sampled, probs = backend.sample(logits, temperature=0.0)
    assert backend.to_numpy(sampled).tolist() == [[1, 0]]
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    expected = [[softmax[0, 0, 1], softmax[0, 1, 0]]]
    assert np.allclose(backend.to_numpy(probs), expected, atol=1e-6)


def test_backend_sample_never_draws_mask_id():
    logits = np.zeros((4, 32, 8), dtype=np.float32)
    logits[..., 2] = 10.0  # the mask id is by far the most likely
    sampled, _ = backend.sample(logits, mask_token_id=2, rng=0)
    assert not np.any(backend.to_numpy(sampled) == 2)


def test_backend_sample_follows_distribution_and_seed():
    row = np.log(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    logits = np.broadcast_to(row, (1, 20000, 3)).copy()
    a, _ = backend.sample(logits, rng=0)
    b, _ = backend.sample(logits, rng=0)
    a = backend.to_numpy(a)
    assert a.shape == (1, 20000)
    assert np.array_equal(a, backend.to_numpy(b))
    freq = np.bincount(a.ravel(), minlength=3) / a.size
    assert np.allclose(freq, [0.7, 0.2, 0.1], atol=0.02)


def test_backend_sample_advances_a_shared_generator():
    logits = np.zeros((1, 64, 8), dtype=np.float32)
    rng = np.random.default_rng(0)
    a, _ = backend.sample(logits, rng=rng)
    b, _ = backend.sample(logits, rng=rng)
    assert not np.array_equal(backend.to_numpy(a), backend.to_numpy(b))
    c, _ = backend.sample(logits, rng=np.random.default_rng(0))
    assert np.array_equal(backend.to_numpy(a), backend.to_numpy(c))


def test_maskgit_decoder_draws_fresh_noise_each_step_and_reset_replays():
    # Equal probs leave the noise alone to pick what stays masked.
    decoder = MaskGITDecoder(MASK, total_steps=4, seed=0)
    tokens = np.full((2, 32), MASK)
    sampled = np.full((2, 32), 25)
    probs = np.full((2, 32), 0.5)
    a = decoder(tokens, sampled, probs, 0, 32)
    b = decoder(tokens, sampled, probs, 0, 32)
    assert not np.array_equal(a, b)
    decoder.reset(0)
    assert np.array_equal(decoder(tokens, sampled, probs, 0, 32), a)
    assert np.array_equal(
        MaskGITDecoder(MASK, total_steps=4, seed=0)(tokens, sampled, probs, 0, 32),
        a,
    )


def test_maskgit_decoder_reset_without_seed_is_fresh_but_replayable():
    decoder = MaskGITDecoder(MASK, total_steps=4, seed=0)
    tokens = np.full((2, 32), MASK)
    sampled = np.full((2, 32), 25)
    probs = np.full((2, 32), 0.5)
    seeded = decoder(tokens, sampled, probs, 0, 32)
    decoder.reset()
    assert decoder.seed != 0
    fresh = decoder(tokens, sampled, probs, 0, 32)
    assert not np.array_equal(fresh, seeded)
    decoder.reset(decoder.seed)
    assert np.array_equal(decoder(tokens, sampled, probs, 0, 32), fresh)


def test_maskgit_decoder_rejects_negative_seed():
    with pytest.raises(ValueError):
        MaskGITDecoder(MASK, total_steps=4, seed=-1)


def test_backend_sample_rejects_negative_temperature():
    with pytest.raises(ValueError):
        backend.sample(np.zeros((1, 2, 3), dtype=np.float32), temperature=-1.0)


def test_backend_sample_drives_decoder_to_completion():
    steps, mask = 6, 3
    decoder = MaskGITDecoder(mask, total_steps=steps, seed=1)
    rng = np.random.default_rng(0)
    tokens = np.full((2, 32), mask)
    for step in range(steps):
        logits = rng.normal(size=(2, 32, 16)).astype(np.float32)
        sampled, probs = backend.sample(logits, mask_token_id=mask,
                                        rng=decoder.rng)
        tokens = decoder(tokens, sampled, probs, step, 32)
    assert not np.any(tokens == mask)


@pytest.mark.skipif(
    ACTIVE_BACKEND != "torch",
    reason=f"needs the torch backend; TAKUMI_MASKING_BACKEND selected "
    f"{ACTIVE_BACKEND!r}",
)
def test_maskgit_decoder_torch_roundtrip():
    torch = pytest.importorskip("torch")
    decoder = MaskGITDecoder(MASK, total_steps=4)
    tokens = np.full((2, 16), MASK)
    sampled = np.random.default_rng(0).integers(10, 60, size=(2, 16))
    probs = np.random.default_rng(1).random((2, 16))
    decoder.reset(3)
    n = decoder(tokens, sampled, probs, 0, 16)
    decoder.reset(3)
    t = decoder(
        torch.as_tensor(tokens), torch.as_tensor(sampled),
        torch.as_tensor(probs), 0, 16,
    )
    assert isinstance(t, torch.Tensor)
    assert np.array_equal(t.numpy(), n)


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


# --------------------------------------------------------------------------
# resuming the curriculum
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "make",
    [
        lambda: LinearScheduler([1.0, 0.0, 0.5], [0.0, 1.0, 0.5], total_steps=50),
        lambda: EMAScheduler([1.0, 0.0, 0.5], [0.0, 1.0, 0.5], beta=0.9),
        lambda: StagedLinearScheduler(num_stages=3, total_steps=30),
    ],
    ids=["linear", "ema", "staged"],
)
def test_scheduler_resumes_from_state_dict(make):
    original = make()
    for _ in range(17):
        original.step()
    # through JSON, as a checkpoint file would store it
    state = json.loads(json.dumps(original.state_dict()))
    resumed = make()
    assert resumed.sample() != original.sample()
    resumed.load_state_dict(state)
    assert resumed.sample() == original.sample()
    for _ in range(5):
        original.step()
        resumed.step()
    assert resumed.sample() == original.sample()


def test_scheduler_rejects_out_of_range_step():
    scheduler = LinearScheduler([1.0], [0.0], total_steps=10)
    for step in (-1, 11):
        with pytest.raises(ValueError):
            scheduler.load_state_dict({"step": step})


def test_sampler_resumes_its_scheduler():
    def make():
        scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=10)
        return MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1],
                                     scheduler)

    def draws(sampler):
        return [sampler.sample(rng=i) for i in range(50)]

    original = make()
    for _ in range(6):
        original.step()
    resumed = make()
    assert draws(resumed) != draws(original)
    resumed.load_state_dict(original.state_dict())
    assert draws(resumed) == draws(original)


def test_sampler_state_must_match_scheduler_presence():
    plain = MLMProbabilitySampler([0.1], [0.8], [0.1])
    assert plain.state_dict() == {}
    plain.load_state_dict({})
    with pytest.raises(ValueError):
        plain.load_state_dict({"scheduler": {"step": 1}})
    scheduled = MLMProbabilitySampler(
        [0.1], [0.8], [0.1], LinearScheduler([1.0], [1.0], total_steps=2)
    )
    with pytest.raises(ValueError):
        scheduled.load_state_dict({})


# --------------------------------------------------------------------------
# ELBO loss weights
# --------------------------------------------------------------------------

BUILTIN_SCHEDULES = [LinearSchedule(), CosineSchedule(), SqrtSchedule()]
SCHEDULE_IDS = ["linear", "cosine", "sqrt"]


@pytest.mark.parametrize("schedule", BUILTIN_SCHEDULES, ids=SCHEDULE_IDS)
def test_closed_form_slope_matches_the_derivative(schedule):
    # MaskSchedule.compute_slope differentiates rate numerically; each
    # built-in closed form must agree with it.
    for step in (0.5, 1, 10, 37.5, 50, 90, 99.9):
        numeric = MaskSchedule.compute_slope(schedule, step, 100)
        assert schedule.compute_slope(step, 100) == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize(
    "schedule, weight",
    [
        (LinearSchedule(), lambda t: 1 / t),
        (CosineSchedule(), lambda t: math.pi / 2 * math.tan((1 - t) * math.pi / 2)),
        (SqrtSchedule(), lambda t: 1 / (2 * t)),
    ],
    ids=SCHEDULE_IDS,
)
def test_continuous_elbo_weight_closed_forms(schedule, weight):
    for t in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert schedule.continuous_elbo_weight(t, 1.0) == pytest.approx(weight(t))
    assert schedule.continuous_elbo_weight(0, 1.0) == 0.0  # clean data


def test_custom_schedule_gets_a_numerical_slope_and_weight():
    cubic = resolve_schedule(lambda step, total: (step / total) ** 3)
    for t in (0.1, 0.5, 0.9):
        assert cubic.compute_slope(t, 1.0) == pytest.approx(3 * t**2, rel=1e-6)
        assert cubic.continuous_elbo_weight(t, 1.0) == pytest.approx(3 / t, rel=1e-6)


@pytest.mark.parametrize("schedule", BUILTIN_SCHEDULES, ids=SCHEDULE_IDS)
def test_discrete_weights_count_each_token_once(schedule):
    # Steps uniform on 1..T: E[w_i * L * p(i)] = L * sum(p(i) - p(i-1)) = L,
    # so every token's prediction is charged once, as in an AR model.
    T, length = 100, 64
    expected_count = sum(
        schedule.discrete_elbo_weight(i, T) * length * schedule.rate(i, T)
        for i in range(1, T + 1)
    ) / T
    assert expected_count == pytest.approx(length, rel=1e-9)


def test_discrete_elbo_weight_values_and_errors():
    linear = LinearSchedule()
    assert linear.discrete_elbo_weight(0, 100) == 0.0  # clean data
    for i in (1, 7, 50, 100):
        assert linear.discrete_elbo_weight(i, 100) == pytest.approx(100 / i)
    shrinking = resolve_schedule(lambda step, total: 0.5 if step < 2 else 0.25)
    with pytest.raises(ValueError):
        shrinking.discrete_elbo_weight(2, 10)


def test_schedule_rate_weights_follow_the_time_argument():
    ctx = CallContext(rng=np.random.default_rng(0), batch_size=3,
                      timestep=np.array([1, 50, 100]))
    discrete = ScheduleRate(100, "cosine", time="discrete")
    draw = discrete(ctx)
    assert draw.timesteps.tolist() == [1, 50, 100]
    expected = [discrete.schedule.discrete_elbo_weight(t, 100) for t in (1, 50, 100)]
    assert draw.weights.tolist() == pytest.approx(expected)

    # the same whole-number steps, declared continuous, get p'(t) / p(t)
    continuous = ScheduleRate(100, "cosine", time="continuous")
    expected = [continuous.schedule.continuous_elbo_weight(t, 100) for t in (1, 50, 100)]
    assert continuous(ctx).weights.tolist() == pytest.approx(expected)

    ctx.timestep = 50  # a scalar applies to every row
    expected = [discrete.schedule.discrete_elbo_weight(50, 100)] * 3
    assert discrete(ctx).weights.tolist() == pytest.approx(expected)


def test_discrete_time_rejects_fractional_timesteps():
    ctx = CallContext(rng=np.random.default_rng(0), batch_size=2,
                      timestep=np.array([10, 50.5]))
    with pytest.raises(ValueError):
        ScheduleRate(100, time="discrete")(ctx)
    ScheduleRate(100, time="continuous")(ctx)  # fine when declared continuous


def test_time_argument_is_required_and_checked():
    with pytest.raises(TypeError):
        ScheduleRate(100)
    with pytest.raises(ValueError):
        ScheduleRate(100, time="integer")
    with pytest.raises(TypeError):
        DiffusionMasker(100, MASK, SPECIALS)


def test_uniform_timestep_rate_reports_its_draw():
    policy = UniformTimestepRate("cosine")
    draw = policy(CallContext(rng=np.random.default_rng(0), batch_size=5))
    for t, rate, weight in zip(draw.timesteps, draw.rate, draw.weights):
        assert rate == pytest.approx(policy.schedule.rate(t, 1.0))
        assert weight == pytest.approx(policy.schedule.continuous_elbo_weight(t, 1.0))


def test_masking_result_carries_timesteps_and_weights():
    seq = _batch(rows=3)
    plain = FixedMasker(0.5, 0.8, 0.1, MASK, VOCAB, SPECIALS)(seq, seed=0)
    assert plain.timesteps is None
    assert plain.weights.shape == seq.shape and np.all(plain.weights == 1.0)

    steps = np.array([10, 50, 90])
    diffusion = DiffusionMasker(100, MASK, SPECIALS, time="discrete")(
        seq, seed=0, timestep=steps)
    assert diffusion.timesteps.tolist() == [10, 50, 90]
    assert diffusion.weights.dtype == np.float32
    assert diffusion.weights.shape == seq.shape  # one per token, like labels
    expected = [CosineSchedule().discrete_elbo_weight(t, 100) for t in steps]
    assert diffusion.weights[:, 0].tolist() == pytest.approx(expected, rel=1e-6)
    assert np.all(diffusion.weights == diffusion.weights[:, :1])  # row-constant

    maskgit = MaskGITMasker(MASK, SPECIALS)(seq, seed=0)
    assert maskgit.timesteps.shape == (3,) and maskgit.weights.shape == seq.shape


def test_corruptor_rejects_weights_of_the_wrong_shape():
    class BadWeights(RatePolicy):
        def draw(self, ctx):
            return RateDraw(0.5, None, np.ones(ctx.batch_size + 1))

    corr = Corruptor(BadWeights(), Bernoulli(), MaskOnly(MASK), SPECIALS)
    with pytest.raises(ValueError):
        corr(_batch(rows=2), seed=0)


@pytest.mark.parametrize(
    "make", [LinearSchedule, CosineSchedule, SqrtSchedule], ids=SCHEDULE_IDS
)
def test_floor_reaches_masking_and_both_weights(make):
    eps = 0.1
    raw, floored = make(), make(eps=eps)

    def p(step, total):
        return eps + (1 - eps) * raw.rate(step, total)

    for t in (0.01, 0.3, 0.9):
        assert floored(t, 1.0) == pytest.approx(p(t, 1.0))
        expected = (1 - eps) * raw.compute_slope(t, 1.0) / p(t, 1.0)
        assert floored.continuous_elbo_weight(t, 1.0) == pytest.approx(expected)
    for i in (1, 50, 100):
        expected = 100 * (p(i, 100) - p(i - 1, 100)) / p(i, 100)
        assert floored.discrete_elbo_weight(i, 100) == pytest.approx(expected)

    # the rate policies mask with the floored curve, never below eps
    draw = UniformTimestepRate(floored)(
        CallContext(rng=np.random.default_rng(0), batch_size=1000)
    )
    assert draw.rate.min() >= eps
    assert np.allclose(draw.rate, [p(t, 1.0) for t in draw.timesteps])
    ctx = CallContext(rng=np.random.default_rng(0), batch_size=2,
                      timestep=np.array([0, 50]))
    rate = ScheduleRate(100, floored, time="discrete")(ctx).rate
    assert rate.tolist() == pytest.approx([eps, p(50, 100)])

    # the price of the floor: p(0) = eps, so each token is charged (1 - eps)
    # times on average instead of once (the t = 0 boundary term is dropped)
    T, length = 100, 64
    expected_count = sum(
        floored.discrete_elbo_weight(i, T) * length * floored(i, T)
        for i in range(1, T + 1)
    ) / T
    assert expected_count == pytest.approx((1 - eps) * length)


def test_eps_is_validated_and_defaults_to_no_floor():
    for eps in (-0.1, 1.0):
        with pytest.raises(ValueError):
            LinearSchedule(eps=eps)

    class OwnInit(MaskSchedule):  # skips super().__init__()
        def __init__(self, power):
            self.power = power

        def rate(self, step, total):
            return (step / total) ** self.power

    assert OwnInit(2)(0.5, 1.0) == 0.25
    assert resolve_schedule(lambda step, total: step / total)(0.5, 1.0) == 0.5


# --------------------------------------------------------------------------
# elbo_loss
# --------------------------------------------------------------------------

def _diffusion_result(rows=8, length=32, padding=0):
    masker = DiffusionMasker(100, MASK, SPECIALS, time="discrete")
    tokens = _batch(rows, length)
    tokens[:, length - padding:] = 0  # special id: never masked, like padding
    steps = np.arange(1, rows + 1) * (100 // rows)  # spread over 1..T
    return masker(tokens, seed=0, timestep=steps)


def _token_loss(shape):
    return np.random.default_rng(1).random(shape)


def _value(loss):
    return float(backend.to_numpy(loss))


def _weighted_sum(ce, result):
    selected = backend.to_numpy(result.labels) != -100
    weights = backend.to_numpy(result.weights)
    return (weights * np.where(selected, ce, 0.0)).sum()


# elbo_loss takes backend tensors, as a training loop has them.
_t = backend.to_tensor


def _elbo(ce, result, attention_mask=None):
    # The (B*L,) layout, from (B, L) arrays.
    mask = None if attention_mask is None else _t(attention_mask)
    return elbo_loss(_t(ce.reshape(-1)), _t(result.labels.reshape(-1)),
                     _t(result.weights.reshape(-1)), attention_mask=mask)


def test_elbo_loss_is_the_nelbo_per_token():
    result = _diffusion_result()
    ce = _token_loss(result.labels.shape)
    expected = _weighted_sum(ce, result) / ce.size
    assert _value(_elbo(ce, result)) == pytest.approx(expected, rel=1e-6)


def test_elbo_loss_attention_mask_leaves_padding_out_of_the_count():
    result = _diffusion_result(padding=8)
    ce = _token_loss(result.labels.shape)
    attention_mask = np.ones(ce.shape, dtype=bool)
    attention_mask[:, -8:] = False
    loss = _elbo(ce, result, attention_mask=attention_mask.reshape(-1))
    expected = _weighted_sum(ce, result) / attention_mask.sum()
    assert _value(loss) == pytest.approx(expected, rel=1e-6)


def test_elbo_loss_takes_every_input_flat_or_as_rows():
    result = _diffusion_result(padding=8)
    ce = _token_loss(result.labels.shape)  # (B, L)
    attention_mask = np.ones(ce.shape, dtype=bool)
    attention_mask[:, -8:] = False
    as_rows = elbo_loss(_t(ce), _t(result.labels), _t(result.weights),
                        attention_mask=_t(attention_mask))
    flat = _elbo(ce, result, attention_mask=attention_mask.reshape(-1))
    assert _value(as_rows) == _value(flat)


def test_elbo_loss_ignores_unlabelled_positions():
    result = _diffusion_result()
    ce = _token_loss(result.labels.shape)
    garbage = np.where(backend.to_numpy(result.labels) != -100, ce, np.nan)
    assert _value(_elbo(garbage, result)) == _value(_elbo(ce, result))


def test_elbo_loss_is_zero_when_nothing_is_masked():
    loss = elbo_loss(_t(np.ones(32)), _t(np.full(32, -100)), _t(np.ones(32)))
    assert _value(loss) == 0.0


def test_elbo_loss_is_zero_when_the_mask_leaves_no_tokens():
    # The labelled position is masked out too, so nothing is summed.
    loss = elbo_loss(_t(np.ones(4)), _t(np.array([5, -100, -100, -100])),
                     _t(np.ones(4)), attention_mask=_t(np.zeros(4)))
    assert _value(loss) == 0.0


def test_elbo_loss_takes_no_loss_outside_the_attention_mask():
    labels = _t(np.array([5, 6, -100, 7]))  # B = 1; the last is padding
    mask = _t(np.array([1, 1, 1, 0]))
    loss = elbo_loss(_t(np.full(4, 2.0)), labels, _t(np.ones(4)),
                     attention_mask=mask)
    assert _value(loss) == pytest.approx(4.0 / 3)  # (2 + 2) / 3 real tokens


@pytest.mark.parametrize("shape", [(), (2,)])
def test_elbo_loss_rejects_a_loss_that_is_not_per_position(shape):
    # () is a reduction="mean" scalar, (2,) one loss per row of a 2 x 3 batch.
    labels = _t(np.array([5, -100, 6, 7, -100, -100]))
    with pytest.raises(ValueError, match="one loss per position"):
        elbo_loss(_t(np.full(shape, 2.0)), labels, _t(np.ones(6)))


@pytest.mark.skipif(
    ACTIVE_BACKEND != "torch",
    reason=f"needs the torch backend; TAKUMI_MASKING_BACKEND selected "
    f"{ACTIVE_BACKEND!r}",
)
def test_elbo_loss_torch_sums_in_float32_and_backpropagates():
    torch = pytest.importorskip("torch")
    result = _diffusion_result()
    ce = torch.as_tensor(_token_loss(256), dtype=torch.bfloat16)
    ce.requires_grad_(True)
    loss = elbo_loss(ce, _t(result.labels.reshape(-1)),
                     _t(result.weights.reshape(-1)))
    assert loss.dtype == torch.float32 and loss.ndim == 0
    loss.backward()
    selected = torch.as_tensor(result.labels.reshape(-1) != -100)
    assert torch.all(ce.grad[~selected] == 0)
    assert torch.all(ce.grad[selected] > 0)


@pytest.mark.skipif(
    ACTIVE_BACKEND != "jax",
    reason=f"needs the jax backend; TAKUMI_MASKING_BACKEND selected "
    f"{ACTIVE_BACKEND!r}",
)
def test_elbo_loss_jax_traces_under_jit_and_grad():
    jax = pytest.importorskip("jax")
    result = _diffusion_result()
    labels = result.labels.reshape(-1)
    weights = result.weights.reshape(-1)

    def loss_fn(c):
        return elbo_loss(c, _t(labels), _t(weights))

    ce = jax.numpy.asarray(_token_loss(256), dtype=jax.numpy.float32)
    assert np.allclose(np.asarray(loss_fn(ce)), np.asarray(jax.jit(loss_fn)(ce)))
    grad = np.asarray(jax.grad(loss_fn)(ce))
    assert np.all(grad[labels == -100] == 0)
    assert np.all(grad[labels != -100] > 0)
