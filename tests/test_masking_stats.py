"""Statistical tests: each random component matches its math.

Every test draws many samples and compares what it sees with a value derived
from the component's definition, never from the code's own output. Checks
allow ``Z = 5`` standard errors: seeds are fixed so the suite is
deterministic, and with a random seed a correct implementation would fail a
given check with probability about 6e-7, so a failure points at a real
deviation rather than bad luck.

Rows of one batch are independent samples only for per-row components
(Bernoulli selection, exact counts, per-row rates). Per-call policies such as
`BetaRate` share one rate across a batch, so those tests make many calls.
"""

from __future__ import annotations

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
from takumi.masking import BetaRate
from takumi.masking import Bernoulli
from takumi.masking import CallContext
from takumi.masking import Corruptor
from takumi.masking import DiffusionMasker
from takumi.masking import ExactCount
from takumi.masking import FocusBiased
from takumi.masking import LinearScheduler
from takumi.masking import MaskGITDecoder
from takumi.masking import MaskGITMasker
from takumi.masking import Mixed
from takumi.masking import MLMProbabilitySampler
from takumi.masking import SampledRate
from takumi.masking import Spans
from takumi.masking import UniformTimestepRate

VOCAB = list(range(10, 60))
SPECIALS = [0]
MASK = 99

Z = 5.0
# Kolmogorov-Smirnov critical value at alpha = 1e-6: sqrt(ln(2 / alpha) / 2).
KS = 2.69


def assert_proportion(hits, n, p):
    """``hits / n`` is within ``Z`` binomial standard errors of ``p``."""
    se = math.sqrt(p * (1.0 - p) / n)
    observed = hits / n
    assert abs(observed - p) <= Z * se + 1e-12, (
        f"observed {observed:.5f}, expected {p:.5f} ± {Z * se:.5f}"
    )


def assert_mean(samples, mean, variance):
    """Sample mean is within ``Z`` standard errors of ``mean``."""
    samples = np.asarray(samples, dtype=np.float64)
    se = math.sqrt(variance / samples.size)
    assert abs(samples.mean() - mean) <= Z * se, (
        f"mean {samples.mean():.5f}, expected {mean:.5f} ± {Z * se:.5f}"
    )


def assert_variance_ratio(observed, expected, n):
    """Sample variance matches; its relative standard error is about
    ``sqrt(2 / n)`` for data no heavier-tailed than a normal."""
    tolerance = Z * math.sqrt(2.0 / n)
    assert abs(observed / expected - 1.0) <= tolerance, (
        f"variance {observed:.5f}, expected {expected:.5f} "
        f"(±{tolerance:.1%})"
    )


def assert_cdf(samples, cdf):
    """Kolmogorov-Smirnov: the empirical CDF stays within ``KS / sqrt(n)``."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = x.size
    expected = cdf(x)
    distance = max(
        np.max(np.arange(1, n + 1) / n - expected),
        np.max(expected - np.arange(n) / n),
    )
    assert distance <= KS / math.sqrt(n), (
        f"KS distance {distance:.5f} > {KS / math.sqrt(n):.5f}"
    )


# --------------------------------------------------------------------------
# rate policies
# --------------------------------------------------------------------------

def test_beta_rate_mean_and_variance():
    mean, concentration = 0.3, 10.0
    policy = BetaRate(mean, concentration)
    rng = np.random.default_rng(0)
    draws = np.array([policy(CallContext(rng=rng)).rate for _ in range(20000)])
    variance = mean * (1.0 - mean) / (concentration + 1.0)
    assert_mean(draws, mean, variance)
    assert_variance_ratio(draws.var(), variance, draws.size)


@pytest.mark.parametrize(
    "name, cdf",
    [
        ("linear", lambda x: x),
        # rate = sqrt(t), so P(rate <= x) = P(t <= x^2)
        ("sqrt", lambda x: x**2),
        # rate = cos((1 - t) pi / 2) = sin(t pi / 2), so
        # P(rate <= x) = P(t <= 2 asin(x) / pi)
        ("cosine", lambda x: 2.0 / np.pi * np.arcsin(x)),
    ],
    ids=["linear", "sqrt", "cosine"],
)
def test_uniform_timestep_rate_follows_its_schedule(name, cdf):
    ctx = CallContext(rng=np.random.default_rng(0), batch_size=50000)
    assert_cdf(UniformTimestepRate(name)(ctx).rate, cdf)


def test_diffusion_masker_masks_at_the_schedule_rate():
    total, rows, length = 1000, 200, 500
    steps = np.array([0, 100, 250, 500, 750, 1000])
    timesteps = np.repeat(steps, rows)
    masker = DiffusionMasker(total, MASK, SPECIALS, time="discrete")
    tokens = np.full((timesteps.size, length), 25)
    masked = masker(tokens, seed=0, timestep=timesteps).masked
    for i, t in enumerate(steps):
        group = masked[i * rows:(i + 1) * rows]
        p = math.cos((1.0 - t / total) * math.pi / 2.0)
        assert_proportion(int(group.sum()), group.size, p)


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------

def test_bernoulli_counts_are_binomial_at_each_row_rate():
    rows_per_rate, length = 1000, 400
    rates = np.repeat([0.1, 0.5, 0.9], rows_per_rate)
    eligible = np.ones((rates.size, length), dtype=bool)
    eligible[:, ::4] = False  # 300 eligible per row
    selected = Bernoulli()(np.random.default_rng(0), rates, eligible, None)
    assert not np.any(selected & ~eligible)
    counts = selected.sum(axis=1)
    n = 300
    for p in (0.1, 0.5, 0.9):
        group = counts[rates == p]
        assert_proportion(int(group.sum()), n * group.size, p)
        # independent positions give row counts the binomial variance
        assert_variance_ratio(group.var(), n * p * (1.0 - p), group.size)


def test_exact_count_draws_subsets_uniformly():
    rows, length = 20000, 12
    eligible = np.ones((rows, length), dtype=bool)
    eligible[:, [0, 11]] = False  # 10 eligible
    selected = ExactCount()(np.random.default_rng(0), 0.25, eligible, None)
    assert np.all(selected.sum(axis=1) == 3)  # ceil(0.25 * 10)
    inner = selected[:, 1:11]
    for j in range(10):
        assert_proportion(int(inner[:, j].sum()), rows, 3 / 10)
    # uniform subsets, not only uniform marginals: P(i and j) = 3/10 * 2/9
    for i, j in ((0, 1), (4, 9)):
        both = int((inner[:, i] & inner[:, j]).sum())
        assert_proportion(both, rows, 3 / 10 * 2 / 9)


def test_force_include_and_multiplicative_focus_rates():
    rows, length = 2000, 200
    eligible = np.ones((rows, length), dtype=bool)
    focus = np.zeros((rows, length), dtype=bool)
    focus[:, :50] = True
    rng = np.random.default_rng(0)

    forced = FocusBiased("force_include")(rng, 0.2, eligible, focus)
    assert forced[:, :50].all()
    assert_proportion(int(forced[:, 50:].sum()), rows * 150, 0.2)

    boosted = FocusBiased("multiplicative", 3.0)(rng, 0.2, eligible, focus)
    assert_proportion(int(boosted[:, :50].sum()), rows * 50, 0.6)
    assert_proportion(int(boosted[:, 50:].sum()), rows * 150, 0.2)

    clipped = FocusBiased("multiplicative", 3.0)(rng, 0.5, eligible, focus)
    assert clipped[:, :50].all()  # min(1, 0.5 * 3)


def test_weighted_focus_pick_is_proportional_to_weight():
    rows, length = 8000, 10
    eligible = np.ones((rows, length), dtype=bool)
    focus = np.zeros((rows, length), dtype=bool)
    focus[:, :2] = True
    selector = FocusBiased("weighted", strength=4.0)
    selected = selector(np.random.default_rng(0), 0.1, eligible, focus)
    assert np.all(selected.sum(axis=1) == 1)  # round(0.1 * 10)
    # one draw over weights (4, 4, 1, ..., 1): 4/16 per focus position
    for j in range(2):
        assert_proportion(int(selected[:, j].sum()), rows, 4 / 16)
    for j in range(2, length):
        assert_proportion(int(selected[:, j].sum()), rows, 1 / 16)


def test_span_lengths_follow_a_capped_geometric():
    p, cap = 0.3, 6
    rows, length = 10000, 1000
    spans = Spans(length_p=p, max_length=cap, span_fraction=1.0)
    eligible = np.ones((rows, length), dtype=bool)
    # a budget of one position per row grows exactly one span
    selected = spans(np.random.default_rng(0), 1 / length, eligible, None)
    runs = selected[:, 0] + (selected[:, 1:] & ~selected[:, :-1]).sum(axis=1)
    assert np.all(runs == 1)
    lengths = selected.sum(axis=1)
    for k in range(1, cap):
        hits = int((lengths == k).sum())
        assert_proportion(hits, rows, (1 - p) ** (k - 1) * p)
    assert_proportion(int((lengths == cap).sum()), rows, (1 - p) ** (cap - 1))


# --------------------------------------------------------------------------
# corruption schemes
# --------------------------------------------------------------------------

def test_mixed_split_and_uniform_substitutes():
    tokens = np.full((400, 1000), 25)
    selected = np.ones(tokens.shape, dtype=bool)
    scheme = Mixed(0.8, 0.1, MASK, VOCAB)
    split = scheme(np.random.default_rng(0), tokens, selected)
    n = selected.size
    assert_proportion(int(split.masked.sum()), n, 0.8)
    assert_proportion(int(split.mutated.sum()), n, 0.1)
    assert_proportion(int(split.kept.sum()), n, 0.1)
    substitutes = tokens[split.mutated]
    counts = np.bincount(substitutes, minlength=max(VOCAB) + 1)[VOCAB]
    for count in counts:
        assert_proportion(int(count), substitutes.size, 1 / len(VOCAB))


# --------------------------------------------------------------------------
# MaskGIT training and decoding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("schedule", ["linear", "cosine", "sqrt"])
def test_weighted_mask_count_averages_to_the_sequence_length(schedule):
    # With steps uniform on 1..T and independent masking, the ELBO weights
    # give E[weight * masked count] = L: each token is charged once. This
    # checks scale and wiring end to end; it can't check the weight's shape
    # (a naive 1/p passes too), which the closed-form tests cover. Integer
    # steps cap the weights at T, so the variance is finite here.
    total, rows, length = 100, 20000, 64
    steps = np.random.default_rng(1).integers(1, total + 1, rows)
    masker = DiffusionMasker(total, MASK, SPECIALS, schedule=schedule,
                             time="discrete")
    result = masker(np.full((rows, length), 25), seed=0, timestep=steps)
    samples = (result.weights.astype(np.float64) * result.masked).sum(axis=1)
    assert_mean(samples, length, samples.var())


def test_maskgit_masker_count_distribution():
    rows, n = 50000, 20
    masked = MaskGITMasker(MASK, SPECIALS)(np.full((rows, n), 25), seed=0).masked
    counts = masked.sum(axis=1)
    assert counts.min() >= 1
    # count = ceil(n * sin(t pi / 2)) with t ~ U(0, 1], so
    # P(count <= k) = P(sin(t pi / 2) <= k / n) = 2 asin(k / n) / pi
    cdf = 2.0 / np.pi * np.arcsin(np.arange(n + 1) / n)
    for k in range(1, n + 1):
        assert_proportion(int((counts == k).sum()), rows, cdf[k] - cdf[k - 1])


def test_remask_count_is_binomial_in_committed_tokens():
    rows, length, p = 4000, 32, 0.3
    decoder = MaskGITDecoder(MASK, total_steps=4, remask_prob=p,
                             remask_window=(0.5, 0.5), seed=1)  # step 1 only
    rng = np.random.default_rng(0)
    tokens = np.full((rows, length), MASK)
    generated = tokens == MASK

    def step(tokens, index):
        sampled = rng.integers(10, 60, (rows, length))
        probs = rng.random((rows, length))
        return decoder(tokens, sampled, probs, index, length,
                       generated=generated)

    tokens = step(tokens, 0)
    committed = tokens != MASK
    remasked = committed & (step(tokens, 1) == MASK)
    assert_proportion(int(remasked.sum()), int(committed.sum()), p)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_backend_sample_follows_tempered_softmax(temperature):
    row = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    logits = np.broadcast_to(row, (1, 50000, 3)).copy()
    sampled, _ = backend.sample(logits, temperature=temperature, rng=0)
    sampled = backend.to_numpy(sampled).ravel()
    expected = np.exp(row / temperature) / np.exp(row / temperature).sum()
    for token, p in enumerate(expected):
        assert_proportion(int((sampled == token).sum()), sampled.size, float(p))


def test_backend_sample_renormalizes_without_the_mask_id():
    logits = np.broadcast_to(
        np.array([1.0, 1.0, 5.0], dtype=np.float32), (1, 50000, 3)
    ).copy()
    sampled, _ = backend.sample(logits, mask_token_id=2, rng=0)
    sampled = backend.to_numpy(sampled).ravel()
    assert not np.any(sampled == 2)
    assert_proportion(int((sampled == 0).sum()), sampled.size, 0.5)


# --------------------------------------------------------------------------
# curriculum sampling
# --------------------------------------------------------------------------

def test_mlm_sampler_draws_in_proportion_to_weights():
    n = 6000
    weights = [1.0, 3.0, 0.0]
    scheduler = LinearScheduler(weights, weights, total_steps=1)
    rates = [0.1, 0.2, 0.3]
    sampler = MLMProbabilitySampler(rates, [0.8] * 3, [0.1] * 3, scheduler)
    picks = np.array([sampler.sample(rng=i)[0] for i in range(n)])
    assert_proportion(int((picks == 0.1).sum()), n, 0.25)
    assert_proportion(int((picks == 0.2).sum()), n, 0.75)
    assert not np.any(picks == 0.3)

    uniform = MLMProbabilitySampler(rates, [0.8] * 3, [0.1] * 3)
    picks = np.array([uniform.sample(rng=i)[0] for i in range(n)])
    for rate in rates:
        assert_proportion(int((picks == rate).sum()), n, 1 / 3)


def test_sampled_rate_pick_is_independent_of_selection():
    # The pick and the selection share the call's generator. Re-seeding a
    # second generator from the call's seed would reuse the pick's uniform
    # for position 0, biasing it (P ~ 0.15 instead of 0.325 here).
    n = 6000
    weights = [0.5, 0.5]
    scheduler = LinearScheduler(weights, weights, total_steps=1)
    sampler = MLMProbabilitySampler([0.15, 0.5], [0.8] * 2, [0.1] * 2,
                                    scheduler)
    corruptor = Corruptor(SampledRate(sampler), Bernoulli(),
                          Mixed(1.0, 0.0, MASK, VOCAB), SPECIALS)
    hits = sum(
        bool(corruptor(np.full((1, 4), 25), seed=i).masked[0, 0])
        for i in range(n)
    )
    assert_proportion(hits, n, 0.5 * 0.15 + 0.5 * 0.5)
