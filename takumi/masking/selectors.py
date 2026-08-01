from __future__ import annotations

import typing

import numpy as np

from takumi.masking import backend

FOCUS_STRATEGIES = ("force_include", "multiplicative", "weighted")


def coerce_focus(
    focus: typing.Optional[typing.Any],
    expected_shape: tuple,
) -> typing.Optional[np.ndarray]:
    """Normalize a caller's focus argument to a NumPy bool array.

    Accepts NumPy arrays, CPU torch/JAX tensors, or nested sequences.
    Returns None when ``focus`` is None. Raises if the shape disagrees with
    ``expected_shape``.
    """
    if focus is None:
        return None
    if not isinstance(focus, np.ndarray):
        focus = backend.to_numpy(focus)
    focus = np.asarray(focus, dtype=bool)
    if focus.shape != expected_shape:
        raise ValueError(
            f"focus shape {focus.shape} does not match sequences shape "
            f"{expected_shape}"
        )
    return focus


def focus_from_reference(
    sequences: typing.Any,
    reference: typing.Any,
) -> typing.Any:
    """Build a focus mask that is True wherever ``sequences != reference``.

    The reference is a per-position template — a base sentence, an original
    snippet, anything domain-appropriate. Shapes follow NumPy broadcasting,
    with the constraint that the reference is never "larger" than the
    sequences: the trailing (length) dimension must match, and a 2-D
    reference batch dimension must be 1 or equal the sequences batch.

    Returns a torch/JAX tensor when ``sequences`` is one, else a NumPy array.
    """
    seq_is_np = isinstance(sequences, np.ndarray)
    seq = sequences if seq_is_np else backend.to_numpy(sequences)
    seq = np.asarray(seq)
    ref = (
        reference
        if isinstance(reference, np.ndarray)
        else backend.to_numpy(reference)
    )
    ref = np.asarray(ref)

    if seq.ndim not in (1, 2):
        raise ValueError(f"sequences must be 1-D or 2-D, got {seq.shape}")
    if ref.ndim not in (1, 2):
        raise ValueError(f"reference must be 1-D or 2-D, got {ref.shape}")
    if seq.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"trailing (length) dimension must match between sequences "
            f"{seq.shape} and reference {ref.shape}"
        )
    if seq.ndim == 2 and ref.ndim == 2 and ref.shape[0] not in (1, seq.shape[0]):
        raise ValueError(
            f"reference batch dim must be 1 or match sequences "
            f"({seq.shape[0]}), got {ref.shape[0]}"
        )
    if seq.ndim == 1 and ref.ndim == 2 and ref.shape[0] != 1:
        raise ValueError(
            f"reference batch dim must be 1 when sequences is 1-D, got "
            f"{ref.shape}"
        )

    result = seq != ref
    return result if seq_is_np else backend.to_tensor(result)


def _weighted_choice_without_replacement(
    rng: np.random.Generator,
    candidates: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> np.ndarray:
    """Draw ``count`` distinct candidates with probability ∝ weights.

    Clamps ``count`` to the number of positive-weight candidates, since
    ``rng.choice(replace=False)`` rejects a request larger than the number
    of non-zero-probability entries (e.g. when a zero focus strength drops
    every focus position to weight 0).
    """
    total = weights.sum()
    if total <= 0:
        return np.empty(0, dtype=candidates.dtype)
    count = min(count, int(np.count_nonzero(weights)))
    if count <= 0:
        return np.empty(0, dtype=candidates.dtype)
    return rng.choice(candidates, size=count, p=weights / total, replace=False)


class Selector:
    """Chooses which eligible positions to corrupt.

    A selector is a callable
    ``(rng, rate, eligible, focus) -> bool array`` where ``eligible`` and
    the result share the ``(B, L)`` shape, and the result is always a subset
    of ``eligible``. ``focus`` is an optional ``(B, L)`` bool mask a selector
    may bias toward; None means no focusing.
    """

    def __call__(
        self,
        rng: np.random.Generator,
        rate: float,
        eligible: np.ndarray,
        focus: typing.Optional[np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError


class Bernoulli(Selector):
    """Independent per-position coin flip at probability ``rate``.

    Ignores ``focus`` — this is the vanilla, focus-free selector.
    """

    def __call__(self, rng, rate, eligible, focus):
        return (rng.random(eligible.shape) < rate) & eligible


class FocusBiased(Selector):
    """Bernoulli selection biased toward ``focus`` positions.

    Strategies (with no ``focus``, all fall back to plain Bernoulli):

    - ``force_include``: focus positions are always selected (still subject
      to eligibility); non-focus positions keep rate ``rate``. Raises the
      expected mask rate by the focus density.
    - ``multiplicative``: focus positions use
      ``min(1, rate * strength)``; non-focus keep ``rate``.
    - ``weighted``: the expected budget ``round(rate * eligible)`` is drawn
      without replacement per row, weighting focus positions by ``strength``
      and others by 1. Preserves the overall expected mask rate.
    """

    def __init__(self, strategy: str = "multiplicative", strength: float = 2.0):
        if strategy not in FOCUS_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {list(FOCUS_STRATEGIES)}, got "
                f"{strategy!r}"
            )
        if strength < 0.0:
            raise ValueError(f"strength must be >= 0, got {strength}")
        self.strategy = strategy
        self.strength = float(strength)

    def __call__(self, rng, rate, eligible, focus):
        if focus is None:
            return (rng.random(eligible.shape) < rate) & eligible

        if self.strategy == "force_include":
            selected = (rng.random(eligible.shape) < rate) & eligible
            selected |= focus & eligible
            return selected

        if self.strategy == "multiplicative":
            boosted = min(max(rate * self.strength, 0.0), 1.0)
            per_position = np.where(focus, boosted, rate)
            return (rng.random(eligible.shape) < per_position) & eligible

        # weighted
        selected = np.zeros(eligible.shape, dtype=bool)
        length = eligible.shape[1]
        for b in range(eligible.shape[0]):
            row_eligible = eligible[b]
            count = int(row_eligible.sum())
            if count == 0:
                continue
            target = min(int(round(rate * count)), count)
            if target == 0:
                continue
            weights = np.where(focus[b], self.strength, 1.0) * row_eligible
            chosen = _weighted_choice_without_replacement(
                rng, np.arange(length), weights, target
            )
            selected[b, chosen] = True
        return selected


class Spans(Selector):
    """Contiguous-span selection with a scattered top-up.

    Per row: the target budget ``round(rate * eligible)`` is split into a
    span share ``alpha`` and a scatter remainder. Spans are grown from
    random starts with a ``Geometric(length_p)`` length capped at
    ``max_length``, clipped at ineligible (special) boundaries so specials
    are never covered. Any shortfall is filled with scattered eligible
    positions, biased toward ``focus`` when ``focus_strength != 1``.

    ``alpha`` is fixed at ``span_fraction`` unless
    ``span_fraction_concentration`` is given, in which case it is drawn per
    call from a Beta with that mean and concentration.
    """

    def __init__(
        self,
        length_p: float,
        max_length: int,
        span_fraction: float = 0.5,
        span_fraction_concentration: typing.Optional[float] = None,
        focus_strength: float = 2.0,
    ):
        if not (0.0 < length_p <= 1.0):
            raise ValueError(f"length_p must be in (0, 1], got {length_p}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        if not (0.0 <= span_fraction <= 1.0):
            raise ValueError(
                f"span_fraction must be in [0, 1], got {span_fraction}"
            )
        if focus_strength < 0.0:
            raise ValueError(
                f"focus_strength must be >= 0, got {focus_strength}"
            )
        self.length_p = float(length_p)
        self.max_length = int(max_length)
        self.focus_strength = float(focus_strength)

        if span_fraction in (0.0, 1.0) or span_fraction_concentration is None:
            self._alpha_fixed: typing.Optional[float] = float(span_fraction)
            self._alpha_a = self._alpha_b = None
        else:
            if span_fraction_concentration <= 0.0:
                raise ValueError(
                    f"span_fraction_concentration must be > 0, got "
                    f"{span_fraction_concentration}"
                )
            self._alpha_fixed = None
            self._alpha_a = span_fraction * span_fraction_concentration
            self._alpha_b = (1.0 - span_fraction) * span_fraction_concentration

    def _sample_alpha(self, rng: np.random.Generator) -> float:
        if self._alpha_fixed is not None:
            return self._alpha_fixed
        return float(rng.beta(self._alpha_a, self._alpha_b))

    def _fill_spans(
        self,
        rng: np.random.Generator,
        row_eligible: np.ndarray,
        span_budget: int,
    ) -> np.ndarray:
        length = row_eligible.shape[0]
        row = np.zeros(length, dtype=bool)
        if span_budget <= 0:
            return row
        covered = 0
        attempts = 0
        max_attempts = max(3 * span_budget, 10)
        while covered < span_budget and attempts < max_attempts:
            attempts += 1
            start = int(rng.integers(0, length))
            if not row_eligible[start]:
                continue
            span = min(int(rng.geometric(self.length_p)), self.max_length)
            end = min(start + span, length)
            for i in range(start, end):
                if not row_eligible[i]:
                    end = i
                    break
            if end <= start:
                continue
            covered += int((~row[start:end]).sum())
            row[start:end] = True
        return row

    def _scatter_fill(
        self,
        rng: np.random.Generator,
        row_eligible: np.ndarray,
        row: np.ndarray,
        shortfall: int,
        focus_row: typing.Optional[np.ndarray],
    ) -> None:
        remaining = np.flatnonzero(row_eligible & ~row)
        if remaining.size == 0 or shortfall <= 0:
            return
        n = min(shortfall, remaining.size)
        if focus_row is None or self.focus_strength == 1.0:
            chosen = rng.choice(remaining, size=n, replace=False)
        else:
            weights = np.where(focus_row[remaining], self.focus_strength, 1.0)
            chosen = _weighted_choice_without_replacement(
                rng, remaining, weights, n
            )
        row[chosen] = True

    def __call__(self, rng, rate, eligible, focus):
        selected = np.zeros(eligible.shape, dtype=bool)
        alpha = self._sample_alpha(rng)
        for b in range(eligible.shape[0]):
            row_eligible = eligible[b]
            count = int(row_eligible.sum())
            if count == 0:
                continue
            target = int(round(rate * count))
            if target == 0:
                continue
            span_budget = int(round(alpha * target))
            row = self._fill_spans(rng, row_eligible, span_budget)
            shortfall = target - int(row.sum())
            self._scatter_fill(
                rng,
                row_eligible,
                row,
                shortfall,
                focus[b] if focus is not None else None,
            )
            selected[b] = row
        return selected
