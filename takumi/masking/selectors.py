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

    Args:
        focus (ArrayLike | Sequence | None, shape (B, L)): Positions to bias
            selection toward; truthy values count as focus.
        expected_shape (tuple[int, int]): The sequences' shape ``(B, L)``.

    Returns:
        np.ndarray of bool, shape (B, L) | None: The focus mask, or None.

    Raises:
        ValueError: If the shape differs from ``expected_shape``.

    Example:
        >>> coerce_focus([[0, 1, 1]], (1, 3)).tolist()
        [[False, True, True]]
        >>> coerce_focus(None, (1, 3)) is None
        True
    """
    if focus is None:
        return None
    focus = np.asarray(backend.to_numpy(focus), dtype=bool)
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

    Args:
        sequences (ArrayLike of int, shape (B, L) or (L,)): Token ids.
        reference (ArrayLike of int, shape (L,), (1, L) or (B, L)): The
            template to compare against.

    Returns:
        ArrayLike of bool, shape of ``sequences``: True where the tokens
        differ from the reference.

    Raises:
        ValueError: If either input is not 1-D or 2-D, the lengths differ,
            or the batch dimensions cannot broadcast.

    Example:
        >>> import numpy as np
        >>> sequences = np.array([[1, 2, 3, 4], [1, 9, 3, 4]])
        >>> focus_from_reference(sequences, np.array([1, 2, 3, 0])).tolist()
        [[False, False, False, True], [False, True, False, True]]
    """
    seq_is_np = isinstance(sequences, np.ndarray)
    seq = backend.to_numpy(sequences)
    ref = backend.to_numpy(reference)

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

    Args:
        rng (np.random.Generator): Source of randomness.
        candidates (np.ndarray of int, shape (N,)): Values to choose from.
        weights (np.ndarray of float, shape (N,)): Non-negative weights.
        count (int): How many to draw.

    Returns:
        np.ndarray of int, shape (K,): The chosen candidates, ``K <= count``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> picks = _weighted_choice_without_replacement(
        ...     rng, np.arange(3), np.array([0.0, 1.0, 1.0]), count=5)
        >>> sorted(picks.tolist())  # only two have positive weight
        [1, 2]
    """
    total = weights.sum()
    if total <= 0:
        return np.empty(0, dtype=candidates.dtype)
    count = min(count, int(np.count_nonzero(weights)))
    if count <= 0:
        return np.empty(0, dtype=candidates.dtype)
    return rng.choice(candidates, size=count, p=weights / total, replace=False)


def _rate_column(rate: typing.Union[float, np.ndarray]) -> np.ndarray:
    """Shape a scalar or ``(B,)`` rate to broadcast against ``(B, L)``.

    Args:
        rate (float | np.ndarray of float, shape (B,)): The rate(s).

    Returns:
        np.ndarray of float, shape (1,) or (B, 1): Broadcastable rate.

    Example:
        >>> import numpy as np
        >>> _rate_column(0.5).shape, _rate_column(np.array([0.1, 0.2])).shape
        ((1,), (2, 1))
    """
    return np.asarray(rate, dtype=np.float64)[..., None]


def _row_rates(
    rate: typing.Union[float, np.ndarray], batch_size: int
) -> np.ndarray:
    """Broadcast a scalar or ``(B,)`` rate to one rate per row.

    Args:
        rate (float | np.ndarray of float, shape (B,)): The rate(s).
        batch_size (int): ``B``.

    Returns:
        np.ndarray of float, shape (B,): A read-only view, one rate per row.

    Raises:
        ValueError: If an array ``rate`` does not have ``batch_size`` rows.

    Example:
        >>> _row_rates(0.5, 3).tolist()
        [0.5, 0.5, 0.5]
    """
    return np.broadcast_to(np.asarray(rate, dtype=np.float64), (batch_size,))


class Selector:
    """Chooses which eligible positions to corrupt.

    A selector is a callable
    ``(rng, rate, eligible, focus) -> bool array`` where ``eligible`` and
    the result share the ``(B, L)`` shape, and the result is always a subset
    of ``eligible``. ``rate`` is a float shared by every row or a ``(B,)``
    array with one rate per row. ``focus`` is an optional ``(B, L)`` bool
    mask a selector may bias toward; None means no focusing.

    Example:
        >>> import numpy as np
        >>> class FirstEligible(Selector):
        ...     def __call__(self, rng, rate, eligible, focus):
        ...         return eligible & (eligible.cumsum(axis=1) == 1)
        >>> eligible = np.array([[False, True, True]])
        >>> FirstEligible()(np.random.default_rng(0), 0.5, eligible, None).tolist()
        [[False, True, False]]
    """

    def __call__(
        self,
        rng: np.random.Generator,
        rate: typing.Union[float, np.ndarray],
        eligible: np.ndarray,
        focus: typing.Optional[np.ndarray],
    ) -> np.ndarray:
        """Select positions to corrupt.

        Args:
            rng (np.random.Generator): The call's shared random generator.
            rate (float | np.ndarray of float, shape (B,)): Fraction of
                eligible positions to select, shared or per row.
            eligible (np.ndarray of bool, shape (B, L)): Positions that may
                be selected (False at special tokens).
            focus (np.ndarray of bool, shape (B, L) | None): Positions to
                bias toward, if the selector supports it.

        Returns:
            np.ndarray of bool, shape (B, L): Selected positions, a subset of
            ``eligible``.

        Example:
            >>> import numpy as np
            >>> eligible = np.array([[True, False, True]])
            >>> Bernoulli()(np.random.default_rng(0), 1.0, eligible, None).tolist()
            [[True, False, True]]
        """
        raise NotImplementedError


class Bernoulli(Selector):
    """Independent per-position coin flip at probability ``rate``.

    Ignores ``focus`` — this is the vanilla, focus-free selector.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Corruptor, FixedRate, MaskOnly
        >>> corruptor = Corruptor(FixedRate(0.15), Bernoulli(), MaskOnly(3), [0])
        >>> result = corruptor(np.full((8, 1000), 7), seed=0)
        >>> abs(float(result.masked.mean()) - 0.15) < 0.02
        True
    """

    def __call__(self, rng, rate, eligible, focus):
        """Select each eligible position independently with prob ``rate``.

        Args:
            rng (np.random.Generator): The call's shared random generator.
            rate (float | np.ndarray of float, shape (B,)): Selection
                probability, shared or per row.
            eligible (np.ndarray of bool, shape (B, L)): Selectable positions.
            focus (np.ndarray of bool, shape (B, L) | None): Ignored.

        Returns:
            np.ndarray of bool, shape (B, L): Selected positions.

        Example:
            >>> import numpy as np
            >>> eligible = np.array([[True, True, False], [True, True, False]])
            >>> rates = np.array([0.0, 1.0])
            >>> Bernoulli()(np.random.default_rng(0), rates, eligible, None).tolist()
            [[False, False, False], [True, True, False]]
        """
        return (rng.random(eligible.shape) < _rate_column(rate)) & eligible


class ExactCount(Selector):
    """Exactly ``ceil(rate * eligible)`` positions per row, chosen uniformly.

    MaskGIT's selection: the rate fixes the count instead of setting a
    per-position probability, so rows at the same rate never differ in how
    much they mask, and rounding up means any rate above float noise
    selects at least one position. Ignores ``focus``.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Corruptor, FixedRate, MaskOnly
        >>> corruptor = Corruptor(FixedRate(0.25), ExactCount(), MaskOnly(3), [0])
        >>> tokens = np.array([[0, 7, 7, 7, 7, 7, 7, 7, 7, 0]])  # 8 eligible
        >>> int(corruptor(tokens, seed=0).masked.sum())
        2
    """

    def __call__(self, rng, rate, eligible, focus):
        """Select ``ceil(rate * eligible count)`` positions in each row.

        Args:
            rng (np.random.Generator): The call's shared random generator.
            rate (float | np.ndarray of float, shape (B,)): Fraction to
                select, shared or per row.
            eligible (np.ndarray of bool, shape (B, L)): Selectable positions.
            focus (np.ndarray of bool, shape (B, L) | None): Ignored.

        Returns:
            np.ndarray of bool, shape (B, L): Selected positions, uniformly
            drawn without replacement within each row.

        Example:
            >>> import numpy as np
            >>> eligible = np.ones((2, 10), dtype=bool)
            >>> rates = np.array([0.3, 0.05])
            >>> selected = ExactCount()(np.random.default_rng(0), rates, eligible, None)
            >>> selected.sum(axis=1).tolist()  # 0.3 * 10 = 3; ceil(0.5) = 1
            [3, 1]
        """
        counts = eligible.sum(axis=1)
        rates = _row_rates(rate, eligible.shape[0])
        # The tolerance absorbs float error: 0.3 * 10 is 3.0000000000000004,
        # which must select 3, not 4.
        targets = np.minimum(np.ceil(rates * counts - 1e-9), counts)
        # Rank positions by a random key; ineligible ones sort last, so the
        # lowest ``target`` ranks are a uniform draw from the eligible set.
        keys = np.where(eligible, rng.random(eligible.shape), np.inf)
        ranks = keys.argsort(axis=1).argsort(axis=1)
        return ranks < targets[:, None]


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

    Args:
        strategy (str): ``"force_include"``, ``"multiplicative"`` or
            ``"weighted"``.
        strength (float): Focus boost, ``>= 0``; unused by
            ``force_include``.

    Raises:
        ValueError: If ``strategy`` is unknown or ``strength`` is negative.

    Example:
        >>> import numpy as np
        >>> selector = FocusBiased("weighted", strength=10.0)
        >>> eligible = np.ones((1, 10), dtype=bool)
        >>> focus = np.zeros((1, 10), dtype=bool)
        >>> focus[0, :2] = True
        >>> selected = selector(np.random.default_rng(0), 0.2, eligible, focus)
        >>> int(selected.sum())  # round(0.2 * 10), whatever the focus
        2
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
        """Select positions, biased toward ``focus`` per the strategy.

        Args:
            rng (np.random.Generator): The call's shared random generator.
            rate (float | np.ndarray of float, shape (B,)): Base selection
                rate, shared or per row.
            eligible (np.ndarray of bool, shape (B, L)): Selectable positions.
            focus (np.ndarray of bool, shape (B, L) | None): Positions to
                favor; None gives plain Bernoulli selection.

        Returns:
            np.ndarray of bool, shape (B, L): Selected positions.

        Example:
            >>> import numpy as np
            >>> eligible = np.ones((1, 5), dtype=bool)
            >>> focus = np.array([[False, True, False, True, False]])
            >>> selector = FocusBiased("force_include")
            >>> selector(np.random.default_rng(0), 0.0, eligible, focus).tolist()
            [[False, True, False, True, False]]
        """
        rate_column = _rate_column(rate)
        if focus is None:
            return (rng.random(eligible.shape) < rate_column) & eligible

        if self.strategy == "force_include":
            selected = (rng.random(eligible.shape) < rate_column) & eligible
            selected |= focus & eligible
            return selected

        if self.strategy == "multiplicative":
            boosted = np.clip(rate_column * self.strength, 0.0, 1.0)
            per_position = np.where(focus, boosted, rate_column)
            return (rng.random(eligible.shape) < per_position) & eligible

        # weighted
        selected = np.zeros(eligible.shape, dtype=bool)
        length = eligible.shape[1]
        rates = _row_rates(rate, eligible.shape[0])
        for b in range(eligible.shape[0]):
            row_eligible = eligible[b]
            count = int(row_eligible.sum())
            if count == 0:
                continue
            target = min(int(round(rates[b] * count)), count)
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
    positions, biased toward ``focus`` when ``focus_strength != 1``. The
    last span can overshoot its share, so a row may select a few more than
    the target.

    ``alpha`` is fixed at ``span_fraction`` unless
    ``span_fraction_concentration`` is given, in which case it is drawn per
    call from a Beta with that mean and concentration.

    Args:
        length_p (float): Geometric success probability, in ``(0, 1]``;
            mean span length is ``1 / length_p``.
        max_length (int): Longest allowed span, ``>= 1``.
        span_fraction (float): Mean share of the budget covered by spans,
            in ``[0, 1]``.
        span_fraction_concentration (float | None): If given, ``alpha`` is
            drawn per call from a Beta with mean ``span_fraction`` and this
            concentration, ``> 0``.
        focus_strength (float): Weight of focus positions in the scatter
            top-up, ``>= 0``.

    Raises:
        ValueError: If any argument is out of range.

    Example:
        >>> import numpy as np
        >>> spans = Spans(length_p=0.3, max_length=4, span_fraction=1.0)
        >>> eligible = np.ones((1, 20), dtype=bool)
        >>> selected = spans(np.random.default_rng(0), 0.25, eligible, None)
        >>> "".join("x" if s else "." for s in selected[0])  # 5 asked, 6 covered
        '.xx.........x....xxx'
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
        """This call's span share of the budget.

        Args:
            rng (np.random.Generator): Used only when ``alpha`` is drawn.

        Returns:
            float: ``span_fraction``, or a Beta draw around it.

        Example:
            >>> import numpy as np
            >>> Spans(0.3, 4, span_fraction=0.5)._sample_alpha(np.random.default_rng(0))
            0.5
        """
        if self._alpha_fixed is not None:
            return self._alpha_fixed
        return float(rng.beta(self._alpha_a, self._alpha_b))

    def _fill_spans(
        self,
        rng: np.random.Generator,
        row_eligible: np.ndarray,
        span_budget: int,
    ) -> np.ndarray:
        """Grow random spans in one row until they cover ``span_budget``.

        Gives up after ``max(3 * span_budget, 10)`` attempts, so a crowded
        row may cover less.

        Args:
            rng (np.random.Generator): Source of randomness.
            row_eligible (np.ndarray of bool, shape (L,)): Selectable
                positions; spans stop at the first ineligible one.
            span_budget (int): Positions to cover.

        Returns:
            np.ndarray of bool, shape (L,): Positions covered by spans.

        Example:
            >>> import numpy as np
            >>> spans = Spans(length_p=0.2, max_length=8)
            >>> row_eligible = np.array([True] * 5 + [False] + [True] * 4)
            >>> row = spans._fill_spans(np.random.default_rng(0), row_eligible, 4)
            >>> bool(row[5])  # never crosses the special at position 5
            False
        """
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
        """Select ``shortfall`` more scattered positions in ``row``, in place.

        Args:
            rng (np.random.Generator): Source of randomness.
            row_eligible (np.ndarray of bool, shape (L,)): Selectable
                positions.
            row (np.ndarray of bool, shape (L,)): Current selection; updated
                in place.
            shortfall (int): Additional positions to select.
            focus_row (np.ndarray of bool, shape (L,) | None): Positions to
                weight by ``focus_strength``.

        Returns:
            None: ``row`` is modified in place.

        Example:
            >>> import numpy as np
            >>> spans = Spans(length_p=0.3, max_length=4)
            >>> row = np.array([True, True, False, False, False])
            >>> spans._scatter_fill(np.random.default_rng(0), np.ones(5, bool), row, 2, None)
            >>> int(row.sum())
            4
        """
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
        """Select spans plus scatter, about ``round(rate * eligible)`` per row.

        Args:
            rng (np.random.Generator): The call's shared random generator.
            rate (float | np.ndarray of float, shape (B,)): Fraction to
                select, shared or per row.
            eligible (np.ndarray of bool, shape (B, L)): Selectable positions.
            focus (np.ndarray of bool, shape (B, L) | None): Positions the
                scatter top-up favors.

        Returns:
            np.ndarray of bool, shape (B, L): Selected positions.

        Example:
            >>> import numpy as np
            >>> spans = Spans(length_p=0.5, max_length=3, span_fraction=0.0)
            >>> eligible = np.ones((2, 10), dtype=bool)
            >>> rates = np.array([0.2, 0.5])
            >>> spans(np.random.default_rng(0), rates, eligible, None).sum(axis=1).tolist()
            [2, 5]
        """
        selected = np.zeros(eligible.shape, dtype=bool)
        alpha = self._sample_alpha(rng)
        rates = _row_rates(rate, eligible.shape[0])
        for b in range(eligible.shape[0]):
            row_eligible = eligible[b]
            count = int(row_eligible.sum())
            if count == 0:
                continue
            target = int(round(rates[b] * count))
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
