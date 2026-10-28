"""Named shortcuts for the common corruptor compositions.

Each preset is a thin `Corruptor` subclass that wires up a rate policy,
selector, and corruption scheme from familiar keyword arguments. They exist
purely for ergonomics — anything a preset does can be expressed directly with
`Corruptor(rate=..., select=..., corrupt=...)`. Calling a preset is calling
`Corruptor.__call__`: it takes ``(B, L)`` token ids and returns a
`MaskingResult` of the same shape and type.
"""

from __future__ import annotations

import typing

from takumi.masking.corruptor import Corruptor
from takumi.masking.rates import BetaRate
from takumi.masking.rates import FixedRate
from takumi.masking.rates import SampledRate
from takumi.masking.rates import ScheduleRate
from takumi.masking.rates import UniformTimestepRate
from takumi.masking.schemes import MaskOnly
from takumi.masking.schemes import Mixed
from takumi.masking.selectors import ExactCount
from takumi.masking.selectors import FocusBiased
from takumi.masking.selectors import Spans


class FixedMasker(Corruptor):
    """80/10/10 masking at a constant rate.

    ``FixedRate(mlm_prob)`` + ``FocusBiased`` + ``Mixed``.

    Args:
        mlm_prob (float): Fraction of eligible positions selected, in
            ``(0, 1)``.
        masking_prob (float): Share of selected positions set to the mask
            token (typically 0.8).
        mutation_prob (float): Share replaced with a random valid token
            (typically 0.1).
        mask_token_id (int): The mask token's id.
        valid_token_ids (Sequence[int]): Substitution pool for mutations.
        special_token_ids (Sequence[int]): Ids never selected.
        focus_strategy (str): `FocusBiased` strategy for call-time ``focus``.
        focus_strength (float): `FocusBiased` strength, ``>= 0``.

    Example:
        >>> import numpy as np
        >>> masker = FixedMasker(
        ...     mlm_prob=0.15, masking_prob=0.8, mutation_prob=0.1,
        ...     mask_token_id=3, valid_token_ids=range(4, 100),
        ...     special_token_ids=[0, 1, 2, 3],
        ... )
        >>> result = masker(np.full((16, 512), 50), seed=0)
        >>> result.labels.shape, abs(float((result.labels != -100).mean()) - 0.15) < 0.01
        ((16, 512), True)
    """

    def __init__(
        self,
        mlm_prob: float,
        masking_prob: float,
        mutation_prob: float,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
        special_token_ids: typing.Sequence[int],
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
    ):
        super().__init__(
            rate=FixedRate(mlm_prob),
            select=FocusBiased(focus_strategy, focus_strength),
            corrupt=Mixed(
                masking_prob, mutation_prob, mask_token_id, valid_token_ids
            ),
            special_token_ids=special_token_ids,
        )


class BetaMasker(Corruptor):
    """80/10/10 masking with a Beta-sampled per-call rate.

    ``BetaRate(mlm_prob, mlm_concentration)`` + ``FocusBiased`` + ``Mixed``.

    Args:
        mlm_prob (float): Mean selection rate, in ``(0, 1)``.
        mlm_concentration (float): Beta concentration, ``> 0``; larger keeps
            the rate closer to ``mlm_prob``.
        masking_prob (float): Share of selected positions set to the mask
            token.
        mutation_prob (float): Share replaced with a random valid token.
        mask_token_id (int): The mask token's id.
        valid_token_ids (Sequence[int]): Substitution pool for mutations.
        special_token_ids (Sequence[int]): Ids never selected.
        focus_strategy (str): `FocusBiased` strategy for call-time ``focus``.
        focus_strength (float): `FocusBiased` strength, ``>= 0``.

    Example:
        >>> import numpy as np
        >>> masker = BetaMasker(0.3, 12.0, 0.8, 0.1, 3, range(4, 100), [0, 1, 2, 3])
        >>> tokens = np.full((4, 64), 50)
        >>> a, b = masker(tokens, seed=1), masker(tokens, seed=2)
        >>> bool((a.labels == b.labels).all())  # the rate differs per call
        False
    """

    def __init__(
        self,
        mlm_prob: float,
        mlm_concentration: float,
        masking_prob: float,
        mutation_prob: float,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
        special_token_ids: typing.Sequence[int],
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
    ):
        super().__init__(
            rate=BetaRate(mlm_prob, mlm_concentration),
            select=FocusBiased(focus_strategy, focus_strength),
            corrupt=Mixed(
                masking_prob, mutation_prob, mask_token_id, valid_token_ids
            ),
            special_token_ids=special_token_ids,
        )


class CategoricalMasker(Corruptor):
    """80/10/10 masking with a per-call rate from an external sampler.

    Note: the corruption split (``masking_prob`` / ``mutation_prob``) is
    fixed here, unlike the legacy categorical masker that also read the split
    from the sampler. If you need the split to vary per call, compose a
    custom `CorruptionScheme` instead.

    Args:
        mlm_prob_sampler (Any): Object with ``sample(rng, index)``, such as
            `MLMProbabilitySampler`; see `SampledRate`.
        mask_token_id (int): The mask token's id.
        valid_token_ids (Sequence[int]): Substitution pool for mutations.
        special_token_ids (Sequence[int]): Ids never selected.
        masking_prob (float): Share of selected positions set to the mask
            token.
        mutation_prob (float): Share replaced with a random valid token.
        focus_strategy (str): `FocusBiased` strategy for call-time ``focus``.
        focus_strength (float): `FocusBiased` strength, ``>= 0``.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import MLMProbabilitySampler
        >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1])
        >>> masker = CategoricalMasker(sampler, 3, range(4, 100), [0, 1, 2, 3])
        >>> result = masker(np.full((8, 1000), 50), seed=0, index=1)  # pin 0.5
        >>> round(float((result.labels != -100).mean()), 1)
        0.5
    """

    def __init__(
        self,
        mlm_prob_sampler: typing.Any,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
        special_token_ids: typing.Sequence[int],
        masking_prob: float = 0.8,
        mutation_prob: float = 0.1,
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
    ):
        super().__init__(
            rate=SampledRate(mlm_prob_sampler),
            select=FocusBiased(focus_strategy, focus_strength),
            corrupt=Mixed(
                masking_prob, mutation_prob, mask_token_id, valid_token_ids
            ),
            special_token_ids=special_token_ids,
        )


class DiffusionMasker(Corruptor):
    """Mask-only corruption at a schedule-driven timestep rate.

    Call with ``timestep=`` in ``[0, total_timesteps]``, a scalar or one per
    row. ``time`` says whether those timesteps are whole-number steps or
    real values, which picks the ELBO weight in ``result.weights``; see
    `ScheduleRate`.

    Args:
        total_timesteps (int): ``T``, ``> 0``.
        mask_token_id (int): The mask token's id.
        special_token_ids (Sequence[int]): Ids never selected.
        schedule (str | MaskSchedule | Callable[[float, float], float]):
            The schedule; see `resolve_schedule`.
        focus_strategy (str): `FocusBiased` strategy for call-time ``focus``.
        focus_strength (float): `FocusBiased` strength, ``>= 0``.
        time (str): ``"discrete"`` or ``"continuous"``; required.

    Example:
        >>> import numpy as np
        >>> masker = DiffusionMasker(1000, mask_token_id=3, special_token_ids=[0],
        ...                          time="discrete")
        >>> tokens = np.full((2, 1000), 50)
        >>> result = masker(tokens, seed=0, timestep=np.array([0, 500]))
        >>> [round(float(m), 1) for m in result.masked.mean(axis=1)]
        [0.0, 0.7]
    """

    def __init__(
        self,
        total_timesteps: int,
        mask_token_id: int,
        special_token_ids: typing.Sequence[int],
        schedule: typing.Union[str, typing.Callable] = "cosine",
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
        *,
        time: str,
    ):
        super().__init__(
            rate=ScheduleRate(total_timesteps, schedule, time=time),
            select=FocusBiased(focus_strategy, focus_strength),
            corrupt=MaskOnly(mask_token_id),
            special_token_ids=special_token_ids,
        )


class MaskGITMasker(Corruptor):
    """MaskGIT training corruption.

    Every row draws its own ``t ~ U(0, 1]``, then exactly
    ``ceil(schedule(t) * eligible)`` uniformly chosen positions are replaced
    with the mask token. Decode with `MaskGITDecoder` on the same schedule.

    Args:
        mask_token_id (int): The mask token's id.
        special_token_ids (Sequence[int]): Ids never selected (for example a
            class-conditioning token).
        schedule (str | MaskSchedule | Callable[[float, float], float]):
            The schedule; see `resolve_schedule`.

    Example:
        >>> import numpy as np
        >>> masker = MaskGITMasker(mask_token_id=3, special_token_ids=[0])
        >>> tokens = np.full((4, 16), 50)
        >>> tokens[:, 0] = 0  # conditioning token, never masked
        >>> result = masker(tokens, seed=0)
        >>> result.masked.sum(axis=1).tolist(), bool(result.masked[:, 0].any())
        ([9, 14, 15, 15], False)
    """

    def __init__(
        self,
        mask_token_id: int,
        special_token_ids: typing.Sequence[int],
        schedule: typing.Union[str, typing.Callable] = "cosine",
    ):
        super().__init__(
            rate=UniformTimestepRate(schedule),
            select=ExactCount(),
            corrupt=MaskOnly(mask_token_id),
            special_token_ids=special_token_ids,
        )


class HybridSpanMasker(Corruptor):
    """80/10/10 masking over Beta-sampled contiguous spans plus scatter.

    ``BetaRate`` + ``Spans`` + ``Mixed``.

    Args:
        mlm_prob (float): Mean selection rate, in ``(0, 1)``.
        mlm_concentration (float): Beta concentration for the rate, ``> 0``.
        span_fraction_mean (float): Mean share of the budget covered by
            spans, in ``[0, 1]``.
        span_fraction_concentration (float): Beta concentration for that
            share, ``> 0``.
        span_length_p (float): Geometric parameter for span lengths, in
            ``(0, 1]``; mean length is ``1 / span_length_p``.
        max_span_length (int): Longest span, ``>= 1``.
        masking_prob (float): Share of selected positions set to the mask
            token.
        mutation_prob (float): Share replaced with a random valid token.
        mask_token_id (int): The mask token's id.
        valid_token_ids (Sequence[int]): Substitution pool for mutations.
        special_token_ids (Sequence[int]): Ids never selected; spans stop
            at them.
        focus_strength (float): Weight of focus positions in the scatter
            top-up, ``>= 0``.

    Example:
        >>> import numpy as np
        >>> masker = HybridSpanMasker(
        ...     mlm_prob=0.3, mlm_concentration=20.0,
        ...     span_fraction_mean=0.9, span_fraction_concentration=50.0,
        ...     span_length_p=0.3, max_span_length=6,
        ...     masking_prob=0.8, mutation_prob=0.1,
        ...     mask_token_id=3, valid_token_ids=range(4, 100),
        ...     special_token_ids=[0, 1, 2, 3],
        ... )
        >>> selected = masker(np.full((1, 40), 50), seed=2).labels[0] != -100
        >>> "".join("x" if s else "." for s in selected)
        '......xxx.xxxxxx......xx.............xxx'
    """

    def __init__(
        self,
        mlm_prob: float,
        mlm_concentration: float,
        span_fraction_mean: float,
        span_fraction_concentration: float,
        span_length_p: float,
        max_span_length: int,
        masking_prob: float,
        mutation_prob: float,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
        special_token_ids: typing.Sequence[int],
        focus_strength: float = 2.0,
    ):
        super().__init__(
            rate=BetaRate(mlm_prob, mlm_concentration),
            select=Spans(
                length_p=span_length_p,
                max_length=max_span_length,
                span_fraction=span_fraction_mean,
                span_fraction_concentration=span_fraction_concentration,
                focus_strength=focus_strength,
            ),
            corrupt=Mixed(
                masking_prob, mutation_prob, mask_token_id, valid_token_ids
            ),
            special_token_ids=special_token_ids,
        )
