"""Named shortcuts for the common corruptor compositions.

Each preset is a thin `Corruptor` subclass that wires up a rate policy,
selector, and corruption scheme from familiar keyword arguments. They exist
purely for ergonomics — anything a preset does can be expressed directly with
`Corruptor(rate=..., select=..., corrupt=...)`.
"""

from __future__ import annotations

import typing

from takumi.masking.corruptor import Corruptor
from takumi.masking.rates import BetaRate
from takumi.masking.rates import FixedRate
from takumi.masking.rates import SampledRate
from takumi.masking.rates import ScheduleRate
from takumi.masking.schemes import MaskOnly
from takumi.masking.schemes import Mixed
from takumi.masking.selectors import FocusBiased
from takumi.masking.selectors import Spans


class FixedMasker(Corruptor):
    """80/10/10 masking at a constant rate."""

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
    """80/10/10 masking with a Beta-sampled per-call rate."""

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

    Call with ``timestep=`` in ``[0, total_timesteps]``.
    """

    def __init__(
        self,
        total_timesteps: int,
        mask_token_id: int,
        special_token_ids: typing.Sequence[int],
        schedule: typing.Union[str, typing.Callable] = "cosine",
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
    ):
        super().__init__(
            rate=ScheduleRate(total_timesteps, schedule),
            select=FocusBiased(focus_strategy, focus_strength),
            corrupt=MaskOnly(mask_token_id),
            special_token_ids=special_token_ids,
        )


class HybridSpanMasker(Corruptor):
    """80/10/10 masking over Beta-sampled contiguous spans plus scatter."""

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
