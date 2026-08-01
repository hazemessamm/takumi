from __future__ import annotations

import typing

from takumi.masking.context import CallContext
from takumi.masking.schedules import resolve_schedule


class RatePolicy:
    """Produces the per-call fraction of eligible positions to select.

    A policy is a callable ``(ctx: CallContext) -> float in [0, 1]``. The
    returned rate is the *only* thing that varies between the fixed, Beta,
    sampler, and schedule maskers — everything downstream (selection,
    corruption) is shared.
    """

    def __call__(self, ctx: CallContext) -> float:
        raise NotImplementedError


class FixedRate(RatePolicy):
    """Constant rate on every call."""

    def __init__(self, rate: float):
        if not (0.0 < rate < 1.0):
            raise ValueError(f"rate must be in (0, 1), got {rate}")
        self._rate = float(rate)

    def __call__(self, ctx: CallContext) -> float:
        return self._rate


class BetaRate(RatePolicy):
    """Rate drawn per call from ``Beta(mean * conc, (1 - mean) * conc)``.

    The distribution has mean ``mean`` and tightens around it as
    ``concentration`` grows.
    """

    def __init__(self, mean: float, concentration: float):
        if not (0.0 < mean < 1.0):
            raise ValueError(f"mean must be in (0, 1), got {mean}")
        if concentration <= 0.0:
            raise ValueError(
                f"concentration must be > 0, got {concentration}"
            )
        self._alpha = float(mean * concentration)
        self._beta = float((1.0 - mean) * concentration)

    def __call__(self, ctx: CallContext) -> float:
        return float(ctx.rng.beta(self._alpha, self._beta))


class ScheduleRate(RatePolicy):
    """Rate read from a diffusion `MaskSchedule` at ``ctx.timestep``.

    Endpoints with the built-in schedules: ``t = 0`` gives ``0`` (nothing
    masked) and ``t = T`` gives ``1`` (every eligible position masked).
    """

    def __init__(
        self,
        total_timesteps: int,
        schedule: typing.Union[str, typing.Callable] = "cosine",
    ):
        if total_timesteps <= 0:
            raise ValueError(
                f"total_timesteps must be > 0, got {total_timesteps}"
            )
        self.total_timesteps = int(total_timesteps)
        self.schedule = resolve_schedule(schedule)

    def rate_at(self, timestep: typing.Union[int, float]) -> float:
        """Rate the schedule prescribes at ``timestep``, range-checked."""
        if not (0 <= timestep <= self.total_timesteps):
            raise ValueError(
                f"timestep must be in [0, {self.total_timesteps}], "
                f"got {timestep}"
            )
        p = float(
            self.schedule.rate(float(timestep), float(self.total_timesteps))
        )
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"schedule returned rate outside [0, 1]: {p}")
        return p

    def __call__(self, ctx: CallContext) -> float:
        if ctx.timestep is None:
            raise ValueError(
                "ScheduleRate needs a timestep; pass timestep=... to the "
                "corruptor call"
            )
        return self.rate_at(ctx.timestep)


class SampledRate(RatePolicy):
    """Rate drawn from an external sampler's ``sample(seed, index)``.

    The sampler may return a scalar or a tuple; only the first element (the
    masking rate) is consumed here. Unlike the legacy categorical masker,
    the 80/10/10 corruption split is *not* taken from the sampler — it lives
    in the corruption scheme. Pass ``index`` to the corruptor call to pin a
    specific candidate (e.g. for evaluation at a fixed rate).
    """

    def __init__(self, sampler: typing.Any):
        if not hasattr(sampler, "sample"):
            raise ValueError("sampler must have a sample(seed, index) method")
        self.sampler = sampler

    def __call__(self, ctx: CallContext) -> float:
        result = self.sampler.sample(seed=ctx.seed, index=ctx.index)
        rate = float(result[0] if isinstance(result, tuple) else result)
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"sampler returned rate outside [0, 1]: {rate}")
        return rate
