from __future__ import annotations

import typing

import numpy as np

from takumi.masking.context import CallContext
from takumi.masking.schedules import MaskSchedule
from takumi.masking.schedules import resolve_schedule


def _checked_rate(schedule: MaskSchedule, step: float, total: float) -> float:
    """Evaluate ``schedule`` at ``step`` and require a rate in ``[0, 1]``.

    Args:
        schedule (MaskSchedule): The schedule to evaluate.
        step (float): Timestep in ``[0, total]``.
        total (float): Number of timesteps, ``> 0``.

    Returns:
        float: The schedule's rate.

    Raises:
        ValueError: If the schedule returns a rate outside ``[0, 1]``.

    Example:
        >>> from takumi.masking import LinearSchedule
        >>> _checked_rate(LinearSchedule(), 1, 4)
        0.25
    """
    p = float(schedule.rate(float(step), float(total)))
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"schedule returned rate outside [0, 1]: {p}")
    return p


class RatePolicy:
    """Produces the per-call fraction of eligible positions to select.

    A policy is a callable ``(ctx: CallContext) -> rate`` where the rate is
    a float in ``[0, 1]`` shared by the whole batch, or a ``(B,)`` array
    with one rate per row. The returned rate is the *only* thing that
    varies between the fixed, Beta, sampler, and schedule maskers —
    everything downstream (selection, corruption) is shared.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import CallContext
        >>> class Alternating(RatePolicy):
        ...     def __call__(self, ctx):
        ...         return np.resize([0.1, 0.9], ctx.batch_size)
        >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=3)
        >>> Alternating()(ctx).tolist()
        [0.1, 0.9, 0.1]
    """

    def __call__(self, ctx: CallContext) -> float:
        """Return this call's rate.

        Args:
            ctx (CallContext): The call's RNG and call-time options.

        Returns:
            float | np.ndarray of float, shape (B,): One rate in ``[0, 1]``
            for the whole batch, or one per row.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext, FixedRate
            >>> FixedRate(0.15)(CallContext(rng=np.random.default_rng(0)))
            0.15
        """
        raise NotImplementedError


class FixedRate(RatePolicy):
    """Constant rate on every call.

    Args:
        rate (float): The rate, in ``(0, 1)``.

    Raises:
        ValueError: If ``rate`` is outside ``(0, 1)``.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Bernoulli, Corruptor, MaskOnly
        >>> corruptor = Corruptor(FixedRate(0.5), Bernoulli(), MaskOnly(3), [0])
        >>> result = corruptor(np.full((4, 1000), 7), seed=0)
        >>> round(float(result.masked.mean()), 1)
        0.5
    """

    def __init__(self, rate: float):
        if not (0.0 < rate < 1.0):
            raise ValueError(f"rate must be in (0, 1), got {rate}")
        self._rate = float(rate)

    def __call__(self, ctx: CallContext) -> float:
        """Return the constant rate.

        Args:
            ctx (CallContext): Ignored.

        Returns:
            float: The rate given at construction.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> FixedRate(0.3)(CallContext(rng=np.random.default_rng(0)))
            0.3
        """
        return self._rate


class BetaRate(RatePolicy):
    """Rate drawn per call from ``Beta(mean * conc, (1 - mean) * conc)``.

    The distribution has mean ``mean`` and tightens around it as
    ``concentration`` grows. One rate is drawn per call and shared by the
    whole batch.

    Args:
        mean (float): Mean rate, in ``(0, 1)``.
        concentration (float): Sharpness, ``> 0``; ``mean = 0.5`` with
            ``concentration = 2`` is uniform on ``[0, 1]``.

    Raises:
        ValueError: If ``mean`` is outside ``(0, 1)`` or ``concentration``
            is not positive.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import CallContext
        >>> policy = BetaRate(mean=0.15, concentration=50.0)
        >>> rng = np.random.default_rng(0)
        >>> draws = [policy(CallContext(rng=rng)) for _ in range(2000)]
        >>> round(float(np.mean(draws)), 2)
        0.15
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
        """Draw one rate from the Beta distribution.

        Args:
            ctx (CallContext): Supplies ``ctx.rng``.

        Returns:
            float: A rate in ``(0, 1)``.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> rate = BetaRate(0.3, 10.0)(CallContext(rng=np.random.default_rng(0)))
            >>> 0.0 < rate < 1.0
            True
        """
        return float(ctx.rng.beta(self._alpha, self._beta))


class ScheduleRate(RatePolicy):
    """Rate read from a diffusion `MaskSchedule` at ``ctx.timestep``.

    Endpoints with the built-in schedules: ``t = 0`` gives ``0`` (nothing
    masked) and ``t = T`` gives ``1`` (every eligible position masked).
    A ``(B,)`` timestep gives one rate per row.

    Args:
        total_timesteps (int): ``T``, the number of timesteps, ``> 0``.
        schedule (str | MaskSchedule | Callable[[float, float], float]):
            The schedule; see `resolve_schedule`.

    Raises:
        ValueError: If ``total_timesteps`` is not positive, or the schedule
            cannot be resolved.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Bernoulli, Corruptor, MaskOnly
        >>> corruptor = Corruptor(ScheduleRate(10), Bernoulli(), MaskOnly(3), [0])
        >>> tokens = np.full((2, 6), 7)
        >>> result = corruptor(tokens, seed=0, timestep=np.array([0, 10]))
        >>> result.masked.sum(axis=1).tolist()
        [0, 6]
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
        """Rate the schedule prescribes at ``timestep``, range-checked.

        Args:
            timestep (int | float): Timestep in ``[0, total_timesteps]``.

        Returns:
            float: Rate in ``[0, 1]``.

        Raises:
            ValueError: If ``timestep`` is out of range, or the schedule
                returns a rate outside ``[0, 1]``.

        Example:
            >>> ScheduleRate(1000, "linear").rate_at(250)
            0.25
        """
        if not (0 <= timestep <= self.total_timesteps):
            raise ValueError(
                f"timestep must be in [0, {self.total_timesteps}], "
                f"got {timestep}"
            )
        return _checked_rate(self.schedule, timestep, self.total_timesteps)

    def __call__(
        self, ctx: CallContext
    ) -> typing.Union[float, np.ndarray]:
        """Rate at ``ctx.timestep``, one per row if the timestep is.

        Args:
            ctx (CallContext): Supplies ``ctx.timestep``, a scalar or a
                ``(B,)`` array of timesteps in ``[0, total_timesteps]``.

        Returns:
            float | np.ndarray of float, shape (B,): The rate(s), matching
            the timestep's shape.

        Raises:
            ValueError: If ``ctx.timestep`` is missing or out of range.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> policy = ScheduleRate(4, "linear")
            >>> ctx = CallContext(rng=np.random.default_rng(0), timestep=np.array([1, 3]))
            >>> policy(ctx).tolist()
            [0.25, 0.75]
        """
        if ctx.timestep is None:
            raise ValueError(
                "ScheduleRate needs a timestep; pass timestep=... to the "
                "corruptor call"
            )
        timesteps = np.asarray(ctx.timestep, dtype=np.float64)
        if timesteps.ndim == 0:
            return self.rate_at(float(timesteps))
        rates = [self.rate_at(t) for t in timesteps.ravel()]
        return np.array(rates).reshape(timesteps.shape)


class UniformTimestepRate(RatePolicy):
    """Per-row rate from a `MaskSchedule` at a fresh ``t ~ U(0, 1]``.

    MaskGIT's training recipe: every sequence gets its own noise level, so
    one batch spans the whole schedule. ``t = 0`` is excluded, so with the
    built-in schedules every row gets a positive rate. Ignores
    ``ctx.timestep``; use `ScheduleRate` to pin the noise level.

    Args:
        schedule (str | MaskSchedule | Callable[[float, float], float]):
            The schedule, evaluated with ``total = 1``; see
            `resolve_schedule`.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Corruptor, ExactCount, MaskOnly
        >>> corruptor = Corruptor(UniformTimestepRate(), ExactCount(), MaskOnly(3), [0])
        >>> result = corruptor(np.full((4, 10), 7), seed=0)
        >>> result.masked.sum(axis=1).tolist()  # each row its own rate
        [6, 10, 10, 10]
    """

    def __init__(
        self,
        schedule: typing.Union[str, typing.Callable] = "cosine",
    ):
        self.schedule = resolve_schedule(schedule)

    def __call__(self, ctx: CallContext) -> np.ndarray:
        """Draw one timestep per row and return the schedule's rates.

        Args:
            ctx (CallContext): Supplies ``ctx.rng`` and ``ctx.batch_size``.

        Returns:
            np.ndarray of float, shape (B,): One rate per row, in
            ``(0, 1]`` for the built-in schedules.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=4)
            >>> rates = UniformTimestepRate("linear")(ctx)
            >>> rates.shape, bool(((rates > 0) & (rates <= 1)).all())
            ((4,), True)
        """
        t = 1.0 - ctx.rng.random(ctx.batch_size)
        return np.array([_checked_rate(self.schedule, x, 1.0) for x in t])


class SampledRate(RatePolicy):
    """Rate drawn from an external sampler's ``sample(seed, index)``.

    The sampler may return a scalar or a tuple; only the first element (the
    masking rate) is consumed here. Unlike the legacy categorical masker,
    the 80/10/10 corruption split is *not* taken from the sampler — it lives
    in the corruption scheme. Pass ``index`` to the corruptor call to pin a
    specific candidate (e.g. for evaluation at a fixed rate).

    Args:
        sampler (Any): An object with a ``sample(seed, index)`` method, such
            as `MLMProbabilitySampler`.

    Raises:
        ValueError: If ``sampler`` has no ``sample`` method.

    Example:
        >>> from takumi.masking import MLMProbabilitySampler
        >>> sampler = MLMProbabilitySampler([0.15, 0.4], [0.8, 0.8], [0.1, 0.1])
        >>> policy = SampledRate(sampler)
        >>> import numpy as np
        >>> from takumi.masking import CallContext
        >>> policy(CallContext(rng=np.random.default_rng(0), index=1))
        0.4
    """

    def __init__(self, sampler: typing.Any):
        if not hasattr(sampler, "sample"):
            raise ValueError("sampler must have a sample(seed, index) method")
        self.sampler = sampler

    def __call__(self, ctx: CallContext) -> float:
        """Ask the sampler for this call's rate.

        Args:
            ctx (CallContext): Supplies ``ctx.seed`` and ``ctx.index``,
                forwarded to ``sampler.sample``.

        Returns:
            float: The sampler's rate (the first element if it returns a
            tuple).

        Raises:
            ValueError: If the sampler returns a rate outside ``[0, 1]``.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> class Half:
            ...     def sample(self, seed=None, index=None):
            ...         return 0.5
            >>> SampledRate(Half())(CallContext(rng=np.random.default_rng(0)))
            0.5
        """
        result = self.sampler.sample(seed=ctx.seed, index=ctx.index)
        rate = float(result[0] if isinstance(result, tuple) else result)
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"sampler returned rate outside [0, 1]: {rate}")
        return rate
