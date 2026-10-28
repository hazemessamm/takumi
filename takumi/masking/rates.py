from __future__ import annotations

import typing

import numpy as np

from takumi.masking.context import CallContext
from takumi.masking.schedules import MaskSchedule
from takumi.masking.schedules import resolve_schedule


def _checked_rate(schedule: MaskSchedule, step: float, total: float) -> float:
    """The masking probability ``schedule(step, total)``, required in ``[0, 1]``.

    Calls the schedule itself rather than its raw ``rate``, so its ``eps``
    floor applies to what gets masked.

    Args:
        schedule (MaskSchedule): The schedule to evaluate.
        step (float): Timestep in ``[0, total]``.
        total (float): Number of timesteps, ``> 0``.

    Returns:
        float: The floored masking probability.

    Raises:
        ValueError: If the schedule returns a rate outside ``[0, 1]``.

    Example:
        >>> from takumi.masking import LinearSchedule
        >>> _checked_rate(LinearSchedule(), 1, 4)
        0.25
        >>> round(_checked_rate(LinearSchedule(eps=0.1), 0, 4), 3)
        0.1
    """
    p = float(schedule(float(step), float(total)))
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"schedule returned rate outside [0, 1]: {p}")
    return p


class RateDraw(typing.NamedTuple):
    """One call's rate, with the timesteps behind it and their ELBO weights.

    This is what calling a `RatePolicy` returns. A policy's `RatePolicy.draw`
    returns one directly when it has a notion of time; otherwise it returns
    just the rate, and the call wraps it with no timesteps and unit weights.

    Attributes:
        rate (float | np.ndarray of float, shape (B,)): The selection rate,
            shared or per row.
        timesteps (np.ndarray of float, shape (B,) | None): Each row's
            timestep in the policy's own units (``[0, total_timesteps]`` for
            `ScheduleRate`, ``(0, 1]`` for `UniformTimestepRate`), or None
            for a policy without time.
        weights (np.ndarray of float, shape (B,) | None): Each row's ELBO
            loss weight. ``draw`` may leave it None, which calling the
            policy turns into ones.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import CallContext
        >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=2,
        ...                   timestep=np.array([25, 50]))
        >>> policy = ScheduleRate(100, "linear", time="discrete")
        >>> rate, timesteps, weights = policy(ctx)
        >>> rate.tolist(), weights.round(6).tolist()
        ([0.25, 0.5], [4.0, 2.0])
    """

    rate: typing.Union[float, np.ndarray]
    timesteps: typing.Optional[np.ndarray]
    weights: typing.Optional[np.ndarray]


class RatePolicy:
    """Produces the per-call fraction of eligible positions to select.

    Subclasses implement `draw`, returning the rate: a float in ``[0, 1]``
    shared by the whole batch, or a ``(B,)`` array with one rate per row.
    A policy with a notion of time returns a `RateDraw` instead, adding
    each row's timestep and ELBO loss weight. Calling the policy runs
    ``draw`` and always returns a checked `RateDraw`, so subclasses leave
    ``__call__`` alone. The rate is the *only* thing that varies between the
    fixed, Beta, sampler, and schedule maskers — everything downstream
    (selection, corruption) is shared.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import CallContext
        >>> class Alternating(RatePolicy):
        ...     def draw(self, ctx):
        ...         return np.resize([0.1, 0.9], ctx.batch_size)
        >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=3)
        >>> result = Alternating()(ctx)
        >>> result.rate.tolist(), result.timesteps, result.weights.tolist()
        ([0.1, 0.9, 0.1], None, [1.0, 1.0, 1.0])
    """

    def __call__(self, ctx: CallContext) -> RateDraw:
        """Run `draw` and return a checked `RateDraw`; `Corruptor` calls this.

        A bare rate from ``draw`` is wrapped with no timesteps, and missing
        weights become ones, so a policy without time trains with plain
        MLM loss.

        Args:
            ctx (CallContext): The call's RNG and call-time options.

        Returns:
            RateDraw: ``(rate, timesteps, weights)``; per-row values have
            shape ``(B,)``.

        Raises:
            ValueError: If a per-row rate, the timesteps, or the weights do
                not have one value per row.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext, FixedRate
            >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=2)
            >>> draw = FixedRate(0.15)(ctx)
            >>> draw.rate, draw.timesteps, draw.weights.tolist()
            (0.15, None, [1.0, 1.0])
        """
        out = self.draw(ctx)
        if not isinstance(out, RateDraw):
            out = RateDraw(out, None, None)
        rows = (ctx.batch_size,)
        if out.weights is None:
            out = out._replace(weights=np.ones(rows))
        if np.ndim(out.rate) != 0 and np.shape(out.rate) != rows:
            raise ValueError(
                f"per-row rate must have shape {rows}, got "
                f"{np.shape(out.rate)}"
            )
        for name in ("timesteps", "weights"):
            values = getattr(out, name)
            if values is not None and np.shape(values) != rows:
                raise ValueError(
                    f"{name} must have shape {rows}, got {np.shape(values)}"
                )
        return out

    def draw(
        self, ctx: CallContext
    ) -> typing.Union[float, np.ndarray, RateDraw]:
        """This call's rate; subclasses implement this.

        Args:
            ctx (CallContext): The call's RNG and call-time options.

        Returns:
            float | np.ndarray of float, shape (B,) | RateDraw: The rate, or
            a `RateDraw` that adds timesteps and ELBO weights.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext, FixedRate
            >>> FixedRate(0.15).draw(CallContext(rng=np.random.default_rng(0)))
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

    def draw(self, ctx: CallContext) -> float:
        """Return the constant rate.

        Args:
            ctx (CallContext): Ignored.

        Returns:
            float: The rate given at construction.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> FixedRate(0.3).draw(CallContext(rng=np.random.default_rng(0)))
            0.3
        """
        return self._rate


class BetaRate(RatePolicy):
    """Rate drawn per call from ``Beta(mean * conc, (1 - mean) * conc)``.

    The distribution has mean ``mean`` and tightens around it as
    ``concentration`` grows. One rate is drawn per call and shared by the
    whole batch.

    Its ELBO weights are always 1, so it trains with plain MLM loss. The
    ELBO needs every masking level, but a Beta rarely draws high ones, so
    their weights would have to be huge. For the masked-diffusion ELBO, use
    `UniformTimestepRate` or `ScheduleRate` instead.

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
        >>> draws = [policy(CallContext(rng=rng)).rate for _ in range(2000)]
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

    def draw(self, ctx: CallContext) -> float:
        """Draw one rate from the Beta distribution.

        Args:
            ctx (CallContext): Supplies ``ctx.rng``.

        Returns:
            float: A rate in ``(0, 1)``.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> rate = BetaRate(0.3, 10.0).draw(CallContext(rng=np.random.default_rng(0)))
            >>> 0.0 < rate < 1.0
            True
        """
        return float(ctx.rng.beta(self._alpha, self._beta))


class ScheduleRate(RatePolicy):
    """Rate read from a diffusion `MaskSchedule` at ``ctx.timestep``.

    Endpoints with the built-in schedules: ``t = 0`` gives ``0`` (nothing
    masked; ``eps`` with a floored schedule) and ``t = T`` gives ``1``
    (every eligible position masked).
    A ``(B,)`` timestep gives one rate per row.

    ``time`` states how your timesteps are drawn, which decides the ELBO
    weight: ``"discrete"`` for whole-number steps drawn from ``1..T`` (the
    schedule's exact ``discrete_elbo_weight``), ``"continuous"`` for
    real-valued steps in ``[0, T]`` (its ``continuous_elbo_weight``,
    ``p'(t) / p(t)``).

    Args:
        total_timesteps (int): ``T``, the number of timesteps, ``> 0``.
        schedule (str | MaskSchedule | Callable[[float, float], float]):
            The schedule; see `resolve_schedule`.
        time (str): ``"discrete"`` or ``"continuous"``; required, so the
            weights always match how timesteps are drawn.

    Raises:
        ValueError: If ``total_timesteps`` is not positive, ``time`` is not
            ``"discrete"`` or ``"continuous"``, or the schedule cannot be
            resolved.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Bernoulli, Corruptor, MaskOnly
        >>> policy = ScheduleRate(10, time="discrete")
        >>> corruptor = Corruptor(policy, Bernoulli(), MaskOnly(3), [0])
        >>> tokens = np.full((2, 6), 7)
        >>> result = corruptor(tokens, seed=0, timestep=np.array([0, 10]))
        >>> result.masked.sum(axis=1).tolist()
        [0, 6]
    """

    def __init__(
        self,
        total_timesteps: int,
        schedule: typing.Union[str, typing.Callable] = "cosine",
        *,
        time: str,
    ):
        if total_timesteps <= 0:
            raise ValueError(
                f"total_timesteps must be > 0, got {total_timesteps}"
            )
        if time not in ("discrete", "continuous"):
            raise ValueError(
                f"time must be 'discrete' or 'continuous', got {time!r}"
            )
        self.total_timesteps = int(total_timesteps)
        self.schedule = resolve_schedule(schedule)
        self.time = time

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
            >>> ScheduleRate(1000, "linear", time="discrete").rate_at(250)
            0.25
        """
        if not (0 <= timestep <= self.total_timesteps):
            raise ValueError(
                f"timestep must be in [0, {self.total_timesteps}], "
                f"got {timestep}"
            )
        return _checked_rate(self.schedule, timestep, self.total_timesteps)

    def draw(self, ctx: CallContext) -> RateDraw:
        """Rate at ``ctx.timestep``, with each row's timestep and ELBO weight.

        A scalar timestep gives one rate shared by every row; a ``(B,)``
        timestep gives one per row. The weights follow ``time``: the exact
        ``schedule.discrete_elbo_weight`` for ``"discrete"``, the schedule's
        ``continuous_elbo_weight`` for ``"continuous"``.

        Args:
            ctx (CallContext): Supplies ``ctx.timestep``, a scalar or a
                ``(B,)`` array in ``[0, total_timesteps]``, and
                ``ctx.batch_size``.

        Returns:
            RateDraw: ``(rate, timesteps, weights)``, with ``timesteps`` in
            ``[0, total_timesteps]``.

        Raises:
            ValueError: If ``ctx.timestep`` is missing or out of range, or
                ``time`` is ``"discrete"`` and a timestep is not a whole
                number.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=2,
            ...                   timestep=np.array([1, 3]))
            >>> ScheduleRate(4, "linear", time="discrete")(ctx).rate.tolist()
            [0.25, 0.75]
            >>> ctx.timestep = np.array([50, 100])
            >>> discrete = ScheduleRate(100, "cosine", time="discrete")
            >>> discrete(ctx).weights.round(3).tolist()
            [1.583, 0.012]
            >>> continuous = ScheduleRate(100, "cosine", time="continuous")
            >>> continuous(ctx).weights.round(3).tolist()  # p'(t) / p(t)
            [1.571, 0.0]
        """
        if ctx.timestep is None:
            raise ValueError(
                "ScheduleRate needs a timestep; pass timestep=... to the "
                "corruptor call"
            )
        timesteps = np.asarray(ctx.timestep, dtype=np.float64)
        if timesteps.ndim == 0:
            rate = self.rate_at(float(timesteps))
            timesteps = np.full(ctx.batch_size, float(timesteps))
        else:
            rates = [self.rate_at(t) for t in timesteps.ravel()]
            rate = np.array(rates).reshape(timesteps.shape)
        if self.time == "discrete":
            if not np.all(timesteps == np.round(timesteps)):
                raise ValueError(
                    "time='discrete' needs whole-number timesteps; use "
                    "time='continuous' for real-valued ones"
                )
            weights = [
                self.schedule.discrete_elbo_weight(t, self.total_timesteps)
                for t in timesteps.ravel()
            ]
        else:
            weights = [
                self.schedule.continuous_elbo_weight(t, self.total_timesteps)
                for t in timesteps.ravel()
            ]
        weights = np.array(weights).reshape(timesteps.shape)
        return RateDraw(rate, timesteps, weights)


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

    def draw(self, ctx: CallContext) -> RateDraw:
        """Draw each row's ``t``; return its rate and continuous ELBO weight.

        Args:
            ctx (CallContext): Supplies ``ctx.rng`` and ``ctx.batch_size``.

        Returns:
            RateDraw: ``(rate, timesteps, weights)``: one rate per row, in
            ``(0, 1]`` for the built-in schedules; the drawn ``t`` in
            ``(0, 1]``; and the schedule's ``continuous_elbo_weight`` at each.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=3)
            >>> rate, t, weights = UniformTimestepRate("linear")(ctx)
            >>> bool(np.allclose(rate, t)), bool(np.allclose(weights, 1 / t))
            (True, True)
        """
        t = 1.0 - ctx.rng.random(ctx.batch_size)
        rate = np.array([_checked_rate(self.schedule, x, 1.0) for x in t])
        weights = np.array(
            [self.schedule.continuous_elbo_weight(x, 1.0) for x in t]
        )
        return RateDraw(rate, t, weights)


class SampledRate(RatePolicy):
    """Rate drawn from an external sampler's ``sample(rng, index)``.

    The sampler may return a scalar or a tuple; only the first element (the
    masking rate) is consumed here. Unlike the legacy categorical masker,
    the 80/10/10 corruption split is *not* taken from the sampler — it lives
    in the corruption scheme. Pass ``index`` to the corruptor call to pin a
    specific candidate (e.g. for evaluation at a fixed rate).

    Args:
        sampler (Any): An object with a ``sample(rng, index)`` method, such
            as `MLMProbabilitySampler`.

    Raises:
        ValueError: If ``sampler`` has no ``sample`` method.

    Example:
        >>> from takumi.masking import MLMProbabilitySampler
        >>> sampler = MLMProbabilitySampler([0.15, 0.4], [0.8, 0.8], [0.1, 0.1])
        >>> policy = SampledRate(sampler)
        >>> import numpy as np
        >>> from takumi.masking import CallContext
        >>> policy(CallContext(rng=np.random.default_rng(0), index=1)).rate
        0.4
    """

    def __init__(self, sampler: typing.Any):
        if not hasattr(sampler, "sample"):
            raise ValueError("sampler must have a sample(rng, index) method")
        self.sampler = sampler

    def draw(self, ctx: CallContext) -> float:
        """Ask the sampler for this call's rate.

        Args:
            ctx (CallContext): Supplies ``ctx.rng`` and ``ctx.index``,
                forwarded to ``sampler.sample``. The sampler draws from the
                call's shared generator, so its pick is independent of the
                selection draws that follow.

        Returns:
            float: The sampler's rate (the first element if it returns a
            tuple).

        Raises:
            ValueError: If the sampler returns a rate outside ``[0, 1]``.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import CallContext
            >>> class Half:
            ...     def sample(self, rng=None, index=None):
            ...         return 0.5
            >>> SampledRate(Half()).draw(CallContext(rng=np.random.default_rng(0)))
            0.5
        """
        result = self.sampler.sample(rng=ctx.rng, index=ctx.index)
        rate = float(result[0] if isinstance(result, tuple) else result)
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"sampler returned rate outside [0, 1]: {rate}")
        return rate
