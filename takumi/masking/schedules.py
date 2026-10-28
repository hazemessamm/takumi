from __future__ import annotations

import math
import typing


class MaskSchedule:
    """Maps a diffusion timestep to a masking probability in ``[0, 1]``.

    Subclasses implement ``rate(step, total)``, the raw curve, and may
    implement ``compute_slope`` with a closed form. Takumi masks with the
    schedule itself, ``schedule(step, total)``, which applies the optional
    floor ``eps + (1 - eps) * rate``; both ELBO weights use the same floored
    curve. A floor keeps nearly-clean rows from masking almost nothing,
    which bounds their ELBO weight near ``1 / eps``. Instances are ordinary
    picklable objects, so a schedule can ride along to spawn-based
    DataLoader workers without the pickling problems that lambdas and
    local closures hit.

    Args:
        eps (float): Floor on the masking probability, in ``[0, 1)``;
            ``0`` (the default) uses the raw curve.

    Raises:
        ValueError: If ``eps`` is outside ``[0, 1)``.

    Example:
        >>> class Quadratic(MaskSchedule):
        ...     def rate(self, step, total):
        ...         return (step / total) ** 2
        >>> Quadratic()(5, 10)
        0.25
        >>> round(Quadratic(eps=0.1)(5, 10), 3)  # 0.1 + 0.9 * 0.25
        0.325
    """

    # A class-level default, so a subclass with its own __init__ that skips
    # super().__init__() still has a floor of 0.
    eps = 0.0

    def __init__(self, eps: float = 0.0):
        if not (0.0 <= eps < 1.0):
            raise ValueError(f"eps must be in [0, 1), got {eps}")
        self.eps = float(eps)

    def rate(self, step: float, total: float) -> float:
        """The raw masking curve at ``step`` out of ``total``, before ``eps``.

        Subclasses implement this. Takumi never masks with it directly: it
        calls the schedule itself, which applies ``eps``.

        Args:
            step (float): Timestep in ``[0, total]``, on the same scale as
                ``total`` (not a fraction): ``0`` is clean, ``total`` fully
                noised.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: Probability in ``[0, 1]``. Built-in schedules give ``0``
            at ``step = 0`` and ``1`` at ``step = total``.

        Example:
            >>> LinearSchedule().rate(250, 1000)
            0.25
        """
        raise NotImplementedError

    def compute_slope(self, step: float, total: float) -> float:
        """The raw curve's slope ``p'(t)`` at ``t = step / total``.

        The derivative is with respect to the fraction ``t``, not the step,
        and of ``rate``, before ``eps``; the floored curve's slope is
        ``(1 - eps)`` times this, which ``continuous_elbo_weight`` applies.
        Together with ``rate`` this is all a schedule has to provide: both
        ELBO weights are generic formulas built from the two. The built-in
        schedules override it with closed forms; this default
        differentiates ``rate`` numerically, so a custom schedule needs no
        extra code.

        Args:
            step (float): Timestep in ``[0, total]``, on the same scale as
                ``total``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: ``dp/dt``; ``>= 0`` for a non-decreasing schedule.

        Example:
            >>> class Quadratic(MaskSchedule):
            ...     def rate(self, step, total):
            ...         return (step / total) ** 2
            >>> round(Quadratic().compute_slope(1, 2), 6)  # 2t at t = 0.5
            1.0
        """
        h = 1e-6 * total
        lo, hi = max(step - h, 0.0), min(step + h, total)
        rise = float(self.rate(hi, total) - self.rate(lo, total))
        return rise / ((hi - lo) / total)

    def continuous_elbo_weight(self, step: float, total: float) -> float:
        """Continuous-time ELBO weight ``p'(t) / p(t)``.

        With ``t`` drawn uniformly from ``(0, 1]``, the average of this
        weight times a row's summed masked-token cross-entropy estimates the
        negative ELBO (MDLM, Sahoo et al. 2024; MD4, Shi et al. 2024). It is
        the floored curve's slope over the floored curve,
        ``(1 - eps) * compute_slope / schedule(step, total)``, so schedules
        override ``rate`` and ``compute_slope``, not this. For whole-number
        timesteps, `discrete_elbo_weight` is exact.

        Args:
            step (float): Timestep in ``[0, total]``, on the same scale as
                ``total``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: Weight ``>= 0``; ``0`` at ``step = 0`` (the clean data)
            and wherever the rate is 0, since nothing is masked there.

        Example:
            >>> LinearSchedule().continuous_elbo_weight(25, 100)  # 1 / t
            4.0
            >>> round(CosineSchedule().continuous_elbo_weight(50, 100), 3)  # (pi/2) tan(pi/4)
            1.571
            >>> LinearSchedule(eps=0.5).continuous_elbo_weight(0, 100)  # clean data
            0.0
        """
        if step <= 0:
            return 0.0
        p = float(self(step, total))
        if p == 0.0:
            return 0.0
        return (1.0 - self.eps) * float(self.compute_slope(step, total)) / p

    def discrete_elbo_weight(self, step: float, total: float) -> float:
        """Exact ELBO weight for whole-number timesteps.

        The weight is ``total * (p(i) - p(i-1)) / p(i)``, where
        ``(p(i) - p(i-1)) / p(i)`` is the probability that a token masked at
        step ``i`` was still clean at step ``i - 1``: the chance it is the one
        revealed when undoing that step. Multiplying by ``total`` makes a
        step drawn uniformly from ``1..total`` estimate the ELBO's sum over
        all steps (the discrete-time ELBO of MD4, Shi et al. 2024). ``p`` is
        the floored curve, ``schedule(step, total)``; the floor's
        ``(1 - eps)`` factor is already inside the difference. It needs only
        the curve, so every schedule gets it as is.

        Args:
            step (float): A whole-number timestep in ``[0, total]``. Step 0
                is the clean data, not one of the ``total`` steps, so its
                weight is 0.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: The weight, ``>= 0``.

        Raises:
            ValueError: If ``p(i - 1)`` and ``p(i)`` are not ordered inside
                ``[0, 1]``; masking is absorbing, so the rate may never go
                down.

        Example:
            >>> round(LinearSchedule().discrete_elbo_weight(50, 100), 6)  # = 100 / 50
            2.0
            >>> round(CosineSchedule().discrete_elbo_weight(50, 100), 3)
            1.583
        """
        if step < 1:
            return 0.0
        p = float(self(step, total))
        if p == 0.0:
            return 0.0
        previous = float(self(step - 1, total))
        if not (0.0 <= previous <= p <= 1.0):
            raise ValueError(
                f"rates at steps {step - 1} and {step} must satisfy "
                f"0 <= p(i-1) <= p(i) <= 1, got {previous} and {p}; masking "
                f"is absorbing, so the schedule may never go down"
            )
        return total * (p - previous) / p

    def __call__(self, step: float, total: float) -> float:
        """Masking probability at ``step`` out of ``total``, floored by ``eps``.

        What takumi masks with: ``eps + (1 - eps) * rate(step, total)``,
        exactly ``rate`` when ``eps`` is 0.

        Args:
            step (float): Timestep in ``[0, total]``, on the same scale as
                ``total``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: Probability in ``[eps, 1]`` for the built-in schedules.

        Example:
            >>> LinearSchedule()(25, 100)
            0.25
            >>> round(LinearSchedule(eps=0.1)(0, 100), 3)  # never below eps
            0.1
        """
        return self.eps + (1.0 - self.eps) * self.rate(step, total)


class LinearSchedule(MaskSchedule):
    """``p(t) = t / T``.

    Takes ``eps``, a floor on the masking probability; see `MaskSchedule`.

    Example:
        >>> [LinearSchedule().rate(t, 4) for t in range(5)]
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """

    def rate(self, step: float, total: float) -> float:
        """``step / total``; see `MaskSchedule.rate`.

        Args:
            step (float): Timestep in ``[0, total]``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: ``step / total``.

        Example:
            >>> LinearSchedule().rate(3, 4)
            0.75
        """
        return step / total

    def compute_slope(self, step: float, total: float) -> float:
        """``1`` everywhere; see `MaskSchedule.compute_slope`.

        A straight line rises at a constant rate, so its ELBO weight is
        ``1 / t``.

        Args:
            step (float): Timestep in ``[0, total]``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: ``1.0`` everywhere.

        Example:
            >>> LinearSchedule().compute_slope(25, 100)
            1.0
        """
        return 1.0


class CosineSchedule(MaskSchedule):
    """MaskGIT's ``gamma(r) = cos(r * pi / 2)``, with ``r = 1 - t / T``.

    MaskGIT counts ``r`` as decoding progress, so ``r = 0`` is fully
    masked; this module counts ``t`` as noise, so ``t = 0`` is unmasked.
    The mean rate over a uniform ``t`` is ``2 / pi``. Takes ``eps``, a
    floor on the masking probability; see `MaskSchedule`.

    Example:
        >>> [round(CosineSchedule().rate(t, 4), 3) for t in range(5)]
        [0.0, 0.383, 0.707, 0.924, 1.0]
    """

    def rate(self, step: float, total: float) -> float:
        """``cos((1 - step / total) * pi / 2)``; see `MaskSchedule.rate`.

        Args:
            step (float): Timestep in ``[0, total]``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: Masking probability in ``[0, 1]``.

        Example:
            >>> round(CosineSchedule().rate(500, 1000), 3)
            0.707
        """
        return math.cos((1.0 - step / total) * (math.pi / 2.0))

    def compute_slope(self, step: float, total: float) -> float:
        """Slope ``(pi / 2) * sin((1 - t) * pi / 2)``.

        See `MaskSchedule.compute_slope`. With ``u = (1 - t) * pi / 2``,
        ``p = cos(u)`` and ``du/dt = -pi / 2``, so the chain rule gives
        ``p' = -sin(u) * du/dt = (pi / 2) * sin(u)``.
        Dividing by ``p`` gives the ELBO weight ``(pi / 2) * tan(u)``.

        Args:
            step (float): Timestep in ``[0, total]``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: From ``pi / 2`` at ``t = 0`` down to ``0`` at ``t = 1``,
            where the curve flattens.

        Example:
            >>> round(CosineSchedule().compute_slope(50, 100), 3)
            1.111
        """
        return (math.pi / 2.0) * math.sin((1.0 - step / total) * (math.pi / 2.0))


class SqrtSchedule(MaskSchedule):
    """``p(t) = sqrt(t / T)``.

    Takes ``eps``, a floor on the masking probability; see `MaskSchedule`.

    Example:
        >>> [SqrtSchedule().rate(t, 4) for t in (0, 1, 4)]
        [0.0, 0.5, 1.0]
    """

    def rate(self, step: float, total: float) -> float:
        """``sqrt(step / total)``; see `MaskSchedule.rate`.

        Args:
            step (float): Timestep in ``[0, total]``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: Masking probability in ``[0, 1]``.

        Example:
            >>> SqrtSchedule().rate(25, 100)
            0.5
        """
        return math.sqrt(step / total)

    def compute_slope(self, step: float, total: float) -> float:
        """``1 / (2 sqrt(t))``; see `MaskSchedule.compute_slope`.

        Dividing by ``p = sqrt(t)`` gives the ELBO weight ``1 / (2t)``.

        Args:
            step (float): Timestep in ``[0, total]``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: The slope; infinite at ``t = 0``, where the curve starts
            vertical.

        Example:
            >>> SqrtSchedule().compute_slope(25, 100)
            1.0
        """
        t = step / total
        return 1.0 / (2.0 * math.sqrt(t)) if t > 0 else math.inf


BUILTIN_SCHEDULES: typing.Dict[str, typing.Type[MaskSchedule]] = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
    "sqrt": SqrtSchedule,
}


class _CallableSchedule(MaskSchedule):
    """Adapts a bare ``(step, total) -> float`` callable to a MaskSchedule.

    Args:
        fn (Callable[[float, float], float]): The schedule function.

    Example:
        >>> _CallableSchedule(lambda step, total: step / total / 2).rate(1, 2)
        0.25
    """

    def __init__(self, fn: typing.Callable[[float, float], float]):
        self._fn = fn

    def rate(self, step: float, total: float) -> float:
        """Call the wrapped function; see `MaskSchedule.rate`.

        Args:
            step (float): Timestep in ``[0, total]``.
            total (float): Number of timesteps, ``> 0``.

        Returns:
            float: The function's result, coerced to ``float``.

        Example:
            >>> _CallableSchedule(lambda step, total: 1).rate(0, 1)
            1.0
        """
        return float(self._fn(step, total))


def resolve_schedule(
    schedule: typing.Union[
        str, MaskSchedule, typing.Callable[[float, float], float]
    ],
) -> MaskSchedule:
    """Normalize a name / instance / callable into a `MaskSchedule`.

    Prefer built-in names or `MaskSchedule` subclasses when running under
    spawn-based DataLoader workers; a wrapped lambda is convenient but not
    picklable.

    Args:
        schedule (str | MaskSchedule | Callable[[float, float], float]): A
            built-in name (``"linear"``, ``"cosine"``, ``"sqrt"``), a
            schedule instance (returned as is), or a ``(step, total) ->
            float`` callable.

    Returns:
        MaskSchedule: The resolved schedule.

    Raises:
        ValueError: If the name is unknown or the argument is none of the
            accepted kinds.

    Example:
        >>> type(resolve_schedule("cosine")).__name__
        'CosineSchedule'
        >>> resolve_schedule(lambda step, total: 0.5).rate(1, 2)
        0.5
    """
    if isinstance(schedule, str):
        try:
            return BUILTIN_SCHEDULES[schedule]()
        except KeyError:
            raise ValueError(
                f"unknown schedule {schedule!r}; choose one of "
                f"{sorted(BUILTIN_SCHEDULES)}, or pass a MaskSchedule "
                f"instance or a callable"
            ) from None
    if isinstance(schedule, MaskSchedule):
        return schedule
    if callable(schedule):
        return _CallableSchedule(schedule)
    raise ValueError(
        "schedule must be a name, a MaskSchedule, or a callable; "
        f"got {type(schedule).__name__}"
    )
