from __future__ import annotations

import math
import typing


class MaskSchedule:
    """Maps a diffusion timestep to a masking probability in ``[0, 1]``.

    Subclasses implement ``rate(step, total)``. Instances are ordinary
    picklable objects, so a schedule can ride along to spawn-based
    DataLoader workers without the pickling problems that lambdas and
    local closures hit.

    Example:
        >>> class Quadratic(MaskSchedule):
        ...     def rate(self, step, total):
        ...         return (step / total) ** 2
        >>> Quadratic().rate(5, 10)
        0.25
    """

    def rate(self, step: float, total: float) -> float:
        """Masking probability at ``step`` out of ``total``.

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


class LinearSchedule(MaskSchedule):
    """``p(t) = t / T``.

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


class CosineSchedule(MaskSchedule):
    """MaskGIT's ``gamma(r) = cos(r * pi / 2)``, with ``r = 1 - t / T``.

    MaskGIT counts ``r`` as decoding progress, so ``r = 0`` is fully
    masked; this module counts ``t`` as noise, so ``t = 0`` is unmasked.
    The mean rate over a uniform ``t`` is ``2 / pi``.

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


class SqrtSchedule(MaskSchedule):
    """``p(t) = sqrt(t / T)``.

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
