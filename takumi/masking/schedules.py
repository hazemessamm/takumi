from __future__ import annotations

import math
import typing


class MaskSchedule:
    """Maps a diffusion timestep to a masking probability in ``[0, 1]``.

    Subclasses implement ``rate(step, total)``. Instances are ordinary
    picklable objects, so a schedule can ride along to spawn-based
    DataLoader workers without the pickling problems that lambdas and
    local closures hit.
    """

    def rate(self, step: float, total: float) -> float:
        raise NotImplementedError


class LinearSchedule(MaskSchedule):
    """``p(t) = t / T``."""

    def rate(self, step: float, total: float) -> float:
        return step / total


class CosineSchedule(MaskSchedule):
    """MaskGIT-style ``p(t) = 1 - cos((t / T) * pi / 2)``."""

    def rate(self, step: float, total: float) -> float:
        return 1.0 - math.cos((step / total) * (math.pi / 2.0))


class SqrtSchedule(MaskSchedule):
    """``p(t) = sqrt(t / T)``."""

    def rate(self, step: float, total: float) -> float:
        return math.sqrt(step / total)


BUILTIN_SCHEDULES: typing.Dict[str, typing.Type[MaskSchedule]] = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
    "sqrt": SqrtSchedule,
}


class _CallableSchedule(MaskSchedule):
    """Adapts a bare ``(step, total) -> float`` callable to a MaskSchedule."""

    def __init__(self, fn: typing.Callable[[float, float], float]):
        self._fn = fn

    def rate(self, step: float, total: float) -> float:
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
