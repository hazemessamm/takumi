from __future__ import annotations

import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class CallContext:
    """Per-call state handed to a corruptor's rate policy.

    Bundles the shared RNG with the optional call-time knobs so each
    component reads what it needs and ignores the rest: ``timestep`` feeds
    schedule-driven rates (a scalar, or one per row), ``index`` pins a
    candidate for sampler-driven rates. ``batch_size`` is the number of
    rows, for policies that draw a separate rate per row.

    `Corruptor` builds one per call; build one by hand only to call a
    `RatePolicy` directly.

    Attributes:
        rng (np.random.Generator): The call's shared random generator.
        timestep (float | np.ndarray of float, shape (B,) | None): Noise
            level for `ScheduleRate`.
        index (int | None): Candidate index for `SampledRate`.
        batch_size (int): Number of rows ``B`` in the batch.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import CallContext, UniformTimestepRate
        >>> ctx = CallContext(rng=np.random.default_rng(0), batch_size=3)
        >>> UniformTimestepRate()(ctx).rate.shape
        (3,)
    """

    rng: np.random.Generator
    timestep: typing.Union[float, np.ndarray, None] = None
    index: typing.Optional[int] = None
    batch_size: int = 1
