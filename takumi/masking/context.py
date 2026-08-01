from __future__ import annotations

import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class CallContext:
    """Per-call state handed to a corruptor's rate policy.

    Bundles the shared RNG with the optional call-time knobs so each
    component reads what it needs and ignores the rest: ``timestep`` feeds
    schedule-driven rates, ``index`` pins a candidate for sampler-driven
    rates. ``seed`` is the raw per-call seed, exposed for policies (such as
    sampler-backed ones) that spin up their own generator from it.
    """

    rng: np.random.Generator
    seed: typing.Optional[int] = None
    timestep: typing.Optional[float] = None
    index: typing.Optional[int] = None
