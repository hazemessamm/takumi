from __future__ import annotations

from typing import List

from takumi.masking.scheduler.common import validate_lengths
from takumi.masking.scheduler.scheduler import Scheduler
from takumi.masking.utils import ensure_non_negative


class EMAScheduler(Scheduler):
    """
    Updates weights using a bias-corrected exponential moving average.
    This provides a smooth, exponential transition from initial to final
    weights. The update rule is equivalent to an exponential interpolation:
    w_t = beta^t * w_0 + (1 - beta^t) * w_final
    Where t is the effective step count.
    """

    def __init__(
        self,
        initial_weights: List[float],
        final_weights: List[float],
        beta: float = 0.999,
        multiplier: float = 1.0,
    ):
        validate_lengths(initial_weights, final_weights)
        ensure_non_negative(initial_weights, name="initial_weights")
        ensure_non_negative(final_weights, name="final_weights")
        if not (0.0 < beta < 1.0):
            raise ValueError("Beta must be between 0 and 1.")
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")

        super().__init__(initial_value=0, final_value=2**31 - 1)

        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.beta = beta
        self.multiplier = multiplier

    def sample(self) -> List[float]:
        """Get the interpolated weights for the current step."""
        effective_step = self.current_step.get() * self.multiplier
        decay_factor = self.beta**effective_step
        return [
            decay_factor * iw + (1 - decay_factor) * fw
            for iw, fw in zip(self.initial_weights, self.final_weights)
        ]
