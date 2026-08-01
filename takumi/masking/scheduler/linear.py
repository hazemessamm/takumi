from __future__ import annotations

from typing import List

from takumi.masking.scheduler.common import validate_lengths
from takumi.masking.scheduler.scheduler import Scheduler
from takumi.masking.utils import ensure_non_negative


class LinearScheduler(Scheduler):
    """
    Linearly interpolates between initial and final weights for
    MLM probabilities over a specified number of training steps.

    This allows for a curriculum learning approach where the distribution of
    masking probabilities can change during training.
    """

    def __init__(
        self,
        initial_weights: List[float],
        final_weights: List[float],
        total_steps: int,
    ):
        validate_lengths(initial_weights, final_weights)
        ensure_non_negative(initial_weights, name="initial_weights")
        ensure_non_negative(final_weights, name="final_weights")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        super().__init__(initial_value=0, final_value=total_steps)
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.total_steps = total_steps

    def sample(self) -> List[float]:
        """Get the interpolated weights for the current step."""
        progress = self.current_step.get() / self.total_steps
        progress = min(progress, 1.0)

        return [
            iw + progress * (fw - iw)
            for iw, fw in zip(self.initial_weights, self.final_weights)
        ]
