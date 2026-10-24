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

    Args:
        initial_weights (list[float], length N): Weights at step 0, ``>= 0``.
        final_weights (list[float], length N): Weights approached as steps
            grow, ``>= 0``.
        beta (float): Decay per effective step, in ``(0, 1)``; closer to 1
            moves more slowly.
        multiplier (float): Effective steps per counter step, ``> 0``.

    Raises:
        ValueError: If the lists differ in length or hold negative values,
            ``beta`` is outside ``(0, 1)``, or ``multiplier`` is not
            positive.

    Example:
        >>> scheduler = EMAScheduler([1.0, 0.0], [0.0, 1.0], beta=0.5)
        >>> scheduler.sample()
        [1.0, 0.0]
        >>> scheduler.step()
        >>> scheduler.sample()
        [0.5, 0.5]
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
        """Get the interpolated weights for the current step.

        Returns:
            list[float], length N: ``beta**t * initial + (1 - beta**t) *
            final``, with ``t = step * multiplier``.

        Example:
            >>> scheduler = EMAScheduler([1.0, 0.0], [0.0, 1.0], beta=0.5, multiplier=2.0)
            >>> scheduler.step()
            >>> scheduler.sample()  # t = 2, so beta**t = 0.25
            [0.25, 0.75]
        """
        effective_step = self.current_step.get() * self.multiplier
        decay_factor = self.beta**effective_step
        return [
            decay_factor * iw + (1 - decay_factor) * fw
            for iw, fw in zip(self.initial_weights, self.final_weights)
        ]
