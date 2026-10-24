from __future__ import annotations

from typing import List

from takumi.masking.scheduler.common import SharedCounter


class Scheduler:
    """Base class for curriculum schedules over candidate weights.

    Holds a process-shared step counter; subclasses implement ``sample`` to
    turn the current step into one weight per candidate. Feed one to
    `MLMProbabilitySampler`, which draws candidates in proportion to the
    weights.

    Args:
        initial_value (int): Counter's starting step.
        final_value (int): Counter's last step; ``step`` stops there.

    Example:
        >>> class Flip(Scheduler):
        ...     def sample(self):
        ...         return [1.0, 0.0] if self.current_step.get() == 0 else [0.0, 1.0]
        >>> scheduler = Flip(initial_value=0, final_value=1)
        >>> scheduler.sample()
        [1.0, 0.0]
        >>> scheduler.step()
        >>> scheduler.sample()
        [0.0, 1.0]
    """

    def __init__(self, initial_value, final_value):
        self.current_step = SharedCounter(
            initial_value=initial_value, final_value=final_value
        )

    def step(self):
        """Advance the shared counter by one.

        Returns:
            None

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> scheduler = LinearScheduler([1.0], [0.0], total_steps=4)
            >>> scheduler.step()
            >>> scheduler.current_step.get()
            1
        """
        self.current_step.step()

    def reset(self):
        """Rewind the shared counter to its initial value.

        Returns:
            None

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> scheduler = LinearScheduler([1.0], [0.0], total_steps=4)
            >>> scheduler.step()
            >>> scheduler.reset()
            >>> scheduler.current_step.get()
            0
        """
        self.current_step.reset()

    def sample(self) -> List[float]:
        """Weights for the current step.

        Returns:
            list[float]: One non-negative weight per candidate; they need
            not sum to 1.

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=4).sample()
            [1.0, 0.0]
        """
        raise NotImplementedError
