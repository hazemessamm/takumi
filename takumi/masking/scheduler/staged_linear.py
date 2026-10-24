from __future__ import annotations

from typing import List

from takumi.masking.scheduler.scheduler import Scheduler


class StagedLinearScheduler(Scheduler):
    """
    Staged curriculum scheduler that introduces masking difficulties
    sequentially. Each stage introduces a new difficulty level, ramping
    its weight linearly from 0 to equal weight with existing difficulties.

    Args:
        num_stages (int): Number of difficulties (candidates), ``> 0``.
        total_steps (int): Steps until every stage is fully ramped, at least
            ``num_stages``.

    Raises:
        ValueError: If either argument is not positive, or ``total_steps <
            num_stages``.

    Example:
        >>> scheduler = StagedLinearScheduler(num_stages=2, total_steps=10)
        >>> scheduler.sample()  # only the first stage at the start
        [1.0, 0.0]
        >>> for _ in range(10):
        ...     scheduler.step()
        >>> scheduler.sample()
        [0.5, 0.5]
    """

    def __init__(
        self,
        num_stages: int,
        total_steps: int,
    ):
        if num_stages <= 0:
            raise ValueError("num_stages must be a positive integer.")
        if total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        if total_steps < num_stages:
            raise ValueError(
                "total_steps must be at least the number of stages."
            )

        super().__init__(initial_value=0, final_value=total_steps)

        self.num_stages = num_stages
        self.total_steps = total_steps

    def sample(self) -> List[float]:
        """Return the staged weights for all masking difficulties.

        Stage ``i`` ramps from 0 to 1 over steps ``[i * T / S, (i + 1) * T /
        S)``; the weights are then normalized to sum to 1.

        Returns:
            list[float], length num_stages: Normalized weights.

        Example:
            >>> scheduler = StagedLinearScheduler(num_stages=2, total_steps=10)
            >>> for _ in range(7):
            ...     scheduler.step()
            >>> [round(w, 3) for w in scheduler.sample()]  # stage 2 at 40%
            [0.714, 0.286]
        """

        current_step = min(self.current_step.get(), self.total_steps)

        weights = []
        for i in range(self.num_stages):
            stage_start = (i * self.total_steps) // self.num_stages
            stage_end = ((i + 1) * self.total_steps) // self.num_stages

            if current_step >= stage_end:
                weights.append(1.0)
            elif current_step <= stage_start:
                weights.append(0.0)
            else:
                stage_progress = (current_step - stage_start) / (
                    stage_end - stage_start
                )
                weights.append(stage_progress)

        total = sum(weights)
        if total == 0:
            weights[0] = 1.0
            total = 1.0
        return [w / total for w in weights]
