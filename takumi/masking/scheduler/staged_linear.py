from __future__ import annotations

from typing import List

from takumi.masking.scheduler.scheduler import Scheduler


class StagedLinearScheduler(Scheduler):
    """
    Staged curriculum scheduler that introduces masking difficulties
    sequentially. Each stage introduces a new difficulty level, ramping
    its weight linearly from 0 to equal weight with existing difficulties.
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
        """Return the staged weights for all masking difficulties."""

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
