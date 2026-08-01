from __future__ import annotations

from typing import List

from takumi.masking.scheduler.common import SharedCounter


class Scheduler:
    def __init__(self, initial_value, final_value):
        self.current_step = SharedCounter(
            initial_value=initial_value, final_value=final_value
        )

    def step(self):
        self.current_step.step()

    def reset(self):
        self.current_step.reset()

    def sample(self) -> List[float]:
        raise NotImplementedError
