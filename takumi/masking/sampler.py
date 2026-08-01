from __future__ import annotations

import typing

import numpy as np

from takumi.masking.scheduler import Scheduler
from takumi.masking.utils import ensure_non_negative


class MLMProbabilitySampler:
    """Samples per-call (mlm_prob, masking_prob, mutation_prob) triples
    from parallel lists of candidate values.

    Sampling strategy, in priority order:
      1. If `index` is passed to `sample()`, return the triple at that
         index. Used for evaluation at a fixed configuration.
      2. Else if a scheduler is configured, draw an index from the
         scheduler's weight distribution.
      3. Else draw an index uniformly at random from the candidates.

    Args:
        mlm_probs: Candidate masking rates.
        masking_probs: Candidate fractions of MLM positions replaced
            with the mask token, paired by index with `mlm_probs`.
        mutation_probs: Candidate fractions of MLM positions replaced
            with a random valid token, paired by index with `mlm_probs`.
        scheduler: Optional object with `sample() -> weights` and
            `step()` / `reset()` methods. Weights are used as the
            categorical distribution over the candidate indices.
    """

    def __init__(
        self,
        mlm_probs: typing.Sequence[float],
        masking_probs: typing.Sequence[float],
        mutation_probs: typing.Sequence[float],
        scheduler: Scheduler | None = None,
    ):
        if not (len(mlm_probs) == len(masking_probs) == len(mutation_probs)):
            raise ValueError(
                f"mlm_probs, masking_probs, mutation_probs must have "
                f"equal length, got {len(mlm_probs)}, {len(masking_probs)}, "
                f"{len(mutation_probs)}"
            )
        if len(mlm_probs) == 0:
            raise ValueError("probability lists must be non-empty")
        ensure_non_negative(mlm_probs, name="mlm_probs")
        ensure_non_negative(masking_probs, name="masking_probs")
        ensure_non_negative(mutation_probs, name="mutation_probs")

        # Store as NumPy arrays so downstream ops are vectorized.
        self.mlm_probs = np.asarray(mlm_probs, dtype=np.float64)
        self.masking_probs = np.asarray(masking_probs, dtype=np.float64)
        self.mutation_probs = np.asarray(mutation_probs, dtype=np.float64)
        self.scheduler = scheduler

    @property
    def num_candidates(self) -> int:
        return len(self.mlm_probs)

    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def reset(self):
        if self.scheduler is not None:
            self.scheduler.reset()

    def sample_from_scheduler(self, seed: int | None = None) -> int:
        rng = np.random.default_rng(seed)
        weights = np.asarray(self.scheduler.sample(), dtype=np.float64)
        ensure_non_negative(weights, name="scheduler weights")
        total = weights.sum()
        if total > 0:
            return int(rng.choice(self.num_candidates, p=weights / total))
        # All-zero weights: fall back to uniform.
        return int(rng.integers(0, self.num_candidates))

    def sample_randomly(self, seed: int | None = None) -> int:
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, self.num_candidates))

    def sample(
        self,
        seed: int | None = None,
        index: int | None = None,
    ) -> typing.Tuple[float, float, float]:
        if index is not None:
            if index < 0 or index >= self.num_candidates:
                raise ValueError(
                    f"index {index} out of range [0, {self.num_candidates})"
                )
            idx = index
        elif self.scheduler is not None:
            idx = self.sample_from_scheduler(seed=seed)
        else:
            idx = self.sample_randomly(seed=seed)

        return (
            float(self.mlm_probs[idx]),
            float(self.masking_probs[idx]),
            float(self.mutation_probs[idx]),
        )
