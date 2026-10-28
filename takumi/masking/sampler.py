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
        mlm_probs (Sequence[float], length N): Candidate masking rates.
        masking_probs (Sequence[float], length N): Candidate fractions of
            MLM positions replaced with the mask token, paired by index with
            `mlm_probs`.
        mutation_probs (Sequence[float], length N): Candidate fractions of
            MLM positions replaced with a random valid token, paired by
            index with `mlm_probs`.
        scheduler (Scheduler | None): Optional object with
            `sample() -> weights` and `step()` / `reset()` methods. Weights
            are used as the categorical distribution over the candidate
            indices.

    Raises:
        ValueError: If the lists are empty, differ in length, or hold
            negative values.

    Example:
        >>> sampler = MLMProbabilitySampler(
        ...     mlm_probs=[0.15, 0.3, 0.5],
        ...     masking_probs=[0.8, 0.8, 1.0],
        ...     mutation_probs=[0.1, 0.1, 0.0],
        ... )
        >>> sampler.sample(index=2)
        (0.5, 1.0, 0.0)
        >>> sampler.sample(rng=0) in [(0.15, 0.8, 0.1), (0.3, 0.8, 0.1), (0.5, 1.0, 0.0)]
        True
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
        """Number of candidate triples, ``N``.

        Returns:
            int: Length of the candidate lists.

        Example:
            >>> MLMProbabilitySampler([0.1, 0.2], [0.8, 0.8], [0.1, 0.1]).num_candidates
            2
        """
        return len(self.mlm_probs)

    def step(self):
        """Advance the scheduler by one step; a no-op without one.

        Returns:
            None

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=1)
            >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1], scheduler)
            >>> sampler.sample(rng=0)[0]
            0.1
            >>> sampler.step()
            >>> sampler.sample(rng=0)[0]
            0.5
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def reset(self):
        """Rewind the scheduler to its first step; a no-op without one.

        Returns:
            None

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=1)
            >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1], scheduler)
            >>> sampler.step()
            >>> sampler.reset()
            >>> sampler.sample(rng=0)[0]
            0.1
        """
        if self.scheduler is not None:
            self.scheduler.reset()

    def state_dict(self) -> typing.Dict[str, typing.Any]:
        """What a checkpoint must save to resume sampling: the scheduler's step.

        Returns:
            dict: ``{"scheduler": scheduler.state_dict()}``, or ``{}`` without
            a scheduler; JSON-serializable.

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=4)
            >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1], scheduler)
            >>> sampler.step()
            >>> sampler.state_dict()
            {'scheduler': {'step': 1}}
        """
        if self.scheduler is None:
            return {}
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state: typing.Dict[str, typing.Any]) -> None:
        """Resume from a ``state_dict``.

        Args:
            state (dict): A dict from ``state_dict``.

        Returns:
            None

        Raises:
            ValueError: If ``state`` and this sampler disagree on whether
                there is a scheduler, or the step is out of range.

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=1)
            >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1], scheduler)
            >>> sampler.load_state_dict({"scheduler": {"step": 1}})
            >>> sampler.sample(rng=0)[0]  # the curriculum's final weights
            0.5
        """
        if self.scheduler is None:
            if state:
                raise ValueError(
                    "state holds scheduler state, but this sampler has no "
                    "scheduler"
                )
            return
        if "scheduler" not in state:
            raise ValueError(
                "this sampler has a scheduler, but state holds no scheduler "
                "state"
            )
        self.scheduler.load_state_dict(state["scheduler"])

    def sample_from_scheduler(
        self, rng: np.random.Generator | int | None = None
    ) -> int:
        """Draw a candidate index from the scheduler's current weights.

        Args:
            rng (np.random.Generator | int | None): Generator to draw from,
                or a seed to build one.

        Returns:
            int: An index in ``[0, N)``. All-zero weights fall back to a
            uniform draw.

        Raises:
            ValueError: If the scheduler returns negative weights.

        Example:
            >>> from takumi.masking import LinearScheduler
            >>> scheduler = LinearScheduler([0.0, 1.0], [0.0, 1.0], total_steps=1)
            >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1], scheduler)
            >>> sampler.sample_from_scheduler(rng=0)
            1
        """
        rng = np.random.default_rng(rng)
        weights = np.asarray(self.scheduler.sample(), dtype=np.float64)
        ensure_non_negative(weights, name="scheduler weights")
        total = weights.sum()
        if total > 0:
            return int(rng.choice(self.num_candidates, p=weights / total))
        # All-zero weights: fall back to uniform.
        return int(rng.integers(0, self.num_candidates))

    def sample_randomly(
        self, rng: np.random.Generator | int | None = None
    ) -> int:
        """Draw a candidate index uniformly at random.

        Args:
            rng (np.random.Generator | int | None): Generator to draw from,
                or a seed to build one.

        Returns:
            int: An index in ``[0, N)``.

        Example:
            >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 0.8], [0.1, 0.1])
            >>> sampler.sample_randomly(rng=0) in (0, 1)
            True
        """
        rng = np.random.default_rng(rng)
        return int(rng.integers(0, self.num_candidates))

    def sample(
        self,
        rng: np.random.Generator | int | None = None,
        index: int | None = None,
    ) -> typing.Tuple[float, float, float]:
        """Pick a candidate triple: pinned, scheduled, or uniform.

        Args:
            rng (np.random.Generator | int | None): Generator to draw from,
                or a seed to build one; unused when ``index`` is given.
            index (int | None): Pins the candidate, in ``[0, N)``.

        Returns:
            tuple[float, float, float]: ``(mlm_prob, masking_prob,
            mutation_prob)``.

        Raises:
            ValueError: If ``index`` is out of range.

        Example:
            >>> sampler = MLMProbabilitySampler([0.1, 0.5], [0.8, 1.0], [0.1, 0.0])
            >>> sampler.sample(index=1)
            (0.5, 1.0, 0.0)
        """
        if index is not None:
            if index < 0 or index >= self.num_candidates:
                raise ValueError(
                    f"index {index} out of range [0, {self.num_candidates})"
                )
            idx = index
        elif self.scheduler is not None:
            idx = self.sample_from_scheduler(rng=rng)
        else:
            idx = self.sample_randomly(rng=rng)

        return (
            float(self.mlm_probs[idx]),
            float(self.masking_probs[idx]),
            float(self.mutation_probs[idx]),
        )
