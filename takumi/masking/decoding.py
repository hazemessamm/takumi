"""Inference-side counterpart to `MaskGITMasker`: iterative re-masking."""

from __future__ import annotations

import typing

import numpy as np

from takumi.masking import backend
from takumi.masking.schedules import resolve_schedule


def _lowest(scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mark the ``counts[b]`` lowest scores in each row ``b``.

    Args:
        scores (np.ndarray of float, shape (B, L)): Scores to rank; use
            ``inf`` for positions that must not be picked.
        counts (np.ndarray of int, shape (B,)): How many to mark per row.

    Returns:
        np.ndarray of bool, shape (B, L): True at the lowest-scoring
        positions; ties go to the earlier position.

    Example:
        >>> import numpy as np
        >>> scores = np.array([[0.5, 0.1, np.inf, 0.3]])
        >>> _lowest(scores, np.array([2])).tolist()
        [[False, True, False, True]]
    """
    ranks = scores.argsort(axis=1, kind="stable").argsort(axis=1)
    return ranks < counts[:, None]


class MaskGITDecoder:
    """One step of MaskGIT's iterative parallel decoding.

    Generation starts from sequences whose unknown positions hold the mask
    token and runs ``total_steps`` steps. Each step the model predicts every
    position; the caller samples a token per position and passes those
    tokens with their probabilities. The step commits the samples, then
    re-masks the least confident of the positions that were unknown, so
    that the masked fraction follows the schedule. Known positions are
    never re-masked, and the last step re-masks nothing.

    Confidence is ``log(prob)`` plus Gumbel noise scaled by
    ``temperature * (1 - progress)``, so early steps pick which tokens to
    keep more randomly and later ones more greedily.

    `backend.sample` turns ``(B, L, V)`` logits into the ``sampled`` /
    ``probs`` pair on the model's device.

    Plain MaskGIT never revisits a committed token, so an early mistake
    stays. ``remask_prob`` turns on ReMDM-style remasking: on steps whose
    progress ``(step + 1) / total_steps`` falls inside ``remask_window``,
    each token committed on an earlier step is sent back to the mask with
    probability ``remask_prob`` and re-predicted later. Pass
    ``current_probs`` (from `backend.token_probs`) to re-mask the tokens the
    model now trusts least instead of random ones. Remasked tokens cost
    extra predictions, so raise ``total_steps`` when turning it on. The last
    step never remasks.

    Args:
        mask_token_id (int): Id that marks unknown positions.
        total_steps (int): Number of decoding steps, ``> 0``.
        schedule (str | MaskSchedule | Callable[[float, float], float]):
            Masking schedule; use the one the model was trained with (see
            `MaskGITMasker`).
        temperature (float): Scale of the confidence noise, ``>= 0``. 4.5 is
            MaskGIT's ImageNet setting; 0 keeps exactly the most confident
            samples.
        remask_prob (float): Per-step probability, in ``[0, 1]``, that a
            committed token is re-masked. ``0`` (the default) is plain
            MaskGIT.
        remask_window (tuple[float, float]): ``(start, end)`` progress range,
            ``0 <= start <= end <= 1``, in which remasking applies.

    Raises:
        ValueError: If ``total_steps`` is not positive, ``temperature`` is
            negative, ``remask_prob`` is outside ``[0, 1]``, or
            ``remask_window`` is not an ordered range within ``[0, 1]``.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import backend
        >>> decoder = MaskGITDecoder(mask_token_id=3, total_steps=4)
        >>> tokens = np.full((2, 8), 3)  # start fully masked
        >>> rng = np.random.default_rng(0)
        >>> for step in range(4):
        ...     logits = rng.normal(size=(2, 8, 16))  # stand-in for model(tokens)
        ...     sampled, probs = backend.sample(logits, mask_token_id=3, seed=step)
        ...     tokens = decoder(tokens, sampled, probs, step, num_unknown=8, seed=step)
        ...     print(step, int((tokens == 3).sum()))  # floor(8 * cos(...)) per row
        0 14
        1 10
        2 6
        3 0

        With remasking between 25% and 75% of the way through:

        >>> decoder = MaskGITDecoder(3, total_steps=8, remask_prob=0.3)
        >>> tokens = np.full((2, 8), 3)
        >>> generated = tokens == 3  # positions being generated
        >>> for step in range(8):
        ...     logits = rng.normal(size=(2, 8, 16))
        ...     sampled, probs = backend.sample(logits, mask_token_id=3, seed=step)
        ...     current = backend.token_probs(logits, tokens)
        ...     tokens = decoder(tokens, sampled, probs, step, num_unknown=8, seed=step,
        ...                      generated=generated, current_probs=current)
        >>> int((tokens == 3).sum())
        0
    """

    def __init__(
        self,
        mask_token_id: int,
        total_steps: int,
        schedule: typing.Union[str, typing.Callable] = "cosine",
        temperature: float = 4.5,
        remask_prob: float = 0.0,
        remask_window: typing.Tuple[float, float] = (0.25, 0.75),
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not (0.0 <= remask_prob <= 1.0):
            raise ValueError(
                f"remask_prob must be in [0, 1], got {remask_prob}"
            )
        start, end = remask_window
        if not (0.0 <= start <= end <= 1.0):
            raise ValueError(
                f"remask_window must satisfy 0 <= start <= end <= 1, got "
                f"{remask_window}"
            )
        self.mask_token_id = int(mask_token_id)
        self.total_steps = int(total_steps)
        self.schedule = resolve_schedule(schedule)
        self.temperature = float(temperature)
        self.remask_prob = float(remask_prob)
        self.remask_window = (float(start), float(end))

    def __call__(
        self,
        sequences: backend.ArrayLike,
        sampled: backend.ArrayLike,
        probs: backend.ArrayLike,
        step: int,
        num_unknown: typing.Union[int, backend.ArrayLike],
        seed: int | None = None,
        generated: typing.Optional[backend.ArrayLike] = None,
        current_probs: typing.Optional[backend.ArrayLike] = None,
    ) -> backend.ArrayLike:
        """Advance decoding by one step.

        Args:
            sequences (ArrayLike of int, shape (B, L)): Current tokens, with
                the mask token at unknown positions.
            sampled (ArrayLike of int, shape (B, L)): Tokens sampled from
                the model's output.
            probs (ArrayLike of float, shape (B, L)): Probability the model
                gave each sampled token.
            step (int): Index of this step, in ``[0, total_steps)``.
            num_unknown (int | ArrayLike of int, shape (B,)): Unknown
                positions per row before step 0. The schedule is a fraction
                of this.
            seed (int | None): If given, makes the step reproducible.
            generated (ArrayLike of bool, shape (B, L) | None): Positions
                that were unknown before step 0, the only ones remasking may
                touch; prompt and conditioning tokens stay put. Required
                when ``remask_prob > 0``.
            current_probs (ArrayLike of float, shape (B, L) | None):
                Probability the model gives the token currently at each
                position (see `backend.token_probs`). If given, remasking
                picks the least confident committed tokens; otherwise
                random ones.

        Returns:
            ArrayLike of int, shape (B, L): The tokens to feed the next step,
            as the type of ``sequences``.

        Raises:
            ValueError: If ``step`` is outside ``[0, total_steps)``, or
                ``remask_prob > 0`` and ``generated`` is missing.

        Example:
            >>> import numpy as np
            >>> decoder = MaskGITDecoder(mask_token_id=3, total_steps=2, temperature=0.0)
            >>> tokens = np.array([[42, 3, 3, 3, 3, 3]])  # 42 is known
            >>> sampled = np.array([[0, 12, 13, 14, 15, 16]])
            >>> probs = np.array([[0.0, 0.9, 0.1, 0.8, 0.2, 0.7]])
            >>> decoder(tokens, sampled, probs, step=0, num_unknown=5).tolist()
            [[42, 12, 3, 14, 3, 3]]
        """
        if not (0 <= step < self.total_steps):
            raise ValueError(
                f"step must be in [0, {self.total_steps}), got {step}"
            )
        if self.remask_prob > 0.0 and generated is None:
            raise ValueError(
                "remask_prob > 0 needs generated=..., the positions that "
                "were unknown before step 0"
            )
        input_is_np = isinstance(sequences, np.ndarray)
        arr = backend.to_numpy(sequences)
        unknown = arr == self.mask_token_id
        out = np.where(unknown, backend.to_numpy(sampled), arr)

        if step < self.total_steps - 1:
            progress = (step + 1) / self.total_steps
            ratio = self.schedule.rate(1.0 - progress, 1.0)
            mask_len = np.floor(
                np.asarray(num_unknown, dtype=np.float64) * ratio
            )
            # Re-mask at least one position for the next step, but keep at
            # least one of this step's samples.
            mask_len = np.maximum(
                1, np.minimum(unknown.sum(axis=1) - 1, mask_len)
            )

            rng = np.random.default_rng(seed)
            with np.errstate(divide="ignore"):
                confidence = np.log(backend.to_numpy(probs).astype(np.float64))
            noise = self.temperature * (1.0 - progress)
            confidence = confidence + noise * rng.gumbel(size=arr.shape)
            confidence = np.where(unknown, confidence, np.inf)
            out[_lowest(confidence, mask_len) & unknown] = self.mask_token_id

            start, end = self.remask_window
            if self.remask_prob > 0.0 and start <= progress <= end:
                # Only tokens committed on earlier steps; this step's samples
                # were just chosen by confidence above.
                committed = backend.to_numpy(generated).astype(bool) & ~unknown
                if current_probs is None:
                    scores = rng.random(arr.shape)
                else:
                    scores = backend.to_numpy(current_probs).astype(np.float64)
                counts = rng.binomial(committed.sum(axis=1), self.remask_prob)
                scores = np.where(committed, scores, np.inf)
                out[_lowest(scores, counts)] = self.mask_token_id

        return out if input_is_np else backend.to_tensor(out)
