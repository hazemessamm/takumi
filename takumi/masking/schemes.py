from __future__ import annotations

import typing

import numpy as np

from takumi.masking.utils import ensure_non_negative


class SplitMasks(typing.NamedTuple):
    """Per-position outcome of a corruption scheme over selected positions.

    The three masks partition ``selected``: they are mutually exclusive and
    their union is exactly the selected set.

    - ``masked``: positions replaced with the mask token.
    - ``mutated``: positions replaced with a random valid token.
    - ``kept``: selected positions left unchanged.

    Attributes:
        masked (np.ndarray of bool, shape (B, L)): Set to the mask token.
        mutated (np.ndarray of bool, shape (B, L)): Set to a random token.
        kept (np.ndarray of bool, shape (B, L)): Selected but unchanged.

    Example:
        >>> import numpy as np
        >>> selected = np.array([[True, True, False]])
        >>> split = MaskOnly(3)(np.random.default_rng(0), np.array([[5, 6, 7]]), selected)
        >>> bool(((split.masked | split.mutated | split.kept) == selected).all())
        True
    """

    masked: np.ndarray
    mutated: np.ndarray
    kept: np.ndarray


class CorruptionScheme:
    """Alters the selected positions of a sequence batch in place.

    A scheme is a callable ``(rng, sequences, selected) -> SplitMasks`` that
    mutates ``sequences`` where ``selected`` is True and reports which
    positions fell into each outcome bucket. Labels are captured by the
    corruptor *before* the scheme runs, so a scheme is free to overwrite the
    original tokens. The scheme owns all corruption-specific state (the mask
    token, the substitution vocabulary).

    Example:
        >>> import numpy as np
        >>> class Zero(CorruptionScheme):
        ...     def __call__(self, rng, sequences, selected):
        ...         sequences[selected] = 0
        ...         none = np.zeros_like(selected)
        ...         return SplitMasks(masked=selected, mutated=none, kept=none.copy())
        >>> seq = np.array([[5, 6, 7]])
        >>> _ = Zero()(np.random.default_rng(0), seq, np.array([[False, True, True]]))
        >>> seq.tolist()
        [[5, 0, 0]]
    """

    def __call__(
        self,
        rng: np.random.Generator,
        sequences: np.ndarray,
        selected: np.ndarray,
    ) -> SplitMasks:
        """Corrupt ``sequences`` in place at ``selected``.

        Args:
            rng (np.random.Generator): The call's shared random generator.
            sequences (np.ndarray of int, shape (B, L)): Token ids, modified
                in place.
            selected (np.ndarray of bool, shape (B, L)): Positions to
                corrupt.

        Returns:
            SplitMasks: Which selected positions were masked, mutated, or
            kept.

        Example:
            >>> import numpy as np
            >>> seq = np.array([[5, 6]])
            >>> _ = MaskOnly(3)(np.random.default_rng(0), seq, np.array([[True, False]]))
            >>> seq.tolist()
            [[3, 6]]
        """
        raise NotImplementedError


class MaskOnly(CorruptionScheme):
    """Replace every selected position with the mask token.

    The absorbing-state corruption used by mask-token diffusion (D3PM,
    MaskGIT): no 80/10/10 split, no random substitution.

    Args:
        mask_token_id (int): Id written at every selected position.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Corruptor, FixedRate, Bernoulli
        >>> corruptor = Corruptor(FixedRate(0.5), Bernoulli(), MaskOnly(3), [0])
        >>> result = corruptor(np.full((2, 50), 7), seed=0)
        >>> bool((result.sequences[result.masked] == 3).all()), bool(result.mutated.any())
        (True, False)
    """

    def __init__(self, mask_token_id: int):
        self.mask_token_id = int(mask_token_id)

    def __call__(self, rng, sequences, selected):
        """Write the mask token at every selected position.

        Args:
            rng (np.random.Generator): Unused; part of the scheme contract.
            sequences (np.ndarray of int, shape (B, L)): Token ids, modified
                in place.
            selected (np.ndarray of bool, shape (B, L)): Positions to mask.

        Returns:
            SplitMasks: ``masked`` equals ``selected``; ``mutated`` and
            ``kept`` are all False.

        Example:
            >>> import numpy as np
            >>> seq = np.array([[5, 6, 7]])
            >>> split = MaskOnly(3)(np.random.default_rng(0), seq, np.array([[True, False, True]]))
            >>> seq.tolist(), split.masked.tolist()
            ([[3, 6, 3]], [[True, False, True]])
        """
        sequences[selected] = self.mask_token_id
        none = np.zeros_like(selected)
        return SplitMasks(masked=selected, mutated=none, kept=none.copy())


class Mixed(CorruptionScheme):
    """BERT-style 80/10/10 corruption among the selected positions.

    Each selected position is, independently: replaced with the mask token
    (prob ``mask_prob``), replaced with a random id from ``valid_token_ids``
    (prob ``mutate_prob``), or left unchanged (the remaining
    ``1 - mask_prob - mutate_prob``). Labels still cover every selected
    position regardless of which branch it took.

    Args:
        mask_prob (float): Fraction of selected positions set to the mask
            token (typically 0.8).
        mutate_prob (float): Fraction replaced with a random valid token
            (typically 0.1).
        mask_token_id (int): Id used for the mask branch.
        valid_token_ids (Sequence[int], length >= 1): Pool of substitution
            targets; the caller should exclude special tokens from this set.

    Raises:
        ValueError: If a probability is negative, they sum above 1, or
            ``valid_token_ids`` is empty.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Corruptor, FixedRate, Bernoulli
        >>> scheme = Mixed(0.8, 0.1, mask_token_id=3, valid_token_ids=range(10, 60))
        >>> corruptor = Corruptor(FixedRate(0.5), Bernoulli(), scheme, [0])
        >>> result = corruptor(np.full((100, 100), 7), seed=0)
        >>> n = int((result.labels != -100).sum())
        >>> [round(int(m.sum()) / n, 1) for m in (result.masked, result.mutated, result.kept)]
        [0.8, 0.1, 0.1]
    """

    def __init__(
        self,
        mask_prob: float,
        mutate_prob: float,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
    ):
        ensure_non_negative(
            [mask_prob, mutate_prob], name="(mask_prob, mutate_prob)"
        )
        if mask_prob + mutate_prob > 1.0:
            raise ValueError(
                f"mask_prob + mutate_prob must be <= 1, got "
                f"{mask_prob + mutate_prob}"
            )
        if valid_token_ids is None or len(valid_token_ids) == 0:
            raise ValueError("valid_token_ids must contain at least one id")
        self.mask_prob = float(mask_prob)
        self.mutate_prob = float(mutate_prob)
        self.mask_token_id = int(mask_token_id)
        self.valid_token_ids = np.asarray(valid_token_ids, dtype=np.int64)

    def __call__(self, rng, sequences, selected):
        """Mask, mutate, or keep each selected position.

        Args:
            rng (np.random.Generator): Draws the branch and the substitutes.
            sequences (np.ndarray of int, shape (B, L)): Token ids, modified
                in place.
            selected (np.ndarray of bool, shape (B, L)): Positions to
                corrupt.

        Returns:
            SplitMasks: The branch each selected position took.

        Example:
            >>> import numpy as np
            >>> scheme = Mixed(0.0, 1.0, mask_token_id=3, valid_token_ids=[42])
            >>> seq = np.array([[5, 6, 7]])
            >>> split = scheme(np.random.default_rng(0), seq, np.array([[True, True, False]]))
            >>> seq.tolist(), split.mutated.tolist()
            ([[42, 42, 7]], [[True, True, False]])
        """
        roll = rng.random(sequences.shape)

        masked = selected & (roll < self.mask_prob)
        sequences[masked] = self.mask_token_id

        upper = self.mask_prob + self.mutate_prob
        mutated = selected & (roll >= self.mask_prob) & (roll < upper)
        if mutated.any():
            picks = rng.integers(
                0, len(self.valid_token_ids), sequences.shape
            )
            sequences[mutated] = self.valid_token_ids[picks[mutated]]

        kept = selected & ~masked & ~mutated
        return SplitMasks(masked=masked, mutated=mutated, kept=kept)
