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
    """

    def __call__(
        self,
        rng: np.random.Generator,
        sequences: np.ndarray,
        selected: np.ndarray,
    ) -> SplitMasks:
        raise NotImplementedError


class MaskOnly(CorruptionScheme):
    """Replace every selected position with the mask token.

    The absorbing-state corruption used by mask-token diffusion (D3PM,
    MaskGIT): no 80/10/10 split, no random substitution.
    """

    def __init__(self, mask_token_id: int):
        self.mask_token_id = int(mask_token_id)

    def __call__(self, rng, sequences, selected):
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
        mask_prob: Fraction of selected positions set to the mask token
            (typically 0.8).
        mutate_prob: Fraction replaced with a random valid token (typically
            0.1).
        mask_token_id: Id used for the mask branch.
        valid_token_ids: Pool of substitution targets; the caller should
            exclude special tokens from this set.
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
