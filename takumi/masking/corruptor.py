from __future__ import annotations

import typing

import numpy as np

from takumi.masking import backend
from takumi.masking.context import CallContext
from takumi.masking.rates import RatePolicy
from takumi.masking.schemes import CorruptionScheme
from takumi.masking.selectors import Selector
from takumi.masking.selectors import coerce_focus
from takumi.masking.utils import build_excluded_mask
from takumi.masking.utils import labels_like


class MaskingResult(typing.NamedTuple):
    """Everything a corruption call produces.

    ``masked``, ``mutated``, and ``kept`` are boolean masks that partition
    the selected positions (mutually exclusive; their union is the full set
    of positions where ``labels != -100``). All five fields share the input's
    shape and type.
    """

    sequences: backend.ArrayLike
    labels: backend.ArrayLike
    masked: backend.ArrayLike
    mutated: backend.ArrayLike
    kept: backend.ArrayLike


class Corruptor:
    """Composable MLM corruptor built from three orthogonal strategies.

    A corruption call is always the same pipeline, with each stage swappable:

    1. **rate** (`RatePolicy`) — the per-call fraction of eligible positions
       to select (fixed / Beta / sampler / diffusion schedule).
    2. **select** (`Selector`) — which eligible positions to take given the
       rate (independent Bernoulli / focus-biased / contiguous spans).
    3. **corrupt** (`CorruptionScheme`) — what to do with them (80/10/10
       mixed / mask-only).

    Every legacy masker is one composition of these; see
    `takumi.masking.presets` for the named shortcuts.

    The corruptor is tokenizer-agnostic: the caller supplies
    ``special_token_ids`` (never corrupted) here, and the substitution
    vocabulary / mask id live on the corruption scheme.

    Args:
        rate: The rate policy.
        select: The position selector.
        corrupt: The corruption scheme.
        special_token_ids: Ids that must never be masked. Pass ``[]`` for
            none. Include every "do not corrupt" token (pad, bos/eos,
            unknown, the mask id itself, tokenizer glyphs).
    """

    def __init__(
        self,
        rate: RatePolicy,
        select: Selector,
        corrupt: CorruptionScheme,
        special_token_ids: typing.Sequence[int],
    ):
        if not isinstance(rate, RatePolicy):
            raise TypeError("rate must be a RatePolicy")
        if not isinstance(select, Selector):
            raise TypeError("select must be a Selector")
        if not isinstance(corrupt, CorruptionScheme):
            raise TypeError("corrupt must be a CorruptionScheme")
        if special_token_ids is None:
            raise ValueError(
                "special_token_ids must be provided (use [] for none)"
            )
        self.rate = rate
        self.select = select
        self.corrupt = corrupt
        self.special_token_ids = np.asarray(special_token_ids, dtype=np.int64)

    def __call__(
        self,
        sequences: backend.ArrayLike,
        seed: int | None = None,
        excluded_ids: typing.Iterable[int] | None = None,
        focus: typing.Optional[backend.ArrayLike] = None,
        timestep: typing.Union[int, float, None] = None,
        index: int | None = None,
    ) -> MaskingResult:
        """Corrupt a batch of token sequences.

        Args:
            sequences: ``(B, L)`` token ids; NumPy array or CPU torch/JAX
                tensor. The same type is returned.
            seed: If given, makes the whole call bit-exact reproducible.
            excluded_ids: Overrides ``special_token_ids`` for this call.
            focus: Optional ``(B, L)`` bool mask of positions to bias
                selection toward; honored per the selector. None disables it.
            timestep: Required by `ScheduleRate`; ignored by other rates.
            index: Pins a candidate for `SampledRate`; ignored otherwise.

        Returns:
            A `MaskingResult` — ``(sequences, labels, masked, mutated,
            kept)`` — all sharing the input's shape and type. ``labels``
            holds the original token at every selected position and ``-100``
            elsewhere; ``masked`` / ``mutated`` / ``kept`` are boolean masks
            partitioning the selected positions by outcome.
        """
        input_is_np = isinstance(sequences, np.ndarray)
        arr = sequences if input_is_np else backend.to_numpy(sequences)
        arr = arr.copy()

        focus_arr = coerce_focus(focus, arr.shape)
        seed = int(seed) if seed is not None else None
        rng = np.random.default_rng(seed)
        ctx = CallContext(
            rng=rng, seed=seed, timestep=timestep, index=index
        )

        rate = self.rate(ctx)

        excluded = (
            self.special_token_ids if excluded_ids is None else excluded_ids
        )
        eligible = ~build_excluded_mask(arr, excluded)

        selected = self.select(rng, rate, eligible, focus_arr)

        labels = labels_like(arr)
        labels[~selected] = -100

        masked, mutated, kept = self.corrupt(rng, arr, selected)

        if not input_is_np:
            arr = backend.to_tensor(arr)
            labels = backend.to_tensor(labels)
            masked = backend.to_tensor(masked)
            mutated = backend.to_tensor(mutated)
            kept = backend.to_tensor(kept)
        return MaskingResult(arr, labels, masked, mutated, kept)
