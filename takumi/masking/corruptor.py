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

    Attributes:
        sequences (ArrayLike of int, shape (B, L)): The corrupted tokens.
        labels (ArrayLike of int, shape (B, L)): The original token at every
            selected position, ``-100`` elsewhere. Unsigned inputs come back
            as int64 so ``-100`` fits.
        masked (ArrayLike of bool, shape (B, L)): Replaced with the mask
            token.
        mutated (ArrayLike of bool, shape (B, L)): Replaced with a random
            valid token.
        kept (ArrayLike of bool, shape (B, L)): Selected but unchanged.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import Bernoulli, FixedRate, MaskOnly
        >>> corruptor = Corruptor(FixedRate(0.5), Bernoulli(), MaskOnly(3), [0])
        >>> sequences, labels, masked, mutated, kept = corruptor(np.full((2, 4), 7), seed=0)
        >>> bool(((labels != -100) == (masked | mutated | kept)).all())
        True
    """

    sequences: backend.ArrayLike
    labels: backend.ArrayLike
    masked: backend.ArrayLike
    mutated: backend.ArrayLike
    kept: backend.ArrayLike


class Corruptor:
    """Composable MLM corruptor built from three orthogonal strategies.

    A corruption call is always the same pipeline, with each stage swappable:

    1. **rate** (`RatePolicy`) — the fraction of eligible positions to
       select, per call or per row (fixed / Beta / sampler / diffusion
       schedule).
    2. **select** (`Selector`) — which eligible positions to take given the
       rate (independent Bernoulli / focus-biased / contiguous spans /
       exact count).
    3. **corrupt** (`CorruptionScheme`) — what to do with them (80/10/10
       mixed / mask-only).

    Every legacy masker is one composition of these; see
    `takumi.masking.presets` for the named shortcuts.

    The corruptor is tokenizer-agnostic: the caller supplies
    ``special_token_ids`` (never corrupted) here, and the substitution
    vocabulary / mask id live on the corruption scheme.

    Args:
        rate (RatePolicy): How much to select.
        select (Selector): Which positions to select.
        corrupt (CorruptionScheme): What to do with them.
        special_token_ids (Sequence[int]): Ids that must never be masked.
            Pass ``[]`` for none. Include every "do not corrupt" token (pad,
            bos/eos, unknown, the mask id itself, tokenizer glyphs).

    Raises:
        TypeError: If a strategy has the wrong base class.
        ValueError: If ``special_token_ids`` is None.

    Example:
        >>> import numpy as np
        >>> from takumi.masking import BetaRate, Bernoulli, Mixed
        >>> corruptor = Corruptor(
        ...     rate=BetaRate(mean=0.15, concentration=4.0),
        ...     select=Bernoulli(),
        ...     corrupt=Mixed(0.8, 0.1, mask_token_id=3, valid_token_ids=range(4, 100)),
        ...     special_token_ids=[0, 1, 2, 3],
        ... )
        >>> tokens = np.array([[1, 40, 41, 42, 43, 44, 45, 2]])
        >>> result = corruptor(tokens, seed=0)
        >>> result.sequences.shape, int(result.labels[0, 0])  # bos never selected
        ((1, 8), -100)
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
        timestep: typing.Union[int, float, backend.ArrayLike, None] = None,
        index: int | None = None,
    ) -> MaskingResult:
        """Corrupt a batch of token sequences.

        Args:
            sequences (ArrayLike of int, shape (B, L)): Token ids; NumPy
                array or CPU torch/JAX tensor. The same type is returned and
                the input is never modified.
            seed (int | None): If given, makes the whole call bit-exact
                reproducible.
            excluded_ids (Iterable[int] | None): Overrides
                ``special_token_ids`` for this call.
            focus (ArrayLike of bool, shape (B, L) | None): Positions to bias
                selection toward; honored per the selector. None disables it.
            timestep (int | float | ArrayLike of float, shape (B,) | None):
                Required by `ScheduleRate`, as a scalar or one per row;
                ignored by other rates.
            index (int | None): Pins a candidate for `SampledRate`; ignored
                otherwise.

        Returns:
            MaskingResult: ``(sequences, labels, masked, mutated, kept)``,
            all of shape ``(B, L)`` and the input's type. ``labels`` holds
            the original token at every selected position and ``-100``
            elsewhere; ``masked`` / ``mutated`` / ``kept`` are boolean masks
            partitioning the selected positions by outcome.

        Raises:
            ValueError: If ``focus`` has the wrong shape, the rate policy
                needs an argument that is missing, or it returns a per-row
                rate whose length is not ``B``.

        Example:
            >>> import numpy as np
            >>> from takumi.masking import Bernoulli, FixedRate, MaskOnly
            >>> corruptor = Corruptor(FixedRate(0.5), Bernoulli(), MaskOnly(3), [0])
            >>> tokens = np.array([[0, 7, 8, 9, 10, 11]])
            >>> a = corruptor(tokens, seed=1)
            >>> b = corruptor(tokens, seed=1)
            >>> bool((a.sequences == b.sequences).all())  # same seed, same result
            True
            >>> bool(a.masked[0, 0])  # special id 0 is never selected
            False
        """
        input_is_np = isinstance(sequences, np.ndarray)
        arr = backend.to_numpy(sequences).copy()

        focus_arr = coerce_focus(focus, arr.shape)
        seed = int(seed) if seed is not None else None
        rng = np.random.default_rng(seed)
        ctx = CallContext(
            rng=rng,
            seed=seed,
            timestep=timestep,
            index=index,
            batch_size=arr.shape[0],
        )

        rate = self.rate(ctx)
        if np.ndim(rate) != 0 and np.shape(rate) != (arr.shape[0],):
            raise ValueError(
                f"per-row rate must have shape ({arr.shape[0]},), got "
                f"{np.shape(rate)}"
            )

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
