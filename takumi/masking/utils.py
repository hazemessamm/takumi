import hashlib
import typing

import numpy as np


def build_excluded_mask(
    sequences: np.ndarray,
    excluded_ids: typing.Iterable[int],
) -> np.ndarray:
    """Return a boolean mask that is True at positions holding an excluded id.

    Vectorized membership test over `excluded_ids` (any iterable, including
    an empty one, in which case the mask is all-False).

    Args:
        sequences (np.ndarray of int, shape (B, L)): Token ids.
        excluded_ids (Iterable[int]): Ids to flag.

    Returns:
        np.ndarray of bool, shape (B, L): True where the token is excluded.

    Example:
        >>> import numpy as np
        >>> build_excluded_mask(np.array([[0, 5, 1, 5]]), [0, 1]).tolist()
        [[True, False, True, False]]
    """
    excluded = np.fromiter(excluded_ids, dtype=np.int64)
    return np.isin(sequences, excluded)


def labels_like(sequences: np.ndarray) -> np.ndarray:
    """Return a copy of `sequences` safe to hold the -100 ignore index.

    Unsigned integer inputs are upcast to int64 so the sentinel does not
    wrap around; signed inputs keep their dtype.

    Args:
        sequences (np.ndarray of int, shape (B, L)): Token ids.

    Returns:
        np.ndarray of int, shape (B, L): A copy; int64 if the input was
        unsigned.

    Example:
        >>> import numpy as np
        >>> labels = labels_like(np.array([[1, 2]], dtype=np.uint16))
        >>> labels[0, 0] = -100
        >>> labels.dtype.name, labels.tolist()
        ('int64', [[-100, 2]])
    """
    if np.issubdtype(sequences.dtype, np.unsignedinteger):
        return sequences.astype(np.int64)
    return sequences.copy()


def ensure_non_negative(values, name: str = "values"):
    """Raise ValueError if any element of `values` is negative.

    Accepts scalars or any iterable that NumPy can convert into a numeric
    array. Returns the original `values` unchanged so it can be used inline.

    Args:
        values (float | Sequence[float] | np.ndarray, any shape): Values to
            check.
        name (str): Name used in the error message.

    Returns:
        The input ``values``, unchanged.

    Raises:
        ValueError: If any element is negative.

    Example:
        >>> ensure_non_negative([0.1, 0.0])
        [0.1, 0.0]
        >>> ensure_non_negative([0.1, -1.0], name="weights")
        Traceback (most recent call last):
        ...
        ValueError: weights must contain only non-negative values; got [0.1, -1.0]
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size > 0 and (arr < 0).any():
        raise ValueError(
            f"{name} must contain only non-negative values; got {values}"
        )
    return values


def seed_from_components(*components: int) -> int:
    """Derive a stable 64-bit seed from several integers.

    Hashes the components with BLAKE2b, so the same components give the same
    seed on every platform and Python version (unlike ``hash()``). Useful
    for per-sample seeds such as ``(base_seed, epoch, index)``.

    Args:
        *components (int): Values to combine; order matters.

    Returns:
        int: A seed in ``[0, 2**64)``.

    Example:
        >>> seed_from_components(42, 0, 7) == seed_from_components(42, 0, 7)
        True
        >>> seed_from_components(42, 0, 7) == seed_from_components(42, 1, 7)
        False
    """
    seed_material = ":".join(str(component) for component in components)
    return int.from_bytes(
        hashlib.blake2b(
            seed_material.encode("utf-8"),
            digest_size=8,
        ).digest(),
        byteorder="big",
    )


def create_default_mlm_weights(
    lower: float,
    upper: float,
    increment: float,
) -> np.ndarray:
    """Evenly spaced candidate rates from ``lower`` to ``upper``, inclusive.

    Args:
        lower (float): First value, ``>= 0``.
        upper (float): Last value, ``>= 0``.
        increment (float): Spacing; rounded so both ends are included.

    Returns:
        np.ndarray of float, shape (N,): ``round((upper - lower) /
        increment) + 1`` values.

    Raises:
        ValueError: If ``lower`` or ``upper`` is negative.

    Example:
        >>> create_default_mlm_weights(0.1, 0.5, 0.1).round(2).tolist()
        [0.1, 0.2, 0.3, 0.4, 0.5]
    """
    ensure_non_negative([lower, upper], name="lower/upper")
    num_steps = int(round((upper - lower) / increment)) + 1
    return np.linspace(lower, upper, num_steps)
