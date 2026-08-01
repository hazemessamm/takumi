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
    """
    excluded = np.fromiter(excluded_ids, dtype=np.int64)
    return np.isin(sequences, excluded)


def labels_like(sequences: np.ndarray) -> np.ndarray:
    """Return a copy of `sequences` safe to hold the -100 ignore index.

    Unsigned integer inputs are upcast to int64 so the sentinel does not
    wrap around; signed inputs keep their dtype.
    """
    if np.issubdtype(sequences.dtype, np.unsignedinteger):
        return sequences.astype(np.int64)
    return sequences.copy()


def ensure_non_negative(values, name: str = "values"):
    """Raise ValueError if any element of `values` is negative.

    Accepts scalars or any iterable that NumPy can convert into a numeric
    array. Returns the original `values` unchanged so it can be used inline.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size > 0 and (arr < 0).any():
        raise ValueError(
            f"{name} must contain only non-negative values; got {values}"
        )
    return values


def seed_from_components(*components: int) -> int:
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
    ensure_non_negative([lower, upper], name="lower/upper")
    num_steps = int(round((upper - lower) / increment)) + 1
    return np.linspace(lower, upper, num_steps)
