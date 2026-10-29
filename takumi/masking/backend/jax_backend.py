"""JAX backend: conversions plus on-device token sampling."""

from __future__ import annotations

import typing

import jax
import numpy as np


def to_tensor(x) -> jax.numpy.ndarray:
    """Wrap an array as a JAX array.

    Args:
        x (np.ndarray | jax.Array, any shape): Data to wrap. A JAX array is
            returned unchanged.

    Returns:
        jax.Array: Same shape and values as ``x``, on the default device.

    Example:
        >>> import numpy as np
        >>> a = to_tensor(np.array([[1, 2], [3, 4]]))
        >>> a.shape
        (2, 2)
    """
    if isinstance(x, jax.numpy.ndarray):
        return x
    return jax.numpy.asarray(x)


def to_numpy(x: jax.numpy.ndarray) -> np.ndarray:
    """Copy a JAX array to the host as a NumPy array.

    Args:
        x (jax.Array | np.ndarray, any shape): Array on any device. A NumPy
            array is returned unchanged.

    Returns:
        np.ndarray: Same shape and values as ``x``.

    Example:
        >>> import jax.numpy as jnp
        >>> to_numpy(jnp.array([[1, 2], [3, 4]])).tolist()
        [[1, 2], [3, 4]]
    """
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def framework():
    """Name of this backend's framework.

    Returns:
        str: Always ``"jax"``.

    Example:
        >>> framework()
        'jax'
    """
    return "jax"


def sample(
    logits: jax.numpy.ndarray,
    mask_token_id: typing.Optional[int] = None,
    temperature: float = 1.0,
    rng: typing.Union[np.random.Generator, int, None] = None,
) -> typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """Sample one token per position from logits, on the logits' device.

    Produces the ``sampled`` / ``probs`` pair `MaskGITDecoder` consumes, so
    the ``(B, L, V)`` logits never leave the device.

    Args:
        logits (jax.Array | np.ndarray of float, shape (B, L, V)): Model
            output scores; any leading shape works, ``V`` must be last.
        mask_token_id (int | None): If given, this id is never sampled. An
            id ``>= V`` (a mask token the output layer does not predict)
            needs no blocking and is ignored.
        temperature (float): Sampling temperature, ``>= 0``. ``0`` takes the
            argmax; higher values flatten the distribution.
        rng (np.random.Generator | int | None): Generator the PRNG key's
            seed is drawn from, or a seed to build one. Pass
            ``decoder.rng`` so sampling shares the decoder's stream. ``None``
            draws a fresh key, so repeated calls differ.

    Returns:
        tuple[jax.Array, jax.Array]: ``(sampled, probs)``, both of shape
        ``(B, L)``. ``sampled`` holds integer token ids; ``probs`` holds the
        float32 probability of each sampled token under the untempered
        softmax, so it stays meaningful as a confidence at any temperature.

    Raises:
        ValueError: If ``temperature`` is negative.

    Example:
        >>> import jax.numpy as jnp
        >>> logits = jnp.array([[[0.0, 5.0, 1.0], [9.0, 0.0, 0.0]]])
        >>> sampled, probs = sample(logits, temperature=0.0)
        >>> sampled.tolist()
        [[1, 0]]
        >>> sampled, _ = sample(logits, mask_token_id=0, temperature=0.0)
        >>> sampled.tolist()
        [[1, 1]]
    """
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    logits = to_tensor(logits).astype(jax.numpy.float32)
    if mask_token_id is not None and mask_token_id < logits.shape[-1]:
        logits = logits.at[..., mask_token_id].set(-jax.numpy.inf)

    if temperature == 0.0:
        sampled = jax.numpy.argmax(logits, axis=-1)
    else:
        # A 64-bit key from two 32-bit draws: without x64, jax.random.key
        # keeps only the low 32 bits of an int seed, so fold in the rest.
        low, high = np.random.default_rng(rng).integers(
            0, 2**32, size=2, dtype=np.uint32
        )
        key = jax.random.fold_in(jax.random.key(int(low)), int(high))
        sampled = jax.random.categorical(key, logits / temperature, axis=-1)

    probs = jax.nn.softmax(logits, axis=-1)
    probs = jax.numpy.take_along_axis(probs, sampled[..., None], axis=-1)
    return sampled, probs[..., 0]


def token_probs(
    logits: jax.numpy.ndarray, tokens: jax.numpy.ndarray
) -> jax.numpy.ndarray:
    """Probability the model gives each given token, on the logits' device.

    Feeds `MaskGITDecoder`'s ``current_probs``: how much the model still
    trusts each token already in the sequence.

    Args:
        logits (jax.Array | np.ndarray of float, shape (B, L, V)): Model
            output scores.
        tokens (jax.Array | np.ndarray of int, shape (B, L)): Token at each
            position.

    Returns:
        jax.Array of float32, shape (B, L): Softmax probability of each
        token. Positions holding an id ``>= V``, such as a mask id the
        output layer does not predict, get 0.

    Example:
        >>> import jax.numpy as jnp
        >>> logits = jnp.array([[[0.0, 0.0], [0.0, 9.0]]])
        >>> probs = token_probs(logits, jnp.array([[0, 1]]))
        >>> [round(p, 3) for p in probs[0].tolist()]
        [0.5, 1.0]
        >>> token_probs(logits, jnp.array([[2, 2]])).tolist()  # 2 >= V
        [[0.0, 0.0]]
    """
    logits = to_tensor(logits).astype(jax.numpy.float32)
    tokens = to_tensor(tokens).astype(jax.numpy.int32)
    vocab = logits.shape[-1]
    probs = jax.nn.softmax(logits, axis=-1)
    safe = jax.numpy.minimum(tokens, vocab - 1)
    picked = jax.numpy.take_along_axis(probs, safe[..., None], axis=-1)
    return jax.numpy.where(tokens < vocab, picked[..., 0], 0.0)


def loss_sums(
    token_loss: jax.numpy.ndarray,
    labels: jax.numpy.ndarray,
    weights: jax.numpy.ndarray,
    attention_mask: typing.Optional[jax.numpy.ndarray] = None,
) -> typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """Weighted loss and token count for `elbo_loss`, on the loss's device.

    Flattens every input and sums in float32, so a bfloat16 loss works as
    it comes. Safe to trace under ``jax.jit`` and ``jax.grad``.

    Args:
        token_loss (jax.Array of float, shape (B*L,) or (B, L)): Loss at
            each position.
        labels (jax.Array of int, shape (B*L,) or (B, L)): ``-100`` where a
            position takes no loss.
        weights (jax.Array of float, shape (B*L,) or (B, L)): Each
            position's loss weight.
        attention_mask (jax.Array, shape (B*L,) or (B, L) | None): Nonzero
            at real tokens; positions outside it take no loss and are not
            counted. None counts every position.

    Returns:
        tuple[jax.Array, jax.Array]: Two scalars: the weighted loss summed
        over the labelled positions, and the number of real tokens.

    Raises:
        ValueError: If ``token_loss`` does not hold one loss per position,
            such as a ``reduction="mean"`` scalar.

    Example:
        >>> import jax.numpy as jnp
        >>> loss = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> labels = jnp.array([5, -100, 5, 5])
        >>> weights = jnp.array([[2.0, 2.0], [1.0, 1.0]])  # (B, L)
        >>> [s.item() for s in loss_sums(loss, labels, weights)]
        [9.0, 4]
    """
    loss = token_loss.reshape(-1).astype(jax.numpy.float32)
    labels = labels.reshape(-1)
    if loss.shape != labels.shape:
        raise ValueError(
            "token_loss must hold one loss per position; use reduction='none'"
        )
    weights = weights.reshape(-1).astype(jax.numpy.float32)
    selected = labels != -100
    if attention_mask is None:
        tokens = jax.numpy.asarray(loss.size)
    else:
        real = attention_mask.reshape(-1).astype(bool)
        selected = selected & real
        tokens = real.sum()
    return (jax.numpy.where(selected, loss, 0.0) * weights).sum(), tokens


ArrayLike: typing.TypeAlias = jax.numpy.ndarray
