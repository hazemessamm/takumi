"""Masked-diffusion training loss from per-token losses and ELBO weights."""

from __future__ import annotations

import typing

from takumi.masking import backend


def elbo_loss(
    token_loss: backend.ArrayLike,
    labels: backend.ArrayLike,
    weights: backend.ArrayLike,
    *,
    attention_mask: typing.Optional[backend.ArrayLike] = None,
) -> backend.ArrayLike:
    """Masked-diffusion training loss: the NELBO per token.

    Multiplies each labelled position's loss (``labels != -100``) by its
    ELBO weight, sums, and divides by the number of real tokens. The weight
    scales the masked tokens' loss up to an estimate for the whole sequence,
    so dividing by the token count gives the NELBO per token, an upper bound
    on the per-token negative log-likelihood (MDLM, Sahoo et al., 2024). Use
    it with the maskers whose results carry timesteps (diffusion, MaskGIT).

    Every input is flattened, so each may be ``(B*L,)``, the layout of
    ``F.cross_entropy(logits.view(-1, V), labels.view(-1),
    reduction="none")``, or ``(B, L)``, like ``MaskingResult.weights``.

    Not for MLM: its weights are 1 and its loss is the mean over the
    selected tokens, ``F.cross_entropy(..., ignore_index=-100)``. Dividing
    by every token, as here, would scale that by the masking fraction.

    Padding carries ``-100`` like any unmasked token, so only
    ``attention_mask`` keeps it out of the token count; without one, a
    padded batch comes out low by its padding fraction.

    ``labels``, ``weights`` and ``attention_mask`` may sit on another device
    than ``token_loss``; they are moved to it.

    Every input is a torch tensor or JAX array, matching the backend.

    Args:
        token_loss (Tensor of float, shape (B*L,) or (B, L)): Loss at each
            position, e.g. ``F.cross_entropy(logits.view(-1, V),
            labels.view(-1), ignore_index=-100, reduction="none")``. Values
            at unlabelled positions are ignored.
        labels (Tensor of int, shape (B*L,) or (B, L)): ``-100`` where a
            position takes no loss, e.g. ``MaskingResult.labels``.
        weights (Tensor of float, shape (B*L,) or (B, L)): Each position's
            ELBO weight, e.g. ``MaskingResult.weights``.
        attention_mask (Tensor, shape (B*L,) or (B, L) | None): Nonzero at
            real tokens; positions outside it take no loss and are not
            counted. None counts every position as a token.

    Returns:
        Tensor: The float32 scalar loss, on ``token_loss``'s device and
        differentiable with respect to it. 0 when there are no tokens.

    Raises:
        ValueError: If ``token_loss`` does not hold one loss per position,
            such as a ``reduction="mean"`` scalar.

    Example:
        >>> t = backend.to_tensor
        >>> labels = t([5, -100, 6, 7, -100, -100])  # B = 2, L = 3
        >>> ce = t([2.0] * 6)  # stand-in per-token cross-entropy
        >>> weights = t([[1.0] * 3, [4.0] * 3])  # (B, L), per token
        >>> loss = elbo_loss(ce, labels, weights)
        >>> float(backend.to_numpy(loss))  # (1 * (2 + 2) + 4 * 2) / 6 tokens
        2.0
        >>> mask = t([1, 1, 1, 1, 0, 0])  # row 2 ends in padding
        >>> padded = elbo_loss(ce, labels, weights, attention_mask=mask)
        >>> float(backend.to_numpy(padded))  # 12 / 4 real tokens
        3.0
    """
    weighted, tokens = backend.loss_sums(
        token_loss, labels, weights, attention_mask
    )
    # No tokens means nothing was summed either: return 0, not 0 / 0.
    return weighted / (tokens + (tokens == 0))
