"""Masked-diffusion training loss from per-token losses and ELBO weights."""

from __future__ import annotations

import typing

import numpy as np

from takumi.masking import backend


def elbo_loss(
    token_loss: backend.ArrayLike,
    labels: backend.ArrayLike,
    weights: backend.ArrayLike,
    *,
    attention_mask: typing.Optional[backend.ArrayLike] = None,
) -> backend.ArrayLike:
    """Masked-diffusion training loss: the NELBO per token.

    Takes the per-token inputs flattened, as ``(B*L,)``: the layout of
    ``F.cross_entropy(logits.view(-1, V), labels.view(-1),
    reduction="none")``. ``weights`` holds one value per row, which is how
    the flat inputs are split back into ``B`` rows.

    Sums each row's loss over its labelled positions (``labels != -100``),
    multiplies the sum by the row's ELBO weight, adds up the rows and
    divides by the number of real tokens. The weight scales a row's
    masked-token loss up to an estimate for the whole sequence, so dividing
    by the token count gives the NELBO per token, an upper bound on the
    per-token negative log-likelihood (MDLM, Sahoo et al., 2024). Use it
    with the maskers whose results carry timesteps (diffusion, MaskGIT).

    Not for MLM: its weights are 1 and its loss is the mean over the
    selected tokens, ``F.cross_entropy(..., ignore_index=-100)``. Dividing
    by every token, as here, would scale that by the masking fraction.

    Padding carries ``-100`` like any unmasked token, so only
    ``attention_mask`` keeps it out of the token count; without one, a
    padded batch comes out low by its padding fraction.

    ``labels``, ``weights`` and ``attention_mask`` may sit on another device
    than ``token_loss``; they are moved to it.

    Args:
        token_loss (ArrayLike of float, shape (B*L,)): Loss at each position,
            flattened row by row, e.g. ``F.cross_entropy(logits.view(-1, V),
            labels.view(-1), ignore_index=-100, reduction="none")``. Values
            at unlabelled positions are ignored.
        labels (ArrayLike of int, shape (B*L,)): ``-100`` where a position
            takes no loss, flattened like ``token_loss``, e.g.
            ``MaskingResult.labels.view(-1)``.
        weights (ArrayLike of float, shape (B,)): Each row's ELBO weight, as
            in ``MaskingResult.weights``.
        attention_mask (ArrayLike, shape (B*L,) | None): Nonzero at real
            tokens, flattened like ``token_loss``; positions outside it take
            no loss and are not counted. None counts every position as a
            token.

    Returns:
        ArrayLike: The float32 scalar loss in the backend's type, on
        ``token_loss``'s device and differentiable with respect to it. 0
        when there are no tokens.

    Raises:
        ValueError: If ``token_loss``, ``labels`` or ``attention_mask`` is
            not flat; ``token_loss`` holds a different number of values than
            ``labels`` (for instance, one per row, or a ``reduction="mean"``
            scalar); or ``weights`` is not 1-D or does not split the
            positions into equal rows.

    Example:
        >>> import numpy as np
        >>> labels = np.array([5, -100, 6, 7, -100, -100])  # B = 2, L = 3
        >>> ce = np.full(6, 2.0)  # stand-in per-token cross-entropy
        >>> weights = np.array([1.0, 4.0])
        >>> loss = elbo_loss(ce, labels, weights)
        >>> float(backend.to_numpy(loss))  # (1 * (2 + 2) + 4 * 2) / 6 tokens
        2.0
        >>> mask = np.array([1, 1, 1, 1, 0, 0])  # row 2 ends in padding
        >>> padded = elbo_loss(ce, labels, weights, attention_mask=mask)
        >>> float(backend.to_numpy(padded))  # 12 / 4 real tokens
        3.0
    """
    if np.ndim(weights) != 1:
        raise ValueError(
            "weights must hold one value per row, shape (B,), got shape "
            f"{tuple(np.shape(weights))}"
        )
    if np.ndim(token_loss) != 1:
        raise ValueError(
            "token_loss must be the flattened per-token loss, shape (B*L,), "
            f"got shape {tuple(np.shape(token_loss))}; compute it as "
            "F.cross_entropy(logits.view(-1, V), labels.view(-1), "
            "reduction='none')"
        )
    if np.ndim(labels) != 1:
        raise ValueError(
            "labels must be flattened like token_loss, shape (B*L,), got "
            f"shape {tuple(np.shape(labels))}; pass labels.view(-1)"
        )
    positions = np.shape(token_loss)[0]
    if np.shape(labels)[0] != positions:
        raise ValueError(
            f"token_loss has {positions} values but labels have "
            f"{np.shape(labels)[0]}; it must hold one loss per position, so "
            "compute the cross-entropy with reduction='none'"
        )
    if attention_mask is not None and tuple(np.shape(attention_mask)) != (
        positions,
    ):
        raise ValueError(
            f"attention_mask must be flattened like token_loss, shape "
            f"({positions},), got shape {tuple(np.shape(attention_mask))}"
        )
    rows = np.shape(weights)[0]
    if rows == 0 or positions % rows:
        raise ValueError(
            f"token_loss has {positions} values, which do not split evenly "
            f"into the {rows} rows of weights"
        )

    shape = (rows, positions // rows)
    if attention_mask is not None:
        attention_mask = attention_mask.reshape(shape)
    weighted, tokens = backend.loss_sums(
        token_loss.reshape(shape), labels.reshape(shape), weights,
        attention_mask,
    )
    # No tokens means nothing was summed either: return 0, not 0 / 0.
    return weighted / (tokens + (tokens == 0))
