"""PyTorch backend: conversions plus on-device token sampling."""

import typing

import numpy as np
import torch


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """Wrap an array as a torch tensor, sharing memory where possible.

    Args:
        x (np.ndarray | torch.Tensor, any shape): Data to wrap. A tensor is
            returned unchanged.

    Returns:
        torch.Tensor: Same shape and values as ``x``, on the CPU when ``x``
        is a NumPy array.

    Example:
        >>> import numpy as np
        >>> t = to_tensor(np.array([[1, 2], [3, 4]]))
        >>> type(t).__name__, tuple(t.shape)
        ('Tensor', (2, 2))
    """
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Copy a tensor to the CPU and view it as a NumPy array.

    Args:
        x (torch.Tensor | np.ndarray | Sequence, any shape): Tensor on any
            device. A NumPy array is returned unchanged; anything else goes
            through ``np.asarray``.

    Returns:
        np.ndarray: Same shape and values as ``x``.

    Example:
        >>> import torch
        >>> to_numpy(torch.tensor([[1, 2], [3, 4]])).tolist()
        [[1, 2], [3, 4]]
        >>> to_numpy([[1, 2]]).shape
        (1, 2)
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def framework(x):
    """Name of this backend's framework.

    Args:
        x (Any): Ignored; accepted so every backend shares one signature.

    Returns:
        str: Always ``"torch"``.

    Example:
        >>> framework(None)
        'torch'
    """
    return "torch"


def sample(
    logits: torch.Tensor,
    mask_token_id: typing.Optional[int] = None,
    temperature: float = 1.0,
    rng: typing.Union[np.random.Generator, int, None] = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per position from logits, on the logits' device.

    Produces the ``sampled`` / ``probs`` pair `MaskGITDecoder` consumes, so
    the ``(B, L, V)`` logits never leave the device.

    Args:
        logits (torch.Tensor | np.ndarray of float, shape (B, L, V)): Model
            output scores; any leading shape works, ``V`` must be last.
        mask_token_id (int | None): If given, this id is never sampled. An
            id ``>= V`` (a mask token the output layer does not predict)
            needs no blocking and is ignored.
        temperature (float): Sampling temperature, ``>= 0``. ``0`` takes the
            argmax; higher values flatten the distribution.
        rng (np.random.Generator | int | None): Generator the torch seed
            is drawn from, or a seed to build one. Pass ``decoder.rng`` so
            sampling shares the decoder's stream. ``None`` uses torch's
            global RNG.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(sampled, probs)``, both of
        shape ``(B, L)`` on the logits' device. ``sampled`` holds int64 token
        ids; ``probs`` holds the float32 probability of each sampled token
        under the untempered softmax, so it stays meaningful as a confidence
        at any temperature.

    Raises:
        ValueError: If ``temperature`` is negative.

    Example:
        >>> import torch
        >>> logits = torch.tensor([[[0.0, 5.0, 1.0], [9.0, 0.0, 0.0]]])
        >>> sampled, probs = sample(logits, temperature=0.0)
        >>> sampled.tolist()
        [[1, 0]]
        >>> sampled, _ = sample(logits, mask_token_id=0, temperature=0.0)
        >>> sampled.tolist()
        [[1, 1]]
    """
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    logits = to_tensor(logits).float()
    if mask_token_id is not None and mask_token_id < logits.shape[-1]:
        logits = logits.clone()
        logits[..., mask_token_id] = float("-inf")

    if temperature == 0.0:
        sampled = logits.argmax(dim=-1)
    else:
        generator = None
        if rng is not None:
            # torch's CPU generator keeps only the low 32 bits of this seed;
            # CUDA's Philox keeps all 64.
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(
                int(np.random.default_rng(rng).integers(2**63))
            )
        tempered = torch.softmax(logits / temperature, dim=-1)
        flat = tempered.reshape(-1, tempered.shape[-1])
        sampled = torch.multinomial(flat, 1, generator=generator)
        sampled = sampled.reshape(logits.shape[:-1])

    probs = torch.softmax(logits, dim=-1)
    probs = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, probs


def token_probs(
    logits: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """Probability the model gives each given token, on the logits' device.

    Feeds `MaskGITDecoder`'s ``current_probs``: how much the model still
    trusts each token already in the sequence.

    Args:
        logits (torch.Tensor | np.ndarray of float, shape (B, L, V)): Model
            output scores.
        tokens (torch.Tensor | np.ndarray of int, shape (B, L)): Token at
            each position.

    Returns:
        torch.Tensor of float32, shape (B, L): Softmax probability of each
        token. Positions holding an id ``>= V``, such as a mask id the
        output layer does not predict, get 0.

    Example:
        >>> import torch
        >>> logits = torch.tensor([[[0.0, 0.0], [0.0, 9.0]]])
        >>> probs = token_probs(logits, torch.tensor([[0, 1]]))
        >>> [round(p, 3) for p in probs[0].tolist()]
        [0.5, 1.0]
        >>> token_probs(logits, torch.tensor([[2, 2]])).tolist()  # 2 >= V
        [[0.0, 0.0]]
    """
    logits = to_tensor(logits).float()
    tokens = to_tensor(tokens).to(device=logits.device, dtype=torch.long)
    vocab = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)
    picked = probs.gather(-1, tokens.clamp(max=vocab - 1).unsqueeze(-1))
    return torch.where(tokens < vocab, picked.squeeze(-1), 0.0)


def loss_sums(
    token_loss: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    attention_mask: typing.Optional[torch.Tensor] = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Weighted loss and token count for `elbo_loss`, on the loss's device.

    ``labels``, ``weights`` and ``attention_mask`` are moved to the loss's
    device and everything is summed in float32, so CPU-side corruption
    outputs and a bfloat16 loss work as they come.

    Args:
        token_loss (torch.Tensor | np.ndarray of float, shape (B, L)): Loss
            at each position.
        labels (torch.Tensor | np.ndarray of int, shape (B, L)): ``-100``
            where a position takes no loss.
        weights (torch.Tensor | np.ndarray of float, shape (B,)): Each row's
            loss weight.
        attention_mask (torch.Tensor | np.ndarray, shape (B, L) | None):
            Nonzero at real tokens; positions outside it take no loss and
            are not counted. None counts every position.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two scalars: the rows' weighted
        sums of their labelled positions' loss, added up; the number of real
        tokens.

    Example:
        >>> import torch
        >>> loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> labels = torch.tensor([[5, -100], [5, 5]])
        >>> sums = loss_sums(loss, labels, torch.tensor([2.0, 1.0]))
        >>> [s.item() for s in sums]  # 2 * 1 + 1 * (3 + 4), 4 tokens
        [9.0, 4]
    """
    loss = to_tensor(token_loss).float()
    device = loss.device
    selected = to_tensor(labels).to(device) != -100
    if attention_mask is None:
        tokens = torch.tensor(loss.numel(), device=device)
    else:
        real = to_tensor(attention_mask).to(device).bool()
        selected = selected & real
        tokens = real.sum()
    rows = torch.where(selected, loss, 0.0).sum(-1)
    weights = to_tensor(weights).to(device=device, dtype=torch.float32)
    return (rows * weights).sum(), tokens


ArrayLike = typing.Union[np.ndarray, torch.Tensor]
