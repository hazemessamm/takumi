import typing

import numpy as np
import torch


def to_tensor(x: np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.cpu().numpy()


def framework(x):
    return "torch"


ArrayLike = typing.Union[np.ndarray, torch.Tensor]
