import logging
import os

logger = logging.getLogger(__name__)

_BACKEND = os.environ.get("TAKUMI_MASKING_BACKEND", "torch")

# The framework is a hard requirement of `takumi.masking` only — `takumi.packing`
# is numpy-only and never reaches this module. Translate the bare
# ModuleNotFoundError into a message that says which extra to install.
if _BACKEND == "torch":
    try:
        from takumi.masking.backend import torch_backend as backend
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "torch":
            raise
        raise ImportError(
            "takumi.masking needs PyTorch for the default 'torch' backend. "
            "Install it with `pip install takumi[torch]`, or set "
            "TAKUMI_MASKING_BACKEND=jax before importing to use JAX instead. "
            "takumi.packing requires neither."
        ) from exc

    logger.debug("Using PyTorch backend for takumi.masking.")
elif _BACKEND in ("JAX", "jax"):
    try:
        from takumi.masking.backend import jax_backend as backend
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "jax":
            raise
        raise ImportError(
            "TAKUMI_MASKING_BACKEND=jax needs JAX. Install it with "
            "`pip install jax`. takumi.packing requires neither backend."
        ) from exc

    logger.debug("Using JAX backend for takumi.masking.")
else:
    raise ValueError(f"Unsupported TAKUMI_MASKING_BACKEND: {_BACKEND!r}")
