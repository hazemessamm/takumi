"""takumi — sequence-packing and masking utilities with C++/nanobind hot paths."""

__version__ = "0.1.0"

from takumi import packing

__all__ = ["packing", "masking"]


def __getattr__(name: str):
    """Import ``takumi.masking`` on first access.

    Packing is numpy-only, but masking pulls in torch (or JAX) at import
    time. Importing it eagerly here would make those a hard requirement of
    ``import takumi.packing`` too, contradicting the optional ``torch``
    extra in pyproject.toml. Deferring it keeps the packers usable with
    neither framework installed.

    Every spelling still works: ``import takumi.masking`` and
    ``from takumi.masking import ...`` go through the normal submodule
    machinery, and ``takumi.masking`` after a bare ``import takumi`` lands
    here. Python caches the submodule on the package after the first
    import, so this runs at most once.

    Args:
        name (str): The attribute being looked up.

    Returns:
        module: ``takumi.masking`` when ``name == "masking"``.

    Raises:
        AttributeError: For any other name.

    Example:
        >>> import takumi
        >>> takumi.masking.__name__
        'takumi.masking'
    """
    if name == "masking":
        import importlib

        return importlib.import_module("takumi.masking")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List the public names, including the lazily imported ``masking``.

    Returns:
        list[str]: Sorted ``__all__``.

    Example:
        >>> import takumi
        >>> dir(takumi)
        ['masking', 'packing']
    """
    return sorted(__all__)
