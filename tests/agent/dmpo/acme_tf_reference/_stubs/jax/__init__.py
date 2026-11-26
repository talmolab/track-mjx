"""Import-time stand-in: acme references jax.Array and jax.numpy only in type hints."""

from . import numpy  # noqa: F401


class Array:
    pass
