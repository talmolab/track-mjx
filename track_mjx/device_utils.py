"""JAX runtime configuration utilities."""

import os
from pathlib import Path

import jax

_JIT_CACHE_CONFIGURED = False


def enable_jit_cache(cache_dir: str | None = None):
    """Enable persistent JAX JIT compilation caching.

    Caches compiled XLA executables to disk so repeated runs skip
    recompilation. The cache is keyed by the computation graph, so
    it is safe across code changes (stale entries are simply ignored).

    Args:
        cache_dir: Directory for the cache. Defaults to
            ``$JAX_CACHE_DIR`` if set, otherwise ``~/.cache/jax``.
    """
    global _JIT_CACHE_CONFIGURED
    if _JIT_CACHE_CONFIGURED:
        return
    _JIT_CACHE_CONFIGURED = True

    if cache_dir is None:
        cache_dir = os.environ.get(
            "JAX_CACHE_DIR",
            str(Path.home() / ".cache" / "jax"),
        )

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
