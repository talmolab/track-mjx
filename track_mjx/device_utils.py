"""JAX device placement and configuration utilities compatible with JAX 0.10.x.

Provides:
- Drop-in replacement for the removed jax.device_put_replicated using
  NamedSharding (official migration guide: https://docs.jax.dev/en/latest/
  migrate_pmap.html#drop-in-replacements)
- Persistent JIT compilation caching for faster repeated experiments
- Brax compatibility monkey-patches
"""

import os
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

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


def replicate_for_pmap(pytree, devices: Sequence[jax.Device]):
    """Replicate a pytree across devices for use with jax.pmap.

    Stacks each leaf along a new leading axis (one copy per device) and
    shards that axis across the given devices via NamedSharding.
    The resulting arrays have shape [len(devices), *original_shape],
    which is the layout jax.pmap expects.

    Args:
        pytree: Arbitrary JAX pytree to replicate.
        devices: Sequence of devices to distribute across.

    Returns:
        A pytree with the same structure, each leaf replicated and placed
        on the target devices.
    """
    mesh = Mesh(np.array(devices), ("devices",))
    sharding = NamedSharding(mesh, P("devices"))
    return jax.tree.map(
        lambda x: jax.device_put(jnp.stack([x] * len(devices)), sharding),
        pytree,
    )


def patch_brax_pmap_compat():
    """Monkey-patch brax.training.pmap.bcast_local_devices for JAX ≥0.10.

    brax 0.14.x calls jax.device_put_replicated which was removed in
    JAX 0.10.0. This patches the function to use the modern API.
    Safe to call multiple times (idempotent).
    """
    try:
        from brax.training import pmap as brax_pmap
    except ImportError:
        return

    def _bcast_local_devices(value, local_devices_to_use=1):
        devices = jax.local_devices()[:local_devices_to_use]
        return replicate_for_pmap(value, devices)

    brax_pmap.bcast_local_devices = _bcast_local_devices
