"""Utilities for handling dictionary observations.

This module provides utilities for working with dictionary observations
where each key maps to either a flat array or a nested dict of arrays.
Key components:

- DictRunningStatisticsState: Holds separate running stats for each observation key
- Normalizer functions: init, update, and normalize for dict observations
- Flattening utilities for nested observation structures
"""

from typing import Mapping, Any

import flax
import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs
from jax import flatten_util


def _flatten_nested_obs(nested: Any) -> jnp.ndarray:
    """Flatten a potentially nested observation, preserving batch dimensions.

    Handles both flat arrays and nested dicts/pytrees. Preserves the first
    dimension (batch) and flattens all trailing dimensions.

    Args:
        nested: Either a flat array or a nested dict of arrays.
            - 1D array (obs_size,): returned as-is (unbatched)
            - 2D array (batch, obs_size): returned as-is (already flat)
            - 3D+ array (batch, d1, d2, ...): flattened to (batch, d1*d2*...)
            - Nested dict: leaves are concatenated along last axis

    Returns:
        Flattened array with shape (obs_size,) or (batch, obs_size).
    """
    if isinstance(nested, jnp.ndarray):
        if nested.ndim <= 2:
            # 1D (unbatched) or 2D (batched, already flat) - return as-is
            return nested
        else:
            # 3D+ array: preserve batch dim (first), flatten the rest
            # (batch, d1, d2, ...) -> (batch, d1*d2*...)
            return nested.reshape(nested.shape[0], -1)

    # For nested dicts/pytrees, flatten the observation structure
    leaves = jax.tree_util.tree_leaves(nested)
    if not leaves:
        return jnp.array([])

    # Find batch dims: leading dimensions that are identical across all leaves
    # E.g., (1,64,5,3) and (1,64,18,5,3) share prefix (1,64)
    ref = min(leaves, key=lambda x: x.ndim)
    n_batch = 0
    for i in range(ref.ndim - 1):
        if len({leaf.shape[i] for leaf in leaves}) == 1:
            n_batch += 1
        else:
            break

    if n_batch == 0:
        # Unbatched: flatten everything to 1D
        flat, _ = flatten_util.ravel_pytree(nested)
        return flat

    # Batched: reshape each leaf to (*batch_shape, -1) and concatenate
    batch_shape = ref.shape[:n_batch]
    return jnp.concatenate([leaf.reshape(*batch_shape, -1) for leaf in leaves], axis=-1)


def flatten_obs_dict(obs: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
    """Flatten each top-level key in an observation dict, preserving batch dim.

    Converts nested observation structures (e.g., from vnl_playground)
    to flat arrays at each key, suitable for normalization and network input.
    Preserves the batch dimension if present.

    Args:
        obs: Observation dict where values may be nested dicts or flat arrays.
            Can be unbatched (each value is 1D) or batched (each value has
            leading batch dimension).

    Returns:
        Dict with the same keys but flattened array values. Shape is
        (obs_size,) for unbatched input or (batch_size, obs_size) for batched.
    """
    return {
        key: val if isinstance(val, jnp.ndarray) else _flatten_nested_obs(val)
        for key, val in obs.items()
    }


@flax.struct.dataclass
class DictRunningStatisticsState:
    """Running statistics state for dictionary observations.

    Holds separate RunningStatisticsState for each observation key,
    enabling independent normalization of different observation components.
    """

    task_obs: running_statistics.RunningStatisticsState
    proprioception: running_statistics.RunningStatisticsState


def init_dict_normalizer(
    obs: Mapping[str, Any],
) -> DictRunningStatisticsState | dict:
    """Initialize running statistics state from an example observation dict.

    Handles nested observations by flattening to determine sizes.
    Returns DictRunningStatisticsState for legacy 2-key obs (task_obs +
    proprioception), or a plain dict for arbitrary keys.

    Args:
        obs: Example observation dict with flat or nested arrays at each key.

    Returns:
        Initialized normalizer state with proper shapes.
    """
    # Exclude kpms_code from normalizer — it's a discrete code, not continuous obs
    obs_sizes = {k: v for k, v in get_obs_sizes(obs).items() if k != "kpms_code"}
    if set(obs_sizes.keys()) == {"task_obs", "proprioception"}:
        return DictRunningStatisticsState(
            task_obs=running_statistics.init_state(
                specs.Array((obs_sizes["task_obs"],), jnp.dtype("float32"))
            ),
            proprioception=running_statistics.init_state(
                specs.Array((obs_sizes["proprioception"],), jnp.dtype("float32"))
            ),
        )
    return {
        key: running_statistics.init_state(
            specs.Array((size,), jnp.dtype("float32"))
        )
        for key, size in obs_sizes.items()
    }


def update_dict_normalizer(
    state: DictRunningStatisticsState | dict,
    obs: Mapping[str, Any],
    pmap_axis_name: str | None = None,
) -> DictRunningStatisticsState | dict:
    """Update running statistics from an observation dict.

    Handles nested observations by flattening before updating.

    Args:
        state: Current running statistics state.
        obs: Observation dict with flat or nested arrays at each key.
        pmap_axis_name: Axis name for pmap aggregation (optional).

    Returns:
        Updated normalizer state.
    """
    flat_obs = flatten_obs_dict(obs)
    if isinstance(state, DictRunningStatisticsState):
        return DictRunningStatisticsState(
            task_obs=running_statistics.update(
                state.task_obs,
                flat_obs["task_obs"],
                pmap_axis_name=pmap_axis_name,
            ),
            proprioception=running_statistics.update(
                state.proprioception,
                flat_obs["proprioception"],
                pmap_axis_name=pmap_axis_name,
            ),
        )
    return {
        key: running_statistics.update(
            state[key], flat_obs[key], pmap_axis_name=pmap_axis_name,
        )
        for key in state
    }


def normalize_dict_obs(
    obs: Mapping[str, Any] | jnp.ndarray,
    state: DictRunningStatisticsState | dict,
) -> dict[str, jnp.ndarray]:
    """Normalize observation dict using running statistics.

    Handles nested observations by flattening each key before normalizing.
    Works with arbitrary observation keys.

    Args:
        obs: Observation dict with flat or nested arrays at each key.
        state: Running statistics state for normalization (DictRunningStatisticsState
            or plain dict mapping keys to RunningStatisticsState).

    Returns:
        Dict with normalized flat observation arrays.
    """
    flat_obs = flatten_obs_dict(obs)

    def _get_state(key):
        if isinstance(state, dict):
            return state[key]
        return getattr(state, key)

    return {
        key: running_statistics.normalize(flat_obs[key], _get_state(key))
        for key in flat_obs
        if (key in state if isinstance(state, dict) else hasattr(state, key))
    }


def concat_flat_dict_obs(obs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Concatenate flat observation dict to single array.

    Concatenates observations in sorted key order.
    Useful for value network or legacy compatibility.

    Note: Expects already-flattened observations. Use flatten_obs_dict first
    if observations may be nested.

    Args:
        obs: Observation dict with flat arrays at each key.

    Returns:
        Single flat array with all observations concatenated.
    """
    return jnp.concatenate([obs[key] for key in sorted(obs.keys())], axis=-1)


def get_obs_sizes(obs: Mapping[str, Any]) -> dict[str, int]:
    """Extract observation sizes from an example observation dict.

    Handles nested observations by flattening to determine sizes.
    Works with any set of observation keys (not just the two normalizer keys).

    For flat arrays (jnp.ndarray), uses shape[-1] directly to avoid
    _flatten_nested_obs collapsing multiple batch dimensions (e.g., from
    pmap/vmap wrapping) into the feature dimension. For nested dicts,
    delegates to _flatten_nested_obs which correctly detects batch dims.

    Args:
        obs: Example observation dict with flat or nested arrays.
            May have leading batch dimensions from vmapping.

    Returns:
        Dict mapping observation keys to their flattened sizes.
    """

    def _feature_dim(value: Any) -> int:
        if isinstance(value, jnp.ndarray):
            # Flat array: last dim is always the feature dim, regardless
            # of how many batch dims precede it.
            return value.shape[-1]
        # Nested dict/pytree: flatten to determine total feature size.
        return _flatten_nested_obs(value).shape[-1]

    return {key: _feature_dim(obs[key]) for key in obs}


def convert_flat_to_dict_normalizer(
    flat_state: running_statistics.RunningStatisticsState,
    reference_obs_size: int,
) -> DictRunningStatisticsState:
    """Convert a flat normalizer state to dict normalizer state.

    Used for loading legacy checkpoints that stored observations as flat arrays.
    Splits the flat normalizer at reference_obs_size to create separate states
    for task_obs and proprioception.

    Args:
        flat_state: Legacy flat RunningStatisticsState covering all observations.
        reference_obs_size: Size of the task_obs portion.

    Returns:
        DictRunningStatisticsState with split statistics.
    """
    # Split array fields at reference_obs_size, copy scalar fields
    return DictRunningStatisticsState(
        task_obs=running_statistics.RunningStatisticsState(
            mean=flat_state.mean[:reference_obs_size],
            std=flat_state.std[:reference_obs_size],
            count=flat_state.count,
            summed_variance=flat_state.summed_variance[:reference_obs_size],
            std_eps=flat_state.std_eps,
            mode=flat_state.mode,
        ),
        proprioception=running_statistics.RunningStatisticsState(
            mean=flat_state.mean[reference_obs_size:],
            std=flat_state.std[reference_obs_size:],
            count=flat_state.count,
            summed_variance=flat_state.summed_variance[reference_obs_size:],
            std_eps=flat_state.std_eps,
            mode=flat_state.mode,
        ),
    )
