"""Utilities for handling dictionary observations.

This module provides utilities for working with dictionary observations
where each key maps to either a flat array or a nested dict of arrays.
Key components:

- DictRunningStatisticsState: Holds separate running stats for each observation key
- Normalizer functions: init, update, and normalize for dict observations
- Flattening utilities for nested observation structures
- normalizer_select: Extracts per-key running statistics from a pytree-structured normalizer
- get_obs_sizes / get_obs_shape: Extract observation metadata from example observations
"""

from typing import Mapping, Any

import flax
import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs
from jax import flatten_util


def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState,
    obs_key: str,
) -> running_statistics.RunningStatisticsState:
    """Extract per-key running statistics from a pytree-structured normalizer.

    When running_statistics.init_state is called with a pytree observation shape,
    the resulting RunningStatisticsState has pytree-structured mean, std, etc.
    This function extracts the statistics for a specific top-level key.

    Args:
        processor_params: RunningStatisticsState with pytree-structured fields
            (mean, std, summed_variance are dicts keyed by observation keys).
        obs_key: Top-level key to extract (e.g., 'state' or 'privileged_state').

    Returns:
        RunningStatisticsState with statistics for just the specified key.
    """
    return running_statistics.RunningStatisticsState(
        count=processor_params.count,
        mean=processor_params.mean[obs_key],
        summed_variance=processor_params.summed_variance[obs_key],
        std=processor_params.std[obs_key],
        std_eps=processor_params.std_eps,
        mode=processor_params.mode,
    )


def _flatten_nested_obs(nested: Any) -> jnp.ndarray:
    """Flatten a potentially nested observation, preserving batch dimensions.

    Handles both flat arrays and nested dicts/pytrees. For plain arrays,
    all dimensions are preserved since the last dim is the observation dim
    and preceding dims are batch dims (supports pmap + vmap stacking).

    Args:
        nested: Either a flat array or a nested dict of arrays.
            - Plain array: returned as-is (last dim = obs, rest = batch)
            - Nested dict: leaves are concatenated along last axis

    Returns:
        Flattened array with shape (..., obs_size).
    """
    if isinstance(nested, jnp.ndarray):
        # Plain arrays: last dim is the observation dim, all preceding
        # dims are batch dims (e.g., device, env, unroll_length).
        # running_statistics.update handles multiple batch dims natively.
        return nested

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

    Handles observations wrapped in state/privileged_state hierarchy
    (as returned by vnl_playground environments) by unwrapping to the
    inner "state" dict. Also handles both "imitation_target" and "task_obs"
    as the reference observation key.

    Args:
        obs: Observation dict where values may be nested dicts or flat arrays.
            Can be unbatched (each value is 1D) or batched (each value has
            leading batch dimension). May optionally be wrapped in
            state/privileged_state hierarchy.

    Returns:
        Dict with the same keys but flattened array values. Shape is
        (obs_size,) for unbatched input or (batch_size, obs_size) for batched.
    """
    # Unwrap state/privileged_state hierarchy if present
    if "state" in obs and "proprioception" not in obs:
        obs = obs["state"]

    flat_proprio = _flatten_nested_obs(obs["proprioception"])

    if "imitation_target" in obs:
        flat_imit = _flatten_nested_obs(obs["imitation_target"])
    elif "task_obs" in obs:
        flat_imit = _flatten_nested_obs(obs["task_obs"])
    else:
        # Zero-sized sentinel: preserves batch dims with obs_size=0.
        # Downstream ops (normalize, concat, update) are no-ops on size-0 arrays.
        batch_shape = flat_proprio.shape[:-1]
        flat_imit = jnp.zeros((*batch_shape, 0))

    result = {
        "imitation_target": flat_imit,
        "proprioception": flat_proprio,
    }
    if "vision" in obs:
        # Keep H,W,C shape for CNN - don't flatten
        result["vision"] = obs["vision"]
    return result


@flax.struct.dataclass
class DictRunningStatisticsState:
    """Running statistics state for dictionary observations.

    Holds separate RunningStatisticsState for each observation key,
    enabling independent normalization of different observation components.
    """

    imitation_target: running_statistics.RunningStatisticsState
    proprioception: running_statistics.RunningStatisticsState


def init_dict_normalizer(
    obs: Mapping[str, Any],
) -> DictRunningStatisticsState:
    """Initialize running statistics state from an example observation dict.

    Handles nested observations by flattening to determine sizes.

    Args:
        obs: Example observation dict with flat or nested arrays at each key.

    Returns:
        Initialized DictRunningStatisticsState with proper shapes.
    """
    flat_obs = flatten_obs_dict(obs)
    return DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(flat_obs["imitation_target"].shape[-1:], jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(flat_obs["proprioception"].shape[-1:], jnp.dtype("float32"))
        ),
    )


def update_dict_normalizer(
    state: DictRunningStatisticsState,
    obs: Mapping[str, Any],
    pmap_axis_name: str | None = None,
) -> DictRunningStatisticsState:
    """Update running statistics from an observation dict.

    Handles nested observations by flattening before updating.

    Args:
        state: Current running statistics state.
        obs: Observation dict with flat or nested arrays at each key.
        pmap_axis_name: Axis name for pmap aggregation (optional).

    Returns:
        Updated DictRunningStatisticsState.
    """
    flat_obs = flatten_obs_dict(obs)
    return DictRunningStatisticsState(
        imitation_target=running_statistics.update(
            state.imitation_target,
            flat_obs["imitation_target"],
            pmap_axis_name=pmap_axis_name,
        ),
        proprioception=running_statistics.update(
            state.proprioception,
            flat_obs["proprioception"],
            pmap_axis_name=pmap_axis_name,
        ),
    )


def normalize_dict_obs(
    obs: Mapping[str, Any],
    state: DictRunningStatisticsState,
) -> dict[str, jnp.ndarray]:
    """Normalize observation dict using running statistics.

    Handles nested observations by flattening each key before normalizing.

    Args:
        obs: Observation dict with flat or nested arrays at each key.
        state: Running statistics state for normalization.

    Returns:
        Dict with normalized flat observation arrays.
    """
    # Flatten nested observations first
    flat_obs = flatten_obs_dict(obs)
    result = {
        "imitation_target": running_statistics.normalize(
            flat_obs["imitation_target"], state.imitation_target
        ),
        "proprioception": running_statistics.normalize(
            flat_obs["proprioception"], state.proprioception
        ),
    }
    if "vision" in flat_obs:
        vision = flat_obs["vision"]
        # Normalize to [0, 1]: divide by 255 if uint8, pass through if already float
        if vision.dtype == jnp.uint8:
            result["vision"] = vision.astype(jnp.float32) / 255.0
        else:
            result["vision"] = vision
    return result


def concat_flat_dict_obs(obs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Concatenate flat observation dict to single array.

    Concatenates imitation_target and proprioception in that order.
    Useful for value network or legacy compatibility.

    Note: Expects already-flattened observations. Use flatten_obs_dict first
    if observations may be nested.

    Args:
        obs: Observation dict with flat arrays at each key.

    Returns:
        Single flat array with all observations concatenated.
    """
    return jnp.concatenate([obs["imitation_target"], obs["proprioception"]], axis=-1)


def get_obs_sizes(obs: Mapping[str, Any]) -> dict[str, int]:
    """Extract observation sizes from an example observation dict.

    Handles nested observations by flattening to determine sizes.

    Args:
        obs: Example observation dict with flat or nested arrays.

    Returns:
        Dict mapping observation keys to their flattened sizes.
    """
    flat_obs = flatten_obs_dict(obs)
    result = {}
    imit_size = flat_obs["imitation_target"].shape[-1]
    if imit_size > 0:
        result["imitation_target"] = imit_size
    result["proprioception"] = flat_obs["proprioception"].shape[-1]
    if "vision" in flat_obs:
        # Vision size is the product of H*W*C
        vision_shape = flat_obs["vision"].shape
        # Skip batch dims: vision shape is (..., H, W, C)
        result["vision"] = int(jnp.prod(jnp.array(vision_shape[-3:])))
    return result


def get_obs_shape(
    obs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, specs.Array]]:
    """Extract observation shapes as a pytree for running_statistics.init_state.

    Preserves the container types (e.g. OrderedDict) of the input so that the
    resulting normalizer pytree is compatible with environment observations in
    ``jax.tree_util.tree_map`` calls.

    Args:
        obs: Example observation dict with flat leaf arrays.

    Returns:
        Nested mapping matching obs structure, with specs.Array for each leaf.
    """

    def get_specs(inner_obs: Mapping[str, Any]) -> Mapping[str, specs.Array]:
        specs_dict = {
            key: specs.Array((val.shape[-1],), jnp.dtype("float32"))
            for key, val in inner_obs.items()
        }
        return type(inner_obs)(specs_dict)

    return type(obs)({key: get_specs(inner) for key, inner in obs.items()})


def convert_flat_to_dict_normalizer(
    flat_state: running_statistics.RunningStatisticsState,
    reference_obs_size: int,
) -> DictRunningStatisticsState:
    """Convert a flat normalizer state to dict normalizer state.

    Used for loading legacy checkpoints that stored observations as flat arrays.
    Splits the flat normalizer at reference_obs_size to create separate states
    for imitation_target and proprioception.

    Args:
        flat_state: Legacy flat RunningStatisticsState covering all observations.
        reference_obs_size: Size of the imitation_target portion.

    Returns:
        DictRunningStatisticsState with split statistics.
    """
    # Split array fields at reference_obs_size, copy scalar fields
    return DictRunningStatisticsState(
        imitation_target=running_statistics.RunningStatisticsState(
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
