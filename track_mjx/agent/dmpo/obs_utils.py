"""Dictionary-observation helpers private to the DMPO agent.

Environments describe an observation as a dict with a proprioception entry and
a reference-observation entry, possibly wrapped under ``"state"`` (next to an
optional ``"privileged_state"``), and either entry can itself be a nested dict
of arrays. The helpers here reduce every such layout to one canonical form::

    {
        "imitation_target": (*batch, n_reference),
        "proprioception": (*batch, n_proprioception),
        "vision": (..., H, W, C),  # only when the environment provides it
    }

The reference entry is read from ``"imitation_target"``, or from ``"task_obs"``
when that is the name the environment uses, and is always returned under
``"imitation_target"``. ``"vision"`` is passed through untouched.

Batch convention: the number of leading batch axes is taken from the
proprioception entry. For a flat proprioception array it is ``ndim - 1``. When
proprioception is itself a nested dict, it is the rank of its lowest-rank leaf
minus one, which assumes at least one leaf is a plain per-sample feature
vector. Every leaf of the reference entry is then reshaped to
``(*batch, -1)`` and concatenated, so extra trailing axes of a reference leaf
are feature axes even when several leaves happen to share them.

The running statistics are kept per half (``imitation_target`` and
``proprioception``), so the reference and proprioception parts of an
observation are normalized independently.
"""

import math
from collections.abc import Mapping
from typing import Any

import flax
import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs

# Names under which an environment may publish its reference observation,
# in order of precedence.
_REFERENCE_KEYS = ("imitation_target", "task_obs")


def _unwrap_state(obs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    """Split an observation into its inner dict and its vision entry.

    ``"vision"`` can sit beside ``"state"`` or inside it, so it is looked up
    before unwrapping and again afterwards.

    Args:
        obs: Observation dict, optionally wrapped under ``"state"``.

    Returns:
        ``(inner_obs, vision)``, where ``vision`` is None when absent.
    """
    vision = obs.get("vision")
    if "state" in obs and "proprioception" not in obs:
        obs = obs["state"]
        if vision is None:
            vision = obs.get("vision")
    return obs, vision


def _reference_key(inner_obs: Mapping[str, Any]) -> str | None:
    """Return the key the reference observation is stored under, if any."""
    for key in _REFERENCE_KEYS:
        if key in inner_obs:
            return key
    return None


def _proprioception_batch_shape(proprioception: Any) -> tuple[int, ...]:
    """Return the leading batch shape implied by a proprioception entry."""
    leaves = jax.tree.leaves(proprioception)
    if not leaves:
        raise ValueError("proprioception has no array leaves to infer a batch from.")
    lowest_rank_leaf = min(leaves, key=jnp.ndim)
    return tuple(jnp.shape(lowest_rank_leaf)[: max(jnp.ndim(lowest_rank_leaf) - 1, 0)])


def _flatten_to_batch(nested: Any, batch_shape: tuple[int, ...], name: str):
    """Reshape every leaf to ``(*batch_shape, -1)`` and join them on the last axis.

    Args:
        nested: A flat array or a nested dict/pytree of arrays.
        batch_shape: Leading axes that every leaf must share.
        name: Observation entry being flattened, for error messages.

    Returns:
        Array of shape ``(*batch_shape, n_features)``; ``n_features`` is 0 when
        ``nested`` has no leaves.

    Raises:
        ValueError: If a leaf does not start with ``batch_shape``.
    """
    leaves = jax.tree.leaves(nested)
    if not leaves:
        return jnp.zeros((*batch_shape, 0), jnp.float32)
    flat_leaves = []
    for leaf in leaves:
        leaf = jnp.asarray(leaf)
        if tuple(leaf.shape[: len(batch_shape)]) != batch_shape:
            raise ValueError(
                f"Every leaf of '{name}' must start with the batch shape "
                f"{batch_shape} taken from proprioception, but got a leaf of "
                f"shape {tuple(leaf.shape)}."
            )
        feature_size = math.prod(leaf.shape[len(batch_shape) :])
        flat_leaves.append(leaf.reshape(*batch_shape, feature_size))
    if len(flat_leaves) == 1:
        return flat_leaves[0]
    return jnp.concatenate(flat_leaves, axis=-1)


def flatten_obs_dict(obs: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
    """Reduce an observation dict to flat per-half arrays.

    The ``"state"`` wrapper is removed, the reference observation (stored as
    ``"imitation_target"`` or ``"task_obs"``) is returned as
    ``"imitation_target"``, and each half is flattened to
    ``(*batch, n_features)`` with the batch shape read from proprioception.
    ``"vision"`` keeps its ``(..., H, W, C)`` layout for the CNN. When the
    observation carries no reference entry, ``"imitation_target"`` is a
    zero-width array that keeps the batch axes, so downstream concatenation
    and normalization remain valid no-ops for that half.

    Args:
        obs: Observation dict, unbatched or batched, optionally wrapped under
            ``"state"`` (as the vnl_playground environments produce it). Either
            half may be a flat array or a nested dict of arrays.

    Returns:
        Dict holding ``"imitation_target"``, ``"proprioception"`` and, when the
        input has one, ``"vision"``.

    Raises:
        KeyError: If the observation has no ``"proprioception"`` entry.
        ValueError: If a reference leaf does not start with the batch shape
            implied by proprioception.
    """
    inner_obs, vision = _unwrap_state(obs)

    proprioception = inner_obs["proprioception"]
    batch_shape = _proprioception_batch_shape(proprioception)
    flat_proprio = _flatten_to_batch(proprioception, batch_shape, "proprioception")

    ref_key = _reference_key(inner_obs)
    if ref_key is None:
        flat_imit = jnp.zeros((*batch_shape, 0), jnp.float32)
    else:
        flat_imit = _flatten_to_batch(inner_obs[ref_key], batch_shape, ref_key)

    result = {
        "imitation_target": flat_imit,
        "proprioception": flat_proprio,
    }
    if vision is not None:
        # Keep H, W, C for the CNN; do not flatten.
        result["vision"] = vision
    return result


@flax.struct.dataclass
class DictRunningStatisticsState:
    """Running statistics with one independent entry per observation half.

    Attributes:
        imitation_target: Statistics of the flattened reference observation.
        proprioception: Statistics of the flattened proprioception.
    """

    imitation_target: running_statistics.RunningStatisticsState
    proprioception: running_statistics.RunningStatisticsState


def init_dict_normalizer(obs_template: Mapping[str, Any]) -> DictRunningStatisticsState:
    """Initialize running statistics sized from an example observation.

    Args:
        obs_template: Example observation in any layout accepted by
            ``flatten_obs_dict``. Only the feature sizes of the two halves are
            used, so the example may be batched or unbatched.

    Returns:
        Zero-count state with one statistics entry per half.
    """
    flat_obs = flatten_obs_dict(obs_template)
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
    """Fold a batch of observations into the running statistics.

    Args:
        state: Current running statistics.
        obs: Observation batch in any layout accepted by ``flatten_obs_dict``.
        pmap_axis_name: Name of the pmapped axis to aggregate over, if any.

    Returns:
        Updated statistics.
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
    """Normalize each observation half with its own running statistics.

    Args:
        obs: Observation in any layout accepted by ``flatten_obs_dict``.
        state: Running statistics to normalize with.

    Returns:
        Dict with the normalized, flattened ``"imitation_target"`` and
        ``"proprioception"``. A ``"vision"`` entry, if present, is scaled to
        [0, 1] when it is uint8 and passed through otherwise.
    """
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
        if vision.dtype == jnp.uint8:
            result["vision"] = vision.astype(jnp.float32) / 255.0
        else:
            result["vision"] = vision
    return result


def get_dmpo_obs_sizes(obs_template: Mapping[str, Any]) -> dict[str, int]:
    """Report the flattened feature size of each observation half.

    The reference entry is reported under the name the environment used
    (``"imitation_target"`` or ``"task_obs"``), because the network builders
    look sizes up by that name, and is omitted when the observation has none.
    The vision size is the ``H * W * C`` of its trailing three axes, so
    leading batch axes never enter it.

    Args:
        obs_template: Example observation, unbatched or batched, either wrapped
            under ``"state"`` or a flat dict with ``"proprioception"``.

    Returns:
        Dict mapping ``"proprioception"``, the reference key and, when present,
        ``"vision"`` to their sizes.

    Raises:
        KeyError: If the observation has neither a ``"state"`` nor a
            ``"proprioception"`` entry.
    """
    if "state" not in obs_template and "proprioception" not in obs_template:
        raise KeyError(
            f"Expected 'state' key in observation dict, got keys: "
            f"{list(obs_template.keys())}. Ensure the environment produces "
            f"nested observations with 'state' and 'privileged_state' keys."
        )

    inner_obs, _ = _unwrap_state(obs_template)
    ref_key = _reference_key(inner_obs)
    flat_obs = flatten_obs_dict(obs_template)

    sizes = {}
    imitation_size = flat_obs["imitation_target"].shape[-1]
    if ref_key is not None and imitation_size > 0:
        sizes[ref_key] = imitation_size
    sizes["proprioception"] = flat_obs["proprioception"].shape[-1]
    if "vision" in flat_obs:
        sizes["vision"] = math.prod(flat_obs["vision"].shape[-3:])
    return sizes
