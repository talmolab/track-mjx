"""Track's rollout conventions for dense and sparse VNL imitation tasks."""

from typing import Any

import jax
from vnl_playground.tasks.rodent.sparse_imitation import SparseImitation


def is_sparse_imitation(env: Any) -> bool:
    """Recognize sparse tasks through observation and training wrappers."""
    while hasattr(env, "env"):
        env = env.env
    return isinstance(env, SparseImitation)


def get_imitation_episode_length(env: Any) -> int:
    """Use a full clip for dense tasks and the configured episode for sparse."""
    if is_sparse_imitation(env):
        return env._config.episode_length
    return env.num_control_steps_per_clip


def get_rollout_reset_kwargs(env: Any) -> dict[str, int]:
    """Sparse reset chooses a default pose and does not accept a start frame."""
    return {} if is_sparse_imitation(env) else {"start_frame": 0}


def get_rollout_reference_qpos(env: Any, data: Any, info: dict) -> jax.Array:
    """Return dense reward targets or sparse cyclic visualization poses.

    Sparse rewards match the whole sequence elastically, so their visual
    reference is not a per-step reward target. Reuse VNL's rendering clock,
    including the clip-range offset and wraparound.
    """
    if is_sparse_imitation(env):
        frame = env._get_cyclic_ref_frame(data)
        return env.reference_clips.qpos[info["reference_clip"], frame]
    return env.get_current_reference_state(data, info).qpos
