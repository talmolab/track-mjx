"""AMP discriminator observation feature extraction."""

from collections.abc import Sequence
from typing import Any

import brax.math
import jax
import jax.numpy as jnp


def _rotate_by_root(vec: jnp.ndarray, root_quat: jnp.ndarray) -> jnp.ndarray:
    return brax.math.rotate(vec, root_quat)


def _compute_single_amp_obs(
    root_pos: jnp.ndarray,
    root_quat: jnp.ndarray,
    root_vel: jnp.ndarray,
    root_ang_vel: jnp.ndarray,
    joints: jnp.ndarray,
    joint_vels: jnp.ndarray,
    key_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Compute one flattened AMP feature vector from motion history."""

    ref_root_pos = root_pos[-1]
    ref_root_quat = root_quat[-1]

    rel_root_pos = jax.vmap(
        lambda pos: _rotate_by_root(pos - ref_root_pos, ref_root_quat)
    )(root_pos)
    rel_root_quat = jax.vmap(lambda quat: brax.math.relative_quat(quat, ref_root_quat))(
        root_quat
    )
    rel_root_vel = jax.vmap(lambda vel: _rotate_by_root(vel, ref_root_quat))(root_vel)
    rel_root_ang_vel = jax.vmap(lambda vel: _rotate_by_root(vel, ref_root_quat))(
        root_ang_vel
    )

    if key_pos.shape[-2] > 0:
        rel_key_pos = key_pos - root_pos[:, None, :]
        rel_key_pos = jax.vmap(
            jax.vmap(lambda pos: _rotate_by_root(pos, ref_root_quat))
        )(rel_key_pos)
    else:
        rel_key_pos = key_pos

    return jnp.concatenate(
        [
            rel_root_pos.reshape(-1),
            rel_root_quat.reshape(-1),
            joints.reshape(-1),
            rel_key_pos.reshape(-1),
            rel_root_vel.reshape(-1),
            rel_root_ang_vel.reshape(-1),
            joint_vels.reshape(-1),
        ],
        axis=-1,
    )


def compute_amp_obs(
    root_pos: jnp.ndarray,
    root_quat: jnp.ndarray,
    root_vel: jnp.ndarray,
    root_ang_vel: jnp.ndarray,
    joints: jnp.ndarray,
    joint_vels: jnp.ndarray,
    key_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Compute flattened AMP observations for optional leading batch dims.

    Inputs have shape ``[..., history, feature]`` except ``key_pos``, which has
    shape ``[..., history, key_bodies, 3]``.
    """

    history_len = root_pos.shape[-2]
    key_body_count = key_pos.shape[-2]
    leading_shape = root_pos.shape[:-2]

    flat_root_pos = root_pos.reshape((-1, history_len, root_pos.shape[-1]))
    flat_root_quat = root_quat.reshape((-1, history_len, root_quat.shape[-1]))
    flat_root_vel = root_vel.reshape((-1, history_len, root_vel.shape[-1]))
    flat_root_ang_vel = root_ang_vel.reshape((-1, history_len, root_ang_vel.shape[-1]))
    flat_joints = joints.reshape((-1, history_len, joints.shape[-1]))
    flat_joint_vels = joint_vels.reshape((-1, history_len, joint_vels.shape[-1]))
    flat_key_pos = key_pos.reshape((-1, history_len, key_body_count, 3))

    flat_obs = jax.vmap(_compute_single_amp_obs)(
        flat_root_pos,
        flat_root_quat,
        flat_root_vel,
        flat_root_ang_vel,
        flat_joints,
        flat_joint_vels,
        flat_key_pos,
    )

    return flat_obs.reshape(leading_shape + (flat_obs.shape[-1],))


def reference_histories(
    reference_clips: Any,
    clip_ids: jnp.ndarray,
    frame_ids: jnp.ndarray,
    key_body_names: Sequence[str] = (),
) -> dict[str, jnp.ndarray]:
    """Gather AMP feature histories from reference clips.

    ``clip_ids`` may be scalar or batched. ``frame_ids`` should have the same
    leading batch dims plus a final history dimension.
    """

    if frame_ids.ndim == clip_ids.ndim + 1:
        clip_index = jnp.expand_dims(clip_ids, axis=-1)
    else:
        clip_index = clip_ids

    if key_body_names:
        key_pos = jnp.stack(
            [
                reference_clips.body_xpos(name)[clip_index, frame_ids]
                for name in key_body_names
            ],
            axis=-2,
        )
    else:
        key_pos = jnp.zeros(frame_ids.shape + (0, 3), dtype=jnp.float32)

    return {
        "root_pos": reference_clips.root_position[clip_index, frame_ids],
        "root_quat": reference_clips.root_quaternion[clip_index, frame_ids],
        "root_vel": reference_clips.velocity[clip_index, frame_ids],
        "root_ang_vel": reference_clips.angular_velocity[clip_index, frame_ids],
        "joints": reference_clips.joints[clip_index, frame_ids],
        "joint_vels": reference_clips.joints_velocity[clip_index, frame_ids],
        "key_pos": key_pos,
    }


def sample_reference_amp_obs(
    reference_clips: Any,
    key: jnp.ndarray,
    num_samples: int,
    num_disc_obs_steps: int,
    key_body_names: Sequence[str] = (),
) -> jnp.ndarray:
    """Sample flattened AMP observations from reference clips."""

    clip_key, frame_key = jax.random.split(key)
    n_clips = reference_clips.qpos.shape[0]
    clip_len = reference_clips.qpos.shape[1]

    clip_ids = jax.random.randint(clip_key, (num_samples,), 0, n_clips)
    latest_frames = jax.random.randint(
        frame_key,
        (num_samples,),
        num_disc_obs_steps - 1,
        clip_len,
    )
    offsets = num_disc_obs_steps - 1 - jnp.arange(num_disc_obs_steps)
    frame_ids = latest_frames[:, None] - offsets[None, :]

    histories = reference_histories(
        reference_clips=reference_clips,
        clip_ids=clip_ids,
        frame_ids=frame_ids,
        key_body_names=key_body_names,
    )
    return compute_amp_obs(**histories)
