"""Rodent SMP feature construction.

The rodent data and VNL environments use MuJoCo quaternion order
``[w, x, y, z]``.  These helpers intentionally do not reuse the MimicKit
humanoid convention, which is ``[x, y, z, w]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_KEY_BODY_NAMES: tuple[str, ...] = (
    "lower_arm_R",
    "lower_arm_L",
    "foot_R",
    "foot_L",
    "skull",
)


@dataclass(frozen=True)
class SMPFeatureSpec:
    """Static layout for rodent SMP observations."""

    num_history_steps: int = 10
    key_body_names: tuple[str, ...] = DEFAULT_KEY_BODY_NAMES
    mocap_hz: float = 50.0
    feature_version: str = "rodent_smp_v1"

    @property
    def per_frame_dim(self) -> int:
        # root_pos(3), root_rot_6d(6), joints(67 by default), key positions,
        # root linear velocity(3), root angular velocity(3).
        return 3 + 6 + self.num_joints + 3 * len(self.key_body_names) + 3 + 3

    @property
    def input_dim(self) -> int:
        return self.num_history_steps * self.per_frame_dim

    @property
    def num_joints(self) -> int:
        # VNL rodent qpos has 7 root coordinates and 67 hinge joints.
        return 67

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["key_body_names"] = list(self.key_body_names)
        out["per_frame_dim"] = self.per_frame_dim
        out["input_dim"] = self.input_dim
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMPFeatureSpec":
        return cls(
            num_history_steps=int(data.get("num_history_steps", 10)),
            key_body_names=tuple(data.get("key_body_names", DEFAULT_KEY_BODY_NAMES)),
            mocap_hz=float(data.get("mocap_hz", 50.0)),
            feature_version=str(data.get("feature_version", "rodent_smp_v1")),
        )


def _safe_normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    return _safe_normalize(q)


def quat_conj(q: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Multiplies two ``[w, x, y, z]`` quaternions."""

    w1, x1, y1, z1 = jnp.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(q2, -1, 0)
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotates vector ``v`` by quaternion ``q`` in ``[w, x, y, z]`` order."""

    q = quat_normalize(q)
    q_xyz = q[..., 1:]
    q_w = q[..., :1]
    t = 2.0 * jnp.cross(q_xyz, v)
    return v + q_w * t + jnp.cross(q_xyz, t)


def yaw_inv_quat(root_quat: jnp.ndarray) -> jnp.ndarray:
    """Returns the inverse heading quaternion for a root orientation."""

    forward = quat_rotate(root_quat, jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))
    heading = jnp.arctan2(forward[..., 1], forward[..., 0])
    half = -0.5 * heading
    zeros = jnp.zeros_like(half)
    return jnp.stack([jnp.cos(half), zeros, zeros, jnp.sin(half)], axis=-1)


def quat_to_6d(q: jnp.ndarray) -> jnp.ndarray:
    """Encodes orientation by rotated tangent and normal vectors."""

    q = quat_normalize(q)
    tangent = quat_rotate(q, jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))
    normal = quat_rotate(q, jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
    return jnp.concatenate([tangent, normal], axis=-1)


def compute_smp_frames(
    root_pos: jnp.ndarray,
    root_quat: jnp.ndarray,
    root_vel: jnp.ndarray,
    root_ang_vel: jnp.ndarray,
    joints: jnp.ndarray,
    key_body_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Builds per-frame SMP features for one or more histories.

    Args:
        root_pos: ``[..., H, 3]`` world root positions.
        root_quat: ``[..., H, 4]`` root quaternions in ``[w, x, y, z]`` order.
        root_vel: ``[..., H, 3]`` world root linear velocities.
        root_ang_vel: ``[..., H, 3]`` world root angular velocities.
        joints: ``[..., H, J]`` hinge joint angles.
        key_body_pos: ``[..., H, K, 3]`` world positions for key bodies.

    Returns:
        Per-frame features with shape ``[..., H, D]``.
    """

    root_pos = jnp.asarray(root_pos, dtype=jnp.float32)
    root_quat = quat_normalize(jnp.asarray(root_quat, dtype=jnp.float32))
    root_vel = jnp.asarray(root_vel, dtype=jnp.float32)
    root_ang_vel = jnp.asarray(root_ang_vel, dtype=jnp.float32)
    joints = jnp.asarray(joints, dtype=jnp.float32)
    key_body_pos = jnp.asarray(key_body_pos, dtype=jnp.float32)

    ref_root_pos = root_pos[..., -1:, :]
    ref_heading_inv = yaw_inv_quat(root_quat[..., -1:, :])

    rel_root_pos = root_pos - ref_root_pos
    local_root_pos = quat_rotate(ref_heading_inv, rel_root_pos)
    local_root_vel = quat_rotate(ref_heading_inv, root_vel)
    local_root_ang_vel = quat_rotate(ref_heading_inv, root_ang_vel)
    local_root_quat = quat_mul(ref_heading_inv, root_quat)
    root_rot_6d = quat_to_6d(local_root_quat)

    rel_key_pos = key_body_pos - ref_root_pos[..., None, :]
    local_key_pos = quat_rotate(ref_heading_inv[..., None, :], rel_key_pos)
    local_key_pos = local_key_pos.reshape(local_key_pos.shape[:-2] + (-1,))

    return jnp.concatenate(
        [
            local_root_pos,
            root_rot_6d,
            joints,
            local_key_pos,
            local_root_vel,
            local_root_ang_vel,
        ],
        axis=-1,
    )


def compute_smp_obs(
    root_pos: jnp.ndarray,
    root_quat: jnp.ndarray,
    root_vel: jnp.ndarray,
    root_ang_vel: jnp.ndarray,
    joints: jnp.ndarray,
    key_body_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Builds flattened SMP observations from history tensors."""

    frames = compute_smp_frames(
        root_pos=root_pos,
        root_quat=root_quat,
        root_vel=root_vel,
        root_ang_vel=root_ang_vel,
        joints=joints,
        key_body_pos=key_body_pos,
    )
    return frames.reshape(frames.shape[:-2] + (-1,))


def _stack_body_positions(clips: Any, key_body_names: Sequence[str]) -> jnp.ndarray:
    return jnp.stack([clips.body_xpos(name) for name in key_body_names], axis=-2)


def reference_histories(
    clips: Any,
    clip_ids: jnp.ndarray,
    end_frame_ids: jnp.ndarray,
    spec: SMPFeatureSpec = SMPFeatureSpec(),
) -> dict[str, jnp.ndarray]:
    """Fetches clipped reference histories from a ``ReferenceClips`` object."""

    clip_ids = jnp.asarray(clip_ids, dtype=jnp.int32)
    end_frame_ids = jnp.asarray(end_frame_ids, dtype=jnp.int32)
    offsets = jnp.arange(spec.num_history_steps, dtype=jnp.int32)
    offsets = offsets - (spec.num_history_steps - 1)
    frame_ids = jnp.clip(end_frame_ids[..., None] + offsets, 0, clips.qpos.shape[1] - 1)

    def take(arr: jnp.ndarray) -> jnp.ndarray:
        return arr[clip_ids[:, None], frame_ids]

    key_body_pos = _stack_body_positions(clips, spec.key_body_names)
    return {
        "root_pos": take(clips.root_position),
        "root_quat": take(clips.root_quaternion),
        "root_vel": take(clips.velocity),
        "root_ang_vel": take(clips.angular_velocity),
        "joints": take(clips.joints),
        "key_body_pos": take(key_body_pos),
    }


def sample_reference_smp_obs(
    clips: Any,
    rng: jax.Array,
    num_samples: int,
    spec: SMPFeatureSpec = SMPFeatureSpec(),
) -> jnp.ndarray:
    """Samples flattened SMP observations from reference clips."""

    rng_clip, rng_frame = jax.random.split(rng)
    n_clips, n_frames = clips.qpos.shape[:2]
    clip_ids = jax.random.randint(rng_clip, (num_samples,), 0, n_clips)
    min_frame = spec.num_history_steps - 1
    end_frame_ids = jax.random.randint(rng_frame, (num_samples,), min_frame, n_frames)
    histories = reference_histories(clips, clip_ids, end_frame_ids, spec)
    return compute_smp_obs(**histories)


def validate_reference_metadata(
    clips: Any,
    expected_joint_names: Sequence[str] | None = None,
    key_body_names: Sequence[str] = DEFAULT_KEY_BODY_NAMES,
) -> None:
    """Validates that reference clips contain the expected rodent layout."""

    if clips.qpos.shape[-1] != 74 or clips.qvel.shape[-1] != 73:
        raise ValueError(
            "Rodent SMP expects qpos=74 and qvel=73, got "
            f"qpos={clips.qpos.shape[-1]} qvel={clips.qvel.shape[-1]}."
        )
    if expected_joint_names is not None and list(clips.joint_names) != list(
        expected_joint_names
    ):
        raise ValueError("Reference joint names do not match the environment model.")
    missing = [name for name in key_body_names if name not in clips.body_names]
    if missing:
        raise ValueError(f"Reference clips are missing key bodies: {missing}")


def metadata_from_reference(
    clips: Any,
    spec: SMPFeatureSpec,
    data_path: str,
) -> dict[str, Any]:
    """Builds serializable metadata for a trained SMP prior."""

    return {
        "data_path": data_path,
        "feature_spec": spec.to_dict(),
        "joint_names": list(clips.joint_names),
        "body_names": list(clips.body_names),
        "num_clips": int(clips.qpos.shape[0]),
        "clip_length": int(clips.qpos.shape[1]),
        "qpos_dim": int(clips.qpos.shape[-1]),
        "qvel_dim": int(clips.qvel.shape[-1]),
        "scale_factor": (
            float(clips._config["model"]["SCALE_FACTOR"])
            if getattr(clips, "_config", None) is not None
            else None
        ),
    }


def numpy_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Converts numpy/JAX scalar values in metadata into JSON-friendly values."""

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    return convert(metadata)
