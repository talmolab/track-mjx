"""Build the per-frame motion descriptor consumed by the latent prior.

Default form follows Eq. 4 of Wang et al. (LatentMimic, 2026):
    m_t = (p, theta, v, q, qdot)         feat_dim = 3 + 4 + 6 + n + n

When ``use_qvel=False``, both the root-velocity block (v) and the joint-
velocity block (qdot) are dropped, leaving only kinematic state:
    m_t = (p, theta, q)                  feat_dim = 3 + 4 + n
"""
from typing import Protocol

import numpy as np


def MOTION_FRAME_DIM(n_joints: int, use_qvel: bool = True) -> int:
    if use_qvel:
        return 3 + 4 + 6 + n_joints + n_joints
    return 3 + 4 + n_joints


class _ClipsLike(Protocol):
    qpos: np.ndarray  # (..., 7 + n_joints)
    qvel: np.ndarray  # (..., 6 + n_joints)


def extract_motion_frames(
    clips: _ClipsLike, n_joints: int, use_qvel: bool = True
) -> np.ndarray:
    """Concatenate motion descriptor components along the last axis.

    Returns
    -------
    np.ndarray
        Same leading shape as ``clips.qpos``, last axis = MOTION_FRAME_DIM.
    """
    qpos = np.asarray(clips.qpos, dtype=np.float32)
    if qpos.shape[-1] != 7 + n_joints:
        raise ValueError(
            f"qpos last axis {qpos.shape[-1]} != 7 + n_joints ({7 + n_joints})"
        )
    p = qpos[..., :3]
    theta = qpos[..., 3:7]
    q = qpos[..., 7:]
    if not use_qvel:
        return np.concatenate([p, theta, q], axis=-1)
    qvel = np.asarray(clips.qvel, dtype=np.float32)
    if qvel.shape[-1] != 6 + n_joints:
        raise ValueError(
            f"qvel last axis {qvel.shape[-1]} != 6 + n_joints ({6 + n_joints})"
        )
    v = qvel[..., :6]
    qdot = qvel[..., 6:]
    return np.concatenate([p, theta, v, q, qdot], axis=-1)
