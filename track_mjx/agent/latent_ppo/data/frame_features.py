"""Build m_t = (p, theta, v, q, qdot) from ReferenceClips arrays.

This matches Eq. 4 of Wang et al. (LatentMimic, 2026): the per-frame motion
descriptor consumed by the encoder and reconstructed by the decoder.
"""
from typing import Protocol

import numpy as np


def MOTION_FRAME_DIM(n_joints: int) -> int:
    return 3 + 4 + 6 + n_joints + n_joints


class _ClipsLike(Protocol):
    qpos: np.ndarray  # (..., 7 + n_joints)
    qvel: np.ndarray  # (..., 6 + n_joints)


def extract_motion_frames(clips: _ClipsLike, n_joints: int) -> np.ndarray:
    """Concatenate (p, theta, v, q, qdot) along the last axis.

    Returns
    -------
    np.ndarray
        Same leading shape as ``clips.qpos``, last axis = MOTION_FRAME_DIM.
    """
    qpos = np.asarray(clips.qpos, dtype=np.float32)
    qvel = np.asarray(clips.qvel, dtype=np.float32)
    if qpos.shape[-1] != 7 + n_joints:
        raise ValueError(
            f"qpos last axis {qpos.shape[-1]} != 7 + n_joints ({7 + n_joints})"
        )
    if qvel.shape[-1] != 6 + n_joints:
        raise ValueError(
            f"qvel last axis {qvel.shape[-1]} != 6 + n_joints ({6 + n_joints})"
        )
    p = qpos[..., :3]
    theta = qpos[..., 3:7]
    v = qvel[..., :6]
    q = qpos[..., 7:]
    qdot = qvel[..., 6:]
    return np.concatenate([p, theta, v, q, qdot], axis=-1)
