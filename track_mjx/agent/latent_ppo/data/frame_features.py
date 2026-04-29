"""Build the per-frame motion descriptor consumed by the latent prior.

Default form follows Eq. 4 of Wang et al. (LatentMimic, 2026):
    m_t = (p, theta, v, q, qdot)         feat_dim = 3 + 4 + 6 + n + n

When ``use_qvel=False``, both the root-velocity block (v) and the joint-
velocity block (qdot) are dropped, leaving only kinematic state:
    m_t = (p, theta, q)                  feat_dim = 3 + 4 + n

When ``active_joints`` is supplied, ``q`` (and ``qdot`` if used) only includes
those indices into the 0..n_joints-1 articulated-joint range, dropping joints
that don't move in the dataset (e.g. fingers/tail vertebrae of the rat).
``feat_dim`` shrinks to 3 + 4 + len(active_joints) (or with the velocity block).
"""
from typing import Optional, Protocol, Sequence

import numpy as np


def MOTION_FRAME_DIM(
    n_joints: int, use_qvel: bool = True, n_active_joints: Optional[int] = None
) -> int:
    n = int(n_active_joints) if n_active_joints is not None else n_joints
    if use_qvel:
        return 3 + 4 + 6 + n + n
    return 3 + 4 + n


class _ClipsLike(Protocol):
    qpos: np.ndarray  # (..., 7 + n_joints)
    qvel: np.ndarray  # (..., 6 + n_joints)


def extract_motion_frames(
    clips: _ClipsLike,
    n_joints: int,
    use_qvel: bool = True,
    active_joints: Optional[Sequence[int]] = None,
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
    q_full = qpos[..., 7:]
    if active_joints is not None:
        idx = np.asarray(active_joints, dtype=np.int64)
        q = q_full[..., idx]
    else:
        q = q_full
    if not use_qvel:
        return np.concatenate([p, theta, q], axis=-1)
    qvel = np.asarray(clips.qvel, dtype=np.float32)
    if qvel.shape[-1] != 6 + n_joints:
        raise ValueError(
            f"qvel last axis {qvel.shape[-1]} != 6 + n_joints ({6 + n_joints})"
        )
    v = qvel[..., :6]
    qdot_full = qvel[..., 6:]
    if active_joints is not None:
        qdot = qdot_full[..., idx]
    else:
        qdot = qdot_full
    return np.concatenate([p, theta, v, q, qdot], axis=-1)


def auto_detect_active_joints(
    clips: _ClipsLike, n_joints: int, std_threshold: float = 1e-3
) -> np.ndarray:
    """Return integer indices into [0, n_joints) of joints whose std exceeds the
    threshold across all frames of all clips. Used to drop dead joints (e.g.
    rat finger / tail vertebrae) before pre-training the latent prior.
    """
    qpos = np.asarray(clips.qpos, dtype=np.float32)
    if qpos.shape[-1] != 7 + n_joints:
        raise ValueError(
            f"qpos last axis {qpos.shape[-1]} != 7 + n_joints ({7 + n_joints})"
        )
    q = qpos[..., 7:].reshape(-1, n_joints)
    std = q.std(axis=0)
    active = np.where(std > float(std_threshold))[0].astype(np.int64)
    return active


# Substrings used to classify rat joints into "limb" vs "axial" body parts.
# Limb = anything in the four limbs (hindlimb scapular/shoulder, forelimb).
# Axial = vertebrae (lumbar + caudal + cervical), atlas/axis/mandible.
_LIMB_SUBSTRINGS = (
    "hip_", "knee_", "ankle_", "toe_",
    "scapula_", "shoulder_", "shoulder_sup_",
    "elbow_", "wrist_", "finger_",
)


def classify_joint_groups(
    joint_names: Sequence[str],
    active_joints: Sequence[int],
) -> dict[str, np.ndarray]:
    """Split active joints into body groups based on joint name patterns.

    Returns
    -------
    dict with keys {"limb", "axial"}, each an int64 numpy array of POSITIONS
    inside the active_joints array (NOT raw joint indices). These positions
    can be used directly to slice the q_active block of the motion-feature
    vector (which is laid out as [root_pos(3), root_quat(4), q_active(...)]).
    """
    limb_pos: list[int] = []
    axial_pos: list[int] = []
    for pos, joint_idx in enumerate(active_joints):
        name = joint_names[int(joint_idx)]
        if any(s in name for s in _LIMB_SUBSTRINGS):
            limb_pos.append(pos)
        else:
            axial_pos.append(pos)
    return {
        "limb": np.asarray(limb_pos, dtype=np.int64),
        "axial": np.asarray(axial_pos, dtype=np.int64),
    }


def feat_indices_for_groups(
    group_active_positions: dict[str, np.ndarray],
    n_active_joints: int,
    use_qvel: bool = False,
    root_as_separate_branch: bool = False,
) -> dict[str, np.ndarray]:
    """Build feat-vector indices for each body group, given group positions in
    the active-joint array.

    Motion feature layout (use_qvel=False):
        [root_pos(3), root_quat(4), q_active(n_active_joints)]
        i.e. feat[0:7] = root, feat[7+pos] = active joint at position `pos`.

    Motion feature layout (use_qvel=True):
        [root_pos(3), root_quat(4), root_vel(6),
         q_active(n_active_joints), qdot_active(n_active_joints)]
        Layout: feat[0:7] = root pos+quat, feat[7:13] = root vel (3 lin + 3 ang),
        feat[13+pos] = joint angle at active position pos,
        feat[13+n_active_joints+pos] = joint vel at active position pos.

    2-way split (`root_as_separate_branch=False`, default):
      - axial branch: root_pos+root_quat (+ root_vel if qvel)
                      + axial_joint_pos (+ axial_joint_vel if qvel)
      - limb  branch: limb_joint_pos (+ limb_joint_vel if qvel)

    3-way split (`root_as_separate_branch=True`):
      - root  branch: root_pos+root_quat (+ root_vel if qvel)        # base only
      - axial branch: axial_joint_pos (+ axial_joint_vel if qvel)    # NO root
      - limb  branch: limb_joint_pos (+ limb_joint_vel if qvel)
      Returned keys are exactly {"root", "axial", "limb"}. The caller is
      responsible for constructing branch_names / branch_input_indices in
      this fixed order.
    """
    n = int(n_active_joints)
    qpos_offset = 7
    if use_qvel:
        qpos_offset = 7 + 6
        qdot_offset = qpos_offset + n

    out: dict[str, np.ndarray] = {}

    # Joint indices per group (axial + limb), shared between 2-way and 3-way.
    for name, positions in group_active_positions.items():
        positions = np.asarray(positions, dtype=np.int64)
        joint_pos_feats = qpos_offset + positions
        parts = [joint_pos_feats]
        if use_qvel:
            parts.append(qdot_offset + positions)
        joint_only = np.concatenate(parts)
        if name == "axial" and not root_as_separate_branch:
            # Legacy 2-way: axial branch absorbs root.
            root_parts = [np.arange(7, dtype=np.int64)]
            if use_qvel:
                root_parts.append(np.arange(7, 13, dtype=np.int64))
            out[name] = np.concatenate(root_parts + parts)
        else:
            # Either limb branch in 2-way mode, OR axial/limb branch in 3-way mode.
            out[name] = joint_only

    if root_as_separate_branch:
        root_parts = [np.arange(7, dtype=np.int64)]
        if use_qvel:
            root_parts.append(np.arange(7, 13, dtype=np.int64))
        out["root"] = np.concatenate(root_parts)

    return out
