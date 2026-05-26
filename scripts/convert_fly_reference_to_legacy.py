"""Convert fly imitation reference H5 from named/semantic format to rat-style legacy flat format.

Named format on disk (under /all_clips/):
    position          (n_clips, n_frames, 3)
    quaternion        (n_clips, n_frames, 4)         [w, x, y, z]
    velocity          (n_clips, n_frames, 3)
    angular_velocity  (n_clips, n_frames, 3)
    joints            (n_clips, n_frames, n_joints)
    joints_velocity   (n_clips, n_frames, n_joints)
    body_positions    (n_clips, n_frames, n_bodies, 3)
    body_quaternions  (n_clips, n_frames, n_bodies, 4)

Legacy format on disk (root level, flat):
    qpos        (n_clips * n_frames, n_qpos)
    qvel        (n_clips * n_frames, n_qvel)
    xpos        (n_clips * n_frames, n_bodies, 3)
    xquat       (n_clips * n_frames, n_bodies, 4)
    names_qpos  (n_qpos,)   strings
    names_xpos  (n_bodies,) strings
    config      ()          YAML string

Usage:
    python -m scripts.convert_fly_reference_to_legacy <input.h5> <output.h5> [--fly-xml PATH]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Tuple

import h5py
import mujoco
import numpy as np
import yaml

DEFAULT_FLY_XML = Path(
    "/home/talmolab/Desktop/SalkResearch/vnl-playground/vnl_playground/tasks/fruitfly/xmls/fruitfly_force.xml"
)


def build_qpos_qvel(
    position: np.ndarray,
    quaternion: np.ndarray,
    joints: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    joints_velocity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose flat qpos / qvel from the named-format root + joint arrays.

    Assumes MuJoCo freejoint convention: qpos[0:3]=xyz, qpos[3:7]=[w,x,y,z], qpos[7:]=joints;
    qvel[0:3]=linvel, qvel[3:6]=angvel, qvel[6:]=joint_velocities.
    """
    qpos = np.concatenate([position, quaternion, joints], axis=-1)
    qvel = np.concatenate([velocity, angular_velocity, joints_velocity], axis=-1)
    return qpos, qvel


def resolve_body_names(
    model,
    model_xpos_frame: np.ndarray,
    model_xquat_frame: np.ndarray,
    h5_xpos_frame: np.ndarray,
    h5_xquat_frame: np.ndarray,
    atol: float = 1e-4,
) -> list[str]:
    """For each H5 body index j, find the unique MJCF body whose (xpos, xquat)
    matches the H5 values to within atol, and return its name.

    When multiple MJCF bodies are coincident (e.g. walker/thorax share the
    same world frame), the tie is broken by assigning each successive H5 body
    to the lowest-indexed unoccupied candidate. This preserves the original
    body ordering from the H5 (which was generated before eye-camera additions
    and stored bodies in MJCF index order).

    Aborts (raises ValueError) on no-match (after tie-breaking) or if the
    resulting mapping is not a permutation.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded fly model.
    model_xpos_frame : np.ndarray, shape (model.nbody, 3)
        Body world positions from `mj_forward` on a specific qpos.
    model_xquat_frame : np.ndarray, shape (model.nbody, 4)
        Body world quaternions [w,x,y,z] from the same `mj_forward`.
    h5_xpos_frame : np.ndarray, shape (n_h5, 3)
        H5-reported body world positions for the corresponding frame.
    h5_xquat_frame : np.ndarray, shape (n_h5, 4)
        H5-reported body world quaternions for the corresponding frame.
    atol : float
        Absolute tolerance for matching positions and quaternions.
    """
    n_h5 = h5_xpos_frame.shape[0]
    used: set[int] = set()  # mjcf indices already assigned
    names: list[str] = []
    for j in range(n_h5):
        pos_match = np.all(np.abs(model_xpos_frame - h5_xpos_frame[j]) <= atol, axis=-1)
        quat_match = np.all(np.abs(model_xquat_frame - h5_xquat_frame[j]) <= atol, axis=-1)
        candidates = np.where(pos_match & quat_match)[0]
        # Filter out already-assigned bodies (tie-breaking for coincident bodies)
        available = [int(c) for c in candidates if int(c) not in used]
        if len(available) == 0:
            if candidates.size == 0:
                raise ValueError(
                    f"H5 body {j} has no MJCF match (xpos={h5_xpos_frame[j]}, "
                    f"xquat={h5_xquat_frame[j]}). Closest MJCF body: "
                    f"{int(np.argmin(np.linalg.norm(model_xpos_frame - h5_xpos_frame[j], axis=-1)))}."
                )
            else:
                raise ValueError(
                    f"H5 body {j} has no MJCF match: all candidates "
                    f"{list(candidates)} already assigned. Not a permutation."
                )
        # Among available candidates, pick the lowest index (preserves MJCF order)
        k = min(available)
        used.add(k)
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, k))
    return names
