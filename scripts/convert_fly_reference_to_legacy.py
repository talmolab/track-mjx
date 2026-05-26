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
