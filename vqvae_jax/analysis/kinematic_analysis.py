"""Kinematic analysis for VQ-VAE codes.

This module provides functions for extracting kinematic features from
motion data and correlating them with code activations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KinematicFeatures:
    """Kinematic features extracted from motion data.

    Attributes:
        linear_velocity: Root linear velocity magnitude, shape [T].
        angular_velocity: Root angular velocity magnitude, shape [T].
        body_height: Root z position (body height), shape [T].
        joint_velocities: Mean absolute joint velocities, shape [T].
    """

    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    body_height: np.ndarray
    joint_velocities: np.ndarray


def extract_kinematic_features(
    qpos: np.ndarray,
    qvel: np.ndarray,
    dt: float = 0.02,
) -> KinematicFeatures:
    """Extract kinematic features from position and velocity data.

    Assumes rodent model layout:
    - qpos[0:3]: root position (x, y, z)
    - qpos[3:7]: root quaternion
    - qpos[7:]: joint angles
    - qvel[0:3]: root linear velocity
    - qvel[3:6]: root angular velocity
    - qvel[6:]: joint velocities

    Args:
        qpos: Generalized positions, shape [T, nq].
        qvel: Generalized velocities, shape [T, nv].
        dt: Timestep for velocity computation.

    Returns:
        KinematicFeatures with extracted features.
    """
    T = qpos.shape[0]

    # Linear velocity from qvel (root linear velocity)
    if qvel.shape[1] >= 3:
        linear_velocity = np.linalg.norm(qvel[:, :3], axis=1)
    else:
        # Fallback: compute from position differences
        dpos = np.diff(qpos[:, :3], axis=0) / dt
        linear_velocity = np.zeros(T)
        linear_velocity[1:] = np.linalg.norm(dpos, axis=1)

    # Angular velocity from qvel
    if qvel.shape[1] >= 6:
        angular_velocity = np.linalg.norm(qvel[:, 3:6], axis=1)
    else:
        angular_velocity = np.zeros(T)

    # Body height (z position)
    body_height = qpos[:, 2]

    # Joint velocities (mean absolute velocity across joints)
    if qvel.shape[1] > 6:
        joint_velocities = np.mean(np.abs(qvel[:, 6:]), axis=1)
    else:
        joint_velocities = np.zeros(T)

    return KinematicFeatures(
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        body_height=body_height,
        joint_velocities=joint_velocities,
    )
