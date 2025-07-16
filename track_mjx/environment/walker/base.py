"""
Defines high level abstracted walker class. All walker will inherit this high level class.
"""

from abc import ABC, abstractmethod
import jax.numpy as jp
import jax
from brax import math as brax_math
from brax.io import mjcf
import mujoco
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class BaseWalker(ABC):
    """
    Abstract base class for different types of walker models (rodents, flies, mice, etc.).
    Defines the interface that all walker implementations must follow.
    """

    # public, constructor‐args
    joint_names: Sequence[str]
    body_names:  Sequence[str]
    end_eff_names: Sequence[str]
    torque_actuators: bool = False
    rescale_factor: float = 1.0

    # private fields that will be assigned later
    _joint_names: list[str]         = field(init=False, repr=False)
    _body_names:  Sequence[str]        = field(init=False, repr=False)
    _end_eff_names: Sequence[str]      = field(init=False, repr=False)
    _body_idxs:   jp.ndarray        = field(init=False, repr=False)
    _endeff_idxs: jp.ndarray        = field(init=False, repr=False)
    _torso_idx:   jp.ndarray        = field(init=False, repr=False)
    _mjcf_model:  Any               = field(init=False, repr=False)
    sys:          mujoco.MjModel    = field(init=False, repr=False)
    _mj_model:    mujoco.MjModel    = field(init=False, repr=False)
    _mj_spec:     mujoco.MjSpec     = field(init=False, repr=False)

    @abstractmethod
    def _build_spec(
        self, torque_actuators: bool, rescale_factor: float
    ) -> mujoco.MjSpec:
        """
        Parse XML → MjSpec, apply optional edits, and return the spec.

        Args:
            torque_actuators (bool): Whether to use torque actuators
            rescale_factor (float): Factor to rescale the model

        Returns:
            mujoco.MjSpec: mujoco spec that contains the model
        """
        pass

    @abstractmethod
    def _initialize_indices(self) -> None:
        """
        Initialize indices for joints, bodies, and end-effectors.
        Each implementation should set at minimum:
        - self._joint_idxs
        - self._body_idxs
        - self._endeff_idxs
        - self._torso_idx
        """
        pass

    @property
    def joint_idxs(self) -> jp.ndarray:
        """Get joint indices."""
        return self._joint_idxs

    @property
    def body_idxs(self) -> jp.ndarray:
        """Get body indices."""
        return self._body_idxs

    @property
    def endeff_idxs(self) -> jp.ndarray:
        """Get end effector indices."""
        return self._endeff_idxs

    @property
    def torso_idx(self) -> jp.ndarray:
        """Get torso index."""
        return self._torso_idx

    def get_joint_positions(self, qpos: jp.ndarray) -> jp.ndarray:
        """
        Get joint positions from state.

        Args:
            qpos: Full position state vector

        Returns:
            Joint positions vector
        """
        return qpos[self.joint_idxs]

    def get_body_positions(self, xpos: jp.ndarray) -> jp.ndarray:
        """
        Get body positions from state.

        Args:
            xpos: Full position state vector

        Returns:
            Body positions vector
        """
        return xpos[self.body_idxs]

    def get_end_effector_positions(self, xpos: jp.ndarray) -> jp.ndarray:
        """
        Get end effector positions from state.

        Args:
            xpos: Full position state vector

        Returns:
            End effector positions vector
        """
        return xpos[self.endeff_idxs]

    def get_torso_position(self, xpos: jp.ndarray) -> jp.ndarray:
        """
        Get torso position from state.

        Args:
            xpos: Full position state vector

        Returns:
            Torso position vector
        """
        return xpos[self.torso_idx]

    def get_root_from_qpos(self, qpos: jp.ndarray) -> jp.ndarray:
        """Extracts the root's positional values (x, y, z) from the state vector.

        Args:
            qpos (jp.ndarray): The full positional state vector of the model.

        Returns:
            jp.ndarray: The root's positional values. Shape (3,).
        """
        return qpos[:3]

    def get_root_quaternion_from_qpos(self, qpos: jp.ndarray) -> jp.ndarray:
        """Extracts the root's orientation quaternion (qw, qx, qy, qz) from the state vector.

        Args:
            qpos (jp.ndarray): The full positional state vector of the model.

        Returns:
            jp.ndarray: The root's orientation quaternion. Shape (4,).
        """
        return qpos[3:7]

    def get_all_loc_joints(self, qpos: jp.ndarray) -> jp.ndarray:
        """Extracts all joint positional values (excluding the root position and orientation) from the state vector.

        Args:
            qpos (jp.ndarray): The full positional state vector of the model.

        Returns:
            jp.ndarray: The joint positions. Shape (num_joints,).
        """
        return qpos[7:]

    def compute_local_track_positions(
        self, ref_positions: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute local position differences for tracking, rotated to align with agent orientation.

        Args:
            ref_positions: Reference positions
            qpos: Full state vector containing orientation quaternion

        Returns:
            Local position differences rotated to align with agent orientation
        """
        root = self.get_root_from_qpos(qpos)
        quat = self.get_root_quaternion_from_qpos(qpos)

        track_pos_local = jax.vmap(
            lambda pos, quat: brax_math.rotate(pos, quat),
            in_axes=(0, None),
        )(ref_positions - root, quat).flatten()

        return track_pos_local

    def compute_quat_distances(
        self, ref_quats: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute quaternion distances for rotational alignment.

        Args:
            ref_quats: Reference quaternions
            qpos: Full state vector containing orientation quaternion

        Returns:
            Quaternion distances
        """
        quat = self.get_root_quaternion_from_qpos(qpos)
        quat_dist = jax.vmap(
            lambda ref_quat, agent_quat: brax_math.relative_quat(ref_quat, agent_quat),
            in_axes=(0, None),
        )(ref_quats, quat).flatten()

        return quat_dist

    def compute_local_joint_distances(
        self, ref_joints: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute joint distances relative to reference joints.

        Args:
            ref_joints: Reference joint positions
            qpos: Full state vector containing joint positions

        Returns:
            Joint distances Shape (num_joints,).
        """
        joints = self.get_all_loc_joints(qpos)
        joint_dist = (ref_joints - joints)[:, self._joint_idxs].flatten()

        return joint_dist

    def compute_local_body_positions(
        self, ref_positions: jp.ndarray, xpos: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute local body positions relative to reference positions, rotated by the agent's orientation.

        Args:
            ref_positions: Reference body positions
            xpos: Agent's current body positions
            qpos: Agent's full state vector, including orientation quaternion

        Returns:
            Local body position differences, rotated to align with agent orientation
        """
        quat = self.get_root_quaternion_from_qpos(qpos)
        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_positions - xpos)[:, self._body_idxs],
            quat,
        ).flatten()

        return body_pos_dist_local
    
    def compute_local_track_velocities_with_qpos(
        self, ref_velocities: jp.ndarray, qvel: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute local velocity differences for tracking, rotated to align with agent orientation.

        Args:
            ref_velocities: Reference linear velocities
            qvel: Full velocity vector containing linear and angular velocities
            qpos: Full position vector (needed for orientation)

        Returns:
            Local velocity differences rotated to align with agent orientation
        """
        root_vel = self.get_root_velocity_from_qvel(qvel)
        quat = self.get_root_quaternion_from_qpos(qpos)

        track_vel_local = jax.vmap(
            lambda vel, quat: brax_math.rotate(vel, quat),
            in_axes=(0, None),
        )(ref_velocities - root_vel, quat).flatten()

        return track_vel_local

    def compute_angular_velocity_distances(
        self, ref_angular_vels: jp.ndarray, qvel: jp.ndarray
    ) -> jp.ndarray:
        """Compute angular velocity distances for rotational tracking.

        Args:
            ref_angular_vels: Reference angular velocities
            qvel: Full velocity vector containing angular velocities

        Returns:
            Angular velocity differences
        """
        angular_vel = self.get_root_angular_velocity_from_qvel(qvel)
        angular_vel_dist = jax.vmap(
            lambda ref_vel, agent_vel: ref_vel - agent_vel,
            in_axes=(0, None),
        )(ref_angular_vels, angular_vel).flatten()

        return angular_vel_dist

    def compute_joint_velocity_distances(
        self, ref_joint_vels: jp.ndarray, qvel: jp.ndarray
    ) -> jp.ndarray:
        """Compute joint velocity distances relative to reference joint velocities.

        Args:
            ref_joint_vels: Reference joint velocities
            qvel: Full velocity vector containing joint velocities

        Returns:
            Joint velocity differences
        """
        joint_vels = self.get_joint_velocities_from_qvel(qvel)
        joint_vel_dist = jax.vmap(
            lambda ref_vel, agent_vel: ref_vel - agent_vel,
            in_axes=(0, None),
        )(ref_joint_vels, joint_vels).flatten()

        return joint_vel_dist

    def get_root_velocity_from_qvel(self, qvel: jp.ndarray) -> jp.ndarray:
        """Extracts the root's linear velocity (vx, vy, vz) from the velocity vector.

        Args:
            qvel (jp.ndarray): The full velocity state vector of the model.

        Returns:
            jp.ndarray: The root's linear velocity. Shape (3,).
        """
        return qvel[:3]

    def get_root_angular_velocity_from_qvel(self, qvel: jp.ndarray) -> jp.ndarray:
        """Extracts the root's angular velocity (ωx, ωy, ωz) from the velocity vector.

        Args:
            qvel (jp.ndarray): The full velocity state vector of the model.

        Returns:
            jp.ndarray: The root's angular velocity. Shape (3,).
        """
        return qvel[3:6]

    def get_joint_velocities_from_qvel(self, qvel: jp.ndarray) -> jp.ndarray:
        """Extracts joint velocities from the velocity vector.

        Args:
            qvel (jp.ndarray): The full velocity state vector of the model.

        Returns:
            jp.ndarray: Joint velocities. Shape (num_joints,).
        """
        return qvel[6:]
