"""
Defines high level abstracted reacher class. All reacher will inherit this high level class.
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
class BaseReacher(ABC):
    """
    Abstract base class for different types of reacher models (mouse arm, etc.).
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

    def get_all_loc_joints(self, qpos: jp.ndarray) -> jp.ndarray:
        """Extracts all joint positional values (excluding the root position and orientation) from the state vector.

        Args:
            qpos (jp.ndarray): The full positional state vector of the model.

        Returns:
            jp.ndarray: The joint positions. Shape (num_joints,).
        """
        return qpos

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
    
    
