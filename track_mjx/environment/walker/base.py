"""
Defines high level abstracted walker class. All walker will inherit this high level class.
"""

from abc import ABC, abstractmethod
import jax.numpy as jp
import jax
from brax import math as brax_math
from brax.io import mjcf
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale


class BaseWalker(ABC):
    """
    Abstract base class for different types of walker models (rodents, flies, mice, etc.).
    Defines the interface that all walker implementations must follow.
    """

    def __init__(
        self,
        joint_names: list[str],
        body_names: list[str],
        end_eff_names: list[str],
        torque_actuators: bool = False,
        rescale_factor: float = 1.0,
    ):
        """
        Initialize the walker with model configuration.

        Args:
            joint_names: List of joint names in the model
            body_names: List of body part names
            end_eff_names: List of end effector names
            torque_actuators: Whether to use torque actuators
            rescale_factor: Factor to rescale the model
        """
        self._joint_names = joint_names
        self._body_names = body_names
        self._end_eff_names = end_eff_names

        # Load and configure the model
        self._mjcf_model = self._load_mjcf_model(torque_actuators, rescale_factor)
        self.sys = mjcf.load_model(self._mjcf_model.model.ptr)
        self._initialize_indices()

    @abstractmethod
    def _load_mjcf_model(
        self, torque_actuators: bool, rescale_factor: float
    ) -> mjcf_dm.Physics:
        """
        Load and configure the MJCF model with optional torque actuators and rescaling.

        Args:
            torque_actuators: Whether to use torque actuators
            rescale_factor: Factor to rescale the model

        Returns:
            Configured MuJoCo physics object
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
        return qpos

    def get_all_loc_joints(self, qpos: jp.ndarray) -> jp.ndarray:
        """Extracts all joint positional values (excluding the root position and orientation) from the state vector.

        Args:
            qpos (jp.ndarray): The full positional state vector of the model.

        Returns:
            jp.ndarray: The joint positions. Shape (num_joints,).
        """
        return qpos

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
            ref_joints: Reference joint positions.
            qpos: Full state vector containing joint positions.

        Returns:
            Joint distances Shape (num_joints,).
        """
        joints = self.get_all_loc_joints(qpos)  # Extract all joint positions

        # 🚀 Debugging outputs
        #print(f"DEBUG: ref_joints.shape = {ref_joints.shape}")  # Expect (4,)
        #print(f"DEBUG: joints.shape = {joints.shape}")  # Expect (4,)
        #print(f"DEBUG: _joint_idxs.shape = {self._joint_idxs.shape}")  # Should match
        #print(f"DEBUG: _joint_idxs = {self._joint_idxs}")  # Ensure not empty

        # 🚀 Catch empty joint indices case
        if self._joint_idxs.shape[0] == 0:
            raise ValueError("BUG: _joint_idxs is empty! Check _initialize_indices().")

        # Ensure ref_joints and joints have the correct shape
        if ref_joints.shape != joints.shape:
            raise ValueError(
                f"Shape mismatch! ref_joints: {ref_joints.shape}, joints: {joints.shape}"
            )

        # 🚀 Ensure indexing is correct
        if self._joint_idxs.max() >= joints.shape[0]:
            raise ValueError(
                f"BUG: _joint_idxs contains out-of-range indices: {self._joint_idxs}"
            )

        joint_dist = (ref_joints - joints)[self._joint_idxs].flatten()

        return joint_dist

    def compute_local_body_positions(
        self, ref_positions: jp.ndarray, xpos: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute local body positions relative to reference positions, rotated by the agent's orientation.

        Args:
            ref_positions: Reference body positions (may have multiple frames).
            xpos: Agent's current body positions.
            qpos: Agent's full state vector, including orientation quaternion.

        Returns:
            Local body position differences, rotated to align with agent orientation.
        """

        # 🚀 Debugging print
        # print(
        #     f"DEBUG: ref_positions.shape = {ref_positions.shape}"
        # )  # Expect (5, 9, 3) or (9, 3)
        #print(f"DEBUG: xpos.shape = {xpos.shape}")  # Expect (9, 3)
        #print(f"DEBUG: qpos.shape = {qpos.shape}")  # Expect (4,)

        # If reference positions have multiple frames, select only the first frame
        if len(ref_positions.shape) == 3:
            ref_positions = ref_positions[0]  # Extract first frame
            print(f"DEBUG: Using first frame of ref_positions: {ref_positions.shape}")

        # Ensure valid body indices
        if self._body_idxs.shape[0] == 0:
            raise ValueError(
                f"BUG: _body_idxs is empty! ref_positions.shape = {ref_positions.shape}"
            )

        # Extract valid body positions
        if ref_positions.shape != xpos.shape:
            raise ValueError(
                f"Mismatch in ref_positions and xpos after correction: {ref_positions.shape} vs {xpos.shape}"
            )

        body_positions = (ref_positions - xpos)[self._body_idxs]

        # 🚀 Debugging print
        #print(f"DEBUG: body_positions.shape = {body_positions.shape}")  # Expect (N, 3)

        # Ensure valid dimensions for rotation
        if body_positions.shape[0] == 0 or body_positions.shape[1] != 3:
            raise ValueError(
                f"Invalid body positions shape! Expected (N,3), got {body_positions.shape}"
            )

        quat = self.get_root_quaternion_from_qpos(qpos)

        # 🚀 Debugging print
        #print(f"DEBUG: quat.shape before reshape = {quat.shape}")  # Expect (4,)
        #print(f"DEBUG: quat before reshape = {quat}")  # Expect (4,)

        # Ensure quaternion is of shape (4,) and not (1,)
        # if quat.shape[0] == 1:
        #    quat = jp.concatenate([quat, jp.zeros(3)])  # Pad with zeros if needed

        #print(f"DEBUG: quat.shape after reshape = {quat.shape}")  # Ensure it's (4,)
        #print(f"DEBUG: quat after reshape = {quat}")  # Ensure it's (4,)

        # Fix rotation step by ensuring correct input shapes
        body_pos_dist_local = jax.vmap(brax_math.rotate, in_axes=(0, None))(
            body_positions, quat
        )

        return body_pos_dist_local.flatten()
