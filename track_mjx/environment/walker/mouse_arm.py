import jax
import jax.numpy as jp
import numpy as np
import os

from pathlib import Path
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math

from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

import mujoco
from mujoco import mjx

from track_mjx.environment.walker.base import BaseWalker

_XML_PATH = "/root/vast/eric/track-mjx/track_mjx/environment/walker/assets/mouse_arm/arm_model_v3_torque.xml"
_PAIR_XML_PATH = "/root/vast/eric/track-mjx/track_mjx/environment/walker/assets/mouse_arm/arm_model_v3_ghostpair.xml"


class MouseArm(BaseWalker):
    """MouseArm class that manages the body structure, joint configurations, and model loading."""

    def __init__(
        self,
        joint_names: list[str],
        body_names: list[str],
        end_eff_names: list[str],
        torque_actuators: bool = False,
        rescale_factor: float = 1.0,
    ):
        """Initialize the MouseArm model with optional torque actuator settings and rescaling.

        Args:
            joint_names: Names of the joints in the model.
            body_names: Names of the bodies in the model.
            end_eff_names: Names of the end effectors in the model.
            torque_actuators: Whether to use torque actuators. Default is False.
            rescale_factor: Factor to rescale the model. Default is 1.0.
        """
        self._joint_names = joint_names
        self._body_names = body_names
        self._end_eff_names = end_eff_names
        self._pair_rendering_xml_path = _PAIR_XML_PATH

        self._mjcf_model = self._load_mjcf_model(torque_actuators, rescale_factor)
        self.sys = mjcf_brax.load_model(self._mjcf_model.model.ptr)

        self._initialize_indices()

    def _load_mjcf_model(
        self, torque_actuators: bool = False, rescale_factor: float = 1.0
    ) -> mjcf_dm.Physics:
        """Load and optionally modify the MJCF model.

        Args:
            torque_actuators: Whether to use torque actuators. Default is False.
            rescale_factor: Factor to rescale the model. Default is 1.0.

        Returns:
            mjcf_dm.Physics: Loaded MJCF physics model.
        """
        path = Path(_XML_PATH)
        root = mjcf_dm.from_path(path)

        # Apply torque actuator modifications if needed
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        # Rescale the entire model
        rescale.rescale_subtree(root, rescale_factor, rescale_factor)
        return mjcf_dm.Physics.from_mjcf_model(root)

    def _initialize_indices(self) -> None:
        """Initialize indices for joints, bodies, and end-effectors based on the loaded model."""

        # Decode model names if they are in bytes
        raw_model_names = self._mjcf_model.model.names
        if isinstance(raw_model_names, bytes):
            model_joint_names = raw_model_names.decode("utf-8").split("\x00")
        else:
            model_joint_names = raw_model_names

        print(f"DEBUG: Available joint names in model = {model_joint_names}")

        # Extract joint indices
        self._joint_idxs = jp.array([
            self._mjcf_model.model.name2id(joint, "joint")
            for joint in self._joint_names
            if joint in model_joint_names
        ])

        # If no joint indices found, raise an error
        if self._joint_idxs.shape[0] == 0:
            raise ValueError(
                f"BUG: _joint_idxs is empty! Joint names = {self._joint_names} "
                f"Available joints = {model_joint_names}"
            )

        # Extract body indices
        self._body_idxs = jp.array([
            self._mjcf_model.model.name2id(body, "body")
            for body in self._body_names
            if body in model_joint_names
        ])

        # Extract end-effector indices
        self._endeff_idxs = jp.array([
            self._mjcf_model.model.name2id(end_eff, "body")
            for end_eff in self._end_eff_names
            if end_eff in model_joint_names
        ])

        print(f"DEBUG: _joint_idxs = {self._joint_idxs}")
        print(f"DEBUG: _body_idxs = {self._body_idxs}")
        print(f"DEBUG: _endeff_idxs = {self._endeff_idxs}")

    def get_all_loc_joints(self, qpos: jp.ndarray) -> jp.ndarray:
        """Extracts all joint positions from state vector.

        Args:
            qpos (jp.ndarray): The full positional state vector of the model.

        Returns:
            jp.ndarray: The joint positions. Shape (num_joints,).
        """
        joints = qpos

        return joints

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

        if joints.shape[0] == 0:
            raise ValueError(f"Shape mismatch! ref_joints: {ref_joints.shape}, joints: {joints.shape}")

        joint_dist = (ref_joints - joints)[self._joint_idxs].flatten()
        return joint_dist
