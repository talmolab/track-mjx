import jax
from jax import numpy as jp
from pathlib import Path

from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

import mujoco
from track_mjx.environment.walker.base import BaseWalker


class FlyBody(BaseWalker):
    """FlyBody class that manages the body structure, joint configurations, and model loading."""

    def __init__(
        self,
        xml_path: str,
        joint_names: list[str],
        body_names: list[str],
        end_eff_names: list[str],
        torque_actuators: bool = False,
        rescale_factor: float = 0.9,
    ):
        """
        Initialize the fly body model with optional torque actuator settings and rescaling.

        Args:
            xml_path: Path to the MJCF XML file describing the model.
            joint_names: Names of the joints in the model.
            body_names: Names of the bodies in the model.
            end_eff_names: Names of the end effectors in the model.
            torque_actuators: Whether to use torque actuators. Default is False.
            rescale_factor: Factor to rescale the model. Default is 0.9.
        """
        self._xml = xml_path
        self._joint_names = joint_names
        self._body_names = body_names
        self._end_eff_names = end_eff_names

        self._mjcf_model = self._load_mjcf_model(torque_actuators, rescale_factor)
        self.sys = mjcf_brax.load_model(self._mjcf_model.model.ptr)

        self._initialize_indices()

    def _load_mjcf_model(
        self, torque_actuators: bool = False, rescale_factor: float = 0.9
    ) -> mjcf_dm.Physics:
        """Load and optionally modify the MJCF model.

        Args:
            torque_actuators: Whether to use torque actuators. Default is False.
            rescale_factor: Factor to rescale the model. Default is 0.9.

        Returns:
            mjcf_dm.Physics: Loaded MJCF physics model.
        """
        _XML_PATH = Path(__file__).resolve().parent / self._xml
        root = mjcf_dm.from_path(_XML_PATH)

        # torque
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        # rescale
        rescale.rescale_subtree(root, rescale_factor, rescale_factor)
        return mjcf_dm.Physics.from_mjcf_model(root)

    def _initialize_indices(self) -> None:
        """Initialize indices for joints, bodies, end-effectors, and thorax."""
        self._joint_idxs = jp.array(
            [
                self._mjcf_model.model.name2id(joint, "joint")
                for joint in self._joint_names
            ]
        )

        self._body_idxs = jp.array(
            [self._mjcf_model.model.name2id(body, "body") for body in self._body_names]
        )

        self._endeff_idxs = jp.array(
            [
                self._mjcf_model.model.name2id(end_eff, "body")
                for end_eff in self._end_eff_names
            ]
        )

        self._thorax_idx = self._mjcf_model.model.name2id("thorax", "body")

    def get_thorax_position(self, xpos: jp.ndarray) -> jp.ndarray:
        """Retrieve the thorax's position.

        Args:
            xpos: The full positional state of the model.

        Returns:
            jp.ndarray: Thorax position.
        """
        return xpos[self._thorax_idx]

    def compute_local_track_positions(
        self, ref_positions: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute local position differences for tracking, rotated to align with fly orientation.

        Args:
            ref_positions: Reference positions.
            qpos: Full state vector containing orientation quaternion.

        Returns:
            Local position differences rotated to align with fly orientation.
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
            ref_quats: Reference quaternions.
            qpos: Full state vector containing orientation quaternion.

        Returns:
            Quaternion distances.
        """
        quat = self.get_root_quaternion_from_qpos(qpos)
        quat_dist = jax.vmap(
            lambda ref_quat, agent_quat: brax_math.relative_quat(ref_quat, agent_quat),
            in_axes=(0, None),
        )(ref_quats, quat).flatten()

        return quat_dist

    def compute_local_body_positions(
        self, ref_positions: jp.ndarray, xpos: jp.ndarray, qpos: jp.ndarray
    ) -> jp.ndarray:
        """Compute local body positions relative to reference positions, rotated by the fly's orientation.

        Args:
            ref_positions: Reference body positions.
            xpos: Fly's current body positions.
            qpos: Fly's full state vector, including orientation quaternion.

        Returns:
            Local body position differences, rotated to align with fly orientation.
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
