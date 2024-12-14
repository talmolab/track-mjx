import jax
from jax import numpy as jp
from pathlib import Path

from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

import mujoco
from track_mjx.environment.walker.base import BaseWalker

_XML_PATH = "assets/fruitfly/fruitfly_force_fast.xml"
_PAIR_XML_PATH = "assets/fruitfly/fruitfly_force_pair.xml"


class Fly(BaseWalker):
    """FlyBody class that manages the body structure, joint configurations, and model loading."""

    def __init__(
        self,
        joint_names: list[str],
        body_names: list[str],
        end_eff_names: list[str],
        torque_actuators: bool = False,
        rescale_factor: float = 1.0,
    ):
        """
        Initialize the fly body model with optional torque actuator settings and rescaling.

        Args:
            joint_names: Names of the joints in the model.
            body_names: Names of the bodies in the model.
            end_eff_names: Names of the end effectors in the model.
            torque_actuators: Whether to use torque actuators. Default is False.
            rescale_factor: Factor to rescale the model. Default is 0.9.
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
        path = Path(__file__).resolve().parent / _XML_PATH
        root = mjcf_dm.from_path(path)

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

        # Treat thorax as torso
        self._torso_idx = self._mjcf_model.model.name2id("thorax", "body")
