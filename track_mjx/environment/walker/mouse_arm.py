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

_XML_PATH = "/root/vast/eric/track-mjx/track_mjx/environment/walker/assets/mouse_arm/akira_torque.xml"
_PAIR_XML_PATH = "/root/vast/eric/track-mjx/track_mjx/environment/walker/assets/mouse_arm/akira_model_ghostpair.xml"


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
            rescale_factor: Factor to rescale the model. Default is 0.9.

        Returns:
            mjcf_dm.Physics: Loaded MJCF physics model.
        """
        path = Path(__file__).resolve().parent / _XML_PATH
        root = mjcf_dm.from_path(path)

        # Apply torque actuator modifications if needed
        # if torque_actuators:
        #     for actuator in root.find_all("actuator"):
        #         actuator.gainprm = [actuator.forcerange[1]]

        # Rescale the entire model
        # rescale.rescale_subtree(root, rescale_factor, rescale_factor)
        return mjcf_dm.Physics.from_mjcf_model(root)

    def _initialize_indices(self) -> None:
        """Initialize indices for joints, bodies, and end-effectors based on the loaded model."""
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

        # Set torso index (assuming first body is torso/reference)
        self._torso_idx = self._body_idxs[0] if len(self._body_idxs) > 0 else 0

        # Storage for previous action to enable smooth action transitions
        self._prev_action = None

    def update_prev_action(self, action):
        """Store the current action for the next observation.

        Args:
            action: Current action to store
        """
        self._prev_action = action
