import jax
from jax import numpy as jp
from pathlib import Path

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

from jax.numpy import inf, ndarray
import mujoco
from mujoco import mjx

import numpy as np
import os

from track_mjx.environment.walker.base import BaseWalker


class Rodent(BaseWalker):
    """Rodent class that manages the body structure,
    joint configurations, and model loading"""

    def __init__(
        self,
        xml_path,
        joint_names,
        body_names,
        end_eff_names,
        torque_actuators=False,
        rescale_factor=0.9,
    ):
        """Initialize the rodent model with optional
        torque actuator settings and rescaling"""

        self._xml = xml_path
        self._joint_names = joint_names
        self._body_names = body_names
        self._end_eff_names = end_eff_names

        self._mjcf_model = self._load_mjcf_model(torque_actuators, rescale_factor)
        self.sys = mjcf_brax.load_model(self._mjcf_model.model.ptr)

        self._initialize_indices()

    def _load_mjcf_model(self, torque_actuators=False, rescale_factor=0.9):
        """Only using this for walker, not its pair"""
        _XML_PATH = Path(__file__).resolve() / self._xml
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

    def _initialize_indices(self):
        """Initialize indices for joints, bodies, and end-effectors based on the loaded model"""
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

        self._torso_idx = self._mjcf_model.model.name2id("torso", "body")

    def get_joint_positions(self, qpos):
        """retrieve walker's joint position values"""
        return qpos[self._joint_idxs]

    def get_body_positions(self, xpos):
        """retrieve walker's body position values"""
        return xpos[self._body_idxs]

    def get_end_effector_positions(self, xpos):
        """retrieve walker's end effectors positions values"""
        return xpos[self._endeff_idxs]

    def get_torso_position(self, xpos):
        """retrieve walker's torso position values"""
        return xpos[self._torso_idx]
