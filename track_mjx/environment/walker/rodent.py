from pathlib import Path
from typing import Sequence

import jax.numpy as jp
import mujoco
from brax.io import mjcf as mjcf_brax
import numpy as np
from track_mjx.environment.walker.base import BaseWalker  # type: ignore
from track_mjx.environment.walker.spec_utils import _scale_body_tree


_XML_PATH = "assets/rodent/rodent.xml"  # relative to this file


class Rodent(BaseWalker):
    """Rodent walker using MuJoCo **MjSpec**"""

    def __init__(
        self,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        end_eff_names: Sequence[str],
        *,
        torque_actuators: bool = False,
        rescale_factor: float = 0.9,
    ):
        """

        Parse XML → MjSpec, apply optional edits, and return the spec.

        Args:
            joint_names (Sequence[str]): The names of the joints to be used in the model.
            body_names (Sequence[str]): The names of the bodies to be used in the model.
            end_eff_names (Sequence[str]): The names of the end effectors to be used in the model.
            torque_actuators (bool, optional): whether modify the model to use torque actuators. Defaults to False.
            rescale_factor (float, optional): the rescale factor for the body model. Defaults to 0.9.
        """
        self._joint_names = joint_names
        self._body_names = body_names
        self._end_eff_names = end_eff_names

        # 1) Build the physics model via MjSpec
        self._mj_spec = self._build_spec(torque_actuators, rescale_factor)
        self._mj_model = self._mj_spec.compile()  # mujoco.mjx.Model wrapper
        self.sys = mjcf_brax.load_model(self._mj_model)
        # 2) Cache index arrays for JIT‑friendly access
        self._initialize_indices()

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
        path = Path(__file__).parent / _XML_PATH
        xml_str = path.read_text()
        spec = mujoco.MjSpec.from_string(xml_str)

        # a) Convert motors to torque‑mode if requested
        if torque_actuators and hasattr(spec, "actuator"):
            for actuator in spec.actuators:  # type: ignore[attr-defined]
                # Set gain to max force; remove bias terms if present
                if actuator.forcerange.size >= 2:
                    actuator.gainprm[0] = actuator.forcerange[1]
                # reset custom bias terms
                actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
                actuator.biasprm = np.zeros((10, 1))

        # b) Uniform rescale (geometry + body positions)
        if rescale_factor != 1.0:
            parent = spec.body("walker")
            _scale_body_tree(parent, rescale_factor)

        return spec

    def _initialize_indices(self) -> None:
        """Create immutable JAX arrays of IDs for joints, bodies, end-effectors."""
        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in self._joint_names
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in self._body_names
            ]
        )

        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, e)
                for e in self._end_eff_names
            ]
        )

        self._torso_idx = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
