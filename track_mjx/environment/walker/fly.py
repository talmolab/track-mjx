import jax
from jax import numpy as jp
from pathlib import Path
from brax.io import mjcf as mjcf_brax
import mujoco
from track_mjx.environment.walker.base import BaseWalker
from track_mjx.environment.walker.spec_utils import _scale_body_tree

_XML_PATH = "assets/fruitfly/fruitfly_force_fast.xml"


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
        self._torso_name = "thorax"
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
        path = Path(__file__).with_suffix("").parent / _XML_PATH
        xml_str = path.read_text()
        spec = mujoco.MjSpec.from_string(xml_str)

        # a) Convert motors to torque‑mode if requested
        if torque_actuators and hasattr(spec, "actuator"):
            for motor in spec.actuator.motors:  # type: ignore[attr-defined]
                # Set gain to max force; remove bias terms if present
                if motor.forcerange.size >= 2:
                    motor.gainprm[0] = motor.forcerange[1]
                # Safely delete attributes that may not exist in spec version
                for attr in ("biastype", "biasprm"):
                    if hasattr(motor, attr):
                        delattr(motor, attr)

        # b) Uniform rescale (geometry + body positions)
        if abs(rescale_factor - 1.0) > 1e-6:
            for top in spec.worldbody.find_child("thorax"):
                _scale_body_tree(top, rescale_factor)

        return spec

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
        self._torso_idx = self._mjcf_model.model.name2id(self.torso_name, "body")
