from pathlib import Path
from typing import Sequence

import jax.numpy as jp
import mujoco
from brax.io import mjcf as mjcf_brax
from track_mjx.environment.walker.base import BaseWalker  # type: ignore
from track_mjx.environment.walker.utils import _scale_body_tree

################################################################################
# Utility functions                                                             #
################################################################################



_XML_PATH = "assets/rodent/rodent.xml"  # relative to this file

class Rodent(BaseWalker):
    """Rodent walker using MuJoCo **MjSpec** instead of dm_control.MJCF."""

    # ---------------------------------------------------------------------
    # Construction ---------------------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(
        self,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        end_eff_names: Sequence[str],
        *,
        torque_actuators: bool = False,
        rescale_factor: float = 0.9,
    ):
        """Load the rodent XML and prepare index buffers.

        Parameters
        ----------
        joint_names, body_names, end_eff_names : lists of strings
            Identifiers used later for indexing into qpos/qvel, body COM, …
        torque_actuators : bool, default False
            If *True*, converts all motor-type actuators to a torque-gain style
            (copies `forcerange[1]` into `gainprm[0]` and wipes bias params).
        rescale_factor : float, default 0.9
            Uniform scale multiplier applied to every geometry and body pose.
        """
        self._joint_names = list(joint_names)
        self._body_names = list(body_names)
        self._end_eff_names = list(end_eff_names)

        # ------------------------------------------------------------------
        # 1) Build the physics model via MjSpec
        # ------------------------------------------------------------------
        self._mj_spec = self._build_spec(torque_actuators, rescale_factor)
        self._mj_model = self._mj_spec.compile()  # mujoco.mjx.Model wrapper

        # Optional: still export to Brax System if your downstream code needs it
        self.sys = mjcf_brax.load_model(self._mj_model)

        # ------------------------------------------------------------------
        # 2) Cache index arrays for JIT‑friendly access
        # ------------------------------------------------------------------
        self._initialize_indices()

    # ------------------------------------------------------------------
    # Internal helpers ---------------------------------------------------
    # ------------------------------------------------------------------

    def _build_spec(
        self, torque_actuators: bool, rescale_factor: float
    ) -> mujoco.MjSpec:
        """Parse XML → MjSpec, apply optional edits, and return the spec."""
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
            for top in spec.worldbody.bodies:
                _scale_body_tree(top, rescale_factor)

        return spec

    def _initialize_indices(self) -> None:
        """Create immutable JAX arrays of IDs for joints, bodies, end‑effectors."""
        self._joint_idxs = jp.array(
            [mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self._joint_names]
        )

        self._body_idxs = jp.array(
            [mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, b) for b in self._body_names]
        )

        self._endeff_idxs = jp.array(
            [mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, e) for e in self._end_eff_names]
        )

        self._torso_idx = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
