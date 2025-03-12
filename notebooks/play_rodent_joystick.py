# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.locomotion.go1 import go1_constants
from mujoco_playground._src.locomotion.go1.base import get_assets
from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad

from track_mjx.environment.task.single_clip_tracking import SingleClipTracking

get_obs = SingleClipTracking._get_obs

_HERE = epath.Path(__file__).parent


class OnnxController:
    """ONNX controller for rodent with only decoder inputs."""

    def __init__(
        self,
        policy_path: str,
        n_substeps: int,
        action_scale: float = 0.5,
        vel_scale_x: float = 1.5,
        vel_scale_y: float = 0.8,
        vel_scale_rot: float = 2 * np.pi,
    ):
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )

        self._action_scale = action_scale

        self._counter = 0
        self._n_substeps = n_substeps

        self._joystick = Gamepad(
            vel_scale_x=vel_scale_x,
            vel_scale_y=vel_scale_y,
            vel_scale_rot=vel_scale_rot,
        )

    def get_obs(self, model, data) -> np.ndarray:
        """

        # TODO: Add obs normalizer here

        Args:
            model (_type_): _description_
            data (_type_): _description_

        Returns:
            np.ndarray: _description_
        """
        intention = np.random.normal(size=60)
        obs = np.concatenate(
            [
                intention, data.qpos, data.qvel 
            ]
        )
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_input = {"obs": obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]

            data.ctrl[:] = onnx_pred[:38]


def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        str(
            _HERE
            / ".."
            / "track_mjx"
            / "environment"
            / "walker"
            / "assets"
            / "rodent"
            / "rodent.xml"
        )
    )

    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.01
    sim_dt = 0.002
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy = OnnxController(
        policy_path=(_HERE / "decoder.onnx").as_posix(),
        n_substeps=n_substeps,
        action_scale=0.5,
        vel_scale_x=1.5,
        vel_scale_y=0.8,
        vel_scale_rot=2 * np.pi,
    )

    mujoco.set_mjcb_control(policy.get_control)

    return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
