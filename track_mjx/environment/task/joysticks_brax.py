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
"""Joystick task for rodents"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from etils import epath

from track_mjx.agent.intention_network import Decoder
from track_mjx.agent.checkpointing import load_checkpoint_for_eval

from brax.envs.base import PipelineEnv, State
from brax.training import distribution
from brax.training.acme import running_statistics

from brax.io import mjcf as mjcf_brax

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.002,
        episode_length=500,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking.
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                # Base reward.
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                # Other.
                dof_pos_limits=-1.0,
                pose=0.5,
                # Other.
                termination=-1.0,
                stand_still=-1.0,
                # Regularization.
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                # Feet.
                feet_clearance=-2.0,
                feet_height=-0.2,
                feet_slip=-0.1,
                feet_air_time=0.1,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=[0.0, 3.0],
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        command_config=config_dict.create(
            # Uniform distribution for command amplitude.
            a=[1.5, 0.8, 1.2],
            # Probability of not zeroing out new command.
            b=[0.9, 0.25, 0.5],
        ),
        decoder_transfer=False,
        decoder_config=config_dict.create(
            # Decoder network parameters.
            layer_sizes=[512, 512, 512],
            intention_size=60,  # will be the output of the encoder (policy module)
            decoder_path="/root/vast/scott-yang/track-mjx/model_checkpoints/250306_194809",
        ),
    )

_HERE = epath.Path(__file__).parent
XML_PATH = _HERE / ".." / "walker" / "assets" / "rodent" / "rodent.xml"
xml_path = str(XML_PATH)

FEET_GEOM = [
    "upper_leg_R0_collision",
    "upper_leg_L0_collision",
    "hand_R_collision",
    "hand_L_collision",
]


FEET_SITES = [
    "finger_L",
    "finger_R",
    "sole_L",
    "sole_R",
]

class RodentJoystick(PipelineEnv):
    """Track a joystick command."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        evaluator=False,
    ):
        
        
        self._config = config.lock()
        if config_overrides:
            self._config.update_from_flattened_dict(config_overrides)

        self._ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt
        self._evaluator = evaluator

        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text()
        )
        self._mj_model.opt.timestep = self._config.sim_dt
        
        sys = mjcf_brax.load_model(self._mj_model)

        super().__init__(sys, n_frames=5, backend="mjx", debug=False)

        self._xml_path = xml_path

        self._post_init()
        self._has_decoder = False
        # if self._config.decoder_transfer:
        #     self._embed_decoder()
        #     self._has_decoder = True

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("stand").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("stand").qpos[7:])

        # Note: First joint is freejoint.
        self._lowers, self._uppers = self._mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body("torso").id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in FEET_GEOM]
        )

        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

    # def _embed_decoder(self) -> None:
    #     """
    #     Embeds the lower level decoder into the environment, to transfer
    #     the knowledge of the motor controller. This will change the action size
    #     of the environment from the action of the biomechanical model to
    #     intention size .
    #     """

    #     ckpt = load_checkpoint_for_eval(self._config.decoder_config.decoder_path)
    #     self._decoder_config = ckpt["cfg"]
    #     network_config = ckpt["cfg"]["network_config"]
    #     ref_obs_size = network_config["reference_obs_size"]

    #     # TODO: add more input validation based on the loaded config
    #     self._decoder = Decoder(
    #         layer_sizes=self._config.decoder_config.layer_sizes
    #         + [network_config["action_size"] * 2]
    #     )

    #     self._normalizer = running_statistics.normalize
    #     # load the normalizer parameters
    #     normalizer_param = ckpt["policy"][0]
    #     # index through the normalizer for only the ego observation
    #     self._normalizer_param = running_statistics.NestedMeanStd(
    #         normalizer_param.mean[ref_obs_size:], normalizer_param.std[ref_obs_size:]
    #     )

    #     # load the decoder parameters
    #     decoder_raw = ckpt["policy"][1]["params"]["decoder"]
    #     self.decoder_param = {"params": decoder_raw}
    #     # initialize the action distribution
    #     self._action_distribution = distribution.NormalTanhDistribution(
    #         event_size=network_config["action_size"]
    #     )

    def reset(self, rng: jax.Array) -> State:
        qpos = self._init_q
        qvel = jp.zeros(self.sys.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = self.pipeline_init(qpos, qvel)


        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        steps_until_next_cmd = jp.round(time_until_next_cmd / self._ctrl_dt).astype(jp.int32)
        cmd = jax.random.uniform(
            key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.sys.nu),
            "last_last_act": jp.zeros(self.sys.nu),
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    # def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
    #   qpos = state.data.qpos
    #   new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
    #   new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
    #   qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
    #   state = state.replace(data=state.data.replace(qpos=qpos))
    #   return state

    def step(self, state: State, action: jax.Array) -> State:
        # if self._has_decoder:
        #     normalized_ego_obs = self._normalizer(
        #         self._get_ego_obs(state.data), self._normalizer_param
        #     )
        #     # now the actions become the intentions
        #     decoder_input = jp.concatenate([action, normalized_ego_obs])
        #     action_param = self._decoder.apply(self.decoder_param, decoder_input)
        #     if self._evaluator:
        #         # use deterministic action for evaluation
        #         action = self._action_distribution.mode(action_param)
        #     else:
        #         action = self._action_distribution.sample(
        #             action_param, seed=state.info["rng"]
        #         )

        data = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self._ctrl_dt, 0.0, 10000.0)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
        state.info["command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0,
            self.sample_command(key1, state.info["command"]),
            state.info["command"],
        )
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self._ctrl_dt).astype(jp.int32),
            state.info["steps_until_next_cmd"],
        )
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        done = done.astype(reward.dtype)
        state = state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_upvector(data)[-1] < 0.0
        return fall_termination

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )

        state = jp.hstack(
            [
                noisy_linvel,  # 3
                noisy_gyro,  # 3
                noisy_joint_angles - self._default_pose,  # 12
                noisy_joint_vel,  # 12
                info["last_act"],  # 12
                info["command"],  # 3
            ]
        )

        accelerometer = self.get_accelerometer(data)
        angvel = self.get_global_angvel(data)
        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                linvel,  # 3
                angvel,  # 3
                joint_angles - self._default_pose,  # 12
                joint_vel,  # 12
                data.actuator_force,  # 12
                data.xfrc_applied[self._torso_body_id, :3], 
            ]
        )

        # The state contains the noisy measurements and privileged state (unnoisy).
        return state
    
    def _get_ego_obs(self, data: mjx.Data) -> jax.Array:
        """
        Args:
            data (mjx.Data): _mjx.Data object containing the simulation data.

        Returns:
            jax.Array: Array containing the concatenated position and velocity data.
        """
        return jp.concatenate([data.qpos, data.qvel])

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], self.get_gyro(data)
            ),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
            "termination": self._cost_termination(done),
            "pose": self._reward_pose(data.qpos[7:]),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        }

    # Tracking rewards.

    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        local_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes).
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw).
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

    # Base-related rewards.

    def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
        # Penalize z axis base linear velocity.
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
        # Penalize xy axes base angular velocity.
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        # Penalize non flat base orientation.
        return jp.sum(jp.square(torso_zaxis[:2]))

    # Energy related rewards.

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques.
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))

    # Other rewards.

    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        # Stay close to the default pose.
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose)))

    def _cost_stand_still(
        self,
        commands: jax.Array,
        qpos: jax.Array,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        # Penalize early termination.
        return done

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        # Penalize joints if they cross soft limits.
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y_k = jax.random.uniform(
            y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        x_kp1 = x_k - w_k * (x_k - y_k * z_k)
        return x_kp1

    # Sensor readings.

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "velocimeter")

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.ACCELEROMETER_SENSOR)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    # Accessors.

    @property
    def xml_path(self) -> str:
        return self._xml_path
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

