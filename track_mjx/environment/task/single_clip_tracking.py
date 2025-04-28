"""Base tracking task heer, base env is brax pipeline env"""

import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
import mujoco
from mujoco import mjx

from typing import Any

from track_mjx.io.load import ReferenceClip
from track_mjx.environment.task.reward import compute_tracking_rewards
from track_mjx.environment.walker.base import BaseWalker
from track_mjx.environment.task.reward import RewardConfig

from jax.flatten_util import ravel_pytree

class SingleClipTracking(PipelineEnv):
    """Tracking task for a continuous reference clip."""

    def __init__(
        self,
        reference_clip: ReferenceClip | None,
        walker: BaseWalker,
        reward_config: RewardConfig,
        physics_steps_per_control_step: int,
        reset_noise_scale: float,
        solver: str,
        iterations: int,
        ls_iterations: int,
        mj_model_timestep: float,
        mocap_hz: int,
        clip_length: int,
        random_init_range: int,
        traj_length: int,
        **kwargs: Any,
    ):
        """Initializes the SingleTracking environment.

        Args:
            reference_clip: The reference trajectory data.
            walker: The base walker model.
            torque_actuators: Whether to use torque actuators.
            reward_config: Reward configuration.
            physics_steps_per_control_step: Number of physics steps per control step.
            reset_noise_scale: Scale of noise for reset.
            solver: Solver type for Mujoco.
            iterations: Maximum number of solver iterations.
            ls_iterations: Maximum number of line search iterations.
            mj_model_timestep: fundamental time increment of the MuJoCo physics simulation
            mocap_hz: cycles per second for the reference data
            clip_length: clip length of the tracking clips
            random_init_range: the initiated range
            traj_length: one trajectory length
            **kwargs: Additional arguments for the PipelineEnv initialization.
        """
        self.walker = walker
        self.walker._initialize_indices()

        mj_model = self.walker._mjcf_model.model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep = mj_model_timestep
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = physics_steps_per_control_step
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._steps_for_cur_frame = (
            1.0 / (mocap_hz * mj_model.opt.timestep)
        ) / physics_steps_per_control_step

        self._mocap_hz = mocap_hz
        self._reward_config = reward_config
        self._reference_clip = reference_clip
        self._ref_len = traj_length
        self._reset_noise_scale = reset_noise_scale
        self._mjx_model = mjx.put_model(self.sys.mj_model)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state.

        Args:
            rng: Random number generator state.

        Returns:
            State: The reset environment state.
        """
        _, start_rng, rng = jax.random.split(rng, 3)

        episode_length = (
            clip_length - random_init_range - traj_length
        ) * self._steps_for_cur_frame

        frame_range = clip_length - episode_length - traj_length
        start_frame = jax.random.randint(start_rng, (), 0, frame_range)

        info = {
            "start_frame": start_frame,
            "prev_ctrl": jp.zeros((self.sys.nv,)),
        }

        return self.reset_from_clip(rng, info, noise=True)

    def reset_from_clip(
        self, rng: jp.ndarray, info: dict[str, Any], noise: bool = True
    ) -> State:
        """Resets the environment using a reference clip.

        Args:
            rng: Random number generator state.
            info: Information dictionary.
            noise: Whether to add noise during reset. Defaults to True.

        Returns:
            State: The reset environment state.
        """
        _, rng1, rng2 = jax.random.split(rng, 3)

        # Get reference clip and select the start frame
        reference_frame = jax.tree.map(
            lambda x: x[info["start_frame"]], self._get_reference_clip(info)
        )
        
        info["reference_frame"] = reference_frame
        
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # # Add pos
        # qpos_with_pos = jp.array(self.sys.qpos0).at[:3].set(reference_frame.position)

        # # Add quat
        # new_qpos = qpos_with_pos.at[3:7].set(reference_frame.quaternion)

        # # Add noise
        # qpos = new_qpos + jp.where(
        #     noise,
        #     jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi),
        #     jp.zeros((self.sys.nq,)),
        # )

        new_qpos = jp.concatenate(
            (
                reference_frame.position,
                reference_frame.quaternion,
                reference_frame.joints,
            ),
            axis=0,
        )
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )

        qvel = jp.where(
            noise,
            jax.random.uniform(rng1, (self.sys.nv,), minval=low, maxval=hi),
            jp.zeros((self.sys.nv,)),
        )

        data = self.pipeline_init(qpos, qvel)

        reference_obs, proprioceptive_obs = self._get_obs(data, info)

        # Used to initialize our intention network
        info["reference_obs_size"] = reference_obs.shape[-1]

        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        reward, done, zero = jp.zeros(3)

        # TODO: Set up a metrics dataclass
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "ctrl_cost": zero,
            "ctrl_diff_cost": zero,
            "energy_cost": zero,
            "done": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "fall": zero,
            "nan": zero,
            "joint_distance": zero,
            "summed_pos_distance": zero,
            "quat_distance": zero,
        }

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Executes one timestep of the environment's dynamics.

        Args:
            state: The current environment state object.
            action: The action to take.

        Returns:
            State: The updated environment state.
        """

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        info = state.info.copy()

        # Gets reference clip and indexes to current frame
        reference_frame = jax.tree.map(
            lambda x: x[self._get_cur_frame(info, data)], self._get_reference_clip(info)
        )

        # reward calculation
        # TODO: Make it so that a list of rewards is returned and a
        # list of terminiation values are returned (distances)
        # So we can sum the whole list to get the total reward
        (
            pos_reward,
            quat_reward,
            joint_reward,
            angvel_reward,
            bodypos_reward,
            endeff_reward,
            ctrl_cost,
            ctrl_diff_cost,
            energy_cost,
            too_far,
            bad_pose,
            bad_quat,
            fall,
            joint_distance,
            summed_pos_distance,
            quat_distance,
        ) = compute_tracking_rewards(
            data=data,
            reference_frame=reference_frame,
            walker=self.walker,
            action=action,
            info=info,
            reward_config=self._reward_config,
        )

        info["prev_ctrl"] = action
        reference_obs, proprioceptive_obs = self._get_obs(data, info)
        obs = jp.concatenate([reference_obs, proprioceptive_obs])
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            - ctrl_cost
            - ctrl_diff_cost
            - energy_cost
        )

        # Raise done flag if terminating
        done = jp.max(jp.array([fall, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            angvel_reward=angvel_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            ctrl_cost=-ctrl_cost,
            ctrl_diff_cost=-ctrl_diff_cost,
            energy_cost=-energy_cost,
            done=done,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=fall,
            nan=nan,
            joint_distance=joint_distance,
            summed_pos_distance=summed_pos_distance,
            quat_distance=quat_distance,
            
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Returns reference clip; to be overridden in child classes"""
        return self._reference_clip

    def _get_reference_trajectory(self, info, data) -> ReferenceClip:
        """Slices ReferenceClip into the observation trajectory"""

        # Get the relevant slice of the reference clip
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    self._get_cur_frame(info, data) + 1,
                    self._ref_len,
                )
            return jp.array([])

        return jax.tree_util.tree_map(f, self._get_reference_clip(info))

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> tuple[jp.ndarray, jp.ndarray]:
        """Constructs the observation for the environment.

        Args:
            data: Current Mujoco simulation data.
            info: Information dictionary containing the current state and reference trajectory details.

        Returns:
            Tuple[jp.ndarray, jp.ndarray]:
                - `reference_obs`: Reference trajectory-based observation.
                - `proprioceptive_obs`: Observations of the agent's internal state (position and velocity).
        """

        ref_traj = self._get_reference_trajectory(info, data)

        # walker methods to compute the necessary distances and differences
        track_pos_local = self.walker.compute_local_track_positions(
            ref_traj.position, data.qpos
        )
        quat_dist = self.walker.compute_quat_distances(ref_traj.quaternion, data.qpos)
        joint_dist = self.walker.compute_local_joint_distances(
            ref_traj.joints, data.qpos
        )
        body_pos_dist_local = self.walker.compute_local_body_positions(
            ref_traj.body_positions, data.xpos, data.qpos
        )

        reference_obs = jp.concatenate(
            [
                track_pos_local,
                quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

        # jax.debug.print("track_pos_local: {}", track_pos_local)
        # jax.debug.print("quat_dist: {}", quat_dist)
        # jax.debug.print("joint_dist: {}", joint_dist)
        # jax.debug.print("body_pos_dist_local: {}", body_pos_dist_local)

        prorioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
            ]
        )
        return reference_obs, prorioceptive_obs

    def _get_cur_frame(self, info, data: mjx.Data) -> int:
        """Returns the current frame index based on the simulation time"""
        return jp.array(jp.floor(data.time * self._mocap_hz + info["start_frame"]), int)
