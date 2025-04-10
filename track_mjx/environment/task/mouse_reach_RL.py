"""Mouse reaching task for reinforcement learning with JAX/MJX."""

import jax
from jax import numpy as jp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
import mujoco
from mujoco import mjx
from track_mjx.environment.task.reward import TaskRewardConfig


from typing import Any, Dict, List, Optional, Tuple, Union

from track_mjx.environment.walker.base import BaseWalker


class MouseReachTask(PipelineEnv):
    """
    A task where a mouse reaches for a target, implemented with JAX/MJX.

    This environment is designed for reinforcement learning rather than imitation learning.
    The mouse earns rewards for reaching a target placed at various positions.
    """

    def __init__(
        self,
        walker: BaseWalker,
        physics_steps_per_control_step: int,
        reset_noise_scale: float,
        solver: str,
        iterations: int,
        ls_iterations: int,
        mj_model_timestep: float,
        reward_scale: float = 1.0,
        target_size: float = 0.003,
        termination_distance: float = 0.02,
        margin: float = 0.006,
        randomize_targets: bool = True,
        reward_config: TaskRewardConfig = None,
        **kwargs: Any,
    ):
        """Initializes the MouseReachTask environment."""
        # Store walker reference for rendering - THIS IS CRITICAL
        self.walker = walker
        self.walker._initialize_indices()

        self._n_clips = 1
        self._steps_for_cur_frame = 1  # One step per frame for RL environment

        # Configure reward parameters
        if reward_config is not None:
            self._reward_scale = reward_config.reward_scale
            self._target_size = reward_config.target_size
            self._termination_distance = reward_config.termination_distance
            self._margin = reward_config.margin
            self._ctrl_cost_weight = reward_config.ctrl_cost_weight
            self._ctrl_diff_cost_weight = reward_config.ctrl_diff_cost_weight
            self._energy_cost_weight = reward_config.energy_cost_weight
            self._jerk_cost_weight = reward_config.jerk_cost_weight
        else:
            self._reward_scale = reward_scale
            self._target_size = target_size
            self._termination_distance = termination_distance
            self._margin = margin
            self._ctrl_cost_weight = 0.0
            self._ctrl_diff_cost_weight = 0.0
            self._energy_cost_weight = 0.0
            self._jerk_cost_weight = 0.0

        # Get the MuJoCo model and modify as needed
        mj_model = self.walker._mjcf_model.model.ptr

        # Configure solver properties
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep = mj_model_timestep
        mj_model.opt.jacobian = 0

        # Add a target site to the model if it doesn't exist already
        self._add_or_configure_target(mj_model, target_size)

        # Load the system for MJX
        sys = mjcf_brax.load_model(mj_model)

        # Filter out unsupported parameters before calling PipelineEnv.__init__
        pipeline_kwargs = {
            "n_frames": kwargs.get("n_frames", physics_steps_per_control_step),
            "backend": "mjx",
        }
        super().__init__(sys, **pipeline_kwargs)

        # Store configuration parameters
        self._reset_noise_scale = reset_noise_scale
        self._reward_scale = reward_scale
        self._target_size = target_size
        self._termination_distance = termination_distance
        self._margin = margin
        self._randomize_targets = randomize_targets

        # Define available target positions (in walker coordinate frame)
        self._target_positions = self._create_target_positions()

        # Cache relevant body and site indices
        self._finger_tip_idx = self._get_geom_index("finger_tip")
        self._target_idx = self._get_geom_index("target")
        self._shoulder_idx = self._get_joint_index("sh_elv")

        self._proprio_obs_size = (
            self.sys.nq + self.sys.nv + 3
        )  # qpos + qvel + vector to target (3D)

    def _add_or_configure_target(self, mj_model, target_size):
        """Ensures the model has a target geom with appropriate properties."""

        # Configure target geom if exists (for visualization)
        target_geom_found = False

        for i in range(mj_model.ngeom):
            geom_name = mj_model.geom(i).name
            if geom_name == "target":
                mj_model.geom(i).size[0] = target_size
                mj_model.geom(i).rgba = [1.0, 0.2, 0.2, 0.8]  # Bright red, more opaque
                target_geom_found = True
                break

        if not target_geom_found:
            print("WARNING: Target geom not found in model!")

    def _create_target_positions(self) -> jp.ndarray:
        """Define possible target positions for the task.

        Returns:
            Array of possible target positions in shape [num_positions, 3]
        """
        # These are example positions - would be tuned for the specific mouse model
        # Similar to the dm_control implementation but as JAX arrays
        return jp.array(
            [
                [0.004, 0.012, -0.006],
                [0.0025355, 0.012, -0.0024645],
                [-0.001, 0.012, -0.001],
                [-0.0045355, 0.012, -0.0024645],
                [-0.006, 0.012, -0.006],
                [-0.0045355, 0.012, -0.0095355],
                [-0.001, 0.012, -0.011],
                [0.0025355, 0.012, -0.0095355],
            ]
        )

    def _get_site_index(self, name: str) -> int:
        """Get the index of a site in the MuJoCo model."""
        model = self.walker._mjcf_model.model.ptr  # This is an MjModel pointer
        for i in range(model.nsite):
            site_name = model.site(i).name
            if site_name == name:
                return i
        raise ValueError(
            f"Site {name} not found in model. Available sites: {[model.site(i).name for i in range(model.nsite)]}"
        )

    def _get_joint_index(self, name: str) -> int:
        """Get the index of a joint in the MuJoCo model."""
        model = self.walker._mjcf_model.model.ptr  # This is an MjModel pointer
        for i in range(model.njnt):
            joint_name = model.joint(i).name
            if joint_name == name:
                return i
        raise ValueError(
            f"Joint {name} not found in model. Available joints: {[model.joint(i).name for i in range(model.njnt)]}"
        )

    def _set_target_position(
        self, data: mjx.Data, rng: jp.ndarray
    ) -> Tuple[mjx.Data, jp.ndarray]:
        """Sets a random target position in the environment."""
        if self._randomize_targets:
            # Randomly select a target position
            idx = jax.random.randint(
                rng, shape=(), minval=0, maxval=len(self._target_positions)
            )
            target_pos = self._target_positions[idx]
        else:
            # Use a fixed target position
            target_pos = self._target_positions[0]

        # We won't rely on geom_xpos for the actual target position in the simulation
        # Instead, we'll just set it once during reset and access it from info
        # This prevents MJX from resetting it during forward dynamics

        # Still update the geom position for rendering purposes
        data = data.replace(
            geom_xpos=data.geom_xpos.at[self._target_idx].set(target_pos)
        )

        return data, target_pos  # Return both the updated data and target position

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state.

        Args:
            rng: Random number generator state.

        Returns:
            State: The reset environment state.
        """
        rng, rng_qpos, rng_qvel, rng_target = jax.random.split(rng, 4)

        # Set initial joint configuration (similar to dmcontrol's mouse.reinitialize_pose)
        qpos = jp.array(self.sys.qpos0)
        qvel = jp.zeros((self.sys.nv,))

        # Add noise to initial pose if desired
        low, high = -self._reset_noise_scale, self._reset_noise_scale
        qpos_noise = jax.random.uniform(
            rng_qpos, (self.sys.nq,), minval=low, maxval=high
        )
        qvel_noise = jax.random.uniform(
            rng_qvel, (self.sys.nv,), minval=low, maxval=high
        )

        qpos += qpos_noise
        qvel += qvel_noise

        # Initialize MJX data
        data = self.pipeline_init(qpos, qvel)

        # Set target position and get the target position
        data, target_position = self._set_target_position(data, rng_target)

        # Get observation with explicit target position
        obs = self._get_obs(data, target_position)

        # Initial reward, done, and metrics
        reward, done = jp.zeros(2)
        metrics = {
            "distance_reward": jp.array(0.0),
            "target_distance": jp.array(0.0),
            "reward": jp.array(0.0),  # Add this key to match step method
        }

        # Store info like current target position
        info = {
            "target_position": target_position,  # Store the explicit target position
            "prev_ctrl": jp.zeros((self.sys.nu,)),
            "reference_obs_size": jp.array(
                len(data.qpos) + len(data.qvel) + 3
            ),  # qpos + qvel + vector to target
        }

        return State(data, obs, reward, done, metrics, info)

    def _get_target_position(self, data: mjx.Data) -> jp.ndarray:
        """Gets the current target position from MJX data."""
        # Change from site_xpos to geom_xpos
        return data.geom_xpos[self._target_idx]

    def _get_fingertip_position(self, data: mjx.Data) -> jp.ndarray:
        """Gets the current fingertip position from MJX data."""
        # Change from site_xpos to geom_xpos
        return data.geom_xpos[self._finger_tip_idx]

    def _get_geom_index(self, name: str) -> int:
        """Get the index of a geom in the MuJoCo model."""
        model = self.walker._mjcf_model.model.ptr
        for i in range(model.ngeom):
            geom_name = model.geom(i).name
            if geom_name == name:
                return i
        raise ValueError(f"Geom {name} not found in model")

    def _get_obs(self, data: mjx.Data, target_position=None) -> jp.ndarray:
        """Constructs the observation for the environment.

        Args:
            data: Current MJX data
            target_position: Optional explicit target position, overrides geom position
        """
        # Get fingertip position
        fingertip_pos = self._get_fingertip_position(data)

        # Use provided target position if available, otherwise get from geom
        target_pos = (
            target_position
            if target_position is not None
            else data.geom_xpos[self._target_idx]
        )

        # Calculate vector from fingertip to target
        fingertip_to_target = target_pos - fingertip_pos

        # Create proprioceptive observation
        proprioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
                fingertip_to_target,
            ]
        )

        # For intention network compatibility
        reference_obs = proprioceptive_obs

        # Concatenate reference and proprioceptive parts
        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        return obs

    def _compute_reward(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> Tuple[jp.ndarray, jp.ndarray, Dict[str, jp.ndarray]]:
        """Computes the reward for the current state.

        Args:
            data: Current MJX data
            info: Information dictionary

        Returns:
            Tuple containing:
                - reward: The scalar reward value
                - done: Boolean indicating if episode is done
                - metrics: Dictionary of metrics for logging
        """
        # Get fingertip position
        fingertip_pos = self._get_fingertip_position(data)

        # Use target position from info dictionary to ensure consistency
        target_pos = info["target_position"]

        # Calculate distance between fingertip and target
        distance = jp.linalg.norm(fingertip_pos - target_pos)

        # Calculate reward using tolerance function (similar to dm_control)
        # A value of 1.0 when distance is 0, decreasing to 0.0 as distance increases
        in_target = jp.less(distance, self._target_size)
        reward_smooth = jp.exp(-distance / self._margin)
        reward = jp.where(in_target, jp.ones_like(distance), reward_smooth)
        reward = reward * self._reward_scale

        # No distance-based termination - let the environment's time limit determine episode end
        done = jp.zeros_like(distance, dtype=bool)

        # Store metrics for logging
        metrics = {
            "distance_reward": reward,
            "target_distance": distance,
            "reward": reward,  # Add this to match the structure in reset
        }

        return reward, done, metrics

    def reset_from_clip(
        self, rng: jp.ndarray, info: Optional[Dict[str, Any]] = None
    ) -> State:
        """Resets the environment using a specific motion clip.

        Since this is an RL task without motion clips, this just calls the regular reset.

        Args:
            rng: Random number generator state.
            info: Optional dictionary with additional information.

        Returns:
            State: The reset environment state.
        """
        # For RL task, we ignore any clip-specific info and just do a regular reset
        return self.reset(rng)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Executes one timestep of the environment's dynamics.

        Args:
            state: The current environment state object.
            action: The action to take.

        Returns:
            State: The updated environment state.
        """
        # Get initial state
        data0 = state.pipeline_state
        info = state.info.copy()

        # Clip action between -0.1 and 0.1
        clipped_action = jp.clip(action, -0.1, 0.1)

        # Get target position from info
        target_position = info["target_position"]

        # Step the physics simulation with clipped action
        data = self.pipeline_step(data0, clipped_action)

        # CRITICAL FIX: Explicitly update the target position in the MJX data
        # This ensures the target doesn't get reset to [0,0,0] during physics steps
        data = data.replace(
            geom_xpos=data.geom_xpos.at[self._target_idx].set(target_position)
        )

        # Calculate reward using the stored target position
        reward, done, metrics = self._compute_reward(data, info)

        # Get observation with explicit target position
        obs = self._get_obs(data, target_position)

        # Update info dictionary - store the clipped action
        info["prev_ctrl"] = clipped_action

        # Check for simulation errors (NaNs)
        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan_detected = jp.where(num_nans > 0, 1.0, 0.0)

        # Done either from task completion or simulation error
        done = jp.maximum(done, nan_detected)

        # Update state with new values
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )
