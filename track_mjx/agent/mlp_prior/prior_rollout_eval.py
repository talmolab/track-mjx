"""
Multi-mode prior rollout evaluation for prior training.

This module provides functionality to evaluate the prior network's ability to
generate plausible actions by running rollouts in 3 different modes:
1. Deterministic: z = prior_mean (no sampling)
2. logvar=0: z sampled with std=1.0
3. logvar=-2: z sampled with std≈0.368

All three modes are evaluated and rendered separately.

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from typing import Callable, Optional, Sequence, Tuple, Any, Dict, List, Union
import functools
import time

import jax
import jax.numpy as jnp
from jax import random
from brax.training import distribution, types
from brax.training.acme import running_statistics

from track_mjx.agent.mlp_prior import prior_networks
from track_mjx.agent.observation_utils import DictRunningStatisticsState


def reparameterize(rng: jax.Array, mean: jax.Array, logvar: jax.Array) -> jax.Array:
    """Sample from a Gaussian distribution using the reparameterization trick."""
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def create_prior_policy(
    prior_network_params: Dict,
    decoder_network_params: Dict,
    normalizer_params: running_statistics.RunningStatisticsState,
    intention_latent_size: int,
    action_size: int,
    proprioceptive_obs_size: int,
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    prior_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    fixed_logvar: float = -2.0,
    deterministic: bool = False,
) -> Callable:
    """
    Create a policy function that uses only the prior and decoder networks.

    Args:
        prior_network_params: Parameters for the prior network.
        decoder_network_params: Parameters for the decoder network.
        normalizer_params: Running statistics for observation normalization.
        intention_latent_size: Size of the intention latent space.
        action_size: Size of the action space.
        proprioceptive_obs_size: Size of proprioceptive observations.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
        prior_hidden_layer_sizes: Hidden layer sizes for prior.
        preprocess_observations_fn: Function to normalize observations.
        fixed_logvar: Fixed log-variance to use for latent sampling.
        deterministic: If True, use mean of prior distribution instead of sampling.

    Returns:
        A policy function that takes (proprioceptive_obs, rng_key) and returns actions.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # Create the prior and decoder modules
    prior_module = prior_networks.Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=intention_latent_size,
    )

    decoder_module = prior_networks.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
    )

    def policy_fn(
        obs: jax.Array,
        rng_key: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """
        Generate actions from proprioceptive observations using prior + decoder.

        Args:
            obs: Observations [..., obs_size].
            rng_key: Random key for sampling.

        Returns:
            Tuple of (actions, extras_dict).
        """
        _, key_sample = random.split(rng_key)

        # Get proprioceptive observations (last part of obs)
        proprioceptive_obs = obs[..., -proprioceptive_obs_size:]

        # Normalize observations
        normalized_obs = preprocess_observations_fn(
            proprioceptive_obs, normalizer_params
        )

        # Get prior distribution
        prior_mean, prior_logvar = prior_module.apply(
            {"params": prior_network_params}, normalized_obs
        )

        # Use fixed logvar for more stable sampling
        fixed_logvar_array = jnp.full_like(prior_mean, fixed_logvar)

        # Sample from prior
        if deterministic:
            z = prior_mean
        else:
            z = reparameterize(key_sample, prior_mean, fixed_logvar_array)

        # Decode to action distribution parameters
        decoder_input = jnp.concatenate([z, normalized_obs], axis=-1)
        logits, _ = decoder_module.apply(
            {"params": decoder_network_params}, decoder_input
        )

        # Always use mode of action distribution for deterministic actions
        action = parametric_action_distribution.mode(logits)

        extras = {
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "intention": z,
            "logits": logits,
        }

        return action, extras

    return policy_fn


def extract_prior_decoder_params(
    policy_params: Tuple,
) -> Tuple[Dict, Dict, DictRunningStatisticsState]:
    """
    Extract prior and decoder parameters from full policy params.

    Args:
        policy_params: Tuple of (normalizer_params, network_params).
            normalizer_params is a DictRunningStatisticsState.

    Returns:
        Tuple of (prior_params, decoder_params, normalizer_params).
    """
    normalizer_params, network_params = policy_params
    prior_params = network_params["params"]["prior"]
    decoder_params = network_params["params"]["decoder"]
    return prior_params, decoder_params, normalizer_params


def check_termination_nan(data: Any) -> jax.Array:
    """
    Check for NaN values in physics state (cheap, run every step).
    """
    from jax import flatten_util

    flattened_vals, _ = flatten_util.ravel_pytree(data.qpos)
    flattened_qvel, _ = flatten_util.ravel_pytree(data.qvel)
    all_vals = jnp.concatenate([flattened_vals, flattened_qvel])
    return jnp.any(jnp.isnan(all_vals))


def compute_world_zaxis_termination(env, datas: Any) -> jax.Array:
    """
    Batch compute world z-axis termination for all timesteps.
    """
    world_zaxes = jax.vmap(env._get_world_zaxis)(datas)  # (max_steps, 3)
    return world_zaxes[:, 2] < 0  # (max_steps,)


class MultiModePriorRolloutEvaluator:
    """
    Evaluator class for running prior-only rollouts in multiple modes.

    This class runs evaluation in 3 modes:
    1. deterministic: z = prior_mean (no sampling)
    2. logvar_0: z sampled with std=1.0 (logvar=0)
    3. logvar_-2: z sampled with std≈0.368 (logvar=-2)

    All modes are rendered separately.
    """

    def __init__(
        self,
        env,
        intention_latent_size: int,
        action_size: int,
        proprioceptive_obs_size: int,
        decoder_hidden_layer_sizes: Sequence[int],
        prior_hidden_layer_sizes: Sequence[int],
        preprocess_observations_fn: types.PreprocessObservationFn,
        max_steps: int = 200,
        eval_interval: int = 1,
        render_fps: int = 50,
        render_camera_name: str = "close_profile",
        model_path: str = "",
    ):
        """
        Initialize the multi-mode prior rollout evaluator.

        Args:
            env: The environment to run rollouts in.
            intention_latent_size: Size of intention latent space.
            action_size: Size of action space.
            proprioceptive_obs_size: Size of proprioceptive observations.
            decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
            prior_hidden_layer_sizes: Hidden layer sizes for prior.
            preprocess_observations_fn: Observation normalization function.
            max_steps: Maximum steps per rollout.
            eval_interval: Run evaluation every N evals (1 = every eval).
            render_fps: FPS for rendered video.
            render_camera_name: Camera name for rendering.
            model_path: Path to save rendered videos.
        """
        self.env = env
        self.intention_latent_size = intention_latent_size
        self.action_size = action_size
        self.proprioceptive_obs_size = proprioceptive_obs_size
        self.decoder_hidden_layer_sizes = decoder_hidden_layer_sizes
        self.prior_hidden_layer_sizes = prior_hidden_layer_sizes
        self.preprocess_observations_fn = preprocess_observations_fn
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.render_fps = render_fps
        self.render_camera_name = f"{render_camera_name}-rodent"
        self.model_path = model_path

        self._key = random.PRNGKey(0)

        # Define 3 evaluation modes
        self.evaluation_modes = [
            {"name": "deterministic", "deterministic": True, "fixed_logvar": -2.0},
            {"name": "logvar_0", "deterministic": False, "fixed_logvar": 0.0},
            {"name": "logvar_-2", "deterministic": False, "fixed_logvar": -2.0},
        ]

    def _build_single_rollout_fn(self, policy_params: Tuple, mode: Dict):
        """Build the jitted single rollout function for a specific mode."""
        prior_params, decoder_params, normalizer_params = extract_prior_decoder_params(
            policy_params
        )

        # Extract proprioceptive-only normalizer from dict normalizer
        proprio_normalizer_params = normalizer_params.proprioception

        # Create policy function for this mode
        policy_fn = create_prior_policy(
            prior_network_params=prior_params,
            decoder_network_params=decoder_params,
            normalizer_params=proprio_normalizer_params,
            intention_latent_size=self.intention_latent_size,
            action_size=self.action_size,
            proprioceptive_obs_size=self.proprioceptive_obs_size,
            decoder_hidden_layer_sizes=self.decoder_hidden_layer_sizes,
            prior_hidden_layer_sizes=self.prior_hidden_layer_sizes,
            preprocess_observations_fn=self.preprocess_observations_fn,
            fixed_logvar=mode["fixed_logvar"],
            deterministic=mode["deterministic"],
        )

        jit_step = jax.jit(self.env.step)

        @jax.jit
        def single_rollout_fn(
            initial_state: Any, rng_key: jax.Array
        ) -> Tuple[jax.Array, jax.Array, Any]:
            """Run a single prior rollout from given initial state."""
            state = initial_state

            def step_fn(carry, _):
                state, key, nan_terminated = carry
                key, key_action = random.split(key)

                # Get proprioceptive observations
                if hasattr(state.obs, "get") or isinstance(state.obs, dict):
                    proprio = state.obs.get("proprioception", state.obs)
                    if isinstance(proprio, dict):
                        from jax import flatten_util

                        proprio, _ = flatten_util.ravel_pytree(proprio)
                else:
                    proprio = state.obs

                # Get action from prior policy
                action, _ = policy_fn(proprio, key_action)

                # Step environment
                next_state = jit_step(state, action)

                # Only check NaN during rollout (cheap)
                step_nan = check_termination_nan(next_state.data)
                new_nan_terminated = jnp.logical_or(nan_terminated, step_nan)

                return (next_state, key, new_nan_terminated), next_state

            initial_carry = (state, rng_key, jnp.array(False))
            (_, _, nan_terminated), all_states = jax.lax.scan(
                step_fn, initial_carry, None, length=self.max_steps
            )

            # Batch compute world z-axis termination AFTER rollout
            upside_down_flags = compute_world_zaxis_termination(
                self.env, all_states.data
            )

            # Find first termination step
            any_upside_down = jnp.any(upside_down_flags)
            first_upside_down_step = jnp.argmax(upside_down_flags)

            # Combine NaN termination with upside-down termination
            terminated = jnp.logical_or(nan_terminated, any_upside_down)
            step_count = jnp.where(
                any_upside_down, first_upside_down_step + 1, self.max_steps
            )

            return step_count, terminated, all_states

        return single_rollout_fn

    def _render_rollout(
        self, states: Any, step_count: int, current_step: int, mode_name: str
    ) -> None:
        """Render a rollout and log to wandb."""
        import wandb
        import imageio

        # Convert stacked states to list of individual State objects
        states_list = []
        for i in range(step_count):
            state_i = jax.tree_util.tree_map(lambda x: x[i], states)
            states_list.append(state_i)

        # Render all frames at once
        frames = self.env.render(states_list, camera=self.render_camera_name)

        if len(frames) > 0:
            video_path = (
                f"{self.model_path}/prior_rollout_{mode_name}_{current_step}.mp4"
            )
            with imageio.get_writer(video_path, fps=self.render_fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

            wandb.log(
                {
                    f"videos/prior_rollout_{mode_name}": wandb.Video(
                        video_path, format="mp4"
                    )
                },
                commit=False,
            )

    def run_evaluation(
        self,
        policy_params: Tuple,
        eval_step: int,
        reset_key: jax.Array,
    ) -> Optional[Dict[str, float]]:
        """
        Run prior rollout evaluation in all 3 modes from a shared initial state.

        Args:
            policy_params: Current policy parameters.
            eval_step: Current evaluation step number.
            reset_key: Random key for resetting environment (shared with other evaluators).

        Returns:
            Dictionary of metrics if evaluation was run, None otherwise.
        """
        # Check if we should run evaluation this step
        if eval_step % self.eval_interval != 0:
            return None

        # Reset environment once with shared key to get initial state
        jit_reset = jax.jit(self.env.reset)
        initial_state = jit_reset(reset_key)

        self._key, rollout_key = random.split(self._key)

        for mode in self.evaluation_modes:
            mode_name = mode["name"]

            # Build single-rollout function for this mode
            single_rollout_fn = self._build_single_rollout_fn(policy_params, mode)

            # Get random key for this mode's rollout
            rollout_key, mode_key = random.split(rollout_key)

            # Run single rollout from shared initial state
            step_count, _, states = single_rollout_fn(initial_state, mode_key)
            step_count = int(step_count)

            # Render the rollout for this mode
            try:
                self._render_rollout(states, step_count, eval_step, mode_name)
            except Exception as e:
                import logging

                logging.warning(f"Failed to render {mode_name} rollout: {e}")

        return {}
