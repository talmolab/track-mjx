"""
Prior-only rollout evaluation for PPO training.

This module provides functionality to evaluate the prior network's ability to
generate plausible actions by running rollouts that only use the prior and decoder
(without the encoder or trajectory observations). This measures how well the
prior network has learned to predict good actions from proprioceptive state alone.

The rollouts start from random clip states and step physics using prior-sampled
actions, checking for terminations based on simple criteria (NaN detection,
torso height range).
"""

from typing import Callable, Optional, Sequence, Tuple, Any, Dict
import functools

import jax
import jax.numpy as jnp
from jax import random
from brax.training import distribution, types
from brax.training.acme import running_statistics

from track_mjx.agent.mlp_ppo import intention_network


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
    
    This extracts the prior and decoder from trained intention network parameters
    and creates a policy that:
    1. Passes proprioceptive observations through the prior to get latent distribution
    2. Samples from the latent distribution (using fixed logvar for stability)
    3. Decodes the latent + proprioceptive obs to produce actions
    
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
    prior_module = intention_network.Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=intention_latent_size,
    )
    
    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [parametric_action_distribution.param_size],
    )
    
    def policy_fn(
        obs: jax.Array,
        rng_key: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """
        Generate actions from proprioceptive observations using prior + decoder.
        
        Args:
            proprioceptive_obs: Proprioceptive observations [..., proprioceptive_obs_size].
            rng_key: Random key for sampling.
            
        Returns:
            Tuple of (actions, extras_dict).
        """
        key_prior, key_sample, key_action = random.split(rng_key, 3)
        
        # Get proprioceptive observations
        proprioceptive_obs = obs[..., -proprioceptive_obs_size:]

        # Normalize observations
        normalized_obs = preprocess_observations_fn(proprioceptive_obs, normalizer_params)
        
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
        
        # Sample action from distribution
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            raw_action = parametric_action_distribution.sample_no_postprocessing(logits, key_action)
            action = parametric_action_distribution.postprocess(raw_action)
        
        extras = {
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "intention": z,
            "logits": logits,
        }
        
        return action, extras
    
    return policy_fn


def extract_prior_decoder_params(policy_params: Tuple) -> Tuple[Dict, Dict, running_statistics.RunningStatisticsState]:
    """
    Extract prior and decoder parameters from full intention network policy params.
    
    Args:
        policy_params: Tuple of (normalizer_params, network_params).
        
    Returns:
        Tuple of (prior_params, decoder_params, normalizer_params).
    """
    normalizer_params, network_params = policy_params
    prior_params = network_params["params"]["prior"]
    decoder_params = network_params["params"]["decoder"]
    return prior_params, decoder_params, normalizer_params


def check_termination(
    data: Any,
    healthy_z_range: Tuple[float, float] = (0.0325, 0.5),
) -> jax.Array:
    """
    Check if a state should be terminated based on simple criteria.
    
    For prior-only rollouts, we use simpler termination criteria that don't
    require reference trajectories:
    1. NaN detection in the physics state
    2. Torso height outside healthy range (rodent fell or jumped too high)
    
    Args:
        data: MJX Data object containing physics state.
        healthy_z_range: Tuple of (min_z, max_z) for healthy torso height.
        
    Returns:
        Boolean array indicating whether the state is terminated.
    """
    from jax import flatten_util
    
    # Check for NaN values
    flattened_vals, _ = flatten_util.ravel_pytree(data.qpos)
    flattened_qvel, _ = flatten_util.ravel_pytree(data.qvel)
    all_vals = jnp.concatenate([flattened_vals, flattened_qvel])
    has_nan = jnp.any(jnp.isnan(all_vals))
    
    # Check torso height (assuming first body is torso/root)
    # The root body z-position is typically at index 2 of qpos for free joints
    torso_z = data.qpos[2]  # z-position of root
    min_z, max_z = healthy_z_range
    z_out_of_range = jnp.logical_or(torso_z < min_z, torso_z > max_z)
    
    return jnp.logical_or(has_nan, z_out_of_range)


def create_neutral_state(env, mjx_model) -> Any:
    """
    Create an environment state initialized to the neutral pose.

    The neutral pose is the default qpos from the MuJoCo model (qpos0),
    with the z-position adjusted to be within the healthy range.

    Args:
        env: The environment (used to get model info and action size).
        mjx_model: The MJX model for creating data.

    Returns:
        mjx_env.State initialized to the neutral pose.
    """
    import collections
    from mujoco import mjx
    from mujoco_playground._src import mjx_env

    # Get neutral qpos from model
    neutral_qpos = jnp.array(env.mj_model.qpos0)
    # Adjust z-position to be within healthy range (neutral has z=0, but healthy range starts at 0.0325)
    neutral_qpos = neutral_qpos.at[2].set(0.1)

    # Create MJX data with neutral pose
    data = mjx.make_data(mjx_model)
    data = data.replace(qpos=neutral_qpos)
    data = data.replace(qvel=jnp.zeros(mjx_model.nv))
    data = mjx.forward(mjx_model, data)

    # Create info dict needed for proprioception computation
    info = {
        "prev_action": jnp.zeros(env.action_size),
        "action": jnp.zeros(env.action_size),
        "start_frame": 0,
        "reference_clip": 1,
    }

    # Compute proprioceptive observation from the neutral pose
    # This is needed for the first step of the rollout
    proprioception = env._get_proprioception(data, info, flatten=False)
    obs = collections.OrderedDict(
        proprioception=proprioception,
    )

    reward = jnp.array(0.0)
    done = jnp.array(0.0)
    metrics = {}

    return mjx_env.State(data, obs, reward, done, metrics, info)



class PriorRolloutEvaluator:
    """
    Evaluator class for running prior-only rollouts during training.
    
    This class manages the evaluation of the prior network's ability to generate
    plausible actions by running rollouts using only the prior and decoder.
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
        num_rollouts: int = 32,
        max_steps: int = 200,
        healthy_z_range: Tuple[float, float] = (0.0325, 0.5),
        fixed_logvar: float = -2.0,
        deterministic: bool = False,
        eval_interval: int = 1,
    ):
        """
        Initialize the prior rollout evaluator.
        
        Args:
            env: The environment to run rollouts in.
            intention_latent_size: Size of intention latent space.
            action_size: Size of action space.
            proprioceptive_obs_size: Size of proprioceptive observations.
            decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
            prior_hidden_layer_sizes: Hidden layer sizes for prior.
            preprocess_observations_fn: Observation normalization function.
            num_rollouts: Number of rollouts per evaluation.
            max_steps: Maximum steps per rollout.
            healthy_z_range: Healthy torso height range for termination.
            fixed_logvar: Fixed log-variance for prior sampling.
            deterministic: Whether to use deterministic policy.
            eval_interval: Run evaluation every N evals (1 = every eval).
        """
        self.env = env
        self.intention_latent_size = intention_latent_size
        self.action_size = action_size
        self.proprioceptive_obs_size = proprioceptive_obs_size
        self.decoder_hidden_layer_sizes = decoder_hidden_layer_sizes
        self.prior_hidden_layer_sizes = prior_hidden_layer_sizes
        self.preprocess_observations_fn = preprocess_observations_fn
        self.num_rollouts = num_rollouts
        self.max_steps = max_steps
        self.healthy_z_range = healthy_z_range
        self.fixed_logvar = fixed_logvar
        self.deterministic = deterministic
        self.eval_interval = eval_interval
        
        self._key = random.PRNGKey(0)
        self._jit_evaluate = None
    
    def _build_evaluate_fn(self, policy_params: Tuple):
        """Build the jitted evaluation function."""
        prior_params, decoder_params, normalizer_params = extract_prior_decoder_params(policy_params)
        
        # Create proprioceptive-only normalizer params
        proprio_normalizer_params = running_statistics.RunningStatisticsState(
            count=normalizer_params.count,
            mean=normalizer_params.mean[-self.proprioceptive_obs_size:],
            summed_variance=normalizer_params.summed_variance[-self.proprioceptive_obs_size:],
            std=normalizer_params.std[-self.proprioceptive_obs_size:],
        )
        
        # Create policy function
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
            fixed_logvar=self.fixed_logvar,
            deterministic=self.deterministic,
        )
        
        jit_step = jax.jit(self.env.step)
        
        def single_rollout_fn(rng_key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            """Run a single prior rollout."""
            key_reset, key_rollout = random.split(rng_key)
            
            # Reset environment
            state = jit_reset(key_reset)
            
            def step_fn(carry, _):
                state, key, terminated, step_count = carry
                key, key_action = random.split(key)
                
                # Get proprioceptive observations
                if hasattr(state.obs, 'get') or isinstance(state.obs, dict):
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
                
                # Check termination
                step_terminated = check_termination(next_state.data, self.healthy_z_range)
                new_terminated = jnp.logical_or(terminated, step_terminated)
                
                # Only increment step count if not already terminated
                new_step_count = jnp.where(terminated, step_count, step_count + 1)
                
                return (next_state, key, new_terminated, new_step_count), None
            
            initial_carry = (state, key_rollout, jnp.array(False), jnp.array(0))
            (_, _, terminated, step_count), _ = jax.lax.scan(
                step_fn, initial_carry, None, length=self.max_steps
            )
            
            return step_count, terminated
        
        # Vmap and jit the rollout function
        vmapped_rollout = jax.vmap(single_rollout_fn)
        
        @jax.jit
        def evaluate_fn(rng_key: jax.Array) -> Dict[str, jax.Array]:
            rollout_keys = random.split(rng_key, self.num_rollouts)
            step_counts, terminated_flags = vmapped_rollout(rollout_keys)
            
            avg_steps = jnp.mean(step_counts.astype(jnp.float32))
            termination_rate = jnp.mean(terminated_flags.astype(jnp.float32))
            max_steps_reached = jnp.mean((step_counts >= self.max_steps).astype(jnp.float32))
            
            return {
                "prior_rollout/avg_steps": avg_steps,
                "prior_rollout/termination_rate": termination_rate,
                "prior_rollout/max_steps_reached": max_steps_reached,
                "prior_rollout/rollouts_min_steps": jnp.min(step_counts).astype(jnp.float32),
                "prior_rollout/rollouts_max_steps": jnp.max(step_counts).astype(jnp.float32),
            }
        
        return evaluate_fn
    
    def run_evaluation(
        self,
        policy_params: Tuple,
        eval_step: int,
    ) -> Optional[Dict[str, float]]:
        """
        Run prior rollout evaluation if it's time to do so.
        
        Args:
            policy_params: Current policy parameters.
            eval_step: Current evaluation step number.
            
        Returns:
            Dictionary of metrics if evaluation was run, None otherwise.
        """
        # Check if we should run evaluation this step
        if eval_step % self.eval_interval != 0:
            return None
        
        # Build the evaluate function with current params
        evaluate_fn = self._build_evaluate_fn(policy_params)
        
        # Get new random key
        self._key, eval_key = random.split(self._key)
        
        # Run evaluation
        import time
        t_start = time.time()
        metrics = evaluate_fn(eval_key)
        # Block until complete and convert to Python floats
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["prior_rollout/eval_time"] = time.time() - t_start
        
        return metrics

class PriorRolloutEvaluatorNeutralState:
    """
    Evaluator class for running prior-only rollouts during training.
    
    This class manages the evaluation of the prior network's ability to generate
    plausible actions by running rollouts using only the prior and decoder.
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
        num_rollouts: int = 32,
        max_steps: int = 200,
        healthy_z_range: Tuple[float, float] = (0.0325, 0.5),
        fixed_logvar: float = -2.0,
        deterministic: bool = False,
        eval_interval: int = 1,
    ):
        """
        Initialize the prior rollout evaluator.
        
        Args:
            env: The environment to run rollouts in.
            intention_latent_size: Size of intention latent space.
            action_size: Size of action space.
            proprioceptive_obs_size: Size of proprioceptive observations.
            decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
            prior_hidden_layer_sizes: Hidden layer sizes for prior.
            preprocess_observations_fn: Observation normalization function.
            num_rollouts: Number of rollouts per evaluation.
            max_steps: Maximum steps per rollout.
            healthy_z_range: Healthy torso height range for termination.
            fixed_logvar: Fixed log-variance for prior sampling.
            deterministic: Whether to use deterministic policy.
            eval_interval: Run evaluation every N evals (1 = every eval).
        """
        self.env = env
        self.mjx_model = env.mjx_model  # Store MJX model for neutral pose initialization
        self.intention_latent_size = intention_latent_size
        self.action_size = action_size
        self.proprioceptive_obs_size = proprioceptive_obs_size
        self.decoder_hidden_layer_sizes = decoder_hidden_layer_sizes
        self.prior_hidden_layer_sizes = prior_hidden_layer_sizes
        self.preprocess_observations_fn = preprocess_observations_fn
        self.num_rollouts = num_rollouts
        self.max_steps = max_steps
        self.healthy_z_range = healthy_z_range
        self.fixed_logvar = fixed_logvar
        self.deterministic = deterministic
        self.eval_interval = eval_interval
        
        self._key = random.PRNGKey(0)
        self._jit_evaluate = None
    
    def _build_evaluate_fn(self, policy_params: Tuple):
        """Build the jitted evaluation function."""
        prior_params, decoder_params, normalizer_params = extract_prior_decoder_params(policy_params)
        
        # Create proprioceptive-only normalizer params
        proprio_normalizer_params = running_statistics.RunningStatisticsState(
            count=normalizer_params.count,
            mean=normalizer_params.mean[-self.proprioceptive_obs_size:],
            summed_variance=normalizer_params.summed_variance[-self.proprioceptive_obs_size:],
            std=normalizer_params.std[-self.proprioceptive_obs_size:],
        )
        
        # Create policy function
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
            fixed_logvar=self.fixed_logvar,
            deterministic=self.deterministic,
        )
        
        jit_step = jax.jit(self.env.step)

        # Create jitted function for neutral state initialization
        jit_create_neutral_state = jax.jit(
            functools.partial(create_neutral_state, self.env, self.mjx_model)
        )

        def single_rollout_fn(rng_key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            """Run a single prior rollout starting from neutral pose."""
            key_rollout = rng_key  # No longer need to split for reset

            # Initialize from neutral pose instead of random clip
            state = jit_create_neutral_state()
            
            def step_fn(carry, _):
                state, key, terminated, step_count = carry
                key, key_action = random.split(key)
                
                # Get proprioceptive observations
                if hasattr(state.obs, 'get') or isinstance(state.obs, dict):
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
                
                # Check termination
                step_terminated = check_termination(next_state.data, self.healthy_z_range)
                new_terminated = jnp.logical_or(terminated, step_terminated)
                
                # Only increment step count if not already terminated
                new_step_count = jnp.where(terminated, step_count, step_count + 1)
                
                return (next_state, key, new_terminated, new_step_count), None
            
            initial_carry = (state, key_rollout, jnp.array(False), jnp.array(0))
            (_, _, terminated, step_count), _ = jax.lax.scan(
                step_fn, initial_carry, None, length=self.max_steps
            )
            
            return step_count, terminated
        
        # Vmap and jit the rollout function
        vmapped_rollout = jax.vmap(single_rollout_fn)
        
        @jax.jit
        def evaluate_fn(rng_key: jax.Array) -> Dict[str, jax.Array]:
            rollout_keys = random.split(rng_key, self.num_rollouts)
            step_counts, terminated_flags = vmapped_rollout(rollout_keys)
            
            avg_steps = jnp.mean(step_counts.astype(jnp.float32))
            termination_rate = jnp.mean(terminated_flags.astype(jnp.float32))
            max_steps_reached = jnp.mean((step_counts >= self.max_steps).astype(jnp.float32))
            
            return {
                "prior_rollout/avg_steps": avg_steps,
                "prior_rollout/termination_rate": termination_rate,
                "prior_rollout/max_steps_reached": max_steps_reached,
                "prior_rollout/rollouts_min_steps": jnp.min(step_counts).astype(jnp.float32),
                "prior_rollout/rollouts_max_steps": jnp.max(step_counts).astype(jnp.float32),
            }
        
        return evaluate_fn
    
    def run_evaluation(
        self,
        policy_params: Tuple,
        eval_step: int,
    ) -> Optional[Dict[str, float]]:
        """
        Run prior rollout evaluation if it's time to do so.
        
        Args:
            policy_params: Current policy parameters.
            eval_step: Current evaluation step number.
            
        Returns:
            Dictionary of metrics if evaluation was run, None otherwise.
        """
        # Check if we should run evaluation this step
        if eval_step % self.eval_interval != 0:
            return None
        
        # Build the evaluate function with current params
        evaluate_fn = self._build_evaluate_fn(policy_params)
        
        # Get new random key
        self._key, eval_key = random.split(self._key)
        
        # Run evaluation
        import time
        t_start = time.time()
        metrics = evaluate_fn(eval_key)
        # Block until complete and convert to Python floats
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["prior_rollout/eval_time"] = time.time() - t_start
        
        return metrics