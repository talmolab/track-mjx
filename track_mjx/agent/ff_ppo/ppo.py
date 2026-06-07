# Copyright 2024 The Brax Authors.
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

"""Proximal Policy Optimization (PPO) training for VNL imitation learning.

This module implements PPO training with support for:
- Intention-based policy networks (VAE-style latent encoding)
- Observation normalization
- Checkpoint saving/restoration with preemption recovery
- Train/test split evaluation
- KL divergence scheduling for VAE training stability

Based on Brax's PPO implementation with modifications for VNL tracking tasks.
"""

import functools
import time
from typing import Any, Callable, Tuple

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import logging
from brax import base, envs
from brax.training import acting, pmap, types
from brax.training.acme import running_statistics
from brax.training.types import Params, PRNGKey
from mujoco_playground import wrapper as mp_wrapper
from track_mjx.agent import checkpointing, gradients
from track_mjx.agent.ff_ppo import losses, ppo_networks
from track_mjx.agent.observation_utils import (
    get_obs_sizes,
    get_obs_shape,
)

# Type aliases
InferenceParams = tuple[running_statistics.RunningStatisticsState, Params]
Metrics = types.Metrics

# Constants
STEPS_IN_THOUSANDS = 1e3
_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Training state for PPO learner.

    Attributes:
        optimizer_state: Optax optimizer state.
        params: PPO network parameters (policy and value).
        normalizer_params: Running statistics for observation normalization.
            Has pytree structure matching observation dict.
        env_steps: Total environment steps taken (in thousands).
    """

    optimizer_state: optax.OptState
    params: losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v: Any) -> Any:
    """Extract first device's values from a pmap'd pytree.

    Args:
        v: Pytree with leading device axis from pmap.

    Returns:
        Pytree with device axis removed (values from first device).
    """
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree: Any) -> Any:
    """Remove weak types from a pytree to prevent JIT recompilation.

    Brax user code is sometimes ambiguous about weak_type, which can cause
    unnecessary JIT recompilations.

    Args:
        tree: Input pytree potentially containing weak-typed arrays.

    Returns:
        Pytree with all arrays converted to their canonical dtype.
    """

    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _agg_fn(metric, fn, to_aggregate, to_normalize, episode_lengths):
    if not to_aggregate:
        return metric
    if to_normalize:
        return fn(metric / episode_lengths)
    return fn(metric)


def run_evaluation(
    self,
    policy_params: InferenceParams,
    training_metrics: Metrics,
    aggregate_episodes: bool = True,
    data_split: str = "",
) -> Metrics:
    """Run one epoch of evaluation.

    Extended version of Brax's Evaluator.run_evaluation that supports
    data_split prefixes for train/test set metrics separation.

    Args:
        self: Evaluator instance.
        policy_params: Tuple of (normalizer_params, policy_params).
        training_metrics: Training metrics to include in output.
        aggregate_episodes: If True, compute mean/std across episodes.
        data_split: Prefix for metric keys (e.g., "test_set").

    Returns:
        Dictionary of evaluation metrics merged with training_metrics.
    """
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(
        self._eval_state_to_donate, policy_params, unroll_key
    )
    self._eval_state_to_donate = eval_state
    eval_metrics = eval_state.info["eval_metrics"]
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    episode_lengths = np.maximum(eval_metrics.episode_steps, 1.0).astype(float)

    metrics = {}
    prefix = f"{data_split}/" if data_split else ""

    for fn in [np.mean, np.std]:
        suffix = "_std" if fn == np.std else ""
        metrics.update(
            {
                f"eval/{prefix}episode_{name}{suffix}": _agg_fn(
                    value, fn, aggregate_episodes, "per_step" in name, episode_lengths
                )
                for name, value in eval_metrics.episode_metrics.items()
            }
        )

    metrics[f"eval/{prefix}avg_episode_length"] = np.mean(eval_metrics.episode_steps)
    metrics[f"eval/{prefix}epoch_eval_time"] = epoch_eval_time
    metrics[f"eval/{prefix}sps"] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        f"eval/{prefix}walltime": self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics


# Monkey-patch Evaluator to support data_split parameter for train/test metrics
acting.Evaluator.run_evaluation = run_evaluation


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    ckpt_mgr: ocp.CheckpointManager,
    config_dict: dict[str, Any],
    checkpoint_to_restore: str | None = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: int | None = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    latent_kl_weight: float = 1e-3,
    latent_ar1_weight: float = 1e-3,
    discounting: float = 0.9,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 20,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    network_factory: Callable[
        ..., ppo_networks.PPOImitationNetworks
    ] = ppo_networks.make_intention_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    eval_env: envs.Env | None = None,
    eval_env_test_set: envs.Env | None = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: (
        Callable[[base.System, jnp.ndarray], tuple[base.System, base.System]] | None
    ) = None,
    get_activation: bool = True,
    use_kl_schedule: bool = True,
    kl_ramp_up_frac: float = 0.25,
    checkpoint_callback: Callable[[int], None] | None = None,
    grad_clip_threshold: float = 20.0,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
) -> tuple[Callable, InferenceParams, Metrics]:
    """Train a PPO agent on the given environment.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      episode_length: the length of an environment episode
      ckpt_mgr: an orbax checkpoint manager for saving policy checkpoints
      config_dict: a dictionary that contains the configuration for the training,
        will be saved to the orbax checkpoint alongside with the policy and training state
      checkpoint_to_restore: Optional path for a checkpoint to load to resume training
      action_repeat: the number of timesteps to repeat an action
      num_envs: the number of parallel environments to use for rollouts
        NOTE: `num_envs` must be divisible by the total number of chips since each
          chip gets `num_envs // total_number_of_chips` environments to roll out
        NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
          data generated by `num_envs` parallel envs gets used for gradient
          updates over `num_minibatches` of data, where each minibatch has a
          leading dimension of `batch_size`
      max_devices_per_host: maximum number of chips to use per host process
      num_eval_envs: the number of envs to use for evluation. Each env will run 1
        episode, and all envs run in parallel during eval.
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy
        of the policy
      discounting: discounting rate
      seed: random seed
      unroll_length: the number of timesteps to unroll in each environment. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new environment rollout
      num_evals: the number of evals to run during the entire training run.
        Increasing the number of evals increases total training time
      num_resets_per_eval: the number of environment resets to run between each
        eval. The environment resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      deterministic_eval: whether to run the eval with a deterministic policy
      network_factory: function that generates networks for policy and value
        functions
      progress_fn: a user-defined callback function for reporting/plotting metrics
      normalize_advantage: whether to normalize advantage estimate
      vf_loss_coefficient: Coefficient for value function loss.
      eval_env: an optional environment for eval only, defaults to `environment`
      policy_params_fn: a user-defined callback function that can be used for
        saving policy checkpoints
      randomization_fn: a user-defined callback function that generates randomized
        environments
      get_activation: boolean argument indicating for getting activations of all of
        the networks
      use_kl_schedule: whether to use a ramping schedule for the kl weight in the PPO loss
        (intention network variational layer)
      kl_ramp_up_frac: the fraction of the total number of evals to ramp up max kl weight
      checkpoint_callback: Callback called after checkpointing to update
        run state JSON for preemption recovery.
      grad_clip_threshold: Maximum gradient norm for clipping.
      wrap_for_training: Function that wraps environment for training.
      use_pmap_on_reset: whether to use pmap instead of vmap for env.reset across devices.
    Returns:
        Tuple of:
            - make_policy: Function to create inference policy from params.
            - params: Trained (normalizer_params, policy_params) tuple.
            - metrics: Final evaluation metrics dictionary.
    """
    assert batch_size * num_minibatches % num_envs == 0, (
        batch_size * num_minibatches % num_envs
    )
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    # TODO (Scott:) this will be dependent of the eval interval,
    # it could be confusing when loading a checkpoint
    # and num_evals is not the same as the one used for training.
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key, it = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            it,
            optimizer_state=optimizer_state,
            params=params,
        )

        return (optimizer_state, params, key, it), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key, it = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad, it),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key, it), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey, int], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey, int], Metrics]:
        training_state, state, key, it = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (training_state.normalizer_params, training_state.params.policy)
        )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params (only if normalization is enabled).
        # When disabled, normalizer stays at identity (mean=0, std=1).
        if normalize_observations:
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
        else:
            normalizer_params = training_state.normalizer_params

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd, it),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=jnp.int32(
                training_state.env_steps
                + env_step_per_training_step / STEPS_IN_THOUSANDS
            ),  # env step in thousands
        )
        return (new_training_state, state, new_key, it), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key, it),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1),
    )

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        step = jnp.ones_like(training_state.env_steps) * it
        result = training_epoch(training_state, env_state, key, step)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        # epoch_training_time times a single training_epoch (one iteration of the
        # `for _ in range(max(num_resets_per_eval, 1))` reset loop in the main
        # training loop). num_training_steps_per_epoch already divides the step
        # budget by num_resets_per_eval, so multiplying the numerator by it again
        # double-counts and over-reports sps by exactly that factor.
        sps = (
            num_training_steps_per_epoch * env_step_per_training_step
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value, policy_params_fn_key = jax.random.split(global_key, 3)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    def reset_fn_donated_env_state(env_state_donated, key_envs):
        return env.reset(key_envs)

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    if local_devices_to_use > 1 or use_pmap_on_reset:
        reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
        env_state = reset_fn_(key_envs)
        reset_fn = jax.pmap(
            reset_fn_donated_env_state,
            axis_name=_PMAP_AXIS_NAME,
            donate_argnums=(0,),
        )
    else:
        reset_fn_ = jax.jit(jax.vmap(env.reset))
        env_state = reset_fn_(key_envs)
        reset_fn = jax.jit(
            reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
        )(key_envs)

    # Extract observation sizes from the dict observation
    obs_sizes = get_obs_sizes(env_state.obs)
    logging.info(f"Observation sizes: {obs_sizes}")

    config_dict["network_config"].update(
        {
            "obs_sizes": obs_sizes,
            "action_size": env.action_size,
            "normalize_observations": normalize_observations,
        }
    )

    ppo_network = network_factory(
        obs_sizes,
        env.action_size,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    make_logging_policy = ppo_networks.make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    grad_clip_threshold = 20.0
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_threshold),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0, eps=1e-5),
    )

    latent_kl_schedule = None
    latent_ar1_schedule = None
    if use_kl_schedule:
        latent_kl_schedule = losses.create_ramp_schedule(
            max_value=latent_kl_weight,
            ramp_steps=int(num_evals * kl_ramp_up_frac),
            schedule="linear",
        )
        latent_ar1_schedule = losses.create_ramp_schedule(
            max_value=latent_ar1_weight,
            ramp_steps=int(num_evals * kl_ramp_up_frac),
            schedule="linear",
        )

    loss_fn = functools.partial(
        losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        latent_kl_weight=latent_kl_weight,
        latent_ar1_weight=latent_ar1_weight,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        vf_coefficient=vf_loss_coefficient,
        latent_kl_schedule=latent_kl_schedule,
        latent_ar1_schedule=latent_ar1_schedule,
    )

    init_params = losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )

    # Initialize normalizer with pytree structure matching observations
    obs_shape = get_obs_shape(env_state.obs)
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(obs_shape),
        env_steps=0,
    )

    # Load the checkpoint if it exists
    if checkpoint_to_restore is not None:
        training_state = checkpointing.load_training_state(
            checkpoint_to_restore, training_state
        )
        logging.info(f"Restored latest checkpoint at {checkpoint_to_restore}")

    # gradient update function with the new optimizer and loss function
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn,
        optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
        clip_threshold=grad_clip_threshold,
    )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    if eval_env is None:
        eval_env = environment
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
        )
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    evaluator_test_set = None
    if eval_env_test_set is not None:
        key_env, key_env_test_set = jax.random.split(key_env, 2)
        eval_env_test_set = wrap_for_training(
            eval_env_test_set,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=None,
        )
        evaluator_test_set = acting.Evaluator(
            eval_env_test_set,
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=key_env_test_set,
        )

    # Logic to restore iteration count from checkpoint
    start_it = 0
    if ckpt_mgr is not None:
        if ckpt_mgr.latest_step() is not None:
            num_evals_after_init -= ckpt_mgr.latest_step()
            start_it = ckpt_mgr.latest_step()
            pass

    logging.info(
        f"Starting at iteration: {start_it} with {num_evals_after_init} evals left"
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1 and start_it == 0:
        logging.info("Running initial evaluation")
        policy_param = _unpmap(
            (training_state.normalizer_params, training_state.params.policy)
        )
        metrics = evaluator.run_evaluation(
            policy_param,
            training_metrics={},
        )
        if evaluator_test_set is not None:
            # run evaluation on hold out test set
            metrics = evaluator_test_set.run_evaluation(
                policy_param,
                training_metrics=metrics,
                data_split="test_set",
            )
        logging.info(metrics)
        progress_fn(start_it, metrics)
        # Save checkpoints
        logging.info("Saving initial checkpoint")
        if ckpt_mgr is not None:
            # new orbax API
            ckpt_mgr.save(
                step=0,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardSave(policy_param),
                    train_state=ocp.args.StandardSave(_unpmap(training_state)),
                    config=ocp.args.JsonSave(config_dict),
                ),
            )
            # Call checkpoint callback for initial save
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(0)
                except Exception as e:
                    logging.warning(f"Initial checkpoint callback failed: {e}")
        else:
            logging.info("Skipping checkpoint save as ckpt_mgr is None")

    training_metrics = {}
    training_walltime = 0
    start_it += 1
    current_step = 0
    for it in range(start_it, num_evals_after_init + start_it):
        logging.info("starting iteration %s %s", it, time.time() - xt)
        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            training_state, env_state, training_metrics = training_epoch_with_timing(
                training_state, env_state, epoch_keys, it
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            if num_resets_per_eval > 0:
                env_state = reset_fn((training_state, env_state), key_envs)

        if process_id == 0:
            # Run evaluation rollout, logging and checkpointing.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_state.normalizer_params, training_state.params.policy)
                ),
                training_metrics,
            )
            if evaluator_test_set is not None:
                # run evaluation on hold out test set
                metrics = evaluator_test_set.run_evaluation(
                    _unpmap(
                        (training_state.normalizer_params, training_state.params.policy)
                    ),
                    metrics,
                    data_split="test_set",
                )

            policy_param = _unpmap(
                (training_state.normalizer_params, training_state.params.policy)
            )
            # Do policy evaluation and logging.
            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            if it % config_dict["render_config"]["render_interval"] == 0:
                # Render video every `render_interval` iterations.
                policy_params_fn(
                    current_step=it,
                    jit_logging_inference_fn=jit_logging_inference_fn,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    render_video=True,
                    ppo_network=ppo_network,
                )
            else:
                policy_params_fn(
                    current_step=it,
                    jit_logging_inference_fn=jit_logging_inference_fn,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    render_video=False,
                    ppo_network=ppo_network,
                )

            # log metrics
            logging.info(metrics)
            progress_fn(current_step, metrics)
            # Save checkpoint
            if ckpt_mgr is not None:
                checkpointing.save(
                    ckpt_mgr,
                    it,
                    policy_param,
                    _unpmap(training_state),
                    config_dict,
                    checkpoint_callback,
                )

    total_steps = current_step
    # TODO: this assert will fail
    # assert (
    #     total_steps >= num_timesteps / STEPS_IN_THOUSANDS
    # ), "Total steps must be at least the number of timesteps scaled to thousands."

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap((training_state.normalizer_params, training_state.params.policy))
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
