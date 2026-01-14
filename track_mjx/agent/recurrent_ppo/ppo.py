"""Proximal Policy Optimization (PPO) training for recurrent intention networks.

This module implements PPO training with recurrent decoder networks for
VNL imitation learning. The policy uses an MLP encoder and RNN decoder,
while the value network remains feedforward.

Key features:
- Recurrent decoder with configurable RNN cell type (SimpleCell, GRU, LSTM)
- Hidden state management across training steps
- VAE losses (KL divergence, AR1 temporal smoothness)
- Observation normalization with dict observations
- Checkpoint saving/restoration with preemption recovery
- Train/test split evaluation
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
from brax.training import pmap, types
from brax.training.types import Params, PRNGKey
from mujoco_playground import wrapper as mp_wrapper

from track_mjx.agent import checkpointing, gradients
from track_mjx.agent.recurrent_ppo import losses, networks
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    get_obs_sizes,
    init_dict_normalizer,
    update_dict_normalizer,
)


# Type aliases
InferenceParams = tuple[DictRunningStatisticsState, Params]
Metrics = types.Metrics
HiddenState = networks.HiddenState

# Constants
STEPS_IN_THOUSANDS = 1e3
_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Training state for recurrent PPO learner.

    Attributes:
        optimizer_state: Optax optimizer state.
        params: PPO network parameters (policy and value).
        normalizer_params: Running statistics for observation normalization
            (per observation key).
        env_steps: Total environment steps taken (in thousands).
    """

    optimizer_state: optax.OptState
    params: losses.RecurrentPPONetworkParams
    normalizer_params: DictRunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v: Any) -> Any:
    """Extract first device's values from a pmap'd pytree."""
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree: Any) -> Any:
    """Remove weak types from a pytree to prevent JIT recompilation."""

    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def actor_step_rnn(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    policy_hidden: HiddenState | list[HiddenState],
    key: PRNGKey,
    extra_fields: tuple = (),
    cell_type: networks.RNNCellType = "gru",
) -> Tuple[envs.State, types.Transition, HiddenState | list[HiddenState]]:
    """Collect one step with recurrent policy.

    Args:
        env: Environment.
        env_state: Current environment state.
        policy: Recurrent policy function.
        policy_hidden: Policy network hidden state.
        key: Random key.
        extra_fields: Extra fields to collect from env state.
        cell_type: RNN cell type for hidden state reset.

    Returns:
        Tuple of (next_state, transition, new_policy_hidden).
    """
    actions, policy_extras, new_policy_hidden = policy(
        env_state.obs, policy_hidden, key
    )
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}

    transition = types.Transition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

    # Reset hidden state where episodes ended
    new_policy_hidden = networks.reset_hidden_on_done(
        new_policy_hidden, nstate.done, cell_type
    )

    return nstate, transition, new_policy_hidden


def generate_unroll_rnn(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    policy_hidden: HiddenState | list[HiddenState],
    key: PRNGKey,
    unroll_length: int,
    extra_fields: tuple = (),
    cell_type: networks.RNNCellType = "gru",
) -> Tuple[envs.State, types.Transition, HiddenState | list[HiddenState]]:
    """Collect trajectory with recurrent policy.

    Args:
        env: Environment.
        env_state: Current environment state.
        policy: Recurrent policy function.
        policy_hidden: Policy network hidden state.
        key: Random key.
        unroll_length: Number of steps to unroll.
        extra_fields: Extra fields to collect.
        cell_type: RNN cell type for hidden state reset.

    Returns:
        Tuple of (final_state, transitions, final_policy_hidden).
    """

    def f(carry, unused_t):
        state, hidden, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition, new_hidden = actor_step_rnn(
            env,
            state,
            policy,
            hidden,
            current_key,
            extra_fields=extra_fields,
            cell_type=cell_type,
        )
        return (nstate, new_hidden, next_key), transition

    (final_state, final_hidden, _), data = jax.lax.scan(
        f, (env_state, policy_hidden, key), (), length=unroll_length
    )
    return final_state, data, final_hidden


class RecurrentEvaluator:
    """Evaluator for recurrent policies."""

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[Params], Callable],
        recurrent_ppo_network: networks.RecurrentPPONetworks,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        """Initialize recurrent evaluator.

        Args:
            eval_env: Evaluation environment.
            eval_policy_fn: Function to create policy from params.
            recurrent_ppo_network: Network container.
            num_eval_envs: Number of evaluation environments.
            episode_length: Episode length.
            action_repeat: Action repeat.
            key: Random key.
        """
        self._key = key
        self._eval_walltime = 0.0
        self._network = recurrent_ppo_network
        self._num_eval_envs = num_eval_envs

        eval_env = envs.training.EvalWrapper(eval_env)
        self._eval_state_to_donate = jax.jit(eval_env.reset)(
            jax.random.split(key, num_eval_envs)
        )

        def generate_eval_unroll(
            eval_env_state_donated: envs.State,
            policy_params: Params,
            key: PRNGKey,
        ) -> envs.State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)

            # Initialize policy hidden state
            policy_hidden = recurrent_ppo_network.policy_network.init_hidden(
                num_eval_envs
            )

            final_state, _, _ = generate_unroll_rnn(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                policy_hidden,
                key,
                unroll_length=episode_length // action_repeat,
                cell_type=recurrent_ppo_network.cell_type,
            )
            return final_state

        self._generate_eval_unroll = jax.jit(
            generate_eval_unroll, donate_argnums=(0,), keep_unused=True
        )
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params: Params,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
        data_split: str = "",
    ) -> Metrics:
        """Run one epoch of evaluation.

        Args:
            policy_params: Policy parameters.
            training_metrics: Training metrics to include.
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

        metrics = {}
        prefix = f"{data_split}/" if data_split else ""

        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"eval/{prefix}episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        metrics[f"eval/{prefix}avg_episode_length"] = np.mean(
            eval_metrics.episode_steps
        )
        metrics[f"eval/{prefix}epoch_eval_time"] = epoch_eval_time
        metrics[f"eval/{prefix}sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            f"eval/{prefix}walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }

        return metrics


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
        ..., networks.RecurrentPPONetworks
    ] = networks.make_recurrent_intention_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    eval_env: envs.Env | None = None,
    eval_env_test_set: envs.Env | None = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: (
        Callable[[base.System, jnp.ndarray], tuple[base.System, base.System]] | None
    ) = None,
    use_kl_schedule: bool = True,
    kl_ramp_up_frac: float = 0.25,
    checkpoint_callback: Callable[[int], None] | None = None,
    grad_clip_threshold: float = 20.0,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
) -> tuple[Callable, InferenceParams, Metrics]:
    """Train a recurrent PPO agent on the given environment.

    Args:
        environment: The environment to train.
        num_timesteps: Total environment steps for training.
        episode_length: Episode length.
        ckpt_mgr: Orbax checkpoint manager.
        config_dict: Configuration dictionary.
        checkpoint_to_restore: Path to checkpoint to restore from.
        action_repeat: Action repeat count.
        num_envs: Number of parallel environments.
        max_devices_per_host: Max chips per host.
        num_eval_envs: Evaluation environments count.
        learning_rate: Learning rate.
        entropy_cost: Entropy bonus coefficient.
        latent_kl_weight: KL divergence weight.
        latent_ar1_weight: AR1 temporal loss weight.
        discounting: Discount factor (gamma).
        seed: Random seed.
        use_pmap_on_reset: Use pmap for env.reset.
        unroll_length: Timesteps per unroll.
        batch_size: Batch size per minibatch.
        num_minibatches: Number of minibatches.
        num_updates_per_batch: Gradient updates per batch.
        num_evals: Number of evaluations during training.
        num_resets_per_eval: Environment resets between evals.
        normalize_observations: Whether to normalize observations.
        reward_scaling: Reward multiplier.
        clipping_epsilon: PPO clipping epsilon.
        gae_lambda: GAE lambda.
        deterministic_eval: Use deterministic policy for eval.
        network_factory: Function to create networks.
        progress_fn: Callback for reporting metrics.
        normalize_advantage: Normalize advantages.
        vf_loss_coefficient: Value function loss coefficient.
        eval_env: Evaluation environment.
        eval_env_test_set: Test set evaluation environment.
        policy_params_fn: Callback for saving policy params.
        randomization_fn: Domain randomization function.
        use_kl_schedule: Use ramping KL schedule.
        kl_ramp_up_frac: Fraction of evals to ramp up KL.
        checkpoint_callback: Callback after checkpointing.
        grad_clip_threshold: Gradient clipping norm.
        wrap_for_training: Environment wrapper function.

    Returns:
        Tuple of (make_policy, params, metrics).
    """
    assert batch_size * num_minibatches % num_envs == 0
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

    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value, policy_params_fn_key = jax.random.split(global_key, 3)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
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

    obs_sizes = get_obs_sizes(env_state.obs)
    logging.info(f"Observation sizes: {obs_sizes}")

    config_dict["network_config"].update(
        {
            "obs_sizes": obs_sizes,
            "action_size": env.action_size,
            "normalize_observations": normalize_observations,
        }
    )

    recurrent_ppo_network = network_factory(
        obs_sizes,
        env.action_size,
    )
    make_policy = networks.make_inference_fn(recurrent_ppo_network)

    # Create logging inference function for evaluation/rollouts
    make_logging_policy = networks.make_logging_inference_fn(recurrent_ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    # Number of environments per device
    envs_per_device = num_envs // device_count

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
        losses.compute_recurrent_ppo_loss,
        recurrent_ppo_network=recurrent_ppo_network,
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

    init_params = losses.RecurrentPPONetworkParams(
        policy=recurrent_ppo_network.policy_network.init(key_policy),
        value=recurrent_ppo_network.value_network.init(key_value),
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=init_dict_normalizer(env_state.obs),
        env_steps=0,
    )

    # Load checkpoint if provided
    if checkpoint_to_restore is not None:
        training_state = checkpointing.load_training_state(
            checkpoint_to_restore, training_state
        )
        logging.info(f"Restored checkpoint from {checkpoint_to_restore}")

    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn,
        optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
        clip_threshold=grad_clip_threshold,
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: DictRunningStatisticsState,
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
        normalizer_params: DictRunningStatisticsState,
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
        carry: Tuple[TrainingState, envs.State, list[HiddenState], PRNGKey, int],
        unused_t,
    ) -> Tuple[
        Tuple[TrainingState, envs.State, list[HiddenState], PRNGKey, int], Metrics
    ]:
        training_state, state, policy_hidden, key, it = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (training_state.normalizer_params, training_state.params.policy)
        )

        def f(carry, unused_t):
            current_state, current_hidden, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            initial_hidden = current_hidden
            next_state, data, new_hidden = generate_unroll_rnn(
                env,
                current_state,
                policy,
                current_hidden,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
                cell_type=recurrent_ppo_network.cell_type,
            )
            return (next_state, new_hidden, next_key), (data, initial_hidden)

        (state, policy_hidden, _), (data, initial_policy_hidden) = jax.lax.scan(
            f,
            (state, policy_hidden, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )

        # Reshape: (num_unrolls, unroll_length, envs_per_device) -> (batch, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        # Reshape initial hidden states similarly
        initial_policy_hidden = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
            initial_policy_hidden,
        )
        assert data.discount.shape[1:] == (unroll_length,)

        # Add initial hidden state to policy extras
        policy_extras = dict(data.extras["policy_extras"])
        policy_extras["initial_policy_hidden"] = initial_policy_hidden
        data = types.Transition(
            observation=data.observation,
            action=data.action,
            reward=data.reward,
            discount=data.discount,
            next_observation=data.next_observation,
            extras={
                "policy_extras": policy_extras,
                "state_extras": data.extras["state_extras"],
            },
        )

        # Update normalization params
        if normalize_observations:
            normalizer_params = update_dict_normalizer(
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
            ),
        )
        return (new_training_state, state, policy_hidden, new_key, it), metrics

    def training_epoch(
        training_state: TrainingState,
        state: envs.State,
        policy_hidden: list[HiddenState],
        key: PRNGKey,
        it: int,
    ) -> Tuple[TrainingState, envs.State, list[HiddenState], Metrics]:
        (training_state, state, policy_hidden, _, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, policy_hidden, key, it),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, policy_hidden, loss_metrics

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1, 2),
    )

    training_walltime = 0

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        policy_hidden: list[HiddenState],
        key: PRNGKey,
        it: int,
    ) -> Tuple[TrainingState, envs.State, list[HiddenState], Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state, policy_hidden = _strip_weak_type(
            (training_state, env_state, policy_hidden)
        )
        step = jnp.ones_like(training_state.env_steps) * it
        result = training_epoch(training_state, env_state, policy_hidden, key, step)
        training_state, env_state, policy_hidden, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, policy_hidden, metrics

    # Replicate training state across devices
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    # Initialize policy hidden state for each device
    # Shape: [devices, envs_per_device, hidden_size]
    policy_hidden = recurrent_ppo_network.policy_network.init_hidden(envs_per_device)
    policy_hidden = jax.device_put_replicated(
        policy_hidden, jax.local_devices()[:local_devices_to_use]
    )

    # Setup evaluation
    if eval_env is None:
        eval_env = environment
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )

    evaluator = RecurrentEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        recurrent_ppo_network,
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
        evaluator_test_set = RecurrentEvaluator(
            eval_env_test_set,
            functools.partial(make_policy, deterministic=deterministic_eval),
            recurrent_ppo_network,
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=key_env_test_set,
        )

    # Handle checkpoint restoration iteration count
    start_it = 0
    if ckpt_mgr is not None:
        latest_step = ckpt_mgr.latest_step()
        if latest_step is not None:
            num_evals_after_init -= latest_step
            start_it = latest_step

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
        metrics = evaluator.run_evaluation(policy_param, training_metrics={})
        if evaluator_test_set is not None:
            metrics = evaluator_test_set.run_evaluation(
                policy_param, metrics, data_split="test_set"
            )
        logging.info(metrics)
        progress_fn(start_it, metrics)

        # Save initial checkpoint
        if ckpt_mgr is not None:
            ckpt_mgr.save(
                step=0,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardSave(policy_param),
                    train_state=ocp.args.StandardSave(_unpmap(training_state)),
                    config=ocp.args.JsonSave(config_dict),
                ),
            )
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(0)
                except Exception as e:
                    logging.warning(f"Initial checkpoint callback failed: {e}")

    training_metrics = {}
    start_it += 1
    current_step = 0

    for it in range(start_it, num_evals_after_init + start_it):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (
                training_state,
                env_state,
                policy_hidden,
                training_metrics,
            ) = training_epoch_with_timing(
                training_state, env_state, policy_hidden, epoch_keys, it
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])

            if num_resets_per_eval > 0:
                env_state = reset_fn((training_state, env_state), key_envs)
                # Reset hidden state on environment reset
                policy_hidden = recurrent_ppo_network.policy_network.init_hidden(
                    envs_per_device
                )
                policy_hidden = jax.device_put_replicated(
                    policy_hidden, jax.local_devices()[:local_devices_to_use]
                )

        if process_id == 0:
            policy_param = _unpmap(
                (training_state.normalizer_params, training_state.params.policy)
            )
            metrics = evaluator.run_evaluation(policy_param, training_metrics)
            if evaluator_test_set is not None:
                metrics = evaluator_test_set.run_evaluation(
                    policy_param, metrics, data_split="test_set"
                )

            # Policy params callback
            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            render_video = (
                it % config_dict.get("render_config", {}).get("render_interval", 10)
                == 0
            )
            policy_params_fn(
                current_step=it,
                jit_logging_inference_fn=jit_logging_inference_fn,
                params=policy_param,
                policy_params_fn_key=policy_params_fn_key,
                render_video=render_video,
                ppo_network=recurrent_ppo_network,
            )

            logging.info(metrics)
            progress_fn(current_step, metrics)

            if ckpt_mgr is not None:
                checkpointing.save(
                    ckpt_mgr,
                    it,
                    policy_param,
                    _unpmap(training_state),
                    config_dict,
                    checkpoint_callback,
                )

    pmap.assert_is_replicated(training_state)
    params = _unpmap((training_state.normalizer_params, training_state.params.policy))
    logging.info("total steps: %s", current_step)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
