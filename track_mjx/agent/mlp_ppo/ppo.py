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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import pmap
from brax.training import types
from brax.training import gradients
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
from track_mjx.agent import network_masks
from track_mjx.agent.mlp_ppo import losses, ppo_networks
from track_mjx.environment import wrappers
from track_mjx.agent import checkpointing
from mujoco_playground import wrapper as mp_wrapper

import flax
from flax import traverse_util
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax.transforms import freeze
import orbax.checkpoint as ocp

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics
STEPS_IN_THOUSANDS = 1e3

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def run_evaluation(
    self,
    policy_params,
    training_metrics: Metrics,
    aggregate_episodes: bool = True,
    data_split: str = "",
) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(policy_params, unroll_key)
    eval_metrics = eval_state.info["eval_metrics"]
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    prefix = f"{data_split}/" if data_split != "" else ""
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
    metrics[f"eval/{prefix}avg_episode_length"] = np.mean(eval_metrics.episode_steps)
    metrics[f"eval/{prefix}epoch_eval_time"] = epoch_eval_time
    metrics[f"eval/{prefix}sps"] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        f"eval/{prefix}walltime": self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray


# Monkey patch the run_evaluation method to include data_split
acting.Evaluator.run_evaluation = run_evaluation


# TODO: Pass in a loss-specific config instead of throwing them all in individually.
def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    ckpt_mgr: ocp.CheckpointManager,
    config_dict: dict,
    checkpoint_to_restore: str | None = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    kl_weight: float = 1e-3,
    discounting: float = 0.9,
    seed: int = 0,
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
    network_factory: types.NetworkFactory[
        ppo_networks.PPOImitationNetworks
    ] = ppo_networks.make_intention_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    eval_env_test_set: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    get_activation: bool = True,
    use_lstm: bool = True,
    use_kl_schedule: bool = True,
    kl_ramp_up_frac: float = 0.25,
    freeze_decoder: bool = False,
    checkpoint_callback: Optional[Callable[[int], None]] = None,
):
    """PPO training.

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
      eval_env: an optional environment for eval only, defaults to `environment`
      policy_params_fn: a user-defined callback function that can be used for
        saving policy checkpoints
      randomization_fn: a user-defined callback function that generates randomized
        environments
      get_activation: boolean argument indicating for getting activations of all of
        the networks
      use_lstm: boolean argument for using an LSTM decoder
      use_kl_schedule: whether to use a ramping schedule for the kl weight in the PPO loss
        (intention network variational layer)
      kl_ramp_up_frac: the fraction of the total number of evals to ramp up max kl weight
      checkpoint_callback: a callback function that is called after checkpointing to update
        the json file which contains the run state for preemption handling


    Returns:
      Tuple of (make_policy function, network params, metrics)
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

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        if (
            proprioceptive_obs_size > 0
            and frozen_proprioceptive_normalizer_params is not None
        ):
            normalizer_params = normalizer_params.replace(
                mean=normalizer_params.mean.at[-proprioceptive_obs_size:].set(
                    frozen_proprioceptive_normalizer_params.mean
                ),
                std=normalizer_params.std.at[-proprioceptive_obs_size:].set(
                    frozen_proprioceptive_normalizer_params.std
                ),
                summed_variance=normalizer_params.summed_variance.at[
                    -proprioceptive_obs_size:
                ].set(frozen_proprioceptive_normalizer_params.summed_variance),
            )

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

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

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

    if isinstance(environment, envs.Env):
        wrap_for_training = wrappers.wrap
    else:
        # adapt to mujoco_playground wrapper
        wrap_for_training = mp_wrapper.wrap_for_brax_training

    # TODO: playground env wrapper should be used here
    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
        # use_lstm=use_lstm,
    )

    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    reference_obs_size = int(_unpmap(env_state.info["reference_obs_size"])[0])
    # might be breaking change for old checkpoint
    if "proprioceptive_obs_size" not in env_state.info:
        proprioceptive_obs_size = 0
    else:
        proprioceptive_obs_size = int(
            _unpmap(env_state.info["proprioceptive_obs_size"])[0]
        )
        logging.info("Proprioceptive observation size: %s", proprioceptive_obs_size)

    # TODO: reference_obs_size should be optional (network factory-dependent)
    config_dict["network_config"].update(
        {
            "observation_size": env_state.obs.shape[-1],
            "action_size": env.action_size,
            "normalize_observations": normalize_observations,
            "reference_obs_size": reference_obs_size,
            "proprioceptive_obs_size": proprioceptive_obs_size,
        }
    )

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        env_state.obs.shape[-1],
        reference_obs_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    make_logging_policy = ppo_networks.make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate=learning_rate),
    )

    kl_schedule = None
    if use_kl_schedule:
        kl_schedule = losses.create_ramp_schedule(
            max_value=kl_weight,
            ramp_steps=int(num_evals * kl_ramp_up_frac),
            schedule="linear",
        )

    loss_fn = functools.partial(
        losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        kl_weight=kl_weight,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        kl_schedule=kl_schedule,
    )

    init_params = losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(
            init_params
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(
            specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
        ),
        env_steps=0,
    )

    frozen_proprioceptive_normalizer_params = None

    # Load the checkpoint if it exists
    if checkpoint_to_restore is not None:
        if not freeze_decoder:
            # we are recovering the full training state
            training_state = checkpointing.load_training_state(
                checkpoint_to_restore, training_state
            )
            logging.info(f"Restored latest checkpoint at {checkpoint_to_restore}")
        if freeze_decoder:
            # first is normalizer
            loaded_checkpoint = checkpointing.load_policy(checkpoint_to_restore)
            loaded_normalizer_params = loaded_checkpoint[0]
            loaded_policy = loaded_checkpoint[1]
            decoder_params = loaded_policy["params"]["decoder"]
            training_state.params.policy["params"]["decoder"] = decoder_params
            logging.info(
                f"Restored decoder parameters from checkpoint at {checkpoint_to_restore}"
            )
            mask = network_masks.create_decoder_mask(init_params)
            optimizer = optax.chain(optimizer, freeze(mask))
            # overwrite the optimizer state with the new optimizer
            training_state = training_state.replace(
                optimizer_state=optimizer.init(init_params)
            )
            logging.info("Freezing decoder parameters")
            # TODO WIP
            if proprioceptive_obs_size == 0:
                raise ValueError(
                    "Proprioceptive observation size is 0, "
                    "but decoder parameters are being frozen."
                )
            mean = loaded_normalizer_params.mean[-proprioceptive_obs_size:]
            std = loaded_normalizer_params.std[-proprioceptive_obs_size:]
            summed_variance = loaded_normalizer_params.summed_variance[
                -proprioceptive_obs_size:
            ]
            # TODO, normalizer implementations
            # this will remain unchanged, and use to set the decoder normalizer
            frozen_proprioceptive_normalizer_params = (
                running_statistics.RunningStatisticsState(
                    count=jnp.zeros(()),
                    mean=mean,
                    summed_variance=summed_variance,
                    std=std,
                )
            )
            training_state = training_state.replace(
                normalizer_params=training_state.normalizer_params.replace(
                    mean=training_state.normalizer_params.mean.at[
                        -proprioceptive_obs_size:
                    ].set(frozen_proprioceptive_normalizer_params.mean),
                    std=training_state.normalizer_params.std.at[
                        -proprioceptive_obs_size:
                    ].set(frozen_proprioceptive_normalizer_params.std),
                    summed_variance=training_state.normalizer_params.summed_variance.at[
                        -proprioceptive_obs_size:
                    ].set(frozen_proprioceptive_normalizer_params.summed_variance),
                )
            )

    # gradient update function with the new optimizer and loss function
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
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
        randomization_fn=v_randomization_fn,
        # use_lstm=use_lstm,
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
            randomization_fn=v_randomization_fn,
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
            # TODO: this is not correct, we need a way to overwrite somehow.
            # num_evals_after_init -= ckpt_mgr.latest_step()
            # start_it = ckpt_mgr.latest_step()
            pass

    print(f"Starting at iteration: {start_it} with {num_evals_after_init} evals left")

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
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
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys, it
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

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
            if it % config_dict["env_config"]["render_interval"] == 0:
                # Render video every `render_interval` iterations.
                policy_params_fn(
                    current_step=it,
                    jit_logging_inference_fn=jit_logging_inference_fn,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    render_video=True,
                )
            else:
                policy_params_fn(
                    current_step=it,
                    jit_logging_inference_fn=jit_logging_inference_fn,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    render_video=False,
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
