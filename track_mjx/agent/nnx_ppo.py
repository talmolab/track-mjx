"""
PPO implementation using flax new API, nnx.

Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union

from absl import logging
import brax
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs

# from brax.training.agents.ppo import losses as ppo_losses
from track_mjx.agent import nnx_ppo_losses as ppo_losses
from track_mjx.agent import nnx_ppo_network
from track_mjx.agent.nnx_ppo_network import make_intention_ppo_networks, PPOTrainConfig, make_policy_fn
from track_mjx.agent.intention_network import IntentionNetwork

# from brax.training.agents.ppo import networks as ppo_networks
from track_mjx.agent import custom_ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

from track_mjx.environment import custom_wrappers

_PMAP_AXIS_NAME = "i"

import flax

def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: None
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray

def train(
    environment: envs.Env,
    config: PPOTrainConfig,
):
    ### training setup ###
    assert config.batch_size * config.num_minibatches % config.num_envs == 0, (
        config.batch_size * config.num_minibatches % config.num_envs
    )
    xt = time.time()
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if config.max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, " "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count
    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        config.batch_size * config.unroll_length * config.num_minibatches * config.action_repeat
    )
    num_evals_after_init = max(config.num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        config.num_timesteps / (num_evals_after_init * env_step_per_training_step * max(config.num_resets_per_eval, 1))
    ).astype(int)

    key = jax.random.PRNGKey(config.seed)
    global_key, local_key = jax.random.split(key)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    assert config.num_envs % device_count == 0

    ### Environment setup ###
    v_randomization_fn = None
    if config.randomization_fn is not None:
        randomization_batch_size = config.num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(config.randomization_fn, rng=randomization_rng)

    wrap_for_training = custom_wrappers.wrap

    env = wrap_for_training(
        environment,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
        randomization_fn=v_randomization_fn,
    )
    reset_fn = jax.jit(jax.vmap(env.reset))
    # reset_fn = jax.vmap(env.reset)
    key_envs = jax.random.split(key_env, config.num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)
    # TODO(Scott): this is some very ugly code, we should refactor this.
    normalize = lambda x, y: x
    if config.normalize_observations:
        normalize = running_statistics.normalize

    ### Policy and Network setup ###

    def _unpmap(v):
        return jax.tree_util.tree_map(lambda x: x[0], v)

    # create the ppo network with policy and value networks
    ppo_network = make_intention_ppo_networks(
        observation_size=env_state.obs.shape[-1],
        reference_obs_size=int(_unpmap(env_state.info["reference_obs_size"])[0]),
        action_size=env.action_size,
        preprocess_observations_fn=normalize,
        intention_latent_size=60,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        value_layer_sizes=config.value_layer_sizes,
    )

    # create the policy function that takes in observation and spits out action
    # policy = make_policy_fn(ppo_network)

    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        entropy_cost=config.entropy_cost,
        kl_weight=config.kl_weight,
        discounting=config.discounting,
        reward_scaling=config.reward_scaling,
        gae_lambda=config.gae_lambda,
        clipping_epsilon=config.clipping_epsilon,
        normalize_advantage=config.normalize_advantage,
    )

    # define policy and value optimizer
    optimizer = nnx.Optimizer(ppo_network, optax.adam(config.learning_rate))

    # define the metrics
    metrics = nnx.MultiMetric(
        total_loss=nnx.metrics.Average(),
    )
    
    num_minibatches = config.num_minibatches
    
    def gradient_update_fn(model, data, loss_fn, optimizer):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (total_loss, details), grads = grad_fn(model, data)
        optimizer.update(grads)
        return total_loss
    
    def minibatch_step(
        data: types.Transition,
    ):
        total_loss = gradient_update_fn(ppo_network, data, loss_fn, optimizer)

        return total_loss

    def sgd_step(
        carry,
        data: types.Transition,
    ):
        key = carry
        key, key_perm = jax.random.split(key, 2)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        total_loss = jax.lax.scan(
            minibatch_step,
            shuffled_data,
            length=num_minibatches,
        )
        return total_loss, key

    def training_step(
        carry: Tuple[envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[envs.State, PRNGKey], Metrics]:
        state, key = carry
        key_sgd, key_generate_unroll = jax.random.split(key, 3)

        policy = make_policy_fn(ppo_network)
        unroll_length = config.unroll_length
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
            length=config.batch_size * num_minibatches // config.num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)


        total_loss, key = jax.lax.scan(
            functools.partial(sgd_step, data=data),
            key_sgd,
            length=config.num_updates_per_batch,
        )

        return total_loss

    def training_epoch(
        state: envs.State, key: PRNGKey
    ) -> Tuple[envs.State, Metrics]:
        (state, _), loss_metrics = jax.lax.scan(
            training_step,
            (state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # # Note that this is NOT a pure jittable method.
    # def training_epoch_with_timing(
    #     training_state: TrainingState, env_state: envs.State, key: PRNGKey
    # ) -> Tuple[TrainingState, envs.State, Metrics]:
    #     nonlocal training_walltime
    #     t = time.time()
    #     training_state, env_state = _strip_weak_type((training_state, env_state))
    #     result = training_epoch(training_state, env_state, key)
    #     training_state, env_state, metrics = _strip_weak_type(result)

    #     metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    #     jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    #     epoch_training_time = time.time() - t
    #     training_walltime += epoch_training_time
    #     sps = (
    #         num_training_steps_per_epoch
    #         * env_step_per_training_step
    #         * max(num_resets_per_eval, 1)
    #     ) / epoch_training_time
    #     metrics = {
    #         "training/sps": sps,
    #         "training/walltime": training_walltime,
    #         **{f"training/{name}": value for name, value in metrics.items()},
    #     }
    #     return (
    #         training_state,
    #         env_state,
    #         metrics,
    #     )  # pytype: disable=bad-return-type  # py311-upgrade

    # init_params = ppo_losses.PPONetworkParams(
    #     policy=ppo_network.policy_network.init(key_policy),
    #     value=ppo_network.value_network.init(key_value),
    # )
    # training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
    #     optimizer_state=optimizer.init(
    #         init_params
    #     ),  # pytype: disable=wrong-arg-types  # numpy-scalars
    #     params=init_params,
    #     normalizer_params=running_statistics.init_state(
    #         specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
    #     ),
    #     env_steps=0,
    # )
    # training_state = jax.device_put_replicated(
    #     training_state, jax.local_devices()[:local_devices_to_use]
    # )

    # if not eval_env:
    #     eval_env = environment
    # if randomization_fn is not None:
    #     v_randomization_fn = functools.partial(
    #         randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
    #     )
    # eval_env = wrap_for_training(
    #     eval_env,
    #     episode_length=episode_length,
    #     action_repeat=action_repeat,
    #     randomization_fn=v_randomization_fn,
    # )

    # evaluator = acting.Evaluator(
    #     eval_env,
    #     functools.partial(make_policy, deterministic=deterministic_eval),
    #     num_eval_envs=num_eval_envs,
    #     episode_length=episode_length,
    #     action_repeat=action_repeat,
    #     key=eval_key,
    # )

    # # Run initial eval
    # metrics = {}
    # if process_id == 0 and num_evals > 1:
    #     metrics = evaluator.run_evaluation(
    #         _unpmap((training_state.normalizer_params, training_state.params.policy)),
    #         training_metrics={},
    #     )
    #     logging.info(metrics)
    #     progress_fn(0, metrics)

    # training_metrics = {}
    # training_walltime = 0
    # current_step = 0
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(100):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            result = training_epoch(env_state, key)
            print(result)