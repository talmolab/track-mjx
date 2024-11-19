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
    policy = make_policy_fn(ppo_network)

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

    @nnx.jit
    def train_step(
        ppo_network: nnx_ppo_network.PPOImitationNetworks, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
    ):
        """Train for a single step."""
        
        acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
        )
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        # Compute loss and gradients. Loss weights are provided in
        # the previous partial function.
        (total_loss, loss_components), grads = grad_fn(ppo_network=ppo_network, data=batch)  
        # metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
        optimizer.update(grads)  # In-place updates.

    @nnx.jit
    def eval_step(model: nnx_ppo_network.PPOImitationNetworks, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(ppo_network=ppo_network, data=batch)
        metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
