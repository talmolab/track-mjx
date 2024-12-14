"""
PPO implementation using flax new API, nnx.

Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

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
# General imports.
import functools
import time
from typing import Callable, Optional, Tuple, Union, Any

# brax related imports.
from absl import logging
from brax import base
from brax import envs
from track_mjx.brax_nnx import gradients
from track_mjx.brax_nnx import acting
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1

# flax & JAX & PPO related imports.
import jax
import jax.numpy as jnp
import numpy as np
import flax
from track_mjx.agent import nnx_ppo_losses as ppo_losses
from track_mjx.agent.nnx_ppo_network import (
    PPOImitationNetworks,
    PPOTrainConfig,
)
from flax import nnx
from flax.training import train_state
import optax
from track_mjx.environment import custom_wrappers
from orbax import checkpoint as ocp


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def train(
    environment: envs.Env,
    config: PPOTrainConfig,
):
    """PPO training.

    Args:
      environment: the environment to train
      config: PPOTrainConfig file that specify the hyperparameters for the PPO
    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    # TODO (Scott) Move this multi-device support to its own util file.

    # parameters validation
    assert config.batch_size % config.num_envs == 0, "Batch size must be divisible by num_envs"
    assert config.batch_size % config.num_minibatches == 0, "Batch size must be divisible by num_minibatches"
    
    logging.info(
        "Batch size: %d, Minibatch size: %d, number of mimibatch: %d",
        config.batch_size,
        config.batch_size // config.num_minibatches,
        config.num_minibatches,
    )

    xt = time.time()

    # Get the number of devices and set up multiple devices supports
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
        config.batch_size * config.unroll_length * config.action_repeat # take out the minibatch * config.num_minibatches
    ) # Scott: This metrics for the environment performance is not true -- minibatch and permutation is bootstrapping
    # data instead of directly
    num_evals_after_init = max(config.num_evals - 1, 1)

    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        config.num_timesteps / (num_evals_after_init * env_step_per_training_step * max(config.num_resets_per_eval, 1))
    ).astype(int)

    # initialize random keys for the MJX environment
    key = jax.random.PRNGKey(config.seed)
    global_key, local_key = jax.random.split(key)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    assert config.num_envs % device_count == 0

    v_randomization_fn = None

    if isinstance(environment, envs.Env):
        wrap_for_training = custom_wrappers.wrap
    else:
        raise ValueError(" Brax V1 envs are deprecated, please use envs.Env")

    env = wrap_for_training(
        environment,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
        randomization_fn=v_randomization_fn,
    )

    # create parallelized environment with dimensions from random number keys
    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, config.num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)
    # need to remove batch dim if not doing pmap
    env_state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), env_state)

    # define PPO network, now it is a nnx module.
    ppo_network = config.network_factory(
        env_state.obs.shape[-1],
        int(_unpmap(env_state.info["reference_obs_size"])),
        env.action_size,
        normalize_obs=config.normalize_observations,
    )

    # define optimizer
    optimizer = nnx.Optimizer(ppo_network, optax.adam(learning_rate=config.learning_rate))

    # create loss function
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

    @nnx.jit
    def generate_experience(
        ppo_network: PPOImitationNetworks,
        state: Union[envs.State, envs_v1.State],
        key: PRNGKey,
    ) -> Tuple[types.Transition, Union[envs.State, envs_v1.State], PRNGKey]:
        """Generate experience data in MuJoCo for training. This is already a jitted function.
        because the jax.lax.scan automatically jits the function.

        Args:
            ppo_network (PPOImitationNetworks): ppo network
            config (PPOTrainConfig): training configuration
            state (Union[envs.State, envs_v1.State]): environment state
            key (PRNGKey): random key

        Returns:
            Tuple[types.Transition, Union[envs.State, envs_v1.State], PRNGKey]: experience data, new state, new key
        """
        key_generate_unroll, new_key = jax.random.split(key, 2)
        policy = ppo_network.policy
        def f(carry, unused_t):
            """
            generate unroll data for training using scan function
            """
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                config.unroll_length,
                extra_fields=("truncation",),
            )
            return (next_state, next_key), data
        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=config.batch_size // config.num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        # TODO: Hardcoded for now for minibatch steps
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, 256) + x.shape[2:]), data)
        # assert data.discount.shape[1:] == (config.unroll_length,)
        print("DEBUG: data shape:", data.observation.shape)
        # update normalization params
        ppo_network.obs_normalizer.update(data.observation)

        return data, state, new_key

    @nnx.jit
    # @nnx.vmap(in_axes=(None, None, 0))
    def gradient_update(
        model: PPOImitationNetworks, optimizer: nnx.Optimizer, data: types.Transition
    ) -> Tuple[Metrics, Metrics]:
        """Update the model using the optimizer and the data, in place

        Args:
            model (PPOImitationNetworks): ppo network
            optimizer (nnx.Optimizer): optimizer
            data (types.Transition): experience data

        Returns:
            _type_: _description_
        """
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (total_loss, metrics), grads = grad_fn(model, data, key)
        optimizer.update(grads)
        return total_loss, metrics

    # Note that this is NOT a pure jittable method.
    def training_step_with_timing(
        model: PPOImitationNetworks,
        config: PPOTrainConfig,
        optimizer: nnx.Optimizer,
        env_states: Union[envs.State, envs_v1.State],
        key: PRNGKey,
    ) -> Tuple[Metrics, Metrics, Union[envs.State, envs_v1.State], PRNGKey]:
        """
        Training step with timing information, separately for experience generation and gradient steps.

        Args:
            model (PPOImitationNetworks): _description_
            config (PPOTrainConfig): _description_
            optimizer (nnx.Optimizer): _description_
            env_states (Union[envs.State, envs_v1.State]): _description_
            key (PRNGKey): _description_

        Returns:
            Tuple[Metrics, Metrics, Union[envs.State, envs_v1.State], PRNGKey]: _description_
        """
        # TODO: normalization later
        # Update normalization params and normalize observations.

        # generate experience data
        mj_time = time.time()
        data, new_env_state, new_key = generate_experience(model, env_states, key)
        mj_time = time.time() - mj_time

        mj_sps = config.batch_size * config.unroll_length / mj_time

        # gradient updates steps
        gradient_time = time.time()
        # Initialize lists to collect total_losses and metrics
        total_losses = []
        metrics_list = []
        # Loop over minibatches
        for i in range(data.observation.shape[0]):
            minibatch = jax.tree_util.tree_map(lambda x: x[i], data)
            total_loss, metrics = gradient_update(model, optimizer, minibatch)
            total_losses.append(total_loss)
            metrics_list.append(metrics)
        # Compute the mean of total_losses and metrics
        total_loss = jnp.mean(jnp.stack(total_losses))
        metrics = {key: jnp.mean(jnp.stack([m[key] for m in metrics_list])) for key in metrics_list[0]}
        gradient_time = time.time() - gradient_time

        metrics = {
            "training/mujoco_sps": mj_sps,
            "training/gradient_sps": config.num_minibatches / gradient_time, # gradient steps per minibatch
            "training/walltime": training_walltime,
            "training/mujoco_step": mujoco_step,
            "training/gradient_step": gradient_step,
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        return total_loss, metrics, new_env_state, new_key

    def training_epoch_with_timing(
        model: PPOImitationNetworks,
        optimizer: nnx.Optimizer,
        config: PPOTrainConfig,
        env_state: Union[envs.State, envs_v1.State],
        metrics: Metrics,
        key: PRNGKey,
    ) -> Tuple[nnx.Module, Union[envs.State, envs_v1.State], Metrics, PRNGKey]:
        nonlocal training_walltime, mujoco_step, gradient_step, it
        t = time.time()
        num_training_steps_per_epoch = 100  # hardcoded for now
        for _ in range(num_training_steps_per_epoch):
            mujoco_step += env_step_per_training_step
            gradient_step += config.num_minibatches
            total_loss, metrics, env_state, key = training_step_with_timing(model, config, optimizer, env_state, key)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            print("METRICS:", metrics)
            config.progress_fn(num_training_steps_per_epoch * it + _, metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        return (model, env_state, metrics, key)  # pytype: disable=bad-return-type  # py311-upgrade

    if not config.eval_env:
        eval_env = environment
    else:
        eval_env = config.eval_env
    if config.randomization_fn is not None:
        v_randomization_fn = functools.partial(
            config.randomization_fn, rng=jax.random.split(eval_key, config.num_eval_envs)
        )
    eval_env = wrap_for_training(
        eval_env,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
        randomization_fn=v_randomization_fn,
    )

    evaluator = acting.Evaluator(
        eval_env,
        num_eval_envs=config.num_eval_envs,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and config.num_evals > 1:
        metrics = evaluator.run_evaluation(
            ppo_network=ppo_network,
            training_metrics={},
        )
        logging.info(metrics)
        config.progress_fn(0, metrics)
        

    ckpt_dir = ocp.test_utils.erase_and_create_empty(config.checkpoint_logdir)

    training_metrics = {}
    training_walltime = 0
    mujoco_step = 0
    gradient_step = 0
    current_step = 0
    num_steps = int(1e5)
    for it in range(num_steps):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        # the following two loops control the ratios between the training and evaluation steps

        for _ in range(max(config.num_resets_per_eval, 1)):

            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            total_loss, new_env_state, metrics, new_key = training_epoch_with_timing(
                ppo_network, optimizer, config, env_state, training_metrics, epoch_key
            )
            current_step = it
            config.progress_fn(current_step, metrics)
            key_envs = jax.vmap(lambda x, s: jax.random.split(x[0], s), in_axes=(0, None))(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs)
            env_state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), env_state)

        if process_id == 0:
            # only run the evaluation on the master process
            metrics = evaluator.run_evaluation(
                ppo_network=ppo_network,
                training_metrics=training_metrics,
            )
            logging.info(metrics)
            config.progress_fn(current_step, metrics)

            # checkpointing - previously using the policy param function, now using orbax
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(ckpt_dir / f"ppo_networks_{it}", nnx.split(ppo_network)[1])

    total_steps = current_step
    assert total_steps >= config.num_timesteps

    return None
