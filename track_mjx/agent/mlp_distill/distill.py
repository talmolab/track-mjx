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

"""
Distillation training for imitation learning.

Trains a student network to mimic a pretrained teacher network using:
1. MSE loss between student and teacher actions
2. Autoregressive loss between consecutive encoder latent means  
3. KL divergence loss between encoder and prior distributions
"""

import functools
import time
from typing import Callable, Optional, Tuple

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey

from track_mjx.agent import gradients
from track_mjx.agent import checkpointing
from track_mjx.agent.mlp_distill import losses, distill_networks
from track_mjx.agent.mlp_ppo import prior_rollout
from track_mjx import utils

from mujoco_playground import wrapper as mp_wrapper

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

Metrics = types.Metrics
STEPS_IN_THOUSANDS = 1e3

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: losses.DistillNetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
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


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    ckpt_mgr: ocp.CheckpointManager,
    config_dict: dict,
    teacher_checkpoint_path: str,
    teacher_checkpoint_step: Optional[int] = None,
    checkpoint_to_restore: Optional[str] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    action_mse_weight: float = 1.0,
    autoregressive_weight: float = 1e-3,
    kl_weight: float = 1e-3,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 20,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    deterministic_eval: bool = False,
    network_factory: Callable[..., distill_networks.DistillNetworks] = distill_networks.make_student_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    eval_env_test_set: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    use_schedule: bool = True,
    schedule_params: Optional[dict] = None,
    checkpoint_callback: Optional[Callable[[int], None]] = None,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False),
    prior_rollout_config: Optional[dict] = None,
):
    """Distillation training.

    Args:
        environment: The environment to train on
        num_timesteps: Total number of environment steps
        episode_length: Length of an episode
        ckpt_mgr: Orbax checkpoint manager
        config_dict: Configuration dictionary for checkpointing
        teacher_checkpoint_path: Path to the pretrained teacher checkpoint
        teacher_checkpoint_step: Optional step to load from teacher checkpoint
        checkpoint_to_restore: Optional path to restore student training from
        action_repeat: Number of times to repeat actions
        num_envs: Number of parallel environments
        max_devices_per_host: Maximum devices per host
        num_eval_envs: Number of evaluation environments
        learning_rate: Learning rate for optimizer
        action_mse_weight: Weight for action MSE loss
        autoregressive_weight: Weight for autoregressive loss
        kl_weight: Weight for KL divergence loss
        seed: Random seed
        unroll_length: Number of timesteps to unroll
        batch_size: Batch size for minibatch SGD
        num_minibatches: Number of minibatches
        num_updates_per_batch: Number of gradient updates per batch
        num_evals: Number of evaluations during training
        num_resets_per_eval: Number of environment resets per eval
        normalize_observations: Whether to normalize observations
        deterministic_eval: Whether to use deterministic policy for eval
        network_factory: Function to create student networks
        progress_fn: Callback for logging progress
        eval_env: Optional separate evaluation environment
        eval_env_test_set: Optional test set evaluation environment
        policy_params_fn: Callback for policy checkpointing
        randomization_fn: Optional domain randomization function
        use_schedule: Whether to use loss weight schedules
        schedule_params: Parameters for loss weight schedules
        checkpoint_callback: Callback after checkpointing
        wrap_for_training: Function to wrap environment for training

    Returns:
        Tuple of (make_policy_fn, final_params, final_metrics)
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

    # Load teacher policy
    logging.info(f"Loading teacher from: {teacher_checkpoint_path}")
    make_teacher_policy, teacher_params, teacher_cfg = distill_networks.create_teacher_inference_fn(
        teacher_checkpoint_path, step=teacher_checkpoint_step
    )
    # Create teacher policy with deterministic=True captured in closure, then jit
    teacher_policy_fn = jax.jit(make_teacher_policy(deterministic=True))
    logging.info("Teacher policy loaded successfully")

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

        # Use student policy for data collection
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
        
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
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
            ),
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
        return training_state, env_state, metrics

    # Initialize keys
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, policy_params_fn_key = jax.random.split(global_key, 2)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    reference_obs_size = int(environment.non_proprioceptive_obs_size)
    proprioceptive_obs_size = int(environment.proprioceptive_obs_size)
    logging.info(f"Reference observation size: {reference_obs_size}")
    logging.info(f"Proprioceptive observation size: {proprioceptive_obs_size}")

    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    # Update config with network info
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

    # Create student network
    student_networks = network_factory(
        env_state.obs.shape[-1],
        reference_obs_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = distill_networks.make_student_inference_fn(student_networks)

    # Setup optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0, eps=1e-5),
    )

    # Setup schedules
    kl_schedule_fn = None
    ar_schedule_fn = None
    if use_schedule and schedule_params is not None:
        if "kl_start_weight" in schedule_params:
            kl_schedule_fn = losses.create_ramp_schedule(
                start_value=schedule_params.get("kl_start_weight", 0.0),
                end_value=schedule_params.get("kl_end_weight", kl_weight),
                total_steps=num_evals,
                start_frac=schedule_params.get("kl_start_ramp", 0.0),
                end_frac=schedule_params.get("kl_end_ramp", 0.5),
                schedule="linear",
            )
        if "ar_start_weight" in schedule_params:
            ar_schedule_fn = losses.create_ramp_schedule(
                start_value=schedule_params.get("ar_start_weight", 0.0),
                end_value=schedule_params.get("ar_end_weight", autoregressive_weight),
                total_steps=num_evals,
                start_frac=schedule_params.get("ar_start_ramp", 0.0),
                end_frac=schedule_params.get("ar_end_ramp", 0.5),
                schedule="linear",
            )

    # Create loss function
    loss_fn = functools.partial(
        losses.compute_distillation_loss,
        student_network=student_networks.student_network,
        teacher_policy_fn=teacher_policy_fn,
        teacher_params=teacher_params,
        action_mse_weight=action_mse_weight,
        autoregressive_weight=autoregressive_weight,
        kl_weight=kl_weight,
        kl_schedule=kl_schedule_fn,
        ar_schedule=ar_schedule_fn,
    )

    # Initialize student parameters
    init_params = losses.DistillNetworkParams(
        policy=student_networks.student_network.init(key_policy),
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(
            specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
        ),
        env_steps=0,
    )

    # Optionally restore from checkpoint
    if checkpoint_to_restore is not None:
        training_state = checkpointing.load_training_state(
            checkpoint_to_restore, training_state
        )
        logging.info(f"Restored checkpoint from {checkpoint_to_restore}")

    # Create gradient update function
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
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

    # Initialize prior rollout evaluator if configured
    prior_rollout_evaluator = None
    if prior_rollout_config is not None and prior_rollout_config.get("enabled", False):
        logging.info("Initializing prior rollout evaluator")
        # Get network config from config_dict
        network_config = config_dict.get("network_config", {})
        prior_layer_sizes = network_config.get(
            "prior_layer_sizes",
            network_config.get("encoder_layer_sizes", [1024, 1024])
        )
        decoder_layer_sizes = network_config.get("decoder_layer_sizes", [1024, 1024])
        intention_size = network_config.get("intention_size", 60)
        
        prior_rollout_evaluator = prior_rollout.PriorRolloutEvaluator(
            env=environment,  # Use unwrapped environment for prior rollouts
            intention_latent_size=intention_size,
            action_size=env.action_size,
            proprioceptive_obs_size=proprioceptive_obs_size,
            decoder_hidden_layer_sizes=tuple(decoder_layer_sizes),
            prior_hidden_layer_sizes=tuple(prior_layer_sizes),
            preprocess_observations_fn=normalize,
            num_rollouts=prior_rollout_config.get("num_rollouts", 32),
            max_steps=prior_rollout_config.get("max_steps", 200),
            healthy_z_range=tuple(prior_rollout_config.get("healthy_z_range", (0.0325, 0.5))),
            fixed_logvar=prior_rollout_config.get("fixed_logvar", -2.0),
            deterministic=prior_rollout_config.get("deterministic", False),
            eval_interval=prior_rollout_config.get("eval_interval", 1),
        )
        logging.info(f"Prior rollout evaluator initialized with {prior_rollout_config.get('num_rollouts', 32)} rollouts")

    # Get starting iteration from checkpoint
    start_it = 0
    if ckpt_mgr is not None and ckpt_mgr.latest_step() is not None:
        num_evals_after_init -= ckpt_mgr.latest_step()
        start_it = ckpt_mgr.latest_step()

    logging.info(f"Starting at iteration: {start_it} with {num_evals_after_init} evals left")

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
                policy_param, training_metrics=metrics, data_split="test_set"
            )

        # Run prior rollout evaluation during initial eval
        if prior_rollout_evaluator is not None:
            prior_metrics = prior_rollout_evaluator.run_evaluation(
                policy_params=policy_param,
                eval_step=0,
            )
            if prior_metrics is not None:
                metrics.update(prior_metrics)
                logging.info(f"Initial prior rollout metrics: {prior_metrics}")

        logging.info(metrics)
        progress_fn(start_it, metrics)
        
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
    training_walltime = 0
    start_it += 1
    current_step = 0
    
    for it in range(start_it, num_evals_after_init + start_it):
        logging.info("starting iteration %s %s", it, time.time() - xt)
        
        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys, it
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id == 0:
            policy_param = _unpmap(
                (training_state.normalizer_params, training_state.params.policy)
            )
            
            metrics = evaluator.run_evaluation(policy_param, training_metrics)
            if evaluator_test_set is not None:
                metrics = evaluator_test_set.run_evaluation(
                    policy_param, metrics, data_split="test_set"
                )

            # Run prior rollout evaluation if configured
            if prior_rollout_evaluator is not None:
                prior_metrics = prior_rollout_evaluator.run_evaluation(
                    policy_params=policy_param,
                    eval_step=it,
                )
                if prior_metrics is not None:
                    metrics.update(prior_metrics)
                    logging.info(f"Prior rollout metrics: {prior_metrics}")

            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            if it % config_dict["render_config"]["render_interval"] == 0:
                # Render video every `render_interval` iterations.
                policy_params_fn(
                    teacher_policy_fn=teacher_policy_fn,
                    teacher_params=teacher_params,
                    current_step=it,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    render_video=True,
                )
            else:
                policy_params_fn(
                    teacher_policy_fn=teacher_policy_fn,
                    teacher_params=teacher_params,
                    current_step=it,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    render_video=False,
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

    total_steps = current_step
    pmap.assert_is_replicated(training_state)
    params = _unpmap((training_state.normalizer_params, training_state.params.policy))
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
