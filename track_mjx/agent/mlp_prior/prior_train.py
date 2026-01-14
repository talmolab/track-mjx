"""
Prior network training.

Trains a prior network to match the encoder distributions from a pretrained
mlp_ppo checkpoint. The encoder and decoder remain frozen; only the prior
is trained using KL divergence loss.

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

import functools
import time
from typing import Callable, Dict, Optional, Tuple

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import PRNGKey

from track_mjx.agent import gradients
from track_mjx.agent import checkpointing
from track_mjx.agent.mlp_prior import losses
from track_mjx.agent.mlp_prior import prior_networks
from track_mjx.agent.mlp_prior import prior_rollout_eval
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    get_obs_sizes,
    init_dict_normalizer,
    update_dict_normalizer,
)

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
class PriorTrainingParams:
    """Contains trainable parameters for prior training."""

    prior: types.Params  # Only prior parameters are trainable


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the prior learner."""

    optimizer_state: optax.OptState
    params: PriorTrainingParams  # Only prior params (trainable)
    frozen_encoder_params: Dict  # Frozen - not in optimizer
    frozen_decoder_params: Dict  # Frozen - not in optimizer
    normalizer_params: DictRunningStatisticsState  # Dict-based normalizer
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    ckpt_mgr: ocp.CheckpointManager,
    config_dict: dict,
    mlp_ppo_checkpoint_path: str,
    mlp_ppo_checkpoint_step: Optional[int] = None,
    checkpoint_to_restore: Optional[str] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    kl_weight: float = 1e-3,
    grad_clip_norm: float = 0.5,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 20,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    deterministic_eval: bool = False,
    prior_hidden_layer_sizes: Tuple[int, ...] = (1024, 1024),
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    eval_env_test_set: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    use_kl_schedule: bool = True,
    kl_schedule_params: Optional[dict] = None,
    checkpoint_callback: Optional[Callable[[int], None]] = None,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
    prior_rollout_config: Optional[dict] = None,
):
    """Prior network training.

    Args:
        environment: The environment to train on
        num_timesteps: Total number of environment steps
        episode_length: Length of an episode
        ckpt_mgr: Orbax checkpoint manager
        config_dict: Configuration dictionary for checkpointing
        mlp_ppo_checkpoint_path: Path to the pretrained mlp_ppo checkpoint
        mlp_ppo_checkpoint_step: Optional step to load from mlp_ppo checkpoint
        checkpoint_to_restore: Optional path to restore prior training from
        action_repeat: Number of times to repeat actions
        num_envs: Number of parallel environments
        max_devices_per_host: Maximum devices per host
        num_eval_envs: Number of evaluation environments
        learning_rate: Learning rate for optimizer
        kl_weight: Weight for KL divergence loss
        grad_clip_norm: Gradient clipping norm
        seed: Random seed
        use_pmap_on_reset: Whether to pmap instead of vmap the env reset function
        unroll_length: Number of timesteps to unroll
        batch_size: Batch size for minibatch SGD
        num_minibatches: Number of minibatches
        num_updates_per_batch: Number of gradient updates per batch
        num_evals: Number of evaluations during training
        num_resets_per_eval: Number of environment resets per eval
        normalize_observations: Whether to normalize observations
        deterministic_eval: Whether to use deterministic policy for eval
        prior_hidden_layer_sizes: Hidden layer sizes for prior network
        progress_fn: Callback for logging progress
        eval_env: Optional separate evaluation environment
        eval_env_test_set: Optional test set evaluation environment
        policy_params_fn: Callback for policy checkpointing
        randomization_fn: Optional domain randomization function
        use_kl_schedule: Whether to use KL weight schedule
        kl_schedule_params: Parameters for KL weight schedule
        checkpoint_callback: Callback after checkpointing
        wrap_for_training: Function to wrap environment for training
        prior_rollout_config: Configuration for prior rollout evaluation

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

    # Load frozen encoder and decoder from mlp_ppo checkpoint
    logging.info(f"Loading encoder/decoder from: {mlp_ppo_checkpoint_path}")
    encoder_params, decoder_params, teacher_normalizer_params, teacher_cfg = (
        prior_networks.load_frozen_encoder_decoder(
            mlp_ppo_checkpoint_path, step=mlp_ppo_checkpoint_step
        )
    )
    logging.info("Encoder and decoder loaded successfully")

    # Extract network config from teacher
    latent_size = teacher_cfg["network_config"]["intention_size"]
    encoder_hidden_layer_sizes = tuple(
        teacher_cfg["network_config"]["encoder_layer_sizes"]
    )
    decoder_hidden_layer_sizes = tuple(
        teacher_cfg["network_config"]["decoder_layer_sizes"]
    )
    action_size = teacher_cfg["network_config"]["action_size"]

    # Get observation sizes - support both new dict format and legacy flat format
    teacher_obs_sizes = teacher_cfg["network_config"].get("obs_sizes", None)
    if teacher_obs_sizes is not None:
        reference_obs_size = teacher_obs_sizes["imitation_target"]
        teacher_proprioceptive_obs_size = teacher_obs_sizes["proprioception"]
    else:
        reference_obs_size = teacher_cfg["network_config"]["reference_obs_size"]
        teacher_observation_size = teacher_cfg["network_config"]["observation_size"]
        teacher_proprioceptive_obs_size = teacher_observation_size - reference_obs_size

    # Initialize keys
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_prior, policy_params_fn_key = jax.random.split(global_key, 3)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    env_proprioceptive_obs_size = int(environment.proprioceptive_obs_size)
    logging.info(f"Reference observation size: {reference_obs_size}")
    logging.info(
        f"Environment proprioceptive observation size: {env_proprioceptive_obs_size}"
    )
    logging.info(
        f"Teacher proprioceptive observation size: {teacher_proprioceptive_obs_size}"
    )
    if env_proprioceptive_obs_size != teacher_proprioceptive_obs_size:
        logging.warning(
            f"Environment proprioceptive_obs_size ({env_proprioceptive_obs_size}) "
            f"differs from teacher ({teacher_proprioceptive_obs_size}). "
            f"Using teacher value for decoder compatibility."
        )
    proprioceptive_obs_size = teacher_proprioceptive_obs_size

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

    # Get observation sizes from environment state
    obs_sizes = get_obs_sizes(env_state.obs)

    # Update config with network info
    config_dict["network_config"].update(
        {
            "obs_sizes": obs_sizes,
            "action_size": action_size,
            "normalize_observations": normalize_observations,
            "intention_size": latent_size,
            "encoder_layer_sizes": list(encoder_hidden_layer_sizes),
            "decoder_layer_sizes": list(decoder_hidden_layer_sizes),
            "prior_layer_sizes": list(prior_hidden_layer_sizes),
        }
    )

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    # Create prior network
    _, prior_network = prior_networks.make_prior_networks(
        latent_size=latent_size,
        proprioceptive_obs_size=proprioceptive_obs_size,
        prior_hidden_layer_sizes=prior_hidden_layer_sizes,
    )

    # Create encoder apply function
    encoder_apply_fn = prior_networks.make_encoder_apply_fn(
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        latent_size=latent_size,
        reference_obs_size=reference_obs_size,
    )

    # Create frozen encoder+decoder policy for data collection
    make_data_collection_policy = functools.partial(
        prior_networks.make_encoder_decoder_inference_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        latent_size=latent_size,
        action_size=action_size,
        deterministic=True,
    )

    # Setup optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0, eps=1e-5),
    )

    # Setup KL schedule
    kl_schedule_fn = None
    if use_kl_schedule and kl_schedule_params is not None:
        kl_schedule_fn = losses.create_ramp_schedule(
            start_value=kl_schedule_params.get("kl_start_weight", 1e-5),
            end_value=kl_schedule_params.get("kl_end_weight", kl_weight),
            total_steps=num_evals,
            start_frac=kl_schedule_params.get("kl_start_ramp", 0.0),
            end_frac=kl_schedule_params.get("kl_end_ramp", 0.25),
            schedule="linear",
        )

    # Create loss function
    loss_fn = functools.partial(
        losses.compute_prior_training_loss,
        encoder_apply_fn=encoder_apply_fn,
        prior_apply_fn=prior_network.apply,
        reference_obs_size=reference_obs_size,
        kl_weight=kl_weight,
        kl_schedule=kl_schedule_fn,
    )

    # Initialize prior parameters
    init_prior_params = prior_network.init(key_prior)

    training_state = TrainingState(
        optimizer_state=optimizer.init(PriorTrainingParams(prior=init_prior_params)),
        params=PriorTrainingParams(prior=init_prior_params),
        frozen_encoder_params=encoder_params,
        frozen_decoder_params=decoder_params,
        normalizer_params=teacher_normalizer_params,  # Use teacher's normalizer
        env_steps=0,
    )

    # Optionally restore from checkpoint
    if checkpoint_to_restore is not None:
        logging.info(f"Restoring checkpoint from {checkpoint_to_restore}")
        # Load the saved training state
        # Note: This needs to match the saved structure
        training_state = checkpointing.load_training_state(
            checkpoint_to_restore, training_state, step_prefix="PriorNetwork"
        )
        logging.info(f"Restored checkpoint from {checkpoint_to_restore}")

    # Define minibatch step
    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: DictRunningStatisticsState,
        frozen_encoder_params: Dict,
    ):
        optimizer_state, params, key, it = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            frozen_encoder_params,
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
        frozen_encoder_params: Dict,
    ):
        optimizer_state, params, key, it = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step,
                normalizer_params=normalizer_params,
                frozen_encoder_params=frozen_encoder_params,
            ),
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

        # Use frozen encoder+decoder policy for data collection
        policy = make_data_collection_policy(
            encoder_params=training_state.frozen_encoder_params,
            decoder_params=training_state.frozen_decoder_params,
            normalizer_params=training_state.normalizer_params,
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

        # Update normalization params (dict-based)
        normalizer_params = update_dict_normalizer(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=data,
                normalizer_params=normalizer_params,
                frozen_encoder_params=training_state.frozen_encoder_params,
            ),
            (training_state.optimizer_state, training_state.params, key_sgd, it),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            frozen_encoder_params=training_state.frozen_encoder_params,
            frozen_decoder_params=training_state.frozen_decoder_params,
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

    # Create gradient update function
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(
            0,
            1,
        ),
    )

    training_walltime = 0

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

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    # Setup evaluation - use encoder+decoder policy for eval
    if eval_env is None:
        eval_env = environment
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )

    # Create evaluation policy factory
    def make_eval_policy(params_tuple, deterministic=True):
        normalizer_params, policy_params = params_tuple
        encoder_params = policy_params["params"]["encoder"]
        decoder_params = policy_params["params"]["decoder"]
        return prior_networks.make_encoder_decoder_inference_fn(
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            normalizer_params=normalizer_params,
            encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
            decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
            latent_size=latent_size,
            action_size=action_size,
            deterministic=deterministic,
        )

    # Initialize multi-mode prior rollout evaluator if configured
    multi_mode_evaluator = None
    if prior_rollout_config is not None and prior_rollout_config.get("enabled", False):
        logging.info("Initializing multi-mode prior rollout evaluator")
        render_config = config_dict.get("render_config", {})

        multi_mode_evaluator = prior_rollout_eval.MultiModePriorRolloutEvaluator(
            env=environment,
            intention_latent_size=latent_size,
            action_size=action_size,
            proprioceptive_obs_size=proprioceptive_obs_size,
            decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
            prior_hidden_layer_sizes=prior_hidden_layer_sizes,
            preprocess_observations_fn=normalize,
            max_steps=prior_rollout_config.get("max_steps", 200),
            eval_interval=prior_rollout_config.get("eval_interval", 1),
            render_fps=render_config.get("render_fps", 50),
            render_camera_name=render_config.get("render_camera_name", "close_profile"),
            model_path=str(ckpt_mgr.directory),
        )
        logging.info("Multi-mode prior rollout evaluator initialized")

    # Helper function to create combined checkpoint params
    def get_policy_params(training_state):
        """Create combined policy params for checkpointing."""
        ts = _unpmap(training_state)
        return prior_networks.create_combined_checkpoint_params(
            encoder_params=ts.frozen_encoder_params,
            decoder_params=ts.frozen_decoder_params,
            prior_params=ts.params.prior,
            normalizer_params=ts.normalizer_params,
        )

    # Get starting iteration from checkpoint
    start_it = 0
    if ckpt_mgr is not None and ckpt_mgr.latest_step() is not None:
        num_evals_after_init -= ckpt_mgr.latest_step()
        start_it = ckpt_mgr.latest_step()

    logging.info(
        f"Starting at iteration: {start_it} with {num_evals_after_init} evals left"
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1 and start_it == 0:
        logging.info("Running initial evaluation")
        policy_param = get_policy_params(training_state)

        # Generate shared reset key for all rollouts (prior + encoder_decoder)
        policy_params_fn_key, reset_key = jax.random.split(policy_params_fn_key)

        # Run multi-mode prior rollout evaluation during initial eval
        if multi_mode_evaluator is not None:
            prior_metrics = multi_mode_evaluator.run_evaluation(
                policy_params=policy_param,
                eval_step=0,
                reset_key=reset_key,
            )
            if prior_metrics is not None:
                metrics.update(prior_metrics)
                logging.info(f"Initial prior rollout metrics: {prior_metrics}")

        # Render initial video to wandb (using same reset_key for same clip)
        _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
        policy_params_fn(
            current_step=0,
            params=policy_param,
            policy_params_fn_key=policy_params_fn_key,
            reset_key=reset_key,
            render_video=True,
        )

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
            if num_resets_per_eval > 0:
                env_state = reset_fn((training_state, env_state), key_envs)

        if process_id == 0:
            policy_param = get_policy_params(training_state)

            metrics = training_metrics.copy()

            # Generate shared reset key for all rollouts (prior + encoder_decoder)
            policy_params_fn_key, reset_key = jax.random.split(policy_params_fn_key)

            # Run multi-mode prior rollout evaluation if configured
            if multi_mode_evaluator is not None:
                prior_metrics = multi_mode_evaluator.run_evaluation(
                    policy_params=policy_param,
                    eval_step=it,
                    reset_key=reset_key,
                )
                if prior_metrics is not None:
                    metrics.update(prior_metrics)
                    logging.info(f"Prior rollout metrics: {prior_metrics}")

            # Encoder-decoder logging (using same reset_key for same clip)
            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            render_config = config_dict.get("render_config", {})
            render_interval = render_config.get("render_interval", 1)
            if it % render_interval == 0:
                policy_params_fn(
                    current_step=it,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    reset_key=reset_key,
                    render_video=True,
                )
            else:
                policy_params_fn(
                    current_step=it,
                    params=policy_param,
                    policy_params_fn_key=policy_params_fn_key,
                    reset_key=reset_key,
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
    params = get_policy_params(training_state)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_eval_policy, params, metrics)
