"""Vision-augmented PPO training with interleaved mujoco_warp rendering.

Workaround 1: Replaces brax's ``acting.generate_unroll`` (which uses
``jax.lax.scan``) with a Python-level for-loop that interleaves GPU-based
rendering via mujoco_warp after each physics step.  This is necessary because
the mujoco_warp renderer operates outside the JAX computation graph and cannot
be called from inside ``lax.scan``.

The key function is ``generate_unroll_with_vision``, which follows the same
contract as ``brax.training.acting.generate_unroll`` -- it returns
``(final_state, data)`` where ``data`` is a ``types.Transition`` with leaves
shaped ``(unroll_length, num_envs, ...)``.

The ``train()`` function mirrors ``ppo.train()`` but replaces the
triple-nested ``jax.lax.scan`` with Python loops so that
``generate_unroll_with_vision`` (which calls the mujoco_warp renderer)
can be interleaved at every physics step.  The minibatch SGD inner loop
remains as ``jax.lax.scan`` inside a ``@jax.jit`` function since it does
not need the renderer.
"""

import functools
import time
from typing import Any, Callable, Sequence, Tuple

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
from optax.transforms import freeze

from track_mjx.agent import checkpointing, gradients, network_masks
from track_mjx.agent.ff_ppo import losses, ppo_networks
from track_mjx.agent.ff_ppo.ppo import TrainingState, _strip_weak_type
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    get_obs_sizes,
    init_dict_normalizer,
    update_dict_normalizer,
)

# ---------------------------------------------------------------------------
# Type aliases (mirrors ppo.py)
# ---------------------------------------------------------------------------
InferenceParams = tuple[DictRunningStatisticsState, Params]
Metrics = types.Metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEPS_IN_THOUSANDS = 1e3
_PMAP_AXIS_NAME = "i"


# ---------------------------------------------------------------------------
# Helper: RGB to grayscale
# ---------------------------------------------------------------------------

def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to single-channel grayscale in [0, 1].

    Uses the standard luminance weights (ITU-R BT.601):
        Y = 0.2989 * R + 0.5870 * G + 0.1140 * B

    Args:
        rgb: NumPy array of shape ``(N, H, W, 3)`` with dtype ``uint8``.

    Returns:
        NumPy array of shape ``(N, H, W, 1)`` with dtype ``float32``,
        values in ``[0, 1]``.
    """
    # Normalize to [0, 1] first, then apply luminance weights
    rgb_float = rgb.astype(np.float32) / 255.0
    gray = (
        0.2989 * rgb_float[..., 0:1]
        + 0.5870 * rgb_float[..., 1:2]
        + 0.1140 * rgb_float[..., 2:3]
    )
    return gray


# ---------------------------------------------------------------------------
# Core: generate_unroll_with_vision
# ---------------------------------------------------------------------------

def generate_unroll_with_vision(
    env: envs.Env,
    env_state: envs.State,
    policy: types.Policy,
    key: PRNGKey,
    unroll_length: int,
    renderer: Any,
    grayscale: bool = True,
    extra_fields: Sequence[str] = (),
    step_fn: Callable | None = None,
) -> Tuple[envs.State, types.Transition]:
    """Collect a trajectory of ``unroll_length`` steps with vision rendering.

    This mirrors :func:`brax.training.acting.generate_unroll` but uses a
    Python ``for`` loop instead of ``jax.lax.scan`` so that we can call the
    mujoco_warp renderer (a non-JAX operation) between each physics step.

    At every step the function:
        1. Syncs the physics state to the renderer
           (``renderer.sync_state(env_state.data)``).
        2. Renders an egocentric RGB image
           (``renderer.render()``).
        3. Optionally converts RGB to grayscale.
        4. Injects the image into ``env_state.obs["vision"]`` so the policy
           can condition on it.
        5. Queries the policy for an action.
        6. Advances the environment by one step.
        7. Records the transition.

    Args:
        env: A brax-wrapped environment.
        env_state: Current batched environment state.
        policy: A callable ``(obs, key) -> (action, extras)``.
        key: JAX PRNG key.
        unroll_length: Number of environment steps to collect.
        renderer: A mujoco_warp renderer object that exposes
            ``sync_state(mjx_data)`` and ``render() -> (rgb, depth)``.
        grayscale: If True, convert rendered RGB to single-channel grayscale.
        extra_fields: Additional fields to extract from ``env_state.info``
            (e.g., ``("truncation",)``).
        step_fn: Optional pre-jitted step function.  When provided, this
            is used instead of ``env.step`` to avoid re-tracing the
            ``lax.scan`` inside the brax wrappers on every call.

    Returns:
        A tuple ``(final_state, data)`` where:
            - ``final_state`` is the environment state after ``unroll_length``
              steps.
            - ``data`` is a :class:`~brax.training.types.Transition`
              whose leaf arrays have shape ``(unroll_length, num_envs, ...)``.
    """
    transitions: list[types.Transition] = []

    current_state = env_state
    current_key = key

    for _ in range(unroll_length):
        current_key, step_key = jax.random.split(current_key)

        # --- 1. Render vision from the current physics state ---------------
        renderer.sync_state(current_state.data)
        rgb, _ = renderer.render()  # (num_envs, H, W, 3) uint8

        # --- 2. Process the image ------------------------------------------
        if grayscale:
            vision = rgb_to_grayscale(rgb)  # (num_envs, H, W, 1) float32
        else:
            vision = rgb.astype(np.float32) / 255.0  # (num_envs, H, W, 3) float32

        # --- 3. Inject vision into the observation dict --------------------
        # Convert to JAX array so it can flow into the policy network.
        vision_jax = jnp.asarray(vision)

        # Build a new observation dict with vision replaced.
        obs_with_vision = {**current_state.obs, "vision": vision_jax}

        # --- 4. Query the policy -------------------------------------------
        actions, policy_extras = policy(obs_with_vision, step_key)

        # --- 5. Step the environment ---------------------------------------
        _step = step_fn if step_fn is not None else env.step
        next_state = _step(current_state, actions)

        # --- 6. Record the transition (same format as acting.actor_step) ---
        state_extras = {x: next_state.info[x] for x in extra_fields}
        transition = types.Transition(
            observation=obs_with_vision,
            action=actions,
            reward=next_state.reward,
            discount=1 - next_state.done,
            next_observation=next_state.obs,
            extras={
                "policy_extras": policy_extras,
                "state_extras": state_extras,
            },
        )
        transitions.append(transition)

        current_state = next_state

    # Stack transitions along a new leading time axis: (unroll_length, num_envs, ...)
    data = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *transitions)

    return current_state, data


# ---------------------------------------------------------------------------
# train() — vision-augmented PPO training loop
# ---------------------------------------------------------------------------

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
    ] = ppo_networks.make_vision_ppo_networks,
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
    freeze_decoder: bool = False,
    checkpoint_callback: Callable[[int], None] | None = None,
    grad_clip_threshold: float = 20.0,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
    # Vision-specific parameters
    vision_width: int = 32,
    vision_height: int = 32,
    grayscale: bool = True,
    camera_name: str = "egocentric-rodent",
) -> tuple[Callable, InferenceParams, Metrics]:
    """Train a vision-augmented PPO agent with interleaved rendering.

    Mirrors :func:`ppo.train` but operates on a single GPU without ``pmap``,
    using Python loops for the training epoch so that
    :func:`generate_unroll_with_vision` (which calls the mujoco_warp renderer)
    can be invoked at every physics step.  The minibatch SGD inner loop is
    still JIT-compiled via ``jax.lax.scan``.

    Args:
        environment: The environment to train on.  Must expose ``mj_model``
            (a ``mujoco.MjModel``) for creating the VisionRenderer.
        num_timesteps: Total environment steps budget.
        episode_length: Maximum episode length.
        ckpt_mgr: Orbax checkpoint manager.
        config_dict: Training configuration dictionary (saved with checkpoints).
        checkpoint_to_restore: Optional checkpoint path for resuming training.
        action_repeat: Steps per action repeat.
        num_envs: Number of parallel environments.
        max_devices_per_host: Unused (kept for API compatibility with ppo.train).
        num_eval_envs: Number of evaluation environments.
        learning_rate: Adam learning rate.
        entropy_cost: Entropy bonus coefficient.
        latent_kl_weight: Maximum KL weight for VAE latent.
        latent_ar1_weight: Maximum AR(1) weight for latent regularisation.
        discounting: Discount factor (gamma).
        seed: Random seed.
        use_pmap_on_reset: Unused (kept for API compatibility).
        unroll_length: Steps per unroll.
        batch_size: Minibatch size for SGD.
        num_minibatches: Number of minibatches per SGD sweep.
        num_updates_per_batch: SGD sweeps per training step.
        num_evals: Total evaluation checkpoints (including initial).
        num_resets_per_eval: Env resets between evaluations.
        normalize_observations: Whether to normalise observations.
        reward_scaling: Reward multiplier.
        clipping_epsilon: PPO clip range.
        gae_lambda: GAE lambda.
        deterministic_eval: Whether eval uses deterministic policy.
        network_factory: Factory for PPO networks.
        progress_fn: Callback ``(step, metrics)`` for logging.
        normalize_advantage: Whether to normalise advantages.
        vf_loss_coefficient: Value-function loss coefficient.
        eval_env: Optional separate evaluation environment.
        eval_env_test_set: Optional held-out test-set evaluation environment.
        policy_params_fn: Callback for policy parameter logging/rendering.
        randomization_fn: Domain-randomisation callback.
        get_activation: Whether to capture network activations.
        use_kl_schedule: Whether to ramp KL weight over training.
        kl_ramp_up_frac: Fraction of training for KL ramp-up.
        freeze_decoder: Whether to freeze decoder weights.
        checkpoint_callback: Post-checkpoint callback.
        grad_clip_threshold: Maximum gradient norm.
        wrap_for_training: Environment wrapper factory.
        vision_width: Rendered image width (pixels).
        vision_height: Rendered image height (pixels).
        grayscale: Convert rendered RGB to single-channel grayscale.
        camera_name: Name of the camera in the MJCF model.

    Returns:
        Tuple of ``(make_policy, params, metrics)`` where *params* is
        ``(normalizer_params, policy_params)``.
    """
    # ------------------------------------------------------------------
    # Batch / step accounting (same as ppo.py but no device dimension)
    # ------------------------------------------------------------------
    assert batch_size * num_minibatches % num_envs == 0, (
        f"batch_size * num_minibatches ({batch_size * num_minibatches}) "
        f"must be divisible by num_envs ({num_envs})"
    )
    xt = time.time()

    # Number of unrolls to collect per training step to fill the batch
    num_unrolls_per_step = batch_size * num_minibatches // num_envs

    # Environment steps consumed per training step
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

    logging.info(
        "Vision PPO — num_envs: %d, unroll_length: %d, "
        "num_unrolls_per_step: %d, num_training_steps_per_epoch: %d, "
        "vision: %dx%d (grayscale=%s, camera=%s)",
        num_envs,
        unroll_length,
        num_unrolls_per_step,
        num_training_steps_per_epoch,
        vision_width,
        vision_height,
        grayscale,
        camera_name,
    )

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value, policy_params_fn_key = jax.random.split(global_key, 3)
    del global_key

    # ------------------------------------------------------------------
    # Environment wrapping (single-GPU, vmap-only — no pmap)
    # ------------------------------------------------------------------
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_rng = jax.random.split(key_env, num_envs)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )

    proprioceptive_obs_size = int(environment.proprioceptive_obs_size)
    logging.info("Proprioceptive observation size: %d", proprioceptive_obs_size)

    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    # Reset: shape (num_envs, ...) — no device dimension
    # wrap_for_training already adds VmapWrapper, so env.reset expects (num_envs, 2)
    key_envs = jax.random.split(key_env, num_envs)
    reset_fn = jax.jit(env.reset)
    env_state = reset_fn(key_envs)

    # Warm up env.step JIT to force warp kernel compilation before other
    # GPU allocations (VisionRenderer, render context, etc.) fragment memory.
    step_fn = jax.jit(env.step)
    _warmup_actions = jnp.zeros((num_envs, env.action_size))
    _warmup_state = step_fn(env_state, _warmup_actions)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), _warmup_state.obs)
    logging.info("Warm-up env.step complete")
    # Re-reset to clean state after warmup
    env_state = reset_fn(key_envs)

    # ------------------------------------------------------------------
    # Observation sizes & network config
    # ------------------------------------------------------------------
    obs_sizes = get_obs_sizes(env_state.obs)
    logging.info("Observation sizes: %s", obs_sizes)

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

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_threshold),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0, eps=1e-5),
    )

    # ------------------------------------------------------------------
    # KL schedule (same as ppo.py)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Initialise training state
    # ------------------------------------------------------------------
    init_params = losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=init_dict_normalizer(env_state.obs),
        env_steps=0,
    )

    frozen_proprioceptive_normalizer_params = None

    # ------------------------------------------------------------------
    # Checkpoint restoration (same logic as ppo.py)
    # ------------------------------------------------------------------
    if checkpoint_to_restore is not None:
        if not freeze_decoder:
            training_state = checkpointing.load_training_state(
                checkpoint_to_restore, training_state
            )
            logging.info("Restored latest checkpoint at %s", checkpoint_to_restore)
        if freeze_decoder:
            loaded_checkpoint = checkpointing.load_policy(checkpoint_to_restore)
            loaded_normalizer_params = loaded_checkpoint[0]
            loaded_policy = loaded_checkpoint[1]
            decoder_params = loaded_policy["params"]["decoder"]
            training_state.params.policy["params"]["decoder"] = decoder_params
            logging.info(
                "Restored decoder parameters from checkpoint at %s",
                checkpoint_to_restore,
            )
            mask = network_masks.create_decoder_mask(init_params)
            optimizer = optax.chain(optimizer, freeze(mask))
            training_state = training_state.replace(
                optimizer_state=optimizer.init(init_params)
            )
            logging.info("Freezing decoder parameters")

            if isinstance(loaded_normalizer_params, DictRunningStatisticsState):
                frozen_proprioceptive_normalizer_params = (
                    loaded_normalizer_params.proprioception
                )
            else:
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
                    proprioception=frozen_proprioceptive_normalizer_params
                )
            )

    # ------------------------------------------------------------------
    # Gradient update function — pmap_axis_name=None (single GPU)
    # ------------------------------------------------------------------
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn,
        optimizer,
        pmap_axis_name=None,
        has_aux=True,
        clip_threshold=grad_clip_threshold,
    )

    # NOTE: No jax.device_put_replicated — training_state stays on a single device.

    # ------------------------------------------------------------------
    # Vision renderer
    # ------------------------------------------------------------------
    # VisionRenderer patches mujoco_warp BLEEDING_EDGE_MUJOCO at import time
    # (see vnl_playground/tasks/rodent/vision.py).
    from vnl_playground.tasks.rodent.vision import VisionRenderer

    renderer = VisionRenderer(
        mj_model=environment.mj_model,
        nworld=num_envs,
        camera_name=camera_name,
        width=vision_width,
        height=vision_height,
    )
    logging.info(
        "Created VisionRenderer: %dx%d, camera=%s, nworld=%d",
        vision_width,
        vision_height,
        camera_name,
        num_envs,
    )

    # ------------------------------------------------------------------
    # JIT'd SGD step (minibatch scan + sgd scan)
    # ------------------------------------------------------------------
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
            functools.partial(
                minibatch_step, normalizer_params=normalizer_params
            ),
            (optimizer_state, params, key_grad, it),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key, it), metrics

    @jax.jit
    def jit_sgd(
        optimizer_state,
        params,
        key_sgd,
        it,
        data: types.Transition,
        normalizer_params,
    ):
        """Run num_updates_per_batch SGD sweeps over the collected data."""
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params
            ),
            (optimizer_state, params, key_sgd, it),
            (),
            length=num_updates_per_batch,
        )
        return optimizer_state, params, metrics

    # ------------------------------------------------------------------
    # Evaluator setup (same as ppo.py, no pmap)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Checkpoint iteration recovery
    # ------------------------------------------------------------------
    start_it = 0
    if ckpt_mgr is not None:
        if ckpt_mgr.latest_step() is not None:
            num_evals_after_init -= ckpt_mgr.latest_step()
            start_it = ckpt_mgr.latest_step()

    logging.info(
        "Starting at iteration: %d with %d evals left",
        start_it,
        num_evals_after_init,
    )

    # ------------------------------------------------------------------
    # Initial evaluation & checkpoint
    # ------------------------------------------------------------------
    metrics: dict = {}
    if num_evals > 1 and start_it == 0:
        logging.info("Running initial evaluation")
        policy_param = (
            training_state.normalizer_params,
            training_state.params.policy,
        )
        metrics = evaluator.run_evaluation(
            policy_param,
            training_metrics={},
        )
        if evaluator_test_set is not None:
            metrics = evaluator_test_set.run_evaluation(
                policy_param,
                training_metrics=metrics,
                data_split="test_set",
            )
        logging.info(metrics)
        progress_fn(start_it, metrics)
        logging.info("Saving initial checkpoint")
        if ckpt_mgr is not None:
            checkpointing.save(
                ckpt_mgr, 0, policy_param, training_state,
                config_dict, checkpoint_callback,
            )
        else:
            logging.info("Skipping checkpoint save as ckpt_mgr is None")

    # ------------------------------------------------------------------
    # Main training loop (Python loops — no pmap, no lax.scan on epoch)
    # ------------------------------------------------------------------
    training_walltime = 0.0
    training_metrics: dict = {}
    start_it += 1
    current_step = 0

    for it in range(start_it, num_evals_after_init + start_it):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _reset in range(max(num_resets_per_eval, 1)):
            t = time.time()
            training_state, env_state = _strip_weak_type(
                (training_state, env_state)
            )
            step_tensor = jnp.int32(it)

            # ---- training steps within one epoch ----
            all_step_metrics = []
            for _ts in range(num_training_steps_per_epoch):
                local_key, key_sgd, key_unroll = jax.random.split(local_key, 3)

                # 1. Build current policy
                policy = make_policy(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                    )
                )

                # 2. Collect num_unrolls_per_step unrolls via Python loop
                all_data = []
                current_unroll_key = key_unroll
                for _u in range(num_unrolls_per_step):
                    current_unroll_key, unroll_key = jax.random.split(
                        current_unroll_key
                    )
                    env_state, data = generate_unroll_with_vision(
                        env,
                        env_state,
                        policy,
                        unroll_key,
                        unroll_length,
                        renderer=renderer,
                        grayscale=grayscale,
                        extra_fields=("truncation",),
                        step_fn=step_fn,
                    )
                    all_data.append(data)

                # 3. Stack and reshape:
                #    each data: (unroll_length, num_envs, ...)
                #    stack -> (N, unroll_length, num_envs, ...)
                #    swapaxes -> (N, num_envs, unroll_length, ...)
                #    reshape -> (N*num_envs, unroll_length, ...)
                data = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs), *all_data
                )
                data = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 1, 2), data
                )
                data = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                )

                # 4. Update normaliser
                if normalize_observations:
                    normalizer_params = update_dict_normalizer(
                        training_state.normalizer_params,
                        data.observation,
                        pmap_axis_name=None,
                    )
                else:
                    normalizer_params = training_state.normalizer_params

                if frozen_proprioceptive_normalizer_params is not None:
                    normalizer_params = normalizer_params.replace(
                        proprioception=frozen_proprioceptive_normalizer_params
                    )

                # 5. JIT'd SGD
                optimizer_state, params, step_metrics = jit_sgd(
                    training_state.optimizer_state,
                    training_state.params,
                    key_sgd,
                    step_tensor,
                    data,
                    normalizer_params,
                )

                # 6. Update training state
                training_state = TrainingState(
                    optimizer_state=optimizer_state,
                    params=params,
                    normalizer_params=normalizer_params,
                    env_steps=jnp.int32(
                        training_state.env_steps
                        + env_step_per_training_step / STEPS_IN_THOUSANDS
                    ),
                )
                all_step_metrics.append(step_metrics)

            # Aggregate training metrics across steps
            epoch_metrics = jax.tree_util.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs)),
                *all_step_metrics,
            )
            epoch_metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), epoch_metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                num_training_steps_per_epoch
                * env_step_per_training_step
                * max(num_resets_per_eval, 1)
            ) / epoch_training_time
            training_metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{
                    f"training/{name}": value
                    for name, value in epoch_metrics.items()
                },
            }

            current_step = int(training_state.env_steps)

            # Reset env if requested
            key_envs = jax.random.split(key_envs[0], num_envs)
            if num_resets_per_eval > 0:
                env_state = reset_fn(key_envs)

        # ----------------------------------------------------------
        # Evaluation, logging, checkpointing
        # ----------------------------------------------------------
        policy_param = (
            training_state.normalizer_params,
            training_state.params.policy,
        )

        metrics = evaluator.run_evaluation(
            policy_param,
            training_metrics,
        )
        if evaluator_test_set is not None:
            metrics = evaluator_test_set.run_evaluation(
                policy_param,
                metrics,
                data_split="test_set",
            )

        # Policy param callback (render video at configured interval)
        _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
        render_interval = (
            config_dict.get("render_config", {}).get("render_interval", 0)
        )
        render_video = render_interval > 0 and it % render_interval == 0
        policy_params_fn(
            current_step=it,
            jit_logging_inference_fn=jit_logging_inference_fn,
            params=policy_param,
            policy_params_fn_key=policy_params_fn_key,
            render_video=render_video,
            ppo_network=ppo_network,
        )

        logging.info(metrics)
        progress_fn(current_step, metrics)

        if ckpt_mgr is not None:
            checkpointing.save(
                ckpt_mgr,
                it,
                policy_param,
                training_state,
                config_dict,
                checkpoint_callback,
            )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total_steps = current_step
    params = (training_state.normalizer_params, training_state.params.policy)
    logging.info("total steps: %s", total_steps)
    return (make_policy, params, metrics)
