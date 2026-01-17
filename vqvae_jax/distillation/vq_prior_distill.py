"""VQ-VAE Prior Distillation training module.

This module trains a Prior network to predict VQ-VAE encoder embeddings
from proprioceptive observations only. The key design principle:

FROZEN (loaded from VQ-VAE checkpoint, NO gradients):
    - VQ-VAE Encoder params
    - VQ-VAE Decoder params
    - Codebook embeddings

TRAINABLE (initialized fresh, receives gradients):
    - Prior Network params

During training:
    - The FROZEN VQ-VAE controls the rodent (generates actions from traj)
    - The Prior learns to predict z_e from proprio (but does NOT act)
    - Loss = MSE(z_p, stop_gradient(z_e))

During freeloop evaluation:
    - The Prior controls the rodent (generates z_p from proprio only)
    - No reference trajectory is used

Reference: track_mjx/agent/mlp_distill/distill.py
"""

import functools
import time
from typing import Any, Callable, Optional, Tuple, Sequence

from absl import logging
from brax import base, envs
from brax.training import acting, pmap, types
from brax.training.types import Params, PRNGKey

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from mujoco_playground import wrapper as mp_wrapper
from omegaconf import DictConfig, OmegaConf

from track_mjx.agent import gradients, checkpointing
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
    flatten_obs_dict,
)

# Local imports from distillation package
from .vq_prior_losses import (
    VQPriorDistillNetworkParams,
    compute_vq_prior_distill_loss,
    create_ar_schedule,
)
from .vq_prior_networks import (
    VQPriorNetworks,
    make_vq_prior_networks,
    make_prior_inference_fn,
)

# Imports from parent vqvae_jax directory
import sys
from pathlib import Path
_DISTILL_DIR = Path(__file__).parent
_VQVAE_DIR = _DISTILL_DIR.parent
if str(_VQVAE_DIR) not in sys.path:
    sys.path.insert(0, str(_VQVAE_DIR))

from vq_intention_network import VQEncoder, Decoder, VectorQuantizer
from analysis.checkpoint_utils import (
    load_vq_checkpoint,
    get_codebook,
    get_encoder_params,
    get_decoder_params,
)


Metrics = types.Metrics
STEPS_IN_THOUSANDS = 1e3
_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class VQPriorDistillTrainingState:
    """Training state for VQ-VAE prior distillation.

    IMPORTANT: Only `prior_params` receives gradient updates!
    The encoder_params, decoder_params, and codebook are FROZEN.
    """

    # ═══════════════════════════════════════════════════════════════════
    # 🔓 TRAINABLE - These get gradient updates via optimizer
    # ═══════════════════════════════════════════════════════════════════
    optimizer_state: optax.OptState
    params: VQPriorDistillNetworkParams  # Prior network weights

    # ═══════════════════════════════════════════════════════════════════
    # 🔒 FROZEN - Loaded from VQ-VAE checkpoint, NEVER updated
    # ═══════════════════════════════════════════════════════════════════
    frozen_encoder_params: Params  # VQ-VAE encoder - forward pass only
    frozen_decoder_params: Params  # VQ-VAE decoder - used for inference
    frozen_codebook: jnp.ndarray  # VQ-VAE codebook [num_codes, latent_dim]

    # ═══════════════════════════════════════════════════════════════════
    # Normalizer and bookkeeping
    # ═══════════════════════════════════════════════════════════════════
    normalizer_params: DictRunningStatisticsState  # From data collection
    env_steps: jnp.ndarray


def load_frozen_vqvae(
    checkpoint_path: str,
    step: int | None = None,
) -> dict[str, Any]:
    """Load frozen VQ-VAE components from checkpoint.

    Loads the encoder, decoder, codebook, and normalizer from a VQ-VAE
    checkpoint. These components will be frozen during prior training.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint directory.
        step: Specific step to load. If None, loads latest.

    Returns:
        Dictionary with:
        - "encoder_params": Encoder weights
        - "decoder_params": Decoder weights
        - "codebook": Codebook embeddings [num_codes, latent_dim]
        - "normalizer_params": Observation normalizer
        - "cfg": Configuration
        - "step": Loaded checkpoint step
    """
    checkpoint = load_vq_checkpoint(checkpoint_path, step=step)
    cfg = checkpoint["cfg"]
    policy_params = checkpoint["policy"]

    normalizer_params, policy_weights = policy_params

    # Extract components
    encoder_params = get_encoder_params(policy_params)
    decoder_params = get_decoder_params(policy_params)
    codebook = get_codebook(policy_params)

    logging.info(f"Loaded VQ-VAE from {checkpoint_path} at step {checkpoint['step']}")
    logging.info(f"  Codebook shape: {codebook.shape}")
    logging.info(f"  Latent dim: {cfg.network_config.latent_dim}")

    return {
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "codebook": codebook,
        "normalizer_params": normalizer_params,
        "cfg": cfg,
        "step": checkpoint["step"],
        "full_policy_params": policy_params,
    }


def create_frozen_encoder(
    cfg: DictConfig,
) -> VQEncoder:
    """Create frozen encoder module from config.

    Args:
        cfg: Configuration with network_config section.

    Returns:
        VQEncoder Flax module.
    """
    return VQEncoder(
        layer_sizes=list(cfg.network_config.encoder_layer_sizes),
        latent_dim=cfg.network_config.latent_dim,
    )


def create_frozen_decoder(
    cfg: DictConfig,
) -> Decoder:
    """Create frozen decoder module from config.

    Args:
        cfg: Configuration with network_config section.

    Returns:
        Decoder Flax module.
    """
    action_size = cfg.network_config.action_size
    decoder_layer_sizes = list(cfg.network_config.decoder_layer_sizes) + [
        action_size * 2
    ]
    return Decoder(layer_sizes=decoder_layer_sizes)


def create_frozen_vqvae_policy(
    frozen_encoder_params: Params,
    frozen_decoder_params: Params,
    frozen_codebook: jnp.ndarray,
    normalizer_params: DictRunningStatisticsState,
    encoder_module: VQEncoder,
    decoder_module: Decoder,
    parametric_action_distribution: Any,
    reference_obs_size: int,
) -> Callable:
    """Create frozen VQ-VAE policy for data collection.

    This policy uses the frozen VQ-VAE (encoder + quantizer + decoder)
    to generate actions during training data collection.

    Args:
        frozen_encoder_params: Frozen encoder weights.
        frozen_decoder_params: Frozen decoder weights.
        frozen_codebook: Frozen codebook [num_codes, latent_dim].
        normalizer_params: Dict observation normalizer (DictRunningStatisticsState).
        encoder_module: VQEncoder Flax module.
        decoder_module: Decoder Flax module.
        parametric_action_distribution: Action distribution.
        reference_obs_size: Size of reference trajectory in observations (unused,
            kept for API compatibility).

    Returns:
        Policy function: (obs, key) -> (action, extras)
    """

    def policy(
        observations: types.Observation,
        key: PRNGKey,
    ) -> Tuple[types.Action, types.Extra]:
        """Frozen VQ-VAE policy for data collection.

        Args:
            observations: Dict observations {"imitation_target": ..., "proprioception": ...}.
            key: JAX random key (unused, VQ is deterministic).

        Returns:
            Tuple of (action, extras) where extras contains z_e for training.
        """
        # Normalize dict observations
        obs_normalized = normalize_dict_obs(observations, normalizer_params)

        # Access by key (not flat array slicing)
        traj = obs_normalized["imitation_target"]
        proprio = obs_normalized["proprioception"]

        # Encode trajectory to z_e
        z_e = encoder_module.apply({"params": frozen_encoder_params}, traj)

        # Quantize to nearest codebook entry
        z_e_flat = z_e.reshape(-1, z_e.shape[-1])
        z_e_sq = jnp.sum(z_e_flat**2, axis=-1, keepdims=True)
        codebook_sq = jnp.sum(frozen_codebook**2, axis=-1)
        cross = jnp.matmul(z_e_flat, frozen_codebook.T)
        distances = z_e_sq + codebook_sq - 2 * cross
        flat_indices = jnp.argmin(distances, axis=-1)
        indices = flat_indices.reshape(z_e.shape[:-1])
        z_q = frozen_codebook[indices]

        # Decode to action
        decoder_input = jnp.concatenate([z_q, proprio], axis=-1)
        action_logits, _ = decoder_module.apply(
            {"params": frozen_decoder_params}, decoder_input
        )

        # Get deterministic action (mode)
        action = parametric_action_distribution.mode(action_logits)

        extras = {
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "log_prob": jnp.zeros(action.shape[:-1]),  # Placeholder for compatibility
            "raw_action": action,
        }

        return jnp.array(action), extras

    return policy


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
    config_dict: dict[str, Any],
    vqvae_checkpoint_path: str,
    vqvae_checkpoint_step: int | None = None,
    checkpoint_to_restore: str | None = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: int | None = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    grad_clip_norm: float = 20.0,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 20,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = True,
    # Loss configuration
    loss_type: str = "mse",
    ar_weight: float = 0.0,
    phi: float = 0.99,
    smooth_l1_delta: float = 1.0,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
    use_ar_schedule: bool = False,
    ar_schedule_params: dict | None = None,
    # Prior network configuration
    prior_layer_sizes: Sequence[int] = (1024, 1024),
    # Callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Callable | None = None,
    checkpoint_callback: Callable[[int], None] | None = None,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
    # Evaluation
    eval_env: envs.Env | None = None,
    eval_env_test_set: envs.Env | None = None,
    deterministic_eval: bool = True,
    # Freeloop evaluation
    freeloop_config: dict | None = None,
):
    """Train a Prior network using VQ-VAE prior distillation.

    This trains a Prior network to predict the frozen VQ-VAE encoder's output
    from proprioceptive observations only. The frozen VQ-VAE controls the
    rodent during data collection, while the Prior learns to match z_e.

    Args:
        environment: The environment to train on.
        num_timesteps: Total number of environment steps.
        episode_length: Length of an episode.
        ckpt_mgr: Orbax checkpoint manager.
        config_dict: Configuration dictionary for checkpointing.
        vqvae_checkpoint_path: Path to frozen VQ-VAE checkpoint.
        vqvae_checkpoint_step: Optional step to load from VQ-VAE checkpoint.
        checkpoint_to_restore: Optional path to restore prior training from.
        action_repeat: Number of times to repeat actions.
        num_envs: Number of parallel environments.
        max_devices_per_host: Maximum devices per host.
        num_eval_envs: Number of evaluation environments.
        learning_rate: Learning rate for optimizer.
        grad_clip_norm: Maximum gradient norm for clipping.
        seed: Random seed.
        use_pmap_on_reset: Whether to pmap instead of vmap env reset.
        unroll_length: Number of timesteps to unroll.
        batch_size: Batch size for minibatch SGD.
        num_minibatches: Number of minibatches.
        num_updates_per_batch: Number of gradient updates per batch.
        num_evals: Number of evaluations during training.
        num_resets_per_eval: Number of environment resets per eval.
        normalize_observations: Whether to normalize observations.
        loss_type: Type of alignment loss ("mse", "l2", etc.).
        ar_weight: Weight for AR(1) temporal smoothness loss.
        phi: AR(1) coefficient.
        smooth_l1_delta: Delta for Smooth L1 loss.
        mse_weight: MSE weight for combined loss.
        cosine_weight: Cosine weight for combined loss.
        use_ar_schedule: Whether to use AR weight schedule.
        ar_schedule_params: Parameters for AR schedule.
        prior_layer_sizes: Hidden layer sizes for Prior network.
        progress_fn: Callback for logging progress.
        policy_params_fn: Callback for policy evaluation/logging.
        randomization_fn: Optional domain randomization function.
        checkpoint_callback: Callback after checkpointing.
        wrap_for_training: Function to wrap environment for training.
        eval_env: Optional separate evaluation environment.
        eval_env_test_set: Optional test set evaluation environment.
        deterministic_eval: Whether to use deterministic policy for eval.
        freeloop_config: Configuration for freeloop evaluation.

    Returns:
        Tuple of (make_policy_fn, final_params, final_metrics).
    """
    from brax.training import distribution

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

    # ═══════════════════════════════════════════════════════════════════
    # Load FROZEN VQ-VAE components
    # ═══════════════════════════════════════════════════════════════════
    logging.info(f"Loading frozen VQ-VAE from: {vqvae_checkpoint_path}")
    frozen_vqvae = load_frozen_vqvae(vqvae_checkpoint_path, vqvae_checkpoint_step)
    vqvae_cfg = frozen_vqvae["cfg"]

    frozen_encoder_params = frozen_vqvae["encoder_params"]
    frozen_decoder_params = frozen_vqvae["decoder_params"]
    frozen_codebook = frozen_vqvae["codebook"]
    frozen_normalizer_params = frozen_vqvae["normalizer_params"]

    latent_dim = vqvae_cfg.network_config.latent_dim
    reference_obs_size = vqvae_cfg.network_config.reference_obs_size

    logging.info(f"VQ-VAE latent_dim: {latent_dim}")
    logging.info(f"VQ-VAE reference_obs_size: {reference_obs_size}")
    logging.info(f"VQ-VAE num_codes: {frozen_codebook.shape[0]}")

    # Create frozen encoder/decoder modules
    encoder_module = create_frozen_encoder(vqvae_cfg)
    decoder_module = create_frozen_decoder(vqvae_cfg)

    # Action distribution
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=environment.action_size
    )

    # ═══════════════════════════════════════════════════════════════════
    # Setup environment
    # ═══════════════════════════════════════════════════════════════════
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_prior, policy_params_fn_key = jax.random.split(global_key, 2)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    proprioceptive_obs_size = int(environment.proprioceptive_obs_size)
    logging.info(f"Proprioceptive observation size: {proprioceptive_obs_size}")

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

    # Compute observation sizes from known dimensions
    # (environment is wrapped and has flat observations, so we use the sizes
    # we already have from the VQ-VAE config and environment properties)
    obs_sizes = {
        "imitation_target": reference_obs_size,
        "proprioception": proprioceptive_obs_size,
    }

    # Update config with network info
    config_dict["network_config"] = {
        "arch_name": "vq_prior_distill",
        "latent_dim": latent_dim,
        "reference_obs_size": reference_obs_size,
        "proprioceptive_obs_size": proprioceptive_obs_size,
        "prior_layer_sizes": list(prior_layer_sizes),
        "action_size": env.action_size,
        "obs_sizes": obs_sizes,
        "normalize_observations": normalize_observations,
        "vqvae_checkpoint_path": vqvae_checkpoint_path,
        "vqvae_checkpoint_step": frozen_vqvae["step"],
        "num_codes": frozen_codebook.shape[0],
    }

    # ═══════════════════════════════════════════════════════════════════
    # Create TRAINABLE Prior network
    # ═══════════════════════════════════════════════════════════════════
    prior_networks = make_vq_prior_networks(
        proprio_size=proprioceptive_obs_size,
        latent_dim=latent_dim,
        layer_sizes=prior_layer_sizes,
        normalize_observations=normalize_observations,
    )
    make_prior_fn = make_prior_inference_fn(prior_networks)

    # ═══════════════════════════════════════════════════════════════════
    # Create FROZEN VQ-VAE policy for data collection
    # ═══════════════════════════════════════════════════════════════════
    # Use VQ-VAE normalizer for data collection
    vqvae_normalizer = frozen_vqvae["normalizer_params"]

    frozen_vqvae_policy = create_frozen_vqvae_policy(
        frozen_encoder_params=frozen_encoder_params,
        frozen_decoder_params=frozen_decoder_params,
        frozen_codebook=frozen_codebook,
        normalizer_params=vqvae_normalizer,
        encoder_module=encoder_module,
        decoder_module=decoder_module,
        parametric_action_distribution=parametric_action_distribution,
        reference_obs_size=reference_obs_size,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Setup optimizer (ONLY for Prior params)
    # ═══════════════════════════════════════════════════════════════════
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate=learning_rate),
    )

    # Setup AR schedule if enabled
    ar_schedule_fn = None
    if use_ar_schedule and ar_schedule_params is not None:
        ar_schedule_fn = create_ar_schedule(
            start_value=ar_schedule_params.get("start_value", 0.0),
            end_value=ar_schedule_params.get("end_value", ar_weight),
            total_steps=num_evals,
            start_frac=ar_schedule_params.get("start_frac", 0.3),
            end_frac=ar_schedule_params.get("end_frac", 0.6),
            schedule_type=ar_schedule_params.get("schedule_type", "linear"),
        )

    # Create loss function
    loss_fn = functools.partial(
        compute_vq_prior_distill_loss,
        prior_network=prior_networks.prior_network,
        frozen_encoder=encoder_module,
        frozen_encoder_params={"params": frozen_encoder_params},
        reference_obs_size=reference_obs_size,
        loss_type=loss_type,
        ar_weight=ar_weight,
        phi=phi,
        ar_schedule=ar_schedule_fn,
        smooth_l1_delta=smooth_l1_delta,
        mse_weight=mse_weight,
        cosine_weight=cosine_weight,
    )

    # Initialize Prior parameters
    init_prior_params = VQPriorDistillNetworkParams(
        prior=prior_networks.prior_network.init(key_prior),
    )

    training_state = VQPriorDistillTrainingState(
        optimizer_state=optimizer.init(init_prior_params),
        params=init_prior_params,
        frozen_encoder_params=frozen_encoder_params,
        frozen_decoder_params=frozen_decoder_params,
        frozen_codebook=frozen_codebook,
        normalizer_params=frozen_normalizer_params,  # Use VQ-VAE's normalizer
        env_steps=jnp.array(0),
    )

    # Optionally restore from checkpoint
    if checkpoint_to_restore is not None:
        training_state = checkpointing.load_training_state(
            checkpoint_to_restore, training_state, step_prefix="VQPriorDistill"
        )
        logging.info(f"Restored checkpoint from {checkpoint_to_restore}")

    # Create gradient update function
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    # ═══════════════════════════════════════════════════════════════════
    # Training functions
    # ═══════════════════════════════════════════════════════════════════
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
        carry: Tuple[VQPriorDistillTrainingState, envs.State, PRNGKey, int], unused_t
    ) -> Tuple[Tuple[VQPriorDistillTrainingState, envs.State, PRNGKey, int], Metrics]:
        training_state, state, key, it = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        # ══════════════════════════════════════════════════════════════
        # CRITICAL: Use FROZEN VQ-VAE for data collection
        # The Prior does NOT control the rodent during training!
        # ══════════════════════════════════════════════════════════════

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                frozen_vqvae_policy,  # FROZEN policy controls!
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

        # Use frozen normalizer (don't update - it should match VQ-VAE training)
        normalizer_params = training_state.normalizer_params

        # ══════════════════════════════════════════════════════════════
        # Train the Prior to predict z_e from proprio
        # ══════════════════════════════════════════════════════════════
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd, it),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = VQPriorDistillTrainingState(
            optimizer_state=optimizer_state,
            params=params,
            frozen_encoder_params=training_state.frozen_encoder_params,
            frozen_decoder_params=training_state.frozen_decoder_params,
            frozen_codebook=training_state.frozen_codebook,
            normalizer_params=normalizer_params,
            env_steps=jnp.int32(
                training_state.env_steps
                + env_step_per_training_step / STEPS_IN_THOUSANDS
            ),
        )
        return (new_training_state, state, new_key, it), metrics

    def training_epoch(
        training_state: VQPriorDistillTrainingState,
        state: envs.State,
        key: PRNGKey,
        it: int,
    ) -> Tuple[VQPriorDistillTrainingState, envs.State, Metrics]:
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

    training_walltime = 0.0

    def training_epoch_with_timing(
        training_state: VQPriorDistillTrainingState,
        env_state: envs.State,
        key: PRNGKey,
        it: int,
    ) -> Tuple[VQPriorDistillTrainingState, envs.State, Metrics]:
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

    # Replicate training state across devices
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    # ═══════════════════════════════════════════════════════════════════
    # Setup evaluation
    # ═══════════════════════════════════════════════════════════════════
    # Note: For evaluation, we use frozen VQ-VAE policy (not Prior)
    # Freeloop evaluation uses Prior - implemented separately

    if eval_env is None:
        eval_env = environment
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )

    # For standard eval, use frozen VQ-VAE policy
    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(lambda: frozen_vqvae_policy),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════════════
    start_it = 0
    if ckpt_mgr is not None and ckpt_mgr.latest_step() is not None:
        num_evals_after_init -= ckpt_mgr.latest_step()
        start_it = ckpt_mgr.latest_step()

    logging.info(
        f"Starting at iteration: {start_it} with {num_evals_after_init} evals left"
    )

    # Initial evaluation
    metrics = {}
    if process_id == 0 and num_evals > 1 and start_it == 0:
        logging.info("Running initial evaluation")
        prior_params = _unpmap(
            (training_state.normalizer_params, training_state.params.prior)
        )

        # Log initial metrics
        progress_fn(0, {"eval/initial": True})

        if ckpt_mgr is not None:
            # Save with prior params for freeloop evaluation
            save_params = {
                "prior": prior_params,
                "frozen_vqvae": {
                    "encoder_params": frozen_encoder_params,
                    "decoder_params": frozen_decoder_params,
                    "codebook": frozen_codebook,
                },
            }
            ckpt_mgr.save(
                step=0,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardSave(save_params),
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
            prior_params = _unpmap(
                (training_state.normalizer_params, training_state.params.prior)
            )

            # Merge training metrics with any eval metrics
            metrics = {**training_metrics}

            # Call policy_params_fn for logging
            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            policy_params_fn(
                current_step=it,
                prior_params=prior_params,
                frozen_decoder_params=frozen_decoder_params,
                frozen_codebook=frozen_codebook,
                policy_params_fn_key=policy_params_fn_key,
            )

            logging.info(metrics)
            progress_fn(current_step, metrics)

            if ckpt_mgr is not None:
                save_params = {
                    "prior": prior_params,
                    "frozen_vqvae": {
                        "encoder_params": frozen_encoder_params,
                        "decoder_params": frozen_decoder_params,
                        "codebook": frozen_codebook,
                    },
                }
                ckpt_mgr.save(
                    step=it,
                    args=ocp.args.Composite(
                        policy=ocp.args.StandardSave(save_params),
                        train_state=ocp.args.StandardSave(_unpmap(training_state)),
                        config=ocp.args.JsonSave(config_dict),
                    ),
                )
                if checkpoint_callback is not None:
                    try:
                        checkpoint_callback(it)
                    except Exception as e:
                        logging.warning(f"Checkpoint callback failed: {e}")

    total_steps = current_step
    pmap.assert_is_replicated(training_state)
    params = _unpmap((training_state.normalizer_params, training_state.params.prior))
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()

    return (make_prior_fn, params, metrics)


# Convenience type alias
Sequence = tuple | list
