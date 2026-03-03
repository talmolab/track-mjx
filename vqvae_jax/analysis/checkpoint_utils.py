"""Checkpoint loading utilities for VQ-VAE models.

This module provides functions for loading VQ-VAE checkpoints and creating
inference functions, handling the VQ-specific network architecture.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics, specs
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vq_intention_network import VQEncoder, ResidualVectorQuantizer, Decoder
from vq_ppo_networks import (
    VQPPOImitationNetworks,
    make_vq_inference_fn,
    make_vq_intention_ppo_networks,
    make_vq_chunked_inference_fn,
    make_vq_chunked_ppo_networks,
)

from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    convert_flat_to_dict_normalizer,
)


def load_config_from_checkpoint(
    checkpoint_path: str,
    step_prefix: str = "VQPPONetwork",
    step: int | None = None,
) -> dict[str, Any]:
    """Load configuration from a VQ-VAE checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        step_prefix: Prefix for checkpoint steps (default: VQPPONetwork).
        step: Specific step to load. If None, loads latest.

    Returns:
        Configuration dictionary from the checkpoint.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading config from {checkpoint_path} at step {step}")
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

        return cfg


def _get_obs_sizes_from_cfg(cfg: DictConfig) -> dict[str, int]:
    """Extract obs_sizes dict from config, handling both old and new formats.

    Args:
        cfg: Configuration with network_config section.

    Returns:
        Dict mapping observation keys to sizes.
    """
    net_cfg = cfg.network_config
    # New format: obs_sizes dict is directly available
    if hasattr(net_cfg, "obs_sizes") and net_cfg.obs_sizes is not None:
        return dict(net_cfg.obs_sizes)

    # Old format: separate observation_size and reference_obs_size
    # Convert to new format
    if hasattr(net_cfg, "observation_size") and hasattr(net_cfg, "reference_obs_size"):
        proprio_size = net_cfg.observation_size - net_cfg.reference_obs_size
        return {
            "imitation_target": net_cfg.reference_obs_size,
            "proprioception": proprio_size,
        }

    raise ValueError(
        "Config must have either 'obs_sizes' or both "
        "'observation_size' and 'reference_obs_size'"
    )


def make_vq_ppo_network_from_cfg(
    cfg: DictConfig,
) -> VQPPOImitationNetworks:
    """Create VQ-VAE PPO networks from configuration.

    Args:
        cfg: Configuration with network_config section.

    Returns:
        VQPPOImitationNetworks with policy, value, and action distribution.

    Raises:
        ValueError: If architecture is not a recognized VQ-VAE variant.
    """
    valid_arch_names = {"vqvae_intention", "vqvae_naive", "vqvae_code_chunk"}
    if cfg.network_config.arch_name not in valid_arch_names:
        raise ValueError(
            f"Expected arch_name in {valid_arch_names}, "
            f"got '{cfg.network_config.arch_name}'"
        )

    obs_sizes = _get_obs_sizes_from_cfg(cfg)

    # Handle stickiness_bias: OmegaConf ListConfig → tuple
    stickiness_bias = cfg.network_config.get("stickiness_bias", 0.0)
    try:
        stickiness_bias = tuple(float(b) for b in stickiness_bias)
    except TypeError:
        stickiness_bias = float(stickiness_bias)

    return make_vq_intention_ppo_networks(
        obs_sizes=obs_sizes,
        action_size=cfg.network_config.action_size,
        latent_dim=cfg.network_config.latent_dim,
        num_codes=cfg.network_config.num_codes,
        commitment_cost=cfg.network_config.commitment_cost,
        codebook_init_scale=cfg.network_config.codebook_init_scale,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        stickiness_bias=stickiness_bias,
        rvq_depth=int(cfg.network_config.get("rvq_depth", 1)),
        use_rotation=bool(cfg.network_config.get("use_rotation", False)),
        coupled_residual_grad=bool(
            cfg.network_config.get("coupled_residual_grad", False)
        ),
        use_continuous_latent=bool(
            cfg.network_config.get("use_continuous_latent", False)
        ),
        continuous_latent_dim=int(
            cfg.network_config.get("continuous_latent_dim", 4)
        ),
    )


def make_abstract_vq_policy(
    cfg: DictConfig,
    seed: int = 1,
) -> tuple[Any, Any]:
    """Create abstract policy structure for checkpoint restoration.

    Args:
        cfg: Configuration with network_config section.
        seed: Random seed for initialization.

    Returns:
        Tuple of (normalizer_state, policy_params) with correct pytree structure.
    """
    ppo_network = make_vq_ppo_network_from_cfg(cfg)
    key_policy, key_value = jax.random.split(jax.random.key(seed))

    init_policy_params = ppo_network.policy_network.init(key_policy)

    obs_sizes = _get_obs_sizes_from_cfg(cfg)
    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(obs_sizes["imitation_target"], jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(obs_sizes["proprioception"], jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, init_policy_params)


def _dict_to_running_statistics_state(
    d: dict,
) -> running_statistics.RunningStatisticsState:
    """Convert a dict restored by orbax to RunningStatisticsState.

    Handles both old and new Brax versions (with/without std_eps, mode fields).
    """
    # Required fields
    state = running_statistics.RunningStatisticsState(
        count=d["count"],
        mean=d["mean"],
        summed_variance=d["summed_variance"],
        std=d["std"],
    )
    # Some versions have additional fields - replace if present
    if "std_eps" in d:
        state = state.replace(std_eps=d["std_eps"])
    if "mode" in d:
        state = state.replace(mode=d["mode"])
    return state


def load_vq_policy(
    checkpoint_path: str,
    cfg: DictConfig | None = None,
    step_prefix: str = "VQPPONetwork",
    step: int | None = None,
) -> tuple[Any, Any]:
    """Load VQ-VAE policy parameters from checkpoint.

    Handles both flat normalizers (legacy) and dict normalizers (current).

    Args:
        checkpoint_path: Path to checkpoint directory.
        cfg: Configuration with env_config.reference_obs_size for flat->dict conversion.
        step_prefix: Prefix for checkpoint steps.
        step: Specific step to load. If None, loads latest.

    Returns:
        Tuple of (normalizer_state, policy_params).
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading VQ policy from {checkpoint_path} at step {step}")

        # Restore without strict template matching to handle Brax version differences
        # in running_statistics structure (e.g., NestedMeanStd count.hi/lo)
        policy = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(None)),
        )["policy"]

        # Convert orbax-restored dicts back to proper dataclass types
        normalizer_dict, policy_params = policy

        # Check if it's a dict normalizer or flat normalizer
        if (
            "imitation_target" in normalizer_dict
            and "proprioception" in normalizer_dict
        ):
            # Already dict normalizer structure
            normalizer_state = DictRunningStatisticsState(
                imitation_target=_dict_to_running_statistics_state(
                    normalizer_dict["imitation_target"]
                ),
                proprioception=_dict_to_running_statistics_state(
                    normalizer_dict["proprioception"]
                ),
            )
        else:
            # Flat normalizer - need to convert
            flat_state = _dict_to_running_statistics_state(normalizer_dict)

            # Get reference_obs_size from config (in network_config, not env_config)
            if cfg is None:
                cfg = OmegaConf.create(
                    load_config_from_checkpoint(checkpoint_path, step_prefix, step)
                )

            # Try to get obs_sizes first (newer format), fallback to reference_obs_size
            net_cfg = cfg.network_config
            if hasattr(net_cfg, "obs_sizes") and net_cfg.obs_sizes is not None:
                reference_obs_size = net_cfg.obs_sizes.get("imitation_target")
            elif hasattr(net_cfg, "reference_obs_size"):
                reference_obs_size = net_cfg.reference_obs_size
            else:
                raise ValueError(
                    "Checkpoint config missing both 'obs_sizes' and 'reference_obs_size'"
                )

            total_obs_size = flat_state.mean.shape[0]
            proprio_size = total_obs_size - reference_obs_size

            logging.info(
                f"Converting flat normalizer to dict normalizer: "
                f"total={total_obs_size}, imitation_target={reference_obs_size}, "
                f"proprioception={proprio_size}"
            )

            normalizer_state = convert_flat_to_dict_normalizer(
                flat_state, reference_obs_size
            )

        return (normalizer_state, policy_params)


def load_vq_checkpoint(
    checkpoint_path: str,
    step_prefix: str = "VQPPONetwork",
    step: int | None = None,
) -> dict[str, Any]:
    """Load VQ-VAE checkpoint for evaluation.

    Convenience function that loads both config and policy parameters.

    Args:
        checkpoint_path: Path to checkpoint directory.
        step_prefix: Prefix for checkpoint steps.
        step: Specific step to load. If None, loads latest.

    Returns:
        Dictionary with keys:
        - "cfg": OmegaConf configuration
        - "policy": Policy parameters (normalizer_state, policy_params)
        - "step": The loaded checkpoint step
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    if step is None:
        step = ckpt_mgr.latest_step()

    logging.info(f"Loading VQ checkpoint from {checkpoint_path} at step {step}")

    cfg = OmegaConf.create(
        load_config_from_checkpoint(checkpoint_path, step_prefix, step)
    )
    policy = load_vq_policy(checkpoint_path, cfg, step_prefix, step)

    return {"cfg": cfg, "policy": policy, "step": step}


def load_vq_inference_fn(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
    deterministic: bool = True,
    get_activation: bool = False,
) -> Callable:
    """Create VQ-VAE policy inference function from loaded parameters.

    Args:
        cfg: Configuration with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).
        deterministic: If True, use mean action (no sampling).
        get_activation: If True, return network activations in extras.

    Returns:
        Inference function: (obs, rng) -> (action, extras)
        where extras contains "z_e" and "indices".
    """
    ppo_network = make_vq_ppo_network_from_cfg(cfg)
    make_policy = make_vq_inference_fn(ppo_network)

    return make_policy(
        policy_params, deterministic=deterministic, get_activation=get_activation
    )


def load_vq_inference_fn_with_stickiness(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
    deterministic: bool = True,
    get_activation: bool = False,
) -> tuple[Callable, float]:
    """Create VQ-VAE inference function that supports stickiness bias.

    Unlike load_vq_inference_fn, this returns a function that accepts
    prev_indices to properly apply the stickiness bias during inference.

    Args:
        cfg: Configuration with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).
        deterministic: If True, use mean action (no sampling).
        get_activation: If True, return network activations in extras.

    Returns:
        Tuple of:
        - Inference function: (obs, rng, prev_indices) -> (action, extras)
          where prev_indices can be None (first step) or the previous code index.
        - stickiness_bias: The stickiness bias value from config (0.0 if not set).
    """
    ppo_network = make_vq_ppo_network_from_cfg(cfg)
    stickiness_bias = getattr(
        ppo_network, "stickiness_bias", cfg.network_config.get("stickiness_bias", 0.0)
    )

    policy_network = ppo_network.policy_network
    parametric_action_distribution = ppo_network.parametric_action_distribution

    def inference_fn(
        observations: jnp.ndarray,
        key: jax.Array,
        prev_indices: tuple[jnp.ndarray, ...] | jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """Run inference with stickiness bias support.

        Args:
            observations: Observation (same format as original inference_fn).
            key: JAX random key.
            prev_indices: Previous timestep's code indices, or None for first
                step. Tuple of D arrays for multi-level RVQ, or single array
                for depth=1 backward compat.

        Returns:
            Tuple of (action, extras) where extras contains "z_e", "indices",
            and "all_indices".
        """
        key, key_network = jax.random.split(key)

        # Apply policy with prev_indices for stickiness
        # Returns 4 values: (logits, z_e, all_indices, logvar)
        # or 5 with get_activation: (logits, z_e, all_indices, logvar, activations)
        if get_activation:
            logits, z_e, all_indices, logvar, activations = policy_network.apply(
                *policy_params,
                observations,
                key_network,
                deterministic=deterministic,
                get_activation=True,
                prev_indices=prev_indices,
            )
        else:
            logits, z_e, all_indices, logvar = policy_network.apply(
                *policy_params,
                observations,
                key_network,
                deterministic=deterministic,
                prev_indices=prev_indices,
            )

        # Primary level for backward compat
        indices = all_indices[0] if isinstance(all_indices, tuple) else all_indices

        if deterministic:
            action = jnp.array(parametric_action_distribution.mode(logits))
        else:
            action = parametric_action_distribution.sample(logits, key)

        extras = {"z_e": z_e, "indices": indices, "all_indices": all_indices}
        if get_activation:
            extras["activations"] = activations

        return action, extras

    logging.info(f"Created stickiness-aware inference fn (bias={stickiness_bias})")
    return inference_fn, stickiness_bias


def load_vq_chunked_inference_fn(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
    commitment_horizon: int,
    deterministic: bool = True,
) -> tuple[Callable, Callable]:
    """Create a stateful chunked VQ-VAE inference function.

    Wraps make_vq_chunked_inference_fn to produce a policy that carries
    chunk_state (held_d0_idx, tau) through the rollout, matching the
    Semi-MDP temporal commitment pattern used during training.

    Args:
        cfg: Configuration with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).
        commitment_horizon: H, number of steps to hold D0 code.
        deterministic: If True, use mean action (no sampling).

    Returns:
        Tuple of:
        - inference_fn: (obs, chunk_state, rng) -> (action, extras, new_chunk_state)
        - initial_chunk_state_fn: () -> (held_d0_idx=0, tau=0)
    """
    obs_sizes = _get_obs_sizes_from_cfg(cfg)

    stickiness_bias = cfg.network_config.get("stickiness_bias", 0.0)
    try:
        stickiness_bias = tuple(float(b) for b in stickiness_bias)
    except TypeError:
        stickiness_bias = float(stickiness_bias)

    ppo_network = make_vq_chunked_ppo_networks(
        obs_sizes=obs_sizes,
        action_size=cfg.network_config.action_size,
        commitment_horizon=commitment_horizon,
        latent_dim=cfg.network_config.latent_dim,
        num_codes=cfg.network_config.num_codes,
        commitment_cost=cfg.network_config.commitment_cost,
        codebook_init_scale=cfg.network_config.codebook_init_scale,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        stickiness_bias=stickiness_bias,
        rvq_depth=int(cfg.network_config.get("rvq_depth", 2)),
        use_rotation=bool(cfg.network_config.get("use_rotation", False)),
        coupled_residual_grad=False,  # Must be False for chunking
        proprio_noise_scale=float(
            cfg.network_config.get("proprio_noise_scale", 0.0)
        ),
        use_continuous_latent=bool(
            cfg.network_config.get("use_continuous_latent", False)
        ),
        continuous_latent_dim=int(
            cfg.network_config.get("continuous_latent_dim", 4)
        ),
    )

    make_policy = make_vq_chunked_inference_fn(ppo_network, commitment_horizon)
    inference_fn = make_policy(policy_params, deterministic=deterministic)

    def initial_chunk_state_fn():
        return (jnp.array(0), jnp.array(0))

    logging.info(
        f"Created chunked inference fn (H={commitment_horizon}, "
        f"deterministic={deterministic})"
    )
    return inference_fn, initial_chunk_state_fn


def get_codebook(policy_params: tuple[Any, Any], depth: int = 0) -> jnp.ndarray:
    """Extract codebook embeddings from policy parameters.

    Supports both old (flat VQ) and new (RVQ) parameter structures.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params).
        depth: Which RVQ level to return (default 0 = primary/coarse).

    Returns:
        Codebook array of shape [num_codes, latent_dim].
    """
    _, params = policy_params
    quantizer = params["params"]["quantizer"]

    # New RVQ structure: quantizer/codebooks_0/embeddings
    codebook_key = f"codebooks_{depth}"
    if codebook_key in quantizer:
        return quantizer[codebook_key]["embeddings"]

    # Legacy flat VQ structure: quantizer/embeddings
    return quantizer["embeddings"]


def get_all_codebooks(policy_params: tuple[Any, Any]) -> list[jnp.ndarray]:
    """Extract all codebook embeddings from policy parameters.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        List of codebook arrays, one per RVQ depth level.
    """
    _, params = policy_params
    quantizer = params["params"]["quantizer"]

    codebooks = []
    d = 0
    while f"codebooks_{d}" in quantizer:
        codebooks.append(quantizer[f"codebooks_{d}"]["embeddings"])
        d += 1

    # Fallback for legacy structure
    if not codebooks and "embeddings" in quantizer:
        codebooks.append(quantizer["embeddings"])

    return codebooks


def get_decoder_params(policy_params: tuple[Any, Any]) -> dict[str, Any]:
    """Extract decoder parameters from policy parameters.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        Decoder parameter dictionary.
    """
    _, params = policy_params
    return params["params"]["decoder"]


def get_encoder_params(policy_params: tuple[Any, Any]) -> dict[str, Any]:
    """Extract encoder parameters from policy parameters.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        Encoder parameter dictionary.
    """
    _, params = policy_params
    return params["params"]["encoder"]


def create_standalone_decoder(
    cfg: DictConfig,
) -> Decoder:
    """Create a standalone Decoder module for decoder-only inference.

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


def create_decoder_apply_fn(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Create a function to apply decoder with loaded parameters.

    Args:
        cfg: Configuration with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        Function (z_q, proprio_obs) -> action_params
    """
    decoder = create_standalone_decoder(cfg)
    decoder_params = get_decoder_params(policy_params)

    def apply_decoder(z_q: jnp.ndarray, proprio_obs: jnp.ndarray) -> jnp.ndarray:
        """Apply decoder to quantized latent and proprioceptive observation.

        Args:
            z_q: Quantized latent, shape [..., latent_dim].
            proprio_obs: Proprioceptive observation, shape [..., proprio_size].

        Returns:
            Action parameters, shape [..., action_size * 2].
        """
        x = jnp.concatenate([z_q, proprio_obs], axis=-1)
        action_params, _ = decoder.apply({"params": decoder_params}, x)
        return action_params

    return apply_decoder
