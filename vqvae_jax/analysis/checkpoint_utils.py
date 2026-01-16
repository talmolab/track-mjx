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

from vq_intention_network import VQEncoder, VectorQuantizer, Decoder
from vq_ppo_networks import (
    VQPPOImitationNetworks,
    make_vq_inference_fn,
    make_vq_intention_ppo_networks,
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


def make_vq_ppo_network_from_cfg(
    cfg: DictConfig,
) -> VQPPOImitationNetworks:
    """Create VQ-VAE PPO networks from configuration.

    Args:
        cfg: Configuration with network_config section.

    Returns:
        VQPPOImitationNetworks with policy, value, and action distribution.

    Raises:
        ValueError: If architecture is not vqvae_intention.
    """
    if cfg.network_config.arch_name != "vqvae_intention":
        raise ValueError(
            f"Expected arch_name='vqvae_intention', got '{cfg.network_config.arch_name}'"
        )

    normalize: Callable = lambda x, y: x
    if cfg.train_setup.train_config.normalize_observations:
        normalize = running_statistics.normalize

    return make_vq_intention_ppo_networks(
        observation_size=cfg.network_config.observation_size,
        reference_obs_size=cfg.network_config.reference_obs_size,
        action_size=cfg.network_config.action_size,
        preprocess_observations_fn=normalize,
        latent_dim=cfg.network_config.latent_dim,
        num_codes=cfg.network_config.num_codes,
        commitment_cost=cfg.network_config.commitment_cost,
        codebook_init_scale=cfg.network_config.codebook_init_scale,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
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

    normalizer_state = running_statistics.init_state(
        specs.Array(cfg.network_config.observation_size, jnp.dtype("float32"))
    )

    return (normalizer_state, init_policy_params)


def load_vq_policy(
    checkpoint_path: str,
    cfg: DictConfig | None = None,
    step_prefix: str = "VQPPONetwork",
    step: int | None = None,
) -> tuple[Any, Any]:
    """Load VQ-VAE policy parameters from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory.
        cfg: Configuration. If None, loaded from checkpoint.
        step_prefix: Prefix for checkpoint steps.
        step: Specific step to load. If None, loads latest.

    Returns:
        Tuple of (normalizer_state, policy_params).
    """
    if cfg is None:
        cfg = OmegaConf.create(
            load_config_from_checkpoint(checkpoint_path, step_prefix, step)
        )

    abstract_policy = make_abstract_vq_policy(cfg)

    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading VQ policy from {checkpoint_path} at step {step}")

        return ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]


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


def get_codebook(policy_params: tuple[Any, Any]) -> jnp.ndarray:
    """Extract codebook embeddings from policy parameters.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        Codebook array of shape [num_codes, latent_dim].
    """
    _, params = policy_params
    # Navigate the nested parameter structure
    # Structure: {'params': {'encoder': ..., 'quantizer': {'embeddings': ...}, 'decoder': ...}}
    return params["params"]["quantizer"]["embeddings"]


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
    decoder_layer_sizes = list(cfg.network_config.decoder_layer_sizes) + [action_size * 2]

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
