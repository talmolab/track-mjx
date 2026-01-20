"""Checkpoint utilities for loading mlp_prior checkpoints for task transfer.

This module provides functions to:
- Load prior and decoder parameters from mlp_prior checkpoints
- Create inference functions for the frozen prior and decoder networks

Only supports dict normalizer format (no legacy flat format).
"""

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics, specs
from omegaconf import OmegaConf

from track_mjx.agent.ff_ppo import intention_network
from track_mjx.agent.mlp_prior.prior_networks import Prior
from track_mjx.agent.observation_utils import DictRunningStatisticsState


def load_prior_checkpoint(
    checkpoint_path: str,
    step: int | None = None,
) -> Tuple[Dict, Dict, DictRunningStatisticsState, Dict]:
    """Load prior and decoder parameters from an mlp_prior checkpoint.

    mlp_prior checkpoints have structure:
    (normalizer_params, {"params": {"encoder": ..., "decoder": ..., "prior": ...}})

    Only supports dict normalizer format.

    Args:
        checkpoint_path: Path to the mlp_prior checkpoint directory.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (prior_params, decoder_params, normalizer_params, config_dict)
        normalizer_params is DictRunningStatisticsState.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix="PriorNetwork")

    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

    cfg = OmegaConf.create(cfg)

    obs_sizes = cfg.network_config.get("obs_sizes", None)
    if obs_sizes is None:
        raise ValueError(
            "Checkpoint does not have obs_sizes in config. "
            "Only dict normalizer format is supported."
        )

    abstract_policy = _create_abstract_prior_policy(cfg)

    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        policy_params = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]

    normalizer_params, network_params = policy_params

    prior_params = network_params["params"]["prior"]
    decoder_params = network_params["params"]["decoder"]

    return (
        prior_params,
        decoder_params,
        normalizer_params,
        OmegaConf.to_container(cfg),
    )


def _create_abstract_prior_policy(cfg: OmegaConf) -> Tuple[Any, Any]:
    """Create abstract policy structure for checkpoint restoration.

    Args:
        cfg: Configuration from checkpoint (must have obs_sizes).

    Returns:
        Tuple of (normalizer_state, combined_policy_params) with correct structure.
    """
    latent_size = cfg.network_config.intention_size
    action_size = cfg.network_config.action_size
    obs_sizes = dict(cfg.network_config.obs_sizes)

    reference_obs_size = obs_sizes["imitation_target"]
    proprioceptive_obs_size = obs_sizes["proprioception"]

    encoder_hidden_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    prior_hidden_layer_sizes = tuple(
        cfg.network_config.get("prior_layer_sizes", [1024, 1024])
    )

    combined_params = _init_network_params(
        latent_size,
        action_size,
        reference_obs_size,
        proprioceptive_obs_size,
        encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes,
        prior_hidden_layer_sizes,
    )

    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(reference_obs_size, jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(proprioceptive_obs_size, jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, combined_params)


def _init_network_params(
    latent_size: int,
    action_size: int,
    reference_obs_size: int,
    proprioceptive_obs_size: int,
    encoder_hidden_layer_sizes: tuple,
    decoder_hidden_layer_sizes: tuple,
    prior_hidden_layer_sizes: tuple,
) -> Dict:
    """Initialize network parameters for checkpoint structure matching."""
    encoder_module = intention_network.Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )

    action_param_size = action_size * 2
    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [action_param_size],
    )

    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    key = jax.random.PRNGKey(0)
    key_enc, key_dec, key_prior = jax.random.split(key, 3)

    dummy_traj_obs = jnp.zeros((1, reference_obs_size))
    dummy_proprio_obs = jnp.zeros((1, proprioceptive_obs_size))
    dummy_decoder_input = jnp.zeros((1, latent_size + proprioceptive_obs_size))

    encoder_init = encoder_module.init(key_enc, dummy_traj_obs)
    decoder_init = decoder_module.init(key_dec, dummy_decoder_input)
    prior_init = prior_module.init(key_prior, dummy_proprio_obs)

    return {
        "params": {
            "encoder": encoder_init["params"],
            "decoder": decoder_init["params"],
            "prior": prior_init["params"],
        }
    }


def make_decoder_inference_fn(
    decoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    config: Dict,
) -> Callable:
    """Create decoder inference function for DecoderHighLevelWrapper.

    The decoder takes [latent, proprio] concatenated input and outputs action.

    Args:
        decoder_params: Frozen decoder parameters.
        normalizer_params: Dict normalizer with proprioception stats.
        config: Checkpoint config dict.

    Returns:
        Function (latent_proprio) -> (action, extras)
        where latent_proprio is [latent, proprio] concatenated.
    """
    action_size = config["network_config"]["action_size"]
    latent_size = config["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = tuple(config["network_config"]["decoder_layer_sizes"])

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
    )

    proprio_normalizer = normalizer_params.proprioception

    def decoder_fn(latent_proprio: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
        """Apply decoder to latent + proprio input.

        Args:
            latent_proprio: Concatenated [latent, proprio] array.

        Returns:
            Tuple of (action, extras_dict).
        """
        latent = latent_proprio[..., :latent_size]
        proprio = latent_proprio[..., latent_size:]

        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        decoder_input = jnp.concatenate([latent, normalized_proprio], axis=-1)

        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)

        action = parametric_action_distribution.mode(logits)

        return action, {"logits": logits}

    return decoder_fn


def make_prior_inference_fn(
    prior_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    config: Dict,
) -> Callable:
    """Create prior inference function for PriorDecoderHighLevelWrapper.

    The prior takes proprio input and outputs latent distribution parameters.

    Args:
        prior_params: Frozen prior parameters.
        normalizer_params: Dict normalizer with proprioception stats.
        config: Checkpoint config dict.

    Returns:
        Function (proprio) -> (mean, logvar)
    """
    latent_size = config["network_config"]["intention_size"]
    prior_hidden_layer_sizes = tuple(
        config["network_config"].get("prior_layer_sizes", [1024, 1024])
    )

    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    proprio_normalizer = normalizer_params.proprioception

    def prior_fn(proprio: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply prior to proprioceptive observations.

        Args:
            proprio: Proprioceptive observations.

        Returns:
            Tuple of (mean, logvar) for latent distribution.
        """
        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        mean, logvar = prior_module.apply({"params": prior_params}, normalized_proprio)

        return mean, logvar

    return prior_fn
