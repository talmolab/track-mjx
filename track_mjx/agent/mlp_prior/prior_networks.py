"""
Network definitions for prior training.

This module provides:
- Loading frozen encoder/decoder from ff_ppo checkpoint
- Creating trainable prior network
- Combining networks for checkpointing in distill-compatible format

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics, specs
from flax import linen as nn
from omegaconf import OmegaConf

from track_mjx.agent.ff_ppo import intention_network
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
from track_mjx.agent.ff_ppo import losses as ff_ppo_losses
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
    flatten_obs_dict,
)


class Prior(nn.Module):
    """Prior network that outputs distributions in latent space from proprioceptive observations.

    Takes only proprioceptive observations as input and outputs mean and log-variance
    of the latent distribution.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latents: Dimension of the latent intention space.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[Tuple[jnp.ndarray, jnp.ndarray], dict]:
        activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        if get_activation:
            activations["mean"] = mean_x
            activations["logvar"] = logvar_x
            return (mean_x, logvar_x), activations
        return mean_x, logvar_x


class Decoder(nn.Module):
    """Decode latent + proprioceptive observations to action distribution parameters.

    Takes concatenated [latent, proprioceptive_obs] as input and outputs action
    distribution parameters.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        activate_final: Whether to apply activation to final layer.
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Tuple[jnp.ndarray, dict]:
        activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
                if get_activation:
                    activations[f"layer_{i}"] = x
        if get_activation:
            return x, activations
        return x, {}


def load_frozen_encoder_decoder(
    checkpoint_path: str,
    step: int | None = None,
) -> Tuple[Dict, Dict, DictRunningStatisticsState, Dict]:
    """Load encoder and decoder parameters from ff_ppo checkpoint.

    Args:
        checkpoint_path: Path to the ff_ppo checkpoint directory.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (encoder_params, decoder_params, normalizer_params, config_dict)
        Note: normalizer_params is now a DictRunningStatisticsState.
    """
    # Load config
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix="PPONetwork")
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        # Load config
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

    cfg = OmegaConf.create(cfg)

    # Check if checkpoint uses dict observations (new format) or flat observations (legacy)
    obs_sizes = cfg.network_config.get("obs_sizes", None)
    if obs_sizes is not None:
        # New dict-based format
        obs_sizes = dict(obs_sizes)
        ppo_network = ff_ppo_networks.make_intention_ppo_networks(
            obs_sizes=obs_sizes,
            action_size=cfg.network_config.action_size,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        )

        key_policy, key_value = jax.random.split(jax.random.key(1))
        init_params = ff_ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )

        # Create abstract dict normalizer
        normalizer_state = DictRunningStatisticsState(
            imitation_target=running_statistics.init_state(
                specs.Array(obs_sizes["imitation_target"], jnp.dtype("float32"))
            ),
            proprioception=running_statistics.init_state(
                specs.Array(obs_sizes["proprioception"], jnp.dtype("float32"))
            ),
        )

        abstract_policy = (normalizer_state, init_params.policy)

        # Load actual policy
        with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
            policy_params = ckpt_mgr.restore(
                step,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardRestore(abstract_policy)
                ),
            )["policy"]

        normalizer_params, network_params = policy_params
    else:
        # Legacy flat format - need to convert
        from track_mjx.agent.observation_utils import convert_flat_to_dict_normalizer

        observation_size = cfg.network_config.observation_size
        reference_obs_size = cfg.network_config.reference_obs_size

        # Create abstract flat normalizer for loading
        normalizer_state = running_statistics.init_state(
            specs.Array(observation_size, jnp.dtype("float32"))
        )

        # Create legacy network for structure matching
        # Note: This uses an older signature that we need for legacy checkpoints
        ppo_network = ff_ppo_networks.make_intention_ppo_networks(
            obs_sizes={
                "imitation_target": reference_obs_size,
                "proprioception": observation_size - reference_obs_size,
            },
            action_size=cfg.network_config.action_size,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        )

        key_policy, key_value = jax.random.split(jax.random.key(1))
        init_params = ff_ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )

        abstract_policy = (normalizer_state, init_params.policy)

        # Load actual policy with flat normalizer
        with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
            policy_params = ckpt_mgr.restore(
                step,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardRestore(abstract_policy)
                ),
            )["policy"]

        flat_normalizer_params, network_params = policy_params

        # Convert flat normalizer to dict format
        normalizer_params = convert_flat_to_dict_normalizer(
            flat_normalizer_params, reference_obs_size
        )

        # Add obs_sizes to config for downstream use
        cfg.network_config.obs_sizes = {
            "imitation_target": reference_obs_size,
            "proprioception": observation_size - reference_obs_size,
        }

    # Extract encoder and decoder params
    encoder_params = network_params["params"]["encoder"]
    decoder_params = network_params["params"]["decoder"]

    return (
        encoder_params,
        decoder_params,
        normalizer_params,
        OmegaConf.to_container(cfg),
    )


def make_prior_networks(
    latent_size: int,
    proprioceptive_obs_size: int,
    prior_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> Tuple[Prior, networks.FeedForwardNetwork]:
    """Create the trainable prior network.

    Args:
        latent_size: Dimension of the latent intention space.
        proprioceptive_obs_size: Size of proprioceptive observations.
        prior_hidden_layer_sizes: Hidden layer sizes for prior MLP.

    Returns:
        Tuple of (prior_module, prior_feedforward_network)
    """
    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    def init_fn(key):
        dummy_obs = jnp.zeros((1, proprioceptive_obs_size))
        return prior_module.init(key, dummy_obs)

    def apply_fn(params, obs):
        return prior_module.apply(params, obs)

    return prior_module, networks.FeedForwardNetwork(init=init_fn, apply=apply_fn)


def make_encoder_apply_fn(
    encoder_hidden_layer_sizes: Sequence[int],
    latent_size: int,
    reference_obs_size: int,
) -> Callable:
    """Create encoder apply function.

    Args:
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder.
        latent_size: Dimension of the latent space.
        reference_obs_size: Size of reference trajectory observations.

    Returns:
        Function (params, obs, key) -> (mean, logvar)
    """
    encoder_module = intention_network.Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )

    def apply_fn(
        params: Dict, obs: jnp.ndarray, key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply encoder to trajectory observations.

        Args:
            params: Encoder parameters.
            obs: Trajectory observations [..., reference_obs_size].
            key: Random key (unused, for API consistency).

        Returns:
            Tuple of (mean, logvar) for latent distribution.
        """
        return encoder_module.apply({"params": params}, obs)

    return apply_fn


def make_encoder_decoder_inference_fn(
    encoder_params: Dict,
    decoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    encoder_hidden_layer_sizes: Sequence[int],
    decoder_hidden_layer_sizes: Sequence[int],
    latent_size: int,
    action_size: int,
    deterministic: bool = True,
) -> Callable:
    """Create frozen encoder+decoder policy for data collection.

    Args:
        encoder_params: Frozen encoder parameters.
        decoder_params: Frozen decoder parameters.
        normalizer_params: Dict observation normalizer parameters.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
        latent_size: Dimension of the latent space.
        action_size: Size of the action space.
        deterministic: If True, use mean of distributions (no sampling).

    Returns:
        Policy function (obs, key) -> (action, extras)
        obs can be either a dict {"imitation_target": ..., "proprioception": ...}
        or will be accessed via the flattening utilities if needed.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # Create encoder and decoder modules
    encoder_module = intention_network.Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )

    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
    )

    def policy_fn(
        obs: Mapping[str, jnp.ndarray], key: jax.Array
    ) -> Tuple[jnp.ndarray, Dict]:
        """Generate actions using frozen encoder+decoder.

        Args:
            obs: Dict observations with "imitation_target" and "proprioception" keys.
            key: Random key for sampling.

        Returns:
            Tuple of (action, extras_dict).
        """
        key_encoder, key_action = jax.random.split(key)

        # Flatten and normalize dict observations
        flat_obs = flatten_obs_dict(obs)
        normalized_obs = normalize_dict_obs(flat_obs, normalizer_params)

        # Access observations by key
        traj_obs = normalized_obs["imitation_target"]
        proprio_obs = normalized_obs["proprioception"]

        # Encode trajectory -> latent distribution
        latent_mean, latent_logvar = encoder_module.apply(
            {"params": encoder_params}, traj_obs
        )

        # Sample or use mean
        if deterministic:
            z = latent_mean
        else:
            z = intention_network.reparameterize(
                key_encoder, latent_mean, latent_logvar
            )

        # Decode to action
        decoder_input = jnp.concatenate([z, proprio_obs], axis=-1)
        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)

        # Sample action
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            raw_action = parametric_action_distribution.sample_no_postprocessing(
                logits, key_action
            )
            action = parametric_action_distribution.postprocess(raw_action)

        extras = {
            "latent_mean": latent_mean,
            "latent_logvar": latent_logvar,
            "intention": z,
            "logits": logits,
        }

        return action, extras

    return policy_fn


def create_combined_checkpoint_params(
    encoder_params: Dict,
    decoder_params: Dict,
    prior_params: Dict,
    normalizer_params: DictRunningStatisticsState,
) -> Tuple[DictRunningStatisticsState, Dict]:
    """Combine parameters into checkpoint-compatible format.

    Creates a checkpoint format compatible with prior_rollout_distill.ipynb:
    (normalizer_params, {"params": {"encoder": ..., "decoder": ..., "prior": ...}})

    Args:
        encoder_params: Encoder parameters (from ff_ppo).
        decoder_params: Decoder parameters (from ff_ppo).
        prior_params: Prior parameters (newly trained).
        normalizer_params: Dict observation normalizer parameters.

    Returns:
        Tuple of (normalizer_params, combined_network_params)
    """
    # Extract prior params from the nested structure if needed
    if "params" in prior_params:
        prior_params_inner = prior_params["params"]
    else:
        prior_params_inner = prior_params

    combined_network_params = {
        "params": {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "prior": prior_params_inner,
        }
    }

    return (normalizer_params, combined_network_params)


def create_abstract_prior_policy(
    cfg: Dict,
    prior_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> Tuple[Any, Any]:
    """Create abstract policy structure for checkpoint restoration.

    Creates the pytree structure needed for loading saved parameters.

    Args:
        cfg: Configuration dictionary from checkpoint.
        prior_hidden_layer_sizes: Hidden layer sizes for prior network.

    Returns:
        Tuple of (normalizer_state, combined_policy_params) with correct structure.
    """
    latent_size = cfg["network_config"]["intention_size"]
    action_size = cfg["network_config"]["action_size"]

    # Check if using dict observations (new format) or flat (legacy)
    obs_sizes = cfg["network_config"].get("obs_sizes", None)
    if obs_sizes is not None:
        reference_obs_size = obs_sizes["imitation_target"]
        proprioceptive_obs_size = obs_sizes["proprioception"]
    else:
        reference_obs_size = cfg["network_config"]["reference_obs_size"]
        observation_size = cfg["network_config"]["observation_size"]
        proprioceptive_obs_size = observation_size - reference_obs_size

    encoder_hidden_layer_sizes = tuple(cfg["network_config"]["encoder_layer_sizes"])
    decoder_hidden_layer_sizes = tuple(cfg["network_config"]["decoder_layer_sizes"])

    # Create encoder
    encoder_module = intention_network.Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )

    # Create decoder
    action_param_size = action_size * 2  # mean and std
    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [action_param_size],
    )

    # Create prior
    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    # Initialize with dummy inputs
    key = jax.random.PRNGKey(0)
    key_enc, key_dec, key_prior = jax.random.split(key, 3)

    dummy_traj_obs = jnp.zeros((1, reference_obs_size))
    dummy_proprio_obs = jnp.zeros((1, proprioceptive_obs_size))
    dummy_decoder_input = jnp.zeros((1, latent_size + proprioceptive_obs_size))

    encoder_init = encoder_module.init(key_enc, dummy_traj_obs)
    decoder_init = decoder_module.init(key_dec, dummy_decoder_input)
    prior_init = prior_module.init(key_prior, dummy_proprio_obs)

    # Combine into expected structure
    combined_params = {
        "params": {
            "encoder": encoder_init["params"],
            "decoder": decoder_init["params"],
            "prior": prior_init["params"],
        }
    }

    # Create dict normalizer state
    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(reference_obs_size, jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(proprioceptive_obs_size, jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, combined_params)
