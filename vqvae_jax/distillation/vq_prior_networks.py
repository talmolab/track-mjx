"""VQ-VAE Prior network definitions for prior distillation.

This module provides the Prior network that predicts encoder embeddings
from proprioceptive observations only. Unlike the VAE Prior which outputs
(mean, logvar) for sampling, this Prior outputs a single continuous
embedding z_p that should match the encoder output z_e.

Key components:
- VQPrior: MLP that maps proprio -> z_p (continuous embedding)
- make_vq_prior_network: Factory function for creating Prior networks
- make_prior_inference_fn: Creates inference functions for freeloop evaluation

Reference: VQ-VAE encoder uses traj -> z_e, Prior uses proprio -> z_p
The training objective is MSE(z_p, z_e) with encoder frozen.
"""

from collections.abc import Callable, Sequence
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.training import networks, types
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey

from track_mjx.agent.observation_utils import flatten_obs_dict


class VQPrior(nn.Module):
    """Prior network that predicts encoder embeddings from proprioception only.

    This network learns to predict what the VQ-VAE encoder would output (z_e)
    given only the proprioceptive state. During freeloop evaluation, the prior
    replaces the encoder to generate actions without reference trajectories.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latent_dim: Dimension of the output embedding (must match VQ-VAE).
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        use_layer_norm: Whether to use LayerNorm after each hidden layer.
    """

    layer_sizes: Sequence[int] = (1024, 1024)
    latent_dim: int = 60
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, proprio: jnp.ndarray) -> jnp.ndarray:
        """Predict latent embedding from proprioceptive observations.

        Args:
            proprio: Proprioceptive observations [..., proprio_dim].

        Returns:
            z_p: Predicted latent embedding [..., latent_dim].
        """
        x = proprio

        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"ln_{i}")(x)

        # Project to latent dimension (no activation on output)
        z_p = nn.Dense(
            self.latent_dim,
            name="latent_projection",
            kernel_init=self.kernel_init,
        )(x)

        return z_p


@flax.struct.dataclass
class VQPriorNetworks:
    """Container for VQ-VAE prior distillation network components.

    Attributes:
        prior_network: The Prior network that predicts z_e from proprio.
        latent_dim: Dimension of the latent embedding space.
        proprio_size: Dimension of proprioceptive observations.
    """

    prior_network: networks.FeedForwardNetwork
    latent_dim: int
    proprio_size: int


def make_vq_prior_network(
    proprio_size: int,
    latent_dim: int,
    layer_sizes: Sequence[int] = (1024, 1024),
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
) -> networks.FeedForwardNetwork:
    """Create a VQ-VAE Prior network.

    The Prior network takes proprioceptive observations and outputs
    a continuous embedding z_p that should match the encoder's z_e.

    Args:
        proprio_size: Dimension of proprioceptive observations.
        latent_dim: Dimension of the latent/codebook embedding space.
        layer_sizes: Hidden layer sizes for the Prior MLP.
        preprocess_observations_fn: Observation preprocessing function.

    Returns:
        FeedForwardNetwork with init and apply methods.
    """
    prior_module = VQPrior(
        layer_sizes=list(layer_sizes),
        latent_dim=latent_dim,
    )

    def apply(
        params,
        normalizer_params,
        proprio: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply Prior network with observation normalization.

        Args:
            params: Prior network weights.
            normalizer_params: Normalization statistics for proprio.
            proprio: Proprioceptive observations [..., proprio_size].

        Returns:
            z_p: Predicted latent embedding [..., latent_dim].
        """
        # Normalize proprioceptive observations
        # Note: We only normalize the proprio part, not the full observation
        proprio_normalized = preprocess_observations_fn(proprio, normalizer_params)
        return prior_module.apply(params, proprio_normalized)

    dummy_proprio = jnp.zeros((1, proprio_size))

    return networks.FeedForwardNetwork(
        init=lambda key: prior_module.init(key, dummy_proprio),
        apply=apply,
    )


def make_vq_prior_networks(
    proprio_size: int,
    latent_dim: int,
    layer_sizes: Sequence[int] = (1024, 1024),
    normalize_observations: bool = True,
) -> VQPriorNetworks:
    """Create VQ-VAE Prior networks container.

    Args:
        proprio_size: Dimension of proprioceptive observations.
        latent_dim: Dimension of the latent embedding space.
        layer_sizes: Hidden layer sizes for the Prior MLP.
        normalize_observations: Whether to normalize observations.

    Returns:
        VQPriorNetworks containing the prior network.
    """
    normalize_fn: Callable = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize

    prior_network = make_vq_prior_network(
        proprio_size=proprio_size,
        latent_dim=latent_dim,
        layer_sizes=layer_sizes,
        preprocess_observations_fn=normalize_fn,
    )

    return VQPriorNetworks(
        prior_network=prior_network,
        latent_dim=latent_dim,
        proprio_size=proprio_size,
    )


def make_prior_inference_fn(
    prior_networks: VQPriorNetworks,
) -> Callable[..., Callable]:
    """Create a prior inference function factory for freeloop evaluation.

    The returned function creates inference functions that:
    1. Take proprioceptive observations
    2. Output predicted embeddings z_p

    Args:
        prior_networks: VQPriorNetworks container.

    Returns:
        A make_prior_fn with signature:
            make_prior_fn(params) -> prior_fn
        where prior_fn has signature:
            (proprio, key) -> z_p
    """

    def make_prior_fn(
        params: tuple[Any, Any],  # (normalizer_params, prior_params)
    ) -> Callable:
        """Create a prior function with fixed parameters.

        Args:
            params: Tuple of (normalizer_params, prior_params).

        Returns:
            Prior function: (proprio, key) -> z_p.
        """
        normalizer_params, prior_params = params
        prior_network = prior_networks.prior_network

        def prior_fn(
            proprio: types.Observation,
            key: PRNGKey,  # Unused, for API compatibility
        ) -> jnp.ndarray:
            """Predict latent embedding from proprioceptive observation.

            Args:
                proprio: Proprioceptive observations [..., proprio_size].
                key: JAX random key (unused, VQ is deterministic).

            Returns:
                z_p: Predicted latent embedding [..., latent_dim].
            """
            # Prior only uses proprioception normalizer
            proprio_normalizer = (
                normalizer_params.proprioception
                if hasattr(normalizer_params, "proprioception")
                else normalizer_params
            )
            return prior_network.apply(prior_params, proprio_normalizer, proprio)

        return prior_fn

    return make_prior_fn


def make_freeloop_policy_fn(
    prior_params: tuple[Any, Any],  # (normalizer_params, prior_params)
    decoder_params: dict[str, Any],
    codebook: jnp.ndarray,
    prior_network: networks.FeedForwardNetwork,
    decoder_module: Any,
    parametric_action_distribution: Any,
    reference_obs_size: int,
    quantize_prior: bool = True,
    deterministic: bool = True,
) -> Callable:
    """Create a freeloop policy function using Prior + Decoder.

    This creates the inference function for freeloop evaluation where:
    1. Prior predicts z_p from proprio (no trajectory needed!)
    2. z_p is optionally quantized to z_q using the codebook
    3. Decoder generates action from z_q + proprio

    Args:
        prior_params: Tuple of (normalizer_params, prior_params).
        decoder_params: Frozen decoder parameters.
        codebook: Frozen VQ-VAE codebook [num_codes, latent_dim].
        prior_network: Prior network apply function.
        decoder_module: Decoder Flax module.
        parametric_action_distribution: Action distribution for sampling.
        reference_obs_size: Size of reference trajectory in observations.
        quantize_prior: Whether to quantize prior output to codebook.
        deterministic: Whether to use deterministic action (mode vs sample).

    Returns:
        Policy function: (obs, key) -> (action, extras)
    """
    normalizer_params, prior_params_inner = prior_params

    def policy(
        observations: types.Observation,
        key: PRNGKey,
    ) -> tuple[types.Action, types.Extra]:
        """Freeloop policy using Prior + Decoder.

        Args:
            observations: Dict observations {"imitation_target": ..., "proprioception": ...}.
            key: JAX random key for action sampling.

        Returns:
            Tuple of (action, extras_dict).
        """
        # Extract proprioceptive observations from dict (ignore trajectory part)
        flat_obs = flatten_obs_dict(observations)
        proprio = flat_obs["proprioception"]

        # Prior predicts z_p from proprio (only uses proprioception normalizer)
        proprio_normalizer = (
            normalizer_params.proprioception
            if hasattr(normalizer_params, "proprioception")
            else normalizer_params
        )
        z_p = prior_network.apply(prior_params_inner, proprio_normalizer, proprio)

        # Optionally quantize to nearest codebook entry
        if quantize_prior:
            # Compute distances to all codebook entries
            # z_p: [..., latent_dim], codebook: [K, latent_dim]
            z_p_expanded = jnp.expand_dims(z_p, axis=-2)  # [..., 1, latent_dim]
            distances = jnp.sum((z_p_expanded - codebook) ** 2, axis=-1)  # [..., K]
            indices = jnp.argmin(distances, axis=-1)  # [...]
            z_q = codebook[indices]  # [..., latent_dim]
        else:
            z_q = z_p
            indices = jnp.full(z_p.shape[:-1], -1, dtype=jnp.int32)

        # Decode to action
        decoder_input = jnp.concatenate([z_q, proprio], axis=-1)
        action_logits, _ = decoder_module.apply(
            {"params": decoder_params}, decoder_input
        )

        # Get action from distribution
        if deterministic:
            action = parametric_action_distribution.mode(action_logits)
        else:
            key, sample_key = jax.random.split(key)
            raw_action = parametric_action_distribution.sample_no_postprocessing(
                action_logits, sample_key
            )
            action = parametric_action_distribution.postprocess(raw_action)

        extras = {
            "z_p": z_p,
            "z_q": z_q,
            "indices": indices,
            "action_logits": action_logits,
        }

        return jnp.array(action), extras

    return policy
