"""PPO network definitions for KPMS decoder-only imitation learning.

Provides network factories, inference functions, and a value network that
are analogous to ``vqvae_jax/vq_ppo_networks.py`` but much simpler:

* No encoder or vector quantizer — codes come from Keypoint-MoSeq.
* The policy network embeds a discrete code and concatenates it with
  proprioception to produce action parameters.
* The value network uses normalized proprioception + imitation target
  (flattened) augmented with the code embedding.
"""

from collections.abc import Callable, Mapping, Sequence

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey

from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    normalize_dict_obs,
)

from moseq_decoder_network import MoSeqEncoderDecoderNetwork

# ---------------------------------------------------------------------------
# Network container
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class MoSeqPPONetworks:
    """Container for MoSeq decoder PPO network components.

    Attributes:
        policy_network: MoSeq decoder policy (FeedForwardNetwork-like).
        value_network: Value function network.
        parametric_action_distribution: Action distribution (NormalTanh).
        num_codes: Number of KPMS syllable codes.
        code_embed_dim: Dimensionality of code embedding.
    """

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    num_codes: int = 32
    code_embed_dim: int = 16


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------


def make_moseq_inference_fn(
    ppo_networks: MoSeqPPONetworks,
) -> Callable[..., types.Policy]:
    """Create a policy factory for inference with the MoSeq decoder.

    Returns:
        ``make_policy(params, deterministic, get_activation) -> policy_fn``
    """

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
        get_activation: bool = False,
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        action_dist = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra]:
            key_sample, key_net = jax.random.split(key_sample)

            action_params, code_idx, cont_mean, cont_logvar = policy_network.apply(
                *params, observations, key_net, deterministic=deterministic
            )

            extras = {"code_idx": code_idx}
            if cont_mean is not None:
                extras["cont_mean"] = cont_mean

            if deterministic:
                action = jnp.array(action_dist.mode(action_params))
                return action, extras

            raw_actions = action_dist.sample_no_postprocessing(
                action_params, key_sample
            )
            log_prob = action_dist.log_prob(action_params, raw_actions)
            action = action_dist.postprocess(raw_actions)

            extras["log_prob"] = log_prob
            extras["raw_action"] = raw_actions
            return jnp.array(action), extras

        return policy

    return make_policy


def make_moseq_logging_inference_fn(
    ppo_networks: MoSeqPPONetworks,
) -> Callable[[bool], Callable]:
    """Create a policy factory for logging/evaluation with explicit params.

    Returns:
        ``make_logging_policy(deterministic) -> logging_policy_fn``

        Where ``logging_policy_fn(params, obs, key, prev_indices) -> (action, extras)``
    """

    def make_logging_policy(deterministic: bool = False) -> Callable:
        policy_network = ppo_networks.policy_network
        action_dist = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            key_sample: PRNGKey,
            prev_indices=None,  # unused, kept for interface compat
        ) -> tuple[types.Action, types.Extra]:
            key_sample, key_net = jax.random.split(key_sample)

            action_params, code_idx, cont_mean, cont_logvar = policy_network.apply(
                *params, observations, key_net, deterministic=deterministic
            )

            extras = {"code_idx": code_idx, "indices": code_idx}
            if cont_mean is not None:
                extras["cont_mean"] = cont_mean

            if deterministic:
                action = jnp.array(action_dist.mode(action_params))
                return action, extras

            raw_actions = action_dist.sample_no_postprocessing(
                action_params, key_sample
            )
            log_prob = action_dist.log_prob(action_params, raw_actions)
            action = action_dist.postprocess(raw_actions)

            extras["log_prob"] = log_prob
            extras["raw_action"] = raw_actions
            return jnp.array(action), extras

        return logging_policy

    return make_logging_policy


# ---------------------------------------------------------------------------
# Policy wrapper (handles normalization + code passthrough)
# ---------------------------------------------------------------------------


def _make_moseq_policy_network(
    module: MoSeqEncoderDecoderNetwork,
    obs_sizes: Mapping[str, int],
) -> networks.FeedForwardNetwork:
    """Build a FeedForwardNetwork that normalizes obs and forwards to *module*.

    The wrapper:
    1. Saves the raw ``kpms_code`` from obs.
    2. Normalizes ``imitation_target`` and ``proprioception`` via
       ``normalize_dict_obs`` (which ignores unknown keys).
    3. Re-attaches the raw ``kpms_code`` to the normalized dict.
    4. Passes the combined dict to the Flax module.
    """
    dummy_obs = {k: jnp.zeros(v) for k, v in obs_sizes.items()}

    def init(key):
        return module.init(key, dummy_obs)

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key=None,
        deterministic: bool = False,
        **kwargs,
    ):
        # Save raw code before normalization
        raw_code = obs.get("kpms_code")

        # Normalize imitation_target + proprioception (ignores kpms_code)
        normalized = normalize_dict_obs(obs, processor_params)

        # Re-attach raw code
        if raw_code is not None:
            normalized["kpms_code"] = raw_code

        return module.apply(
            policy_params, normalized, key=key, deterministic=deterministic
        )

    return networks.FeedForwardNetwork(init=init, apply=apply)


# ---------------------------------------------------------------------------
# Value network (augmented with code embedding)
# ---------------------------------------------------------------------------


def make_moseq_value_network(
    obs_sizes: Mapping[str, int],
    num_codes: int,
    code_embed_dim: int,
    hidden_layer_sizes: Sequence[int] = (512, 512, 256, 256),
) -> networks.FeedForwardNetwork:
    """Value network: normalized flat obs + code embedding -> scalar.

    The code index is extracted from ``obs["kpms_code"]`` (raw float),
    embedded via a separate learned embedding table, and concatenated with
    the normalized flat observation.
    """
    # Obs sizes without kpms_code (only imitation_target + proprioception)
    core_obs_size = sum(v for k, v in obs_sizes.items() if k != "kpms_code")
    augmented_input_size = core_obs_size + code_embed_dim

    base_value = networks.make_value_network(
        augmented_input_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    # Separate embedding table for value network (not shared with policy)
    import flax.linen as nn

    value_embed = nn.Embed(num_embeddings=num_codes, features=code_embed_dim)
    embed_params = value_embed.init(jax.random.PRNGKey(0), jnp.zeros((), jnp.int32))

    def init(key):
        base_params = base_value.init(key)
        return {**base_params, "value_embed": embed_params}

    def apply(
        processor_params: DictRunningStatisticsState,
        value_params,
        obs: Mapping[str, jnp.ndarray],
        **kwargs,
    ):
        # Save raw code
        raw_code = obs.get("kpms_code")

        # Normalize and flatten (ignores kpms_code)
        normalized = normalize_dict_obs(obs, processor_params)
        flat_obs = concat_flat_dict_obs(normalized)

        # Embed code
        if raw_code is not None:
            code_idx = jnp.round(raw_code[..., 0]).astype(jnp.int32)
        else:
            code_idx = jnp.zeros(flat_obs.shape[:-1], dtype=jnp.int32)

        code_emb = value_embed.apply(value_params["value_embed"], code_idx)
        augmented = jnp.concatenate([flat_obs, code_emb], axis=-1)

        # Forward through base MLP (processor_params already handled)
        base_params = {k: v for k, v in value_params.items() if k != "value_embed"}
        return base_value.apply((), base_params, augmented)

    return networks.FeedForwardNetwork(init=init, apply=apply)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_moseq_decoder_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    num_codes: int = 32,
    code_embed_dim: int = 16,
    decoder_hidden_layer_sizes: tuple[int, ...] = (512, 512, 256, 256),
    value_hidden_layer_sizes: tuple[int, ...] = (512, 512, 256, 256),
    use_continuous_encoder: bool = False,
    encoder_layer_sizes: tuple[int, ...] = (256, 128),
    continuous_latent_dim: int = 16,
) -> MoSeqPPONetworks:
    """Create MoSeq decoder PPO networks for imitation learning.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        action_size: Action dimension.
        num_codes: Number of KPMS syllable codes.
        code_embed_dim: Code embedding dimension.
        decoder_hidden_layer_sizes: Decoder MLP layer sizes.
        value_hidden_layer_sizes: Value MLP layer sizes.
        use_continuous_encoder: Whether to add a continuous encoder.
        encoder_layer_sizes: Encoder MLP layer sizes.
        continuous_latent_dim: Continuous latent dimension (= code_embed_dim).

    Returns:
        MoSeqPPONetworks containing policy, value, and action distribution.
    """
    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    module = MoSeqEncoderDecoderNetwork(
        num_codes=num_codes,
        code_embed_dim=code_embed_dim,
        decoder_layer_sizes=decoder_hidden_layer_sizes,
        action_param_size=action_dist.param_size,
        use_continuous_encoder=use_continuous_encoder,
        encoder_layer_sizes=encoder_layer_sizes,
        continuous_latent_dim=continuous_latent_dim,
    )

    # Build the obs_sizes dict for init (include kpms_code)
    init_obs_sizes = {**obs_sizes}
    if "kpms_code" not in init_obs_sizes:
        init_obs_sizes["kpms_code"] = 1

    policy_network = _make_moseq_policy_network(module, init_obs_sizes)

    value_network = make_moseq_value_network(
        obs_sizes=obs_sizes,
        num_codes=num_codes,
        code_embed_dim=code_embed_dim,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return MoSeqPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=action_dist,
        num_codes=num_codes,
        code_embed_dim=code_embed_dim,
    )
