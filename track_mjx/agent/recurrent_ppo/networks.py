"""Recurrent PPO networks with VAE-style intention encoding.

This module provides network architectures for recurrent PPO training where
the policy uses an MLP encoder and RNN decoder. The encoder maps trajectory
observations to latent intentions, and the decoder uses an RNN to process
the latent intention along with proprioceptive observations.

Key components:
- RecurrentDecoder: RNN-based decoder with configurable cell type
- RecurrentIntentionNetwork: Full VAE with MLP encoder and RNN decoder
- Factory functions for creating recurrent intention PPO networks

Supported RNN cell types: SimpleCell, GRU, LSTM

Observations are expected as nested dictionaries:
    {
        'state': {'task_obs': ..., 'proprioception': ...},
        'privileged_state': {'task_obs': ..., 'proprioception': ...}
    }

The policy uses 'state' for both encoder and decoder.
The value network uses 'state' by default (configurable via value_obs_key).
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Union

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen as nn

from brax.training.acme import running_statistics

from track_mjx.agent.ff_ppo.intention_network import Encoder, reparameterize
from track_mjx.agent.networks import make_dict_value_network
from track_mjx.agent.observation_utils import normalizer_select

# Type aliases
RNNCellType = Literal["simple", "gru", "lstm"]
HiddenState = Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


_RNN_CELL_CLASSES: dict[RNNCellType, type[nn.RNNCellBase]] = {
    "simple": nn.SimpleCell,
    "gru": nn.GRUCell,
    "lstm": nn.LSTMCell,
}


def get_rnn_cell(
    cell_type: RNNCellType,
    hidden_size: int,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
) -> nn.RNNCellBase:
    """Get the appropriate RNN cell based on cell_type.

    Args:
        cell_type: Type of RNN cell ('simple', 'gru', 'lstm').
        hidden_size: Hidden size of the RNN cell.
        kernel_init: Kernel initializer for the cell weights.

    Returns:
        Flax RNN cell module.

    Raises:
        ValueError: If cell_type is not one of 'simple', 'gru', 'lstm'.
    """
    if cell_type not in _RNN_CELL_CLASSES:
        raise ValueError(
            f"Unsupported RNN cell type: {cell_type}. "
            f"Must be one of {list(_RNN_CELL_CLASSES.keys())}."
        )
    return _RNN_CELL_CLASSES[cell_type](features=hidden_size, kernel_init=kernel_init)


def init_hidden_state(
    cell_type: RNNCellType,
    hidden_size: int,
    batch_size: int,
) -> HiddenState:
    """Initialize hidden state for the given RNN cell type.

    Args:
        cell_type: Type of RNN cell.
        hidden_size: Hidden size of the RNN cell.
        batch_size: Batch size for the hidden state.

    Returns:
        Initialized hidden state (zeros). For LSTM, returns tuple of
        (cell_state, hidden_state). For SimpleCell/GRU, returns single array.
    """
    zeros = jnp.zeros((batch_size, hidden_size))
    return (zeros, zeros) if cell_type == "lstm" else zeros


def _reset_single_hidden(
    hidden: HiddenState,
    done_expanded: jnp.ndarray,
    cell_type: RNNCellType,
) -> HiddenState:
    """Reset a single hidden state where episodes ended."""
    if cell_type == "lstm":
        c, h = hidden
        return (
            jnp.where(done_expanded, 0.0, c),
            jnp.where(done_expanded, 0.0, h),
        )
    return jnp.where(done_expanded, 0.0, hidden)


def reset_hidden_on_done(
    hidden: HiddenState | list[HiddenState],
    done: jnp.ndarray,
    cell_type: RNNCellType,
) -> HiddenState | list[HiddenState]:
    """Reset hidden state to zeros where episodes ended.

    Args:
        hidden: Hidden state(s) - single state or list for multi-layer RNNs.
        done: Boolean array indicating episode termination, shape [batch].
        cell_type: Type of RNN cell.

    Returns:
        Hidden state with zeros where done is True.
    """
    done_expanded = done[..., None]
    if isinstance(hidden, list):
        return [_reset_single_hidden(h, done_expanded, cell_type) for h in hidden]
    return _reset_single_hidden(hidden, done_expanded, cell_type)


class RecurrentDecoder(nn.Module):
    """RNN-based decoder for recurrent intention policy.

    Processes concatenated [latent_z, proprioception] through stacked RNN cells
    to produce action distribution parameters.

    Attributes:
        output_size: Output dimension (action distribution param size).
        rnn_hidden_sizes: Hidden sizes for each RNN layer, e.g. [512, 256].
        cell_type: Type of RNN cell ('simple', 'gru', 'lstm').
        kernel_init: Weight initializer.
    """

    output_size: int
    rnn_hidden_sizes: Sequence[int] = (256,)
    cell_type: RNNCellType = "gru"
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    def setup(self):
        """Initialize RNN cells and final output layer."""
        # Create stacked RNN cells with different hidden sizes
        self.rnn_cells = [
            get_rnn_cell(self.cell_type, hidden_size, self.kernel_init)
            for hidden_size in self.rnn_hidden_sizes
        ]

        # Final output layer (no activation)
        self.final_layer = nn.Dense(
            self.output_size, name="final", kernel_init=self.kernel_init
        )

    @property
    def num_rnn_layers(self) -> int:
        """Number of RNN layers."""
        return len(self.rnn_hidden_sizes)

    def __call__(
        self,
        x: jnp.ndarray,
        hidden: HiddenState | list[HiddenState],
    ) -> tuple[jnp.ndarray, list[HiddenState]]:
        """Forward pass for single timestep.

        Args:
            x: Input tensor [batch, input_dim] (concatenated [z, proprioception]).
            hidden: Hidden state(s) from previous timestep. For single layer,
                can be a single HiddenState. For multiple layers, list of states.

        Returns:
            Tuple of (output [batch, output_size], new_hidden_states).
        """
        # Handle single vs multiple layer hidden states
        if self.num_rnn_layers == 1 and not isinstance(hidden, list):
            hidden_list = [hidden]
        else:
            hidden_list = hidden

        new_hidden_list = []
        rnn_input = x

        for i, (cell, h) in enumerate(zip(self.rnn_cells, hidden_list)):
            new_h, _ = cell(h, rnn_input)
            new_hidden_list.append(new_h)
            # For LSTM, new_h is the carry tuple (c, h); extract h for next layer input
            if self.cell_type == "lstm":
                rnn_input = new_h[1]  # h from (c, h) carry tuple
            else:
                rnn_input = new_h

        # Final output layer (no activation)
        output = self.final_layer(rnn_input)

        return output, new_hidden_list


class RecurrentIntentionNetwork(nn.Module):
    """VAE-style policy with MLP encoder and RNN decoder.

    The network receives inner observation dicts (already selected and normalized
    by the outer apply function). The encoder processes obs['task_obs'] to produce
    latent intention distribution parameters (mean, logvar). The decoder uses an
    RNN to process the sampled latent along with obs['proprioception'].

    Attributes:
        output_size: Output dimension (action distribution param size).
        encoder_layers: Hidden layer sizes for encoder MLP.
        latents: Dimension of latent intention space.
        rnn_hidden_sizes: Hidden sizes for each RNN layer, e.g. [512, 256].
        cell_type: Type of RNN cell ('simple', 'gru', 'lstm').
    """

    output_size: int
    encoder_layers: Sequence[int]
    latents: int = 60
    rnn_hidden_sizes: Sequence[int] = (256,)
    cell_type: RNNCellType = "gru"

    def setup(self):
        """Initialize encoder and decoder submodules."""
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = RecurrentDecoder(
            output_size=self.output_size,
            rnn_hidden_sizes=self.rnn_hidden_sizes,
            cell_type=self.cell_type,
        )

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        hidden: HiddenState | list[HiddenState],
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[HiddenState]]:
        """Forward pass for single timestep.

        Args:
            obs: Inner observation dict (already selected and normalized):
                {'task_obs': ..., 'proprioception': ...}
            hidden: RNN hidden state(s) from previous timestep.
            key: JAX random key for sampling. Either shape [2] (single key
                for all samples) or [batch_size, 2] (per-sample keys for
                deterministic replay).
            deterministic: If True, use latent mean instead of sampling.

        Returns:
            Tuple of (action_params, latent_mean, latent_logvar, new_hidden).
        """
        # Encode trajectory observations to latent distribution
        traj = obs["task_obs"]
        egocentric_obs = obs["proprioception"]

        # Check if observations are actually batched (based on obs shape)
        obs_is_batched = traj.ndim >= 2

        # Handle key splitting based on both key shape AND observation shape
        if key.ndim == 1:
            # Single key - split for encoder
            _, encoder_rng = jax.random.split(key)
        elif not obs_is_batched:
            # Per-sample keys but unbatched observation - use first key
            _, encoder_rng = jax.random.split(key[0])
        else:
            # Per-sample keys [batch_size, 2] - vmap split over batch
            _, encoder_rng = jax.vmap(jax.random.split)(key).swapaxes(0, 1)

        latent_mean, latent_logvar = self.encoder(traj, get_activation=False)

        # Sample or use mean
        if deterministic:
            z = latent_mean
        else:
            z = reparameterize(encoder_rng, latent_mean, latent_logvar)

        # Decode with RNN
        decoder_input = jnp.concatenate([z, egocentric_obs], axis=-1)
        action_params, new_hidden = self.decoder(decoder_input, hidden)

        return action_params, latent_mean, latent_logvar, new_hidden


@dataclasses.dataclass
class RecurrentNetwork:
    """Container for recurrent network functions.

    Attributes:
        init: Function to initialize network parameters.
        apply: Function for single timestep forward pass.
        apply_sequence: Function for sequence forward pass (using scan).
        init_hidden: Function to initialize hidden state for a batch.
    """

    init: Callable[..., Any]
    apply: Callable[..., Any]
    apply_sequence: Callable[..., Any]
    init_hidden: Callable[[int], HiddenState | list[HiddenState]]


@flax.struct.dataclass
class RecurrentPPONetworks:
    """Container for recurrent PPO network components.

    Attributes:
        policy_network: Recurrent intention policy network.
        value_network: Feedforward value network.
        parametric_action_distribution: Action distribution (NormalTanh).
        rnn_hidden_sizes: Hidden sizes for each RNN layer, e.g. (512, 256).
        cell_type: RNN cell type ('simple', 'gru', 'lstm').
    """

    policy_network: RecurrentNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    rnn_hidden_sizes: tuple[int, ...]
    cell_type: RNNCellType


def make_inference_fn(
    recurrent_ppo_networks: RecurrentPPONetworks,
) -> Callable[..., types.Policy]:
    """Create a policy factory function for recurrent inference.

    Returns a function that creates policy functions with fixed parameters.
    The policy function maps (observations, hidden, key) to (actions, extras, new_hidden).

    Args:
        recurrent_ppo_networks: Recurrent PPO network components.

    Returns:
        A make_policy function with signature:
            make_policy(params, deterministic) -> policy_fn
    """

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> Callable:
        """Create a recurrent policy function with fixed parameters.

        Args:
            params: Tuple of (normalizer_params, policy_params).
            deterministic: If True, return mode of action distribution.

        Returns:
            Policy function: (obs, hidden, key) -> (action, extras, new_hidden).
        """
        policy_network = recurrent_ppo_networks.policy_network
        parametric_action_distribution = (
            recurrent_ppo_networks.parametric_action_distribution
        )

        def policy(
            observations: types.Observation,
            hidden: HiddenState | list[HiddenState],
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra, HiddenState | list[HiddenState]]:
            key_sample, key_network = jax.random.split(key_sample)

            # Check if observations are batched (ndim >= 2) or unbatched (ndim == 1)
            # Get first leaf array from nested observation structure
            obs_leaves = jax.tree_util.tree_leaves(observations)
            obs_leaf = obs_leaves[0]
            if obs_leaf.ndim >= 2:
                # Batched observations - generate per-sample keys for deterministic replay
                batch_size = obs_leaf.shape[0]
                per_sample_keys = jax.random.split(key_network, batch_size)
            else:
                # Unbatched observation - use single key
                per_sample_keys = key_network

            logits, latent_mean, latent_logvar, new_hidden = policy_network.apply(
                *params,
                observations,
                hidden,
                per_sample_keys,
                deterministic=deterministic,
            )

            if deterministic:
                action = jnp.array(parametric_action_distribution.mode(logits))
                extras = {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                }
                return action, extras, new_hidden

            # Sample action from distribution
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return (
                jnp.array(postprocessed_actions),
                {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                    "logits": logits,
                    "policy_rng": per_sample_keys,
                },
                new_hidden,
            )

        return policy

    return make_policy


def make_logging_inference_fn(
    recurrent_ppo_networks: RecurrentPPONetworks,
) -> Callable[[bool], Callable]:
    """Create a logging policy factory for recurrent inference with explicit params.

    Unlike make_inference_fn, the returned policy takes params as an argument,
    allowing evaluation with different parameter sets without recreating the policy.

    Args:
        recurrent_ppo_networks: Recurrent PPO network components.

    Returns:
        A make_logging_policy function with signature:
            make_logging_policy(deterministic) -> logging_policy_fn

        Where logging_policy_fn has signature:
            (params, obs, hidden, key) -> (action, extras, new_hidden)
    """

    def make_logging_policy(deterministic: bool = False) -> Callable:
        """Create a logging policy that takes params as input.

        Args:
            deterministic: If True, return mode of action distribution.

        Returns:
            Policy function: (params, obs, hidden, key) -> (action, extras, new_hidden).
        """
        policy_network = recurrent_ppo_networks.policy_network
        parametric_action_distribution = (
            recurrent_ppo_networks.parametric_action_distribution
        )

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            hidden: HiddenState | list[HiddenState],
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra, HiddenState | list[HiddenState]]:
            key_sample, key_network = jax.random.split(key_sample)

            logits, latent_mean, latent_logvar, new_hidden = policy_network.apply(
                *params, observations, hidden, key_network, deterministic=deterministic
            )

            if deterministic:
                action = parametric_action_distribution.mode(logits)
            else:
                action = parametric_action_distribution.sample(logits, key_sample)

            extras = {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
            }

            return jnp.array(action), extras, new_hidden

        return logging_policy

    return make_logging_policy


def make_recurrent_intention_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    rnn_type: RNNCellType = "gru",
    rnn_hidden_sizes: Sequence[int] = (256,),
    value_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
) -> RecurrentPPONetworks:
    """Create recurrent intention-based PPO networks.

    Creates an encoder-decoder policy network where the encoder is an MLP
    that processes obs[policy_obs_key]['task_obs'], and the decoder is an
    RNN that processes the latent intention along with obs[policy_obs_key]['proprioception'].

    The value network uses obs[value_obs_key].

    Args:
        obs_sizes: Dict with 'task_obs' and 'proprioception' sizes.
        action_size: Action dimension.
        intention_latent_size: Dimension of VAE latent space.
        encoder_hidden_layer_sizes: MLP layer sizes for encoder.
        rnn_type: Type of RNN cell ('simple', 'gru', 'lstm').
        rnn_hidden_sizes: Hidden sizes for each RNN layer, e.g. (512, 256).
        value_hidden_layer_sizes: MLP layer sizes for value network.
        policy_obs_key: Top-level observation key for policy (default: 'state').
        value_obs_key: Top-level observation key for value network (default: 'state').

    Returns:
        RecurrentPPONetworks containing policy, value, and action distribution.
    """
    rnn_hidden_sizes = tuple(rnn_hidden_sizes)

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_module = RecurrentIntentionNetwork(
        output_size=parametric_action_distribution.param_size,
        encoder_layers=list(encoder_hidden_layer_sizes),
        latents=intention_latent_size,
        rnn_hidden_sizes=rnn_hidden_sizes,
        cell_type=rnn_type,
    )

    def policy_apply(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, Mapping[str, jnp.ndarray]],
        hidden: HiddenState | list[HiddenState],
        key: jax.Array,
        deterministic: bool = False,
    ):
        """Apply policy with observation normalization."""
        policy_normalizer = normalizer_select(processor_params, policy_obs_key)
        normalized_obs = running_statistics.normalize(
            obs[policy_obs_key], policy_normalizer
        )
        return policy_module.apply(
            policy_params,
            obs=normalized_obs,
            hidden=hidden,
            key=key,
            deterministic=deterministic,
        )

    def policy_apply_sequence(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs_seq: Mapping[str, Mapping[str, jnp.ndarray]],
        initial_hidden: HiddenState | list[HiddenState],
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        stored_keys: jax.Array | None = None,
    ):
        """Apply policy over sequence with hidden state reset on done.

        Args:
            processor_params: Normalizer parameters (pytree structure).
            policy_params: Policy network parameters.
            obs_seq: Nested observations with shape [T, B, ...] for inner arrays.
            initial_hidden: Initial hidden state(s).
            done_seq: Done flags with shape [T, B].
            key: Random key (used only if stored_keys is None).
            deterministic: If True, use latent mean instead of sampling.
            stored_keys: Optional pre-stored RNG keys with shape [T, B, 2].
                If provided, uses these for deterministic replay of stochastic
                layers instead of generating fresh keys.

        Returns:
            Tuple of (logits, latent_mean, latent_logvar, final_hidden).
            Each output has shape [T, B, ...].
        """
        policy_normalizer = normalizer_select(processor_params, policy_obs_key)
        obs_seq_normalized = running_statistics.normalize(
            obs_seq[policy_obs_key], policy_normalizer
        )

        # Validate stored_keys shape if provided
        if stored_keys is not None:
            # Get expected shape from observations [T, B, ...]
            ref_obs = obs_seq_normalized["task_obs"]
            expected_shape = (ref_obs.shape[0], ref_obs.shape[1], 2)
            if stored_keys.shape != expected_shape:
                raise ValueError(
                    f"stored_keys has shape {stored_keys.shape}, expected {expected_shape}. "
                    f"stored_keys must have shape [T, B, 2] matching the observation sequence."
                )

        if stored_keys is not None:
            # Deterministic replay: use stored per-timestep, per-sample keys
            # Pass observations directly to scan (not via closure) for memory efficiency
            def scan_step_stored(carry, inputs):
                hidden = carry
                obs_t, done_t, keys_t = inputs

                logits, mean, logvar, new_hidden = policy_module.apply(
                    policy_params,
                    obs=obs_t,
                    hidden=hidden,
                    key=keys_t,
                    deterministic=deterministic,
                )
                new_hidden = reset_hidden_on_done(new_hidden, done_t, rnn_type)
                return new_hidden, (logits, mean, logvar)

            final_hidden, (logits, means, logvars) = jax.lax.scan(
                scan_step_stored,
                initial_hidden,
                (obs_seq_normalized, done_seq, stored_keys),
            )
        else:
            # Standard path: generate fresh keys at each timestep
            # Pass observations directly to scan (not via closure) for memory efficiency
            def scan_step_fresh(carry, inputs):
                hidden, step_key = carry
                step_key, next_key = jax.random.split(step_key)
                obs_t, done_t = inputs

                logits, mean, logvar, new_hidden = policy_module.apply(
                    policy_params,
                    obs=obs_t,
                    hidden=hidden,
                    key=step_key,
                    deterministic=deterministic,
                )
                new_hidden = reset_hidden_on_done(new_hidden, done_t, rnn_type)
                return (new_hidden, next_key), (logits, mean, logvar)

            (final_hidden, _), (logits, means, logvars) = jax.lax.scan(
                scan_step_fresh, (initial_hidden, key), (obs_seq_normalized, done_seq)
            )

        return logits, means, logvars, final_hidden

    # Create dummy inner observations for initialization (already selected/normalized)
    dummy_obs = {
        "task_obs": jnp.zeros((1, obs_sizes["task_obs"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    dummy_key = jax.random.PRNGKey(0)

    def policy_init(key):
        dummy_hidden = [
            init_hidden_state(rnn_type, hidden_size, 1)
            for hidden_size in rnn_hidden_sizes
        ]
        return policy_module.init(key, dummy_obs, dummy_hidden, dummy_key)

    def policy_init_hidden(batch_size: int) -> list[HiddenState]:
        return [
            init_hidden_state(rnn_type, hidden_size, batch_size)
            for hidden_size in rnn_hidden_sizes
        ]

    policy_network = RecurrentNetwork(
        init=policy_init,
        apply=policy_apply,
        apply_sequence=policy_apply_sequence,
        init_hidden=policy_init_hidden,
    )

    value_network = make_dict_value_network(
        obs_sizes=obs_sizes,
        hidden_layer_sizes=value_hidden_layer_sizes,
        value_obs_key=value_obs_key,
    )

    return RecurrentPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        rnn_hidden_sizes=rnn_hidden_sizes,
        cell_type=rnn_type,
    )
