"""Recurrent binocular vision PPO networks with shared CNN+GRU backbone.

This module provides a recurrent binocular vision policy for training from
stereo egocentric camera images using a shared Siamese-CNN+GRU backbone with
separate policy and value heads. It is designed to be compatible with the
recurrent PPO training infrastructure in this package.

The key difference from the monocular ``RecurrentSharedVisionModule`` is the
use of ``BinocularVisionEncoder`` (Siamese CNN for stereo vision) instead of
``VisionEncoder``. This produces 2*cnn_feature_size features from the
channel-stacked binocular input, enabling binocular disparity and motion
parallax depth estimation through temporal integration in the GRU.

Architecture::

    binocular_vision (H,W,2C) --> [BinocularVisionEncoder] --> bino_features [2*feat_size]
    [bino_features, imitation_target] --> [shared GRUCell] --> gru_output
                            |
            +---------------+---------------+
            |                               |
      Policy head                     Value head
      gru_output --> MLP              [gru_output, task_obs] --> MLP
            |                               |
      action_params                   scalar value

All parameters live in a single ``RecurrentBinocularSharedVisionModule``.
In the ``RecurrentPPONetworkParams`` dataclass, ``params.policy`` holds the
full module (CNN + GRU + policy head + value head) and ``params.value`` is
empty.

Key design decisions:
- The Siamese CNN and GRU are shared between policy and value, so both losses
  flow gradients through the full backbone.
- The value head receives additional task observation input (imitation_target)
  concatenated with the GRU output for richer state estimation.
- The ``apply`` method returns ``(logits, dummy_latent_mean, dummy_latent_logvar,
  new_hidden)`` for interface compatibility with the existing recurrent PPO
  training infrastructure that expects VAE-style outputs.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from brax.training import distribution, networks
from flax import linen as nn

from track_mjx.agent.ff_ppo.binocular_vision_encoder import BinocularVisionEncoder
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)
from track_mjx.agent.recurrent_ppo.networks import (
    RecurrentNetwork,
    RecurrentPPONetworks,
    reset_hidden_on_done,
)


# ---------------------------------------------------------------------------
# Core Flax module
# ---------------------------------------------------------------------------


class RecurrentBinocularSharedVisionModule(nn.Module):
    """Shared Siamese-CNN+GRU backbone with policy and value heads.

    The BinocularVisionEncoder processes channel-stacked stereo images
    (left + right eye) through a Siamese CNN to produce compact binocular
    feature vectors. These features are concatenated with the imitation
    target (task observation) and fed through a GRU cell for temporal
    integration. The GRU output is then routed to two heads:

    - **Policy head**: MLP that maps GRU output to action distribution
      parameters.
    - **Value head**: MLP that maps [GRU output, task_obs] to a scalar
      value estimate. The task observation provides additional context
      for value estimation.

    Attributes:
        action_param_size: Output dimension for action distribution parameters
            (typically 2 * action_size for NormalTanh).
        gru_hidden_size: Dimensionality of the GRU hidden state.
        cnn_feature_size: Output dimension per eye of the BinocularVisionEncoder.
            Total CNN output is 2 * cnn_feature_size.
        cnn_channels: Channel sizes for each conv layer in the CNN encoder(s).
        mono_channels: Number of channels per eye (1 for grayscale, 3 for RGB).
        shared_weights: If True, use shared-weight Siamese CNN (default).
        policy_hidden_sizes: Hidden layer sizes for the policy MLP head.
        value_hidden_sizes: Hidden layer sizes for the value MLP head.
    """

    action_param_size: int
    gru_hidden_size: int = 256
    cnn_feature_size: int = 32
    cnn_channels: Sequence[int] = (4, 8, 16, 32)
    mono_channels: int = 1
    shared_weights: bool = True
    policy_hidden_sizes: Sequence[int] = (256,)
    value_hidden_sizes: Sequence[int] = (256, 128)

    def setup(self):
        """Initialize shared backbone and heads."""
        # --- Shared binocular CNN encoder ---
        self.vision_encoder = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=self.cnn_channels,
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )

        # --- Shared GRU cell ---
        self.gru_cell = nn.GRUCell(features=self.gru_hidden_size)

        # --- Policy head: GRU output -> MLP -> action params ---
        policy_layers = []
        policy_norms = []
        for h in self.policy_hidden_sizes:
            policy_layers.append(nn.Dense(h))
            policy_norms.append(nn.LayerNorm())
        self.policy_layers = policy_layers
        self.policy_norms = policy_norms
        self.policy_out = nn.Dense(self.action_param_size, name="policy_out")

        # --- Value head: [GRU output, task_obs] -> MLP -> scalar ---
        value_layers = []
        value_norms = []
        for h in self.value_hidden_sizes:
            value_layers.append(nn.Dense(h))
            value_norms.append(nn.LayerNorm())
        self.value_layers = value_layers
        self.value_norms = value_norms
        self.value_out = nn.Dense(1, name="value_out")

    # ------------------------------------------------------------------ #
    #  Forward passes                                                      #
    # ------------------------------------------------------------------ #

    def _shared_forward(
        self,
        obs: Mapping[str, jnp.ndarray],
        carry: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run shared binocular CNN + GRU backbone.

        Args:
            obs: Dict with "vision" and "imitation_target" keys.
            carry: GRU hidden state, shape [..., gru_hidden_size].

        Returns:
            Tuple of (gru_output, new_carry).
        """
        vision = obs["vision"]
        task_obs = obs["imitation_target"]

        # Binocular CNN encode vision (produces 2*cnn_feature_size features)
        vision_features = self.vision_encoder(vision)

        # Concatenate vision features + task observations
        gru_input = jnp.concatenate([vision_features, task_obs], axis=-1)

        # GRU temporal integration
        new_carry, gru_output = self.gru_cell(carry, gru_input)

        return gru_output, new_carry

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        carry: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Full forward: shared backbone -> policy output + value output.

        Args:
            obs: Dict with keys "vision", "imitation_target", "proprioception".
                - "vision": shape [..., H, W, 2*C] (channel-stacked binocular)
                - "imitation_target": shape [..., task_obs_dim]
                - "proprioception": shape [..., proprio_dim] (may be zeros)
            carry: GRU hidden state, shape [..., gru_hidden_size].

        Returns:
            Tuple of (action_params, value, new_carry).
            - action_params: shape [..., action_param_size]
            - value: shape [...]  (scalar per sample)
            - new_carry: shape [..., gru_hidden_size]
        """
        task_obs = obs["imitation_target"]

        gru_output, new_carry = self._shared_forward(obs, carry)

        # --- Policy head ---
        p = gru_output
        for dense, norm in zip(self.policy_layers, self.policy_norms):
            p = nn.silu(dense(p))
            p = norm(p)
        action_params = self.policy_out(p)

        # --- Value head (GRU output + task_obs for richer estimation) ---
        v_input = jnp.concatenate([gru_output, task_obs], axis=-1)
        v = v_input
        for dense, norm in zip(self.value_layers, self.value_norms):
            v = nn.silu(dense(v))
            v = norm(v)
        value = self.value_out(v)
        value = jnp.squeeze(value, axis=-1)

        return action_params, value, new_carry

    def policy_forward(
        self,
        obs: Mapping[str, jnp.ndarray],
        carry: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Policy-only forward (for inference, no value computation).

        Args:
            obs: Dict with keys "vision", "imitation_target", "proprioception".
            carry: GRU hidden state, shape [..., gru_hidden_size].

        Returns:
            Tuple of (action_params, new_carry).
        """
        gru_output, new_carry = self._shared_forward(obs, carry)

        # --- Policy head ---
        p = gru_output
        for dense, norm in zip(self.policy_layers, self.policy_norms):
            p = nn.silu(dense(p))
            p = norm(p)
        action_params = self.policy_out(p)

        return action_params, new_carry


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_recurrent_binocular_vision_highlvl_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    vision_shape: tuple[int, int, int],
    gru_hidden_size: int = 256,
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    policy_hidden_sizes: Sequence[int] = (256,),
    value_hidden_sizes: Sequence[int] = (256, 128),
) -> tuple[RecurrentPPONetworks, RecurrentBinocularSharedVisionModule]:
    """Create recurrent binocular vision PPO networks with shared CNN+GRU backbone.

    Constructs a ``RecurrentBinocularSharedVisionModule`` and wraps it as a
    ``RecurrentPPONetworks`` for compatibility with the recurrent PPO
    training pipeline.

    The policy network's ``apply`` returns
    ``(logits, dummy_latent_mean, dummy_latent_logvar, new_hidden)``
    for interface compatibility with the existing recurrent PPO loss
    functions that expect VAE-style outputs.

    The value network is a stub (the actual value computation happens
    inside the shared module during loss computation).

    Args:
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 100, "proprioception": 226, "vision": 12288}.
        action_size: Action dimension.
        vision_shape: Spatial shape of the binocular image (H, W, 2*C).
        gru_hidden_size: Dimensionality of the GRU hidden state.
        cnn_feature_size: Output dimension per eye of the BinocularVisionEncoder.
        cnn_channels: Channel sizes for each conv layer in the CNN encoder(s).
        mono_channels: Number of channels per eye (1 for grayscale, 3 for RGB).
        shared_weights: If True, use shared-weight Siamese architecture.
        policy_hidden_sizes: Hidden layer sizes for the policy MLP head.
        value_hidden_sizes: Hidden layer sizes for the value MLP head.

    Returns:
        Tuple of (RecurrentPPONetworks, RecurrentBinocularSharedVisionModule).
        The shared module is needed by the custom loss function.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    shared_module = RecurrentBinocularSharedVisionModule(
        action_param_size=parametric_action_distribution.param_size,
        gru_hidden_size=gru_hidden_size,
        cnn_feature_size=cnn_feature_size,
        cnn_channels=list(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        policy_hidden_sizes=list(policy_hidden_sizes),
        value_hidden_sizes=list(value_hidden_sizes),
    )

    # ------------------------------------------------------------------ #
    #  policy_apply: single-timestep forward pass                          #
    # ------------------------------------------------------------------ #

    def policy_apply(
        processor_params: DictRunningStatisticsState,
        policy_params: Any,
        obs: Mapping[str, jnp.ndarray],
        hidden: list[jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
    ):
        """Apply shared policy for a single timestep.

        Returns (logits, dummy_latent_mean, dummy_latent_logvar, new_hidden)
        for interface compatibility with the recurrent PPO infrastructure.

        The ``key`` and ``deterministic`` parameters are accepted for
        interface compatibility but unused (no stochastic encoder).
        """
        obs = normalize_dict_obs(obs, processor_params)
        carry = hidden[0]  # Single-layer GRU: hidden is [carry]

        action_params, new_carry = shared_module.apply(
            policy_params,
            obs=obs,
            carry=carry,
            method="policy_forward",
        )

        # Dummy latent outputs for interface compatibility
        dummy_latent_mean = jnp.zeros_like(action_params[..., :1])
        dummy_latent_logvar = jnp.full_like(dummy_latent_mean, -20.0)

        return action_params, dummy_latent_mean, dummy_latent_logvar, [new_carry]

    # ------------------------------------------------------------------ #
    #  policy_apply_sequence: scan over time dimension                      #
    # ------------------------------------------------------------------ #

    def policy_apply_sequence(
        processor_params: DictRunningStatisticsState,
        policy_params: Any,
        obs_seq: Mapping[str, jnp.ndarray],
        initial_hidden: list[jnp.ndarray],
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        stored_keys: jax.Array | None = None,
    ):
        """Apply policy over a time sequence using jax.lax.scan.

        Args:
            processor_params: Normalizer parameters.
            policy_params: Policy network parameters.
            obs_seq: Observations with shape [T, B, ...] for each key.
            initial_hidden: Initial hidden state, list of [carry] where
                carry has shape [B, gru_hidden_size].
            done_seq: Done flags with shape [T, B].
            key: Random key (unused but accepted for interface compat).
            deterministic: Unused but accepted for interface compat.
            stored_keys: Unused but accepted for interface compat.

        Returns:
            Tuple of (logits, dummy_means, dummy_logvars, final_hidden).
            Each output has shape [T, B, ...].
        """
        obs_seq = normalize_dict_obs(obs_seq, processor_params)
        initial_carry = initial_hidden[0]  # [B, gru_hidden_size]

        def step(carry, inputs):
            obs_t, done_t = inputs

            # Run single timestep through shared module (policy only)
            action_params, new_carry = shared_module.apply(
                policy_params,
                obs=obs_t,
                carry=carry,
                method="policy_forward",
            )

            # Reset hidden state where episodes ended
            new_carry = reset_hidden_on_done(new_carry, done_t, "gru")

            return new_carry, action_params

        # Build per-timestep obs dicts for scan
        # obs_seq has shape {key: [T, B, ...]} -- scan will slice along axis 0
        final_carry, logits_seq = jax.lax.scan(
            step, initial_carry, (obs_seq, done_seq)
        )

        # Dummy latent outputs: shape [T, B, 1]
        dummy_means = jnp.zeros(logits_seq.shape[:2] + (1,))
        dummy_logvars = jnp.full_like(dummy_means, -20.0)

        return logits_seq, dummy_means, dummy_logvars, [final_carry]

    # ------------------------------------------------------------------ #
    #  init + init_hidden                                                   #
    # ------------------------------------------------------------------ #

    # Dummy inputs for Flax parameter initialization
    dummy_obs = {
        "imitation_target": jnp.zeros(
            (1, obs_sizes.get("imitation_target", 0))
        ),
        "proprioception": jnp.zeros(
            (1, obs_sizes.get("proprioception", 0))
        ),
        "vision": jnp.zeros((1,) + vision_shape),
    }
    dummy_carry = jnp.zeros((1, gru_hidden_size))

    def policy_init(key: jax.Array):
        """Initialize all shared module parameters."""
        return shared_module.init(key, dummy_obs, dummy_carry)

    def policy_init_hidden(batch_size: int) -> list[jnp.ndarray]:
        """Create zero-initialized GRU hidden state.

        Returns:
            List containing a single carry array of shape
            [batch_size, gru_hidden_size].
        """
        return [jnp.zeros((batch_size, gru_hidden_size))]

    # ------------------------------------------------------------------ #
    #  Assemble RecurrentPPONetworks                                        #
    # ------------------------------------------------------------------ #

    policy_network = RecurrentNetwork(
        init=policy_init,
        apply=policy_apply,
        apply_sequence=policy_apply_sequence,
        init_hidden=policy_init_hidden,
    )

    # Stub value network (actual value computed in shared loss function)
    def value_apply(processor_params, value_params, obs):
        raise RuntimeError(
            "RecurrentBinocularSharedVision value stub should never be called "
            "directly. Use the custom shared-vision loss function instead."
        )

    value_network = networks.FeedForwardNetwork(
        init=lambda key: {},
        apply=value_apply,
    )

    recurrent_ppo_networks = RecurrentPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        rnn_hidden_sizes=(gru_hidden_size,),
        cell_type="gru",
    )

    return recurrent_ppo_networks, shared_module
