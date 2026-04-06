"""RNN intention network and PPO wiring for MoSeq high-level transfer.

The high-level policy maps KPMS codes + proprioception to latent intentions.
A frozen pretrained decoder (external to this file) converts those intentions
into low-level motor commands.

Network architecture:
- ``MoSeqIntentionRNN``: Flax module — embeds KPMS codes, concatenates with
  proprioception, and runs through stacked GRU layers to produce intention
  parameters (mean + logstd for NormalTanh distribution).
- ``MoSeqHighLevelRecurrentNetwork``: Container wrapping init/apply/apply_sequence
  with observation normalization.
- ``MoSeqHighLevelPPONetworks``: Top-level container grouping the recurrent
  policy, feedforward value network, and action distribution.

Inference functions follow the same patterns as ``moseq_ppo_networks.py``.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey

from track_mjx.agent.observation_utils import normalizer_select
from moseq_ppo_networks import make_moseq_value_network


# ---------------------------------------------------------------------------
# Flax module
# ---------------------------------------------------------------------------


class MoSeqIntentionRNN(nn.Module):
    """RNN intention network: code embedding + proprio -> GRU -> intention.

    Attributes:
        num_codes: Number of KPMS syllable codes (embedding table rows).
        code_embed_dim: Dimensionality of the code embedding.
        rnn_hidden_sizes: Hidden sizes for stacked GRU layers.
        intention_param_size: Output dimension (2 * intention_dim for NormalTanh).
        activation: Activation function.
        kernel_init: Initializer for Dense layers.
        code_stack_size: Number of stacked codes in obs (already stacked).
    """

    num_codes: int = 32
    code_embed_dim: int = 16
    rnn_hidden_sizes: Sequence[int] = (256,)
    intention_param_size: int = 1
    activation: Callable = nn.silu
    kernel_init: Callable = nn.initializers.lecun_uniform()
    code_stack_size: int = 1

    def setup(self):
        self.code_embedding = nn.Embed(
            num_embeddings=self.num_codes,
            features=self.code_embed_dim,
        )
        self.rnn_cells = [
            nn.GRUCell(features=h, kernel_init=self.kernel_init)
            for h in self.rnn_hidden_sizes
        ]
        self.intention_head = nn.Dense(
            self.intention_param_size,
            kernel_init=self.kernel_init,
            name="intention_head",
        )

    def _decode_rnn(
        self,
        x: jnp.ndarray,
        hidden: list[jnp.ndarray],
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Run one timestep through the stacked GRU and intention head."""
        new_hidden = []
        rnn_input = x
        for cell, h in zip(self.rnn_cells, hidden):
            new_h, _ = cell(h, rnn_input)
            new_hidden.append(new_h)
            rnn_input = new_h
        return rnn_input, new_hidden

    def __call__(
        self,
        obs: dict[str, jnp.ndarray],
        hidden: list[jnp.ndarray],
        key=None,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
        """Single-timestep forward pass.

        Args:
            obs: Dict with ``kpms_code`` and ``proprioception``.
            hidden: List of GRU hidden states.
            key: PRNG key (unused, kept for interface compatibility).
            deterministic: Unused, kept for interface compatibility.

        Returns:
            ``(intention_params, code_idx, new_hidden)``.
        """
        kpms_code = obs["kpms_code"]
        code_idx = jnp.round(kpms_code[..., 0]).astype(jnp.int32)
        all_code_idx = jnp.round(kpms_code).astype(jnp.int32)
        all_emb = self.code_embedding(all_code_idx)  # [..., N, code_embed_dim]
        code_emb = all_emb.reshape(*all_emb.shape[:-2], -1)  # flatten stack

        proprio = obs["proprioception"]
        rnn_input = jnp.concatenate([code_emb, proprio], axis=-1)

        output, new_hidden = self._decode_rnn(rnn_input, hidden)
        intention_params = self.intention_head(output)
        return intention_params, code_idx, new_hidden

    def apply_sequence(
        self,
        obs_seq: dict[str, jnp.ndarray],
        initial_hidden: list[jnp.ndarray],
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        stored_keys: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Forward pass over a time sequence using jax.lax.scan.

        Args:
            obs_seq: Observations with shape ``[T, B, ...]`` per key.
            initial_hidden: Initial GRU hidden states (list of ``[B, H]``).
            done_seq: Episode-done flags ``[T, B]``.
            key: PRNG key (unused, kept for interface compatibility).
            deterministic: Unused, kept for interface compatibility.
            stored_keys: Unused, kept for interface compatibility.

        Returns:
            ``(logits, final_hidden)`` where logits is ``[T, B, intention_param_size]``.
        """

        def step_fn(carry, inputs):
            hidden_list = carry
            obs_t = {k: inputs[0][k] for k in inputs[0]}
            done_t = inputs[1]

            # Encode and run RNN
            kpms_code = obs_t["kpms_code"]
            all_code_idx = jnp.round(kpms_code).astype(jnp.int32)
            all_emb = self.code_embedding(all_code_idx)
            code_emb = all_emb.reshape(*all_emb.shape[:-2], -1)

            proprio = obs_t["proprioception"]
            rnn_input = jnp.concatenate([code_emb, proprio], axis=-1)

            output, new_hidden = self._decode_rnn(rnn_input, hidden_list)

            # Reset hidden on done
            done_expanded = done_t[..., None]
            new_hidden = [jnp.where(done_expanded, 0.0, h) for h in new_hidden]

            intention_params = self.intention_head(output)
            return new_hidden, intention_params

        final_hidden, logits = jax.lax.scan(
            step_fn, initial_hidden, (obs_seq, done_seq)
        )
        return logits, final_hidden


# ---------------------------------------------------------------------------
# Network containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MoSeqHighLevelRecurrentNetwork:
    """Container for recurrent high-level network functions.

    Attributes:
        init: ``(key) -> params``.
        apply: Single-step: ``(proc, pol, obs, hidden, key, det) -> (intention_params, code_idx, new_hidden)``.
        apply_sequence: Scan: ``(proc, pol, obs_seq, init_h, done, key, det, stored_keys) -> (logits, final_h)``.
        init_hidden: ``(batch_size) -> list[jnp.zeros]``.
    """

    init: Callable[..., Any]
    apply: Callable[..., Any]
    apply_sequence: Callable[..., Any]
    init_hidden: Callable[[int], list[jnp.ndarray]]


@dataclasses.dataclass
class MoSeqHighLevelPPONetworks:
    """Container for high-level PPO network components.

    Attributes:
        policy_network: Recurrent intention policy network.
        value_network: Feedforward value network.
        parametric_action_distribution: Action distribution (NormalTanh).
        num_codes: Number of KPMS syllable codes.
        code_embed_dim: Code embedding dimension.
        rnn_hidden_sizes: GRU hidden sizes per layer.
        intention_size: Dimension of the intention (action) space.
    """

    policy_network: MoSeqHighLevelRecurrentNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    num_codes: int = 32
    code_embed_dim: int = 16
    rnn_hidden_sizes: tuple[int, ...] = (256,)
    intention_size: int = 60


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def _normalize_and_reattach_code(
    obs: Mapping[str, Any],
    processor_params: running_statistics.RunningStatisticsState,
) -> dict[str, jnp.ndarray]:
    """Normalize obs["state"], preserve raw kpms_code."""
    state_obs = obs["state"]
    raw_code = state_obs.get("kpms_code")
    state_normalizer = normalizer_select(processor_params, "state")
    normalized = dict(running_statistics.normalize(state_obs, state_normalizer))
    if raw_code is not None:
        normalized["kpms_code"] = raw_code
    return normalized


# ---------------------------------------------------------------------------
# Policy wrapper (handles normalization + code passthrough)
# ---------------------------------------------------------------------------


def _make_moseq_highlvl_recurrent_policy_network(
    module: MoSeqIntentionRNN,
    obs_sizes: Mapping[str, int],
    rnn_hidden_sizes: tuple[int, ...],
) -> MoSeqHighLevelRecurrentNetwork:
    """Build a MoSeqHighLevelRecurrentNetwork wrapping the Flax RNN module."""
    dummy_obs = {k: jnp.zeros((1, v)) for k, v in obs_sizes.items()}
    dummy_hidden = [jnp.zeros((1, h)) for h in rnn_hidden_sizes]
    dummy_key = jax.random.PRNGKey(0)

    def init(key):
        return module.init(key, dummy_obs, dummy_hidden, dummy_key)

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        hidden: list[jnp.ndarray],
        key=None,
        deterministic: bool = False,
    ):
        normalized = _normalize_and_reattach_code(obs, processor_params)
        return module.apply(
            policy_params,
            normalized,
            hidden,
            key=key,
            deterministic=deterministic,
        )

    def apply_sequence(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs_seq: Mapping[str, jnp.ndarray],
        initial_hidden: list[jnp.ndarray],
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        stored_keys: jax.Array | None = None,
    ):
        normalized_seq = _normalize_and_reattach_code(obs_seq, processor_params)
        return module.apply(
            policy_params,
            normalized_seq,
            initial_hidden,
            done_seq,
            key,
            deterministic=deterministic,
            stored_keys=stored_keys,
            method=module.apply_sequence,
        )

    def init_hidden(batch_size: int) -> list[jnp.ndarray]:
        return [jnp.zeros((batch_size, h)) for h in rnn_hidden_sizes]

    return MoSeqHighLevelRecurrentNetwork(
        init=init,
        apply=apply,
        apply_sequence=apply_sequence,
        init_hidden=init_hidden,
    )


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------


def make_moseq_highlvl_inference_fn(
    ppo_networks: MoSeqHighLevelPPONetworks,
) -> Callable[..., types.Policy]:
    """Create a policy factory for the high-level intention network.

    For ``acting.Evaluator`` compatibility the returned policy has the
    standard ``(obs, key) -> (action, extras)`` signature.  It creates
    fresh-zero hidden state each call (memoryless single-step).  Accurate
    RNN evaluation with hidden carry happens in the logging inference fn.
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

            # Infer batch size from an obs leaf
            obs_leaf = jax.tree_util.tree_leaves(observations)[0]
            batch_size = obs_leaf.shape[0] if obs_leaf.ndim >= 2 else 1
            hidden = policy_network.init_hidden(batch_size)

            intention_params, code_idx, _new_hidden = policy_network.apply(
                *params, observations, hidden, key_net, deterministic=deterministic
            )

            extras = {"code_idx": code_idx}

            if deterministic:
                action = jnp.array(action_dist.mode(intention_params))
                return action, extras

            raw_actions = action_dist.sample_no_postprocessing(
                intention_params, key_sample
            )
            log_prob = action_dist.log_prob(intention_params, raw_actions)
            action = action_dist.postprocess(raw_actions)

            extras["log_prob"] = log_prob
            extras["raw_action"] = raw_actions
            return jnp.array(action), extras

        return policy

    return make_policy


def make_moseq_highlvl_logging_inference_fn(
    ppo_networks: MoSeqHighLevelPPONetworks,
) -> Callable[[bool], Callable]:
    """Create a logging policy that carries hidden state.

    Returns:
        ``make_policy(deterministic) -> logging_policy``

        Where ``logging_policy(params, obs, hidden, key) -> (action, extras, new_hidden)``
    """

    def make_logging_policy(
        deterministic: bool = False,
    ) -> Callable:
        policy_network = ppo_networks.policy_network
        action_dist = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            hidden: list[jnp.ndarray],
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra, list[jnp.ndarray]]:
            key_sample, key_net = jax.random.split(key_sample)

            intention_params, code_idx, new_hidden = policy_network.apply(
                *params,
                observations,
                hidden,
                key_net,
                deterministic=deterministic,
            )

            extras = {"code_idx": code_idx, "indices": code_idx}

            if deterministic:
                action = jnp.array(action_dist.mode(intention_params))
                return action, extras, new_hidden

            raw_actions = action_dist.sample_no_postprocessing(
                intention_params, key_sample
            )
            log_prob = action_dist.log_prob(intention_params, raw_actions)
            action = action_dist.postprocess(raw_actions)

            extras["log_prob"] = log_prob
            extras["raw_action"] = raw_actions
            return jnp.array(action), extras, new_hidden

        return logging_policy

    return make_logging_policy


# ---------------------------------------------------------------------------
# Carry-aware rollout policy (plugs into ppo.train carry path)
# ---------------------------------------------------------------------------


def make_moseq_highlvl_rnn_rollout_policy_fn(
    ppo_networks: MoSeqHighLevelPPONetworks,
) -> Callable:
    """Build a factory for carry-aware rollout policies.

    Returns:
        ``make_policy(params) -> policy(obs, carry, key) -> (action, extras, new_carry)``

    This is passed to ``ppo.train(make_rollout_policy_fn=...)``.
    """
    policy_network = ppo_networks.policy_network
    action_dist = ppo_networks.parametric_action_distribution

    def make_policy(params: types.PolicyParams) -> Callable:
        def policy(
            observations: types.Observation,
            carry: list[jnp.ndarray],
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra, list[jnp.ndarray]]:
            key_sample, key_net = jax.random.split(key_sample)

            intention_params, code_idx, new_hidden = policy_network.apply(
                *params,
                observations,
                carry,
                key_net,
                deterministic=False,
            )

            extras = {"code_idx": code_idx}

            raw_actions = action_dist.sample_no_postprocessing(
                intention_params, key_sample
            )
            log_prob = action_dist.log_prob(intention_params, raw_actions)
            action = action_dist.postprocess(raw_actions)

            extras["log_prob"] = log_prob
            extras["raw_action"] = raw_actions
            return jnp.array(action), extras, new_hidden

        return policy

    return make_policy


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_moseq_highlvl_rnn_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    num_codes: int = 32,
    code_embed_dim: int = 16,
    rnn_hidden_sizes: tuple[int, ...] = (256,),
    value_hidden_layer_sizes: tuple[int, ...] = (512, 512, 256, 256),
    code_stack_size: int = 1,
) -> MoSeqHighLevelPPONetworks:
    """Create high-level RNN intention PPO networks.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes
            (should contain ``kpms_code`` and ``proprioception``).
        action_size: Intention dimension (= intention_size).
        num_codes: Number of KPMS syllable codes.
        code_embed_dim: Code embedding dimension.
        rnn_hidden_sizes: GRU hidden sizes per layer.
        value_hidden_layer_sizes: Value MLP layer sizes.
        code_stack_size: Number of stacked codes in obs.

    Returns:
        MoSeqHighLevelPPONetworks with recurrent policy + FF value network.
    """
    rnn_hidden_sizes = tuple(rnn_hidden_sizes)
    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    module = MoSeqIntentionRNN(
        num_codes=num_codes,
        code_embed_dim=code_embed_dim,
        rnn_hidden_sizes=rnn_hidden_sizes,
        intention_param_size=action_dist.param_size,
        code_stack_size=code_stack_size,
    )

    # Build obs_sizes dict for init (include kpms_code)
    init_obs_sizes = {**obs_sizes}
    if "kpms_code" not in init_obs_sizes:
        init_obs_sizes["kpms_code"] = code_stack_size

    policy_network = _make_moseq_highlvl_recurrent_policy_network(
        module, init_obs_sizes, rnn_hidden_sizes
    )

    value_network = make_moseq_value_network(
        obs_sizes=obs_sizes,
        num_codes=num_codes,
        code_embed_dim=code_embed_dim,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return MoSeqHighLevelPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=action_dist,
        num_codes=num_codes,
        code_embed_dim=code_embed_dim,
        rnn_hidden_sizes=rnn_hidden_sizes,
        intention_size=action_size,
    )
