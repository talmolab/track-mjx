"""Load MoSeq checkpoints and create inference functions.

Works with both feedforward (MLP) and recurrent (RNN/GRU) decoder variants.
"""

from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf

# Ensure repo root and moseq_jax are importable
_MOSEQ_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _MOSEQ_DIR.parent
for _p in (str(_MOSEQ_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from track_mjx.agent import checkpointing
from track_mjx.agent.observation_utils import init_dict_normalizer
from moseq_ppo_networks import (
    make_moseq_decoder_ppo_networks,
    make_moseq_recurrent_decoder_ppo_networks,
    make_moseq_logging_inference_fn,
    make_moseq_recurrent_logging_inference_fn,
)

STEP_PREFIX = "MoSeqPPONetwork"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_ppo_networks(net_cfg: DictConfig) -> tuple[Any, bool, dict]:
    """Build PPO networks from a network config section.

    Returns:
        ``(ppo_networks, use_rnn, common_kwargs)``
    """
    obs_sizes = dict(net_cfg.obs_sizes)
    action_size = int(net_cfg.action_size)
    use_rnn = bool(net_cfg.get("use_rnn_decoder", False))

    common_kwargs = dict(
        obs_sizes=obs_sizes,
        action_size=action_size,
        num_codes=int(net_cfg.num_codes),
        code_embed_dim=int(net_cfg.code_embed_dim),
        value_hidden_layer_sizes=tuple(net_cfg.critic_layer_sizes),
        use_continuous_encoder=bool(net_cfg.get("use_continuous_encoder", False)),
        encoder_layer_sizes=tuple(net_cfg.get("encoder_layer_sizes", [256, 128])),
        continuous_latent_dim=int(net_cfg.get("continuous_latent_dim", 16)),
        z_e_dropout_rate=float(net_cfg.get("z_e_dropout_rate", 0.0)),
    )

    if use_rnn:
        ppo_networks = make_moseq_recurrent_decoder_ppo_networks(
            **common_kwargs,
            rnn_hidden_sizes=tuple(net_cfg.get("rnn_hidden_sizes", [256])),
            rnn_cell_type=str(net_cfg.get("rnn_cell_type", "gru")),
        )
    else:
        ppo_networks = make_moseq_decoder_ppo_networks(
            **common_kwargs,
            decoder_hidden_layer_sizes=tuple(net_cfg.decoder_layer_sizes),
        )

    return ppo_networks, use_rnn, common_kwargs


def _make_moseq_abstract_policy(
    cfg: DictConfig,
    ppo_networks: Any,
    use_rnn: bool,
) -> tuple:
    """Create an abstract (randomly initialised) policy for orbax restore.

    The generic ``checkpointing.make_abstract_policy`` does not recognise
    ``moseq_decoder`` as an architecture, so we build the template here.

    Returns:
        ``(normalizer_state, init_policy_params)`` — same pytree shape as
        what was saved during training.
    """
    net_cfg = cfg.network_config
    obs_sizes = dict(net_cfg.obs_sizes)

    key_policy, key_value = jax.random.split(jax.random.key(1))

    # Initialise normalizer (dict-based)
    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes["imitation_target"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    normalizer_state = init_dict_normalizer(dummy_obs)

    # Initialise policy and value params
    init_policy = ppo_networks.policy_network.init(key_policy)
    init_value = ppo_networks.value_network.init(key_value)

    # The checkpoint stores params as a flat dict {"policy": ..., "value": ...}
    # wrapped in a PPONetworkParams-like structure. We replicate that here.
    from brax.training.agents.ppo import losses as ppo_losses

    init_params = ppo_losses.PPONetworkParams(
        policy=init_policy,
        value=init_value,
    )

    return (normalizer_state, init_params.policy)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_moseq_checkpoint(
    checkpoint_path: str | Path,
    step: int | None = None,
) -> tuple[DictConfig, Any, Any, Any]:
    """Load a MoSeq checkpoint and reconstruct the PPO networks.

    Args:
        checkpoint_path: Directory containing ``MoSeqPPONetwork_*`` folders.
        step: Checkpoint step to restore (``None`` = latest).

    Returns:
        ``(cfg, normalizer_state, policy_params, ppo_networks)``
    """
    checkpoint_path = str(checkpoint_path)

    # 1. Load config
    cfg = checkpointing.load_config_from_checkpoint(
        checkpoint_path, step_prefix=STEP_PREFIX, step=step,
    )
    cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg

    # 2. Build network architecture (needed for abstract template)
    net_cfg = cfg.network_config
    ppo_networks, use_rnn, common_kwargs = _build_ppo_networks(net_cfg)

    # 3. Create abstract template for orbax restore
    abstract_policy = _make_moseq_abstract_policy(cfg, ppo_networks, use_rnn)

    # 4. Restore policy params via orbax
    mgr_options = ocp.CheckpointManagerOptions(
        create=False, step_prefix=STEP_PREFIX,
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)
    restore_step = step if step is not None else ckpt_mgr.latest_step()

    normalizer_state, policy_params = ckpt_mgr.restore(
        restore_step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardRestore(abstract_policy),
        ),
    )["policy"]

    logging.info(
        f"Loaded MoSeq checkpoint step {restore_step}: "
        f"{'RNN' if use_rnn else 'MLP'} decoder, "
        f"codes={net_cfg.num_codes}, "
        f"z_e={'on' if common_kwargs['use_continuous_encoder'] else 'off'}"
    )

    return cfg, normalizer_state, policy_params, ppo_networks


# ---------------------------------------------------------------------------
# Inference function factory
# ---------------------------------------------------------------------------


def make_inference_fn(
    ppo_networks: Any,
    use_rnn: bool,
    deterministic: bool = True,
    z_e_scale: float = 1.0,
) -> Callable:
    """Create a JIT-compiled inference function.

    The returned function has a **uniform API** regardless of architecture:

    - **RNN**: ``fn(params, obs_batched, hidden, key) -> (action, extras, new_hidden)``
    - **MLP**: ``fn(params, obs, key) -> (action, extras)``

    Args:
        ppo_networks: ``MoSeqPPONetworks`` or ``MoSeqRecurrentPPONetworks``.
        use_rnn: Whether this is an RNN architecture.
        deterministic: Use deterministic (mode) actions.
        z_e_scale: Continuous-encoder scaling (0.0 = code-only).

    Returns:
        JIT-compiled inference function.
    """
    if use_rnn:
        make_logging = make_moseq_recurrent_logging_inference_fn(ppo_networks)
        raw_fn = make_logging(deterministic=deterministic, z_e_scale=z_e_scale)
        return jax.jit(raw_fn)
    else:
        make_logging = make_moseq_logging_inference_fn(ppo_networks)
        raw_fn = make_logging(deterministic=deterministic, z_e_scale=z_e_scale)
        return jax.jit(raw_fn)


# ---------------------------------------------------------------------------
# Rollout helper
# ---------------------------------------------------------------------------


def run_rollout(
    env,
    inference_fn: Callable,
    params: tuple,
    ppo_networks: Any,
    use_rnn: bool,
    key: jax.Array,
    max_steps: int = 500,
    code_override: np.ndarray | None = None,
    initial_qpos: np.ndarray | None = None,
    jit_reset: Callable | None = None,
    jit_step: Callable | None = None,
) -> dict[str, Any]:
    """Run a single evaluation rollout and collect trajectory data.

    Args:
        env: ``MoSeqImitation`` environment (JIT-ready).
        inference_fn: From :func:`make_inference_fn`.
        params: ``(normalizer_state, policy_params)`` tuple.
        ppo_networks: Network object (needed for RNN hidden init).
        use_rnn: Whether this is an RNN architecture.
        key: PRNG key.
        max_steps: Maximum episode length.
        code_override: If provided, ``[max_steps]`` int array to override
            the environment's KPMS codes at each step.
        initial_qpos: If provided, override the initial qpos after reset.
        jit_reset: Pre-compiled ``jax.jit(env.reset)``. Created if ``None``.
        jit_step: Pre-compiled ``jax.jit(env.step)``. Created if ``None``.

    Returns:
        Dict with keys: ``qpos``, ``rewards``, ``code_indices``,
        ``per_step_metrics``, ``survival``.
    """
    if jit_reset is None:
        jit_reset = jax.jit(env.reset)
    if jit_step is None:
        jit_step = jax.jit(env.step)

    key, reset_key = jax.random.split(key)
    state = jit_reset(reset_key)

    # Override initial qpos if requested
    if initial_qpos is not None:
        new_data = state.data.replace(qpos=jnp.array(initial_qpos))
        state = state.replace(data=new_data)
        # Re-forward to update obs
        import mujoco
        from mujoco import mjx
        # We need mj_forward equivalent — simplest: step with zero action
        # Actually, just do a step with the current action to sync physics
        # This is tricky in JAX; instead we re-do reset and accept the default pose
        # TODO: proper qpos override — for now use env reset

    # Initialize RNN hidden state
    hidden = None
    if use_rnn:
        hidden = ppo_networks.policy_network.init_hidden(1)

    qpos_list: list[np.ndarray] = []
    reward_list: list[float] = []
    code_list: list[int] = []
    metrics_list: list[dict[str, float]] = []

    for t in range(max_steps):
        # Override code if requested
        if code_override is not None:
            desired_code = int(code_override[min(t, len(code_override) - 1)])
            # CRITICAL: preserve OrderedDict type to avoid JAX recompilation
            # on every step (OrderedDict vs dict are different pytree types)
            from collections import OrderedDict
            new_obs = OrderedDict((k, v) for k, v in state.obs.items())
            new_obs["kpms_code"] = jnp.array([desired_code], dtype=jnp.float32)
            state = state.replace(obs=new_obs)

        key, subkey = jax.random.split(key)

        if use_rnn:
            batched_obs = jax.tree.map(lambda x: x[None], state.obs)
            action, extras, hidden = inference_fn(params, batched_obs, hidden, subkey)
            action = jax.tree.map(lambda x: x[0], action)
            extras = jax.tree.map(
                lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x,
                extras,
            )
        else:
            action, extras = inference_fn(params, state.obs, subkey, None)

        # Extract code index
        if "code_idx" in extras:
            code_list.append(int(extras["code_idx"]))
        elif "indices" in extras:
            code_list.append(int(extras["indices"]))

        # Collect qpos before step
        qpos_list.append(np.array(state.data.qpos))

        # Step environment
        state = jit_step(state, action)
        reward_list.append(float(state.reward))

        # Collect per-term metrics
        step_metrics = {}
        if hasattr(state, "metrics") and state.metrics:
            for mk, mv in state.metrics.items():
                if mk.startswith("rewards/"):
                    step_metrics[mk] = float(mv)
        metrics_list.append(step_metrics)

        if state.done:
            if use_rnn:
                hidden = ppo_networks.policy_network.init_hidden(1)
            break

    # Append final qpos
    qpos_list.append(np.array(state.data.qpos))

    return {
        "qpos": np.array(qpos_list),
        "rewards": np.array(reward_list),
        "code_indices": np.array(code_list),
        "per_step_metrics": metrics_list,
        "survival": len(reward_list),
    }
