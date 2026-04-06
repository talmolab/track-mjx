"""Load MoSeq and mimic-mjx checkpoints and create inference functions.

Works with:
- MoSeq feedforward (MLP) and recurrent (RNN/GRU) decoder variants.
- Mimic-mjx IntentionNetwork (encoder-decoder VAE) for oracle baseline.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
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
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
from moseq_ppo_networks import (
    make_moseq_decoder_ppo_networks,
    make_moseq_recurrent_decoder_ppo_networks,
    make_moseq_logging_inference_fn,
    make_moseq_recurrent_logging_inference_fn,
)

STEP_PREFIX = "MoSeqPPONetwork"
MIMIC_STEP_PREFIX = "MimicEncoder"


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
        # Distillation / pretrained decoder params — must match checkpoint
        # architecture or orbax restore will fail with pytree mismatch.
        # Config key "distillation_head_layer_sizes" maps to function param
        # "distill_head_layer_sizes".
        distill_logvar_min = net_cfg.get("distill_logvar_min", None)
        distill_logvar_max = net_cfg.get("distill_logvar_max", None)
        if distill_logvar_min is not None:
            distill_logvar_min = float(distill_logvar_min)
        if distill_logvar_max is not None:
            distill_logvar_max = float(distill_logvar_max)

        ppo_networks = make_moseq_recurrent_decoder_ppo_networks(
            **common_kwargs,
            rnn_hidden_sizes=tuple(net_cfg.get("rnn_hidden_sizes", [256])),
            rnn_cell_type=str(net_cfg.get("rnn_cell_type", "gru")),
            z_e_at_action_head=bool(net_cfg.get("z_e_at_action_head", False)),
            reinit_hidden_on_code=bool(net_cfg.get("reinit_hidden_on_code", False)),
            learned_hidden_init=bool(net_cfg.get("learned_hidden_init", False)),
            use_distillation_head=bool(net_cfg.get("use_distillation_head", False)),
            distill_head_layer_sizes=tuple(
                net_cfg.get("distillation_head_layer_sizes", [256, 128])
            ),
            distill_logvar_min=distill_logvar_min,
            distill_logvar_max=distill_logvar_max,
            use_pretrained_decoder=bool(net_cfg.get("use_pretrained_decoder", False)),
            decoder_layer_sizes_vae=tuple(
                net_cfg.get("decoder_layer_sizes_vae", [512, 256, 256, 256])
            ),
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
        "task_obs": jnp.zeros((1, obs_sizes["task_obs"])),
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
# Mimic-MJX (IntentionNetwork VAE) checkpoint loading
# ---------------------------------------------------------------------------


def load_mimic_checkpoint(
    checkpoint_path: str | Path,
    step: int | None = None,
) -> tuple[DictConfig, Any, Any, Any]:
    """Load a mimic-mjx (IntentionNetwork VAE) checkpoint.

    The mimic-mjx model is a standard encoder-decoder VAE that maps
    reference trajectories to actions.  It serves as the oracle upper
    bound in experiment comparisons since it sees the full reference.

    Args:
        checkpoint_path: Directory containing ``MimicEncoder_*`` folders.
        step: Checkpoint step to restore (``None`` = latest).

    Returns:
        ``(cfg, normalizer_state, policy_params, ppo_networks)``
    """
    checkpoint_path = str(checkpoint_path)

    eval_data = checkpointing.load_checkpoint_for_eval(
        checkpoint_path, step_prefix=MIMIC_STEP_PREFIX, step=step,
    )
    cfg = eval_data["cfg"]
    normalizer_state, policy_params = eval_data["policy"]

    net_cfg = cfg.network_config
    ppo_networks = ff_ppo_networks.make_intention_ppo_networks(
        obs_sizes=dict(net_cfg.obs_sizes),
        action_size=int(net_cfg.action_size),
        intention_latent_size=int(net_cfg.intention_size),
        encoder_hidden_layer_sizes=tuple(net_cfg.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(net_cfg.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(net_cfg.critic_layer_sizes),
    )

    logging.info(
        f"Loaded mimic-mjx checkpoint from {checkpoint_path}: "
        f"latent_dim={net_cfg.intention_size}"
    )

    return cfg, normalizer_state, policy_params, ppo_networks


def make_mimic_inference_fn(
    ppo_networks: Any,
    deterministic: bool = True,
) -> Callable:
    """Create a JIT-compiled inference function for the mimic-mjx model.

    The returned function matches the MoSeq MLP logging interface so it
    can be used interchangeably with :func:`make_inference_fn` in
    :func:`run_rollout`.

    Signature: ``fn(params, obs, key, prev_indices=None) -> (action, extras)``

    Args:
        ppo_networks: ``PPOImitationNetworks`` from mimic-mjx checkpoint.
        deterministic: Use deterministic (mode) actions.

    Returns:
        JIT-compiled inference function.
    """
    make_logging = ff_ppo_networks.make_logging_inference_fn(ppo_networks)
    raw_fn = make_logging(deterministic=deterministic)

    def adapted_fn(params, obs, key, prev_indices=None):
        return raw_fn(params, obs, key)

    return jax.jit(adapted_fn)


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
    model_type: str = "code2act",
) -> dict[str, Any]:
    """Run a single evaluation rollout and collect trajectory data.

    Args:
        env: ``MoSeqImitation`` environment (JIT-ready).
        inference_fn: From :func:`make_inference_fn` or
            :func:`make_mimic_inference_fn`.
        params: ``(normalizer_state, policy_params)`` tuple.
        ppo_networks: Network object (needed for RNN hidden init).
        use_rnn: Whether this is an RNN architecture.
        key: PRNG key.
        max_steps: Maximum episode length.
        code_override: If provided, ``[max_steps]`` int array to override
            the environment's KPMS codes at each step.  Ignored when
            ``model_type="mimic_mjx"``.
        initial_qpos: If provided, override the initial qpos after reset.
        jit_reset: Pre-compiled ``jax.jit(env.reset)``. Created if ``None``.
        jit_step: Pre-compiled ``jax.jit(env.step)``. Created if ``None``.
        model_type: ``"code2act"`` for MoSeq decoder or ``"mimic_mjx"``
            for the IntentionNetwork VAE oracle.

    Returns:
        Dict with keys: ``qpos``, ``rewards``, ``code_indices``,
        ``per_step_metrics``, ``survival``.
    """
    if jit_reset is None:
        jit_reset = jax.jit(env.reset)
    if jit_step is None:
        jit_step = jax.jit(env.step)

    is_mimic = model_type == "mimic_mjx"

    key, reset_key = jax.random.split(key)
    state = jit_reset(reset_key)

    # Override initial qpos if requested
    if initial_qpos is not None:
        new_data = state.data.replace(qpos=jnp.array(initial_qpos))
        state = state.replace(data=new_data)

    # Initialize RNN hidden state
    hidden = None
    if use_rnn and not is_mimic:
        hidden = ppo_networks.policy_network.init_hidden(1)

    qpos_list: list[np.ndarray] = []
    reward_list: list[float] = []
    code_list: list[int] = []
    metrics_list: list[dict[str, float]] = []

    # Determine whether code_override applies (only for code2act)
    apply_code_override = code_override is not None and not is_mimic

    for t in range(max_steps):
        # Override code if requested (code2act only)
        if apply_code_override:
            desired_code = int(code_override[min(t, len(code_override) - 1)])
            # Preserve OrderedDict type to avoid JAX recompilation
            new_obs = OrderedDict(state.obs)
            new_obs["kpms_code"] = jnp.array([desired_code], dtype=jnp.float32)
            state = state.replace(obs=new_obs)

        key, subkey = jax.random.split(key)

        if use_rnn and not is_mimic:
            batched_obs = jax.tree.map(lambda x: x[None], state.obs)
            action, extras, hidden = inference_fn(params, batched_obs, hidden, subkey)
            action = jax.tree.map(lambda x: x[0], action)
            extras = jax.tree.map(
                lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x,
                extras,
            )
        else:
            action, extras = inference_fn(params, state.obs, subkey, None)

        # Extract code index (code2act only; mimic-mjx has no codes)
        if not is_mimic:
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
            if use_rnn and not is_mimic:
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
