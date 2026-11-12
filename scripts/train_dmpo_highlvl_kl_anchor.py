"""DMPO high-level transfer entry: trainable prior + decoder with an action anchor.

Loads a frozen SCAMPER prior + decoder, warm-starts the B-aggressive kl-anchor
networks from it (prior and decoder blocks trainable at their own learning
rates), wraps the vnl-playground env in ``KLAnchorPriorDecoderWrapper`` + brax
training wrappers + ``BinocularVisionRenderWrapper``, and runs the DMPO
training loop. ``transfer.mode: from_scratch`` swaps the prior/decoder for an
end-to-end recurrent policy; ``network_config.policy_head_rnn`` /
``policy_head_slow_core`` select a recurrent policy head; and
``transfer.graft_ff_checkpoint`` grafts a feed-forward DMPO checkpoint into a
fresh (possibly wider) state.

Usage:
    python scripts/train_dmpo_highlvl_kl_anchor.py --config-name=<group>/<arm>
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
import vnl_playground.naccdmax_patch  # noqa: F401
from omegaconf import DictConfig, OmegaConf
from vnl_playground import tasks
from vnl_playground.tasks.wrappers_kl_anchor import KLAnchorPriorDecoderWrapper
from vnl_playground.tasks.wrappers_truncation import wrap_for_dmpo_training

from track_mjx.agent.dmpo.checkpoint import (
    check_alignment,
    check_resume_alignment,
    make_checkpointer,
    record_alignment,
    saved_alignment,
)
from track_mjx.agent.dmpo.checkpoint import restore as restore_ckpt
from track_mjx.agent.dmpo.checkpoint import save as save_ckpt
from track_mjx.agent.dmpo.config import (
    DMPOConfig,
    realized_ratios,
    resolve_sgd_steps_per_rollout,
)
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks
from track_mjx.agent.dmpo.networks_kl_anchor_rnn import make_dmpo_kl_anchor_rnn_networks
from track_mjx.agent.dmpo.networks_kl_anchor_slow import (
    make_dmpo_kl_anchor_slow_networks,
)
from track_mjx.agent.dmpo.networks_scratch_rnn import make_dmpo_scratch_rnn_networks
from track_mjx.agent.dmpo.normalizer_seeding import seed_proprio_from_imit
from track_mjx.agent.dmpo.optim_kl_anchor import make_kl_anchor_optimizers
from track_mjx.agent.dmpo.prior_utils import (
    load_prior_checkpoint,
    make_decoder_logits_fn,
    make_prior_inference_fn,
)
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.schedules import env_steps_estimate, reward_anneal_lambda
from track_mjx.agent.dmpo.train import (
    _filter_dmpo_kwargs,
    _VnlPlaygroundEnvAdapter,
)
from track_mjx.agent.dmpo.train_dmpo_eval import (
    compute_batch_rollout_metrics,
    compute_rollout_metrics,
    compute_vision_sensitivity,
    render_eval_video,
    run_eval_rollout_envzero,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    detect_git_sha,
    make_run_id,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    load_wandb_state as load_dmpo_wandb_state,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    save_wandb_state as save_dmpo_wandb_state,
)
from track_mjx.agent.dmpo.training_loop import run as run_training_loop
from track_mjx.agent.dmpo.warm_start_graft import (
    graft_normalizer,
    graft_params,
    load_graft_source,
)

log = logging.getLogger(__name__)

try:
    import wandb

    _WANDB_IMPORTED = True
except ImportError:
    _WANDB_IMPORTED = False
    wandb = None  # type: ignore


# ---------------------------------------------------------------------------
# Optional top-level `wrappers:` block
# ---------------------------------------------------------------------------
# Every knob is default-off: a config with no `wrappers:` block gets
# `full_reset=False` and no info wrapper, i.e. exactly the historical hardcoded
# stack.
#
#   wrappers:
#     full_reset: true                 # BraxAutoResetWrapper calls env.reset()
#     info_reset_on_done: true         # InfoResetOnDoneWrapper (info keys only)
#     info_reset_keys: [a, b, ...]
#     allow_frozen_layout: true        # deliberate ablation; skips the guard
#
# `full_reset: false` (the historical hardcoded value) makes
# BraxAutoResetWrapper restore `data`/`obs` from the FIRST reset and never
# touch `state.info`.  For an env whose per-episode variation lives in `Data`
# (spawn pose, object positions written into qpos) that means `env.reset` runs
# exactly once per env index for the whole run: every worker replays one frozen
# layout, and info-derived masks/counters ratchet forever.  See
# vnl_playground/tasks/wrappers_info_reset.py for the ratchet half of that
# story, and vnl_playground/tasks/rodent/maze_forage_vision.py for the
# frozen-layout half.


def _wrapper_settings(cfg, env_info_keys=None):
    """Reads the optional top-level ``wrappers:`` block.

    Args:
        cfg: The hydra config.
        env_info_keys: The keys of the inner (not yet brax-wrapped) env's
            ``state.info`` after a reset. When given, every ``info_reset_keys``
            entry must be one of them.

    Returns:
        ``(full_reset, info_reset_on_done, info_reset_keys)``; the keys tuple is
        ``None`` when the config does not name any.

    Raises:
        ValueError: If ``env_info_keys`` is given and an ``info_reset_keys``
            entry is not among them. ``InfoResetOnDoneWrapper`` would also
            reject such a key, with a KeyError at the first reset; this is an
            earlier error, raised before the env is wrapped, that names the
            keys the inner env does hold.
    """
    wrap_cfg = cfg.get("wrappers", None)
    if not wrap_cfg:
        return False, False, None
    keys = wrap_cfg.get("info_reset_keys", None)
    keys = tuple(keys) if keys else None
    if keys is not None and env_info_keys is not None:
        unknown = [k for k in keys if k not in set(env_info_keys)]
        if unknown:
            raise ValueError(
                f"wrappers.info_reset_keys names {unknown}, which the env's "
                f"state.info does not hold (it holds {sorted(env_info_keys)}). "
                "InfoResetOnDoneWrapper would reject them at the first reset."
            )
    return (
        bool(wrap_cfg.get("full_reset", False)),
        bool(wrap_cfg.get("info_reset_on_done", False)),
        keys,
    )


def _env_info_keys(environment) -> tuple[str, ...]:
    """The keys of ``environment.reset(rng).info``, from an abstract trace.

    ``jax.eval_shape`` traces the reset without running it, so no env step or
    device computation happens.
    """
    state = jax.eval_shape(environment.reset, jax.random.PRNGKey(0))
    return tuple(state.info)


def _wrap_for_brax_training(
    environment,
    cfg,
    *,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn=None,
):
    """``wrap_for_dmpo_training`` honouring the ``wrappers:`` block.

    ``wrap_for_dmpo_training`` is the brax training stack plus a
    ``TruncationStashWrapper``, which keeps the time-limit flag in
    ``state.info["AutoResetWrapper_preserve_info"]["truncation"]`` through the
    auto-reset. ``InfoResetOnDoneWrapper`` is applied OUTSIDE the brax wrappers
    on purpose: it has to see the post-swap ``done`` flag.

    Args:
        environment: Env to wrap.
        cfg: The hydra config (read for the ``wrappers:`` block).
        episode_length: Truncation horizon for ``EpisodeWrapper``.
        action_repeat: Action repeat for ``EpisodeWrapper``.
        randomization_fn: Optional domain-randomization function.

    Returns:
        The wrapped environment.

    Raises:
        ValueError: If ``wrappers.info_reset_keys`` names a key that
            ``environment``'s ``state.info`` does not hold.
    """
    wrap_cfg = cfg.get("wrappers", None) or {}
    env_info_keys = (
        _env_info_keys(environment) if wrap_cfg.get("info_reset_keys", None) else None
    )
    full_reset, info_reset_on_done, info_reset_keys = _wrapper_settings(
        cfg, env_info_keys
    )
    brax_env = wrap_for_dmpo_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
        full_reset=full_reset,
    )
    if full_reset:
        log.info(
            "BraxAutoResetWrapper full_reset=True: env.reset() runs on every "
            "done, so Data-side per-episode randomisation and state.info both "
            "reset properly (costs one extra reset per step)."
        )
    if info_reset_on_done:
        from vnl_playground.tasks.wrappers_info_reset import (
            DEFAULT_RUN_GAP_KEYS,
            InfoResetOnDoneWrapper,
        )

        keys = info_reset_keys or DEFAULT_RUN_GAP_KEYS
        brax_env = InfoResetOnDoneWrapper(brax_env, keys=keys)
        log.info(
            "InfoResetOnDoneWrapper ACTIVE: restoring info keys %s to their "
            "reset-time values on every done.",
            keys,
        )
    return brax_env


def _validate_reset_requirements(cfg, env, env_name: str) -> None:
    """Refuses to launch an env that needs real resets inside a cached-reset stack.

    An env opts in by setting ``requires_per_episode_reset = True`` (and,
    optionally, ``info_reset_keys``).  Without ``wrappers.full_reset`` such an
    env trains on one frozen layout per worker and its ``info`` masks ratchet --
    a silently dead run rather than a crash, which is exactly the failure this
    guard exists to convert into an exception.

    Args:
        cfg: The hydra config.
        env: The loaded (possibly wrapped) environment.
        env_name: Env name, for the error message.

    Raises:
        ValueError: If the env demands per-episode resets and the config does
            not provide them.
    """
    raw_env = env
    while hasattr(raw_env, "env"):
        raw_env = raw_env.env
    if not bool(getattr(raw_env, "requires_per_episode_reset", False)):
        return

    full_reset, info_reset_on_done, info_reset_keys = _wrapper_settings(cfg)
    if full_reset:
        return

    required_keys = tuple(getattr(raw_env, "info_reset_keys", ()) or ())
    wrap_cfg = cfg.get("wrappers", None) or {}
    allow_frozen = bool(wrap_cfg.get("allow_frozen_layout", False))
    covered = info_reset_on_done and not (
        set(required_keys) - set(info_reset_keys or ())
    )
    if allow_frozen and covered:
        log.warning(
            "%s: wrappers.allow_frozen_layout=true -- env.reset() will run "
            "ONCE per env index for the whole run, so every worker replays a "
            "single frozen layout. Only the info ratchet is fixed. This is an "
            "ablation setting, not the intended training recipe.",
            env_name,
        )
        return

    raise ValueError(
        f"{env_name} sets requires_per_episode_reset=True, but this config "
        "does not enable per-episode resets.\n"
        "BraxAutoResetWrapper(full_reset=False) restores data/obs from the "
        "FIRST reset and never clears state.info, so env.reset() would run "
        "exactly once per env index for the entire run: every worker replays "
        "one frozen spawn/object layout and info masks ratchet until the env "
        "emits 1-step, zero-reward episodes forever.\n"
        "Add to the config:\n"
        "  wrappers:\n"
        "    full_reset: true\n"
        "Deliberate frozen-layout ablation instead? Set\n"
        "  wrappers:\n"
        "    allow_frozen_layout: true\n"
        "    info_reset_on_done: true\n"
        f"    info_reset_keys: {list(required_keys)}"
    )


def _replace_cfg_fields(cfg: DMPOConfig, **fields) -> DMPOConfig:
    """``dataclasses.replace`` that refuses a field the installed DMPOConfig lacks.

    Use this for a value the entry computes and the learner MUST see, one that
    is not a ``train_config`` key: a slow-core arm launched against a track-mjx
    without ``rnn_tick_stride`` would otherwise train the per-step learner on a
    strided policy and look healthy.

    Raises:
        ValueError: naming every missing field.
    """
    known = {f.name for f in dataclasses.fields(cfg)}
    missing = sorted(set(fields) - known)
    if missing:
        raise ValueError(
            f"DMPOConfig has no field(s) {missing} in the installed track-mjx "
            f"({DMPOConfig.__module__}). This config needs a learner that reads "
            "them; refusing to launch rather than silently dropping them."
        )
    return dataclasses.replace(cfg, **fields)


# ---- transfer.graft_ff_checkpoint ------------------------------------------
# Shape-checked FF -> superset warm start (track-mjx warm_start_graft). Unlike
# warm_start_dmpo_checkpoint (flax from_state_dict, no shape checks, policy +
# normalizer only) it also grafts the critic, and it zero-fills the rows a
# widened kernel gained (slow-core hidden_0, privileged critic Dense_0), so the
# grafted nets equal the source nets at step 0 whatever the new inputs are.

# Config values that change the network FUNCTION without changing any param
# shape (graft_params cannot see them), or that decide which frozen prior the
# anchor wrapper compares against. This entry's own defaults fill a key either
# side omits, exactly as main() reads it. `critic_use_proprio` changes what the
# critic reads, and vmin/vmax move the categorical critic's atom support, so a
# grafted critic would no longer compute the source critic's value.
_GRAFT_CONFIG_KEYS = (
    ("network_config", "residual_mode", "sigma_tanh"),
    ("network_config", "residual_scale", 2.0),
    ("network_config", "critic_use_proprio", False),
    ("train_config", "vmin", -150.0),
    ("train_config", "vmax", 150.0),
    ("transfer", "prior_checkpoint_path", None),
    ("transfer", "prior_checkpoint_step", None),
)

# `transfer.prior_checkpoint_path` is compared by PRIOR IDENTITY, not by path:
# the same prior is named by different paths once configs point at
# ${oc.env:DMPO_PRIOR_ROOT}/<id> while a graft source recorded the absolute
# path it was trained with. A prior's id is its directory name, with the
# byte-identical aliases of the prior manifest mapped to the id they copy.
_PRIOR_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "track_mjx"
    / "config"
    / "dmpo"
    / "priors"
    / "MANIFEST.md"
)
_MANIFEST_ALIAS_ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \| yes \|$")


def _prior_aliases(manifest_path: Path = _PRIOR_MANIFEST) -> dict[str, str]:
    """Alias name -> prior id, read from the manifest's ``## Aliases`` table.

    Only rows marked byte-identical (``yes``) are aliases.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    aliases: dict[str, str] = {}
    section = ""
    for line in Path(manifest_path).read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif section == "Aliases" and (m := _MANIFEST_ALIAS_ROW.match(line)):
            aliases[m.group(1)] = m.group(2)
    return aliases


def _prior_id(path, aliases: dict[str, str]):
    """The manifest id a prior checkpoint path names (None stays None)."""
    if path is None:
        return None
    name = Path(str(path).rstrip("/")).name
    return aliases.get(name, name)


def _check_graft_source_config(graft_path: str, cfg_dict: dict) -> None:
    """Fail loud if the graft source was trained under a different function.

    Reads the source's saved ``config`` item and compares `_GRAFT_CONFIG_KEYS`.
    ``transfer.prior_checkpoint_path`` is compared by prior id (`_prior_id`),
    so a relocated or aliased copy of the same prior matches; ``cfg_dict`` is
    the resolved config, i.e. after ``${oc.env:...}`` interpolation.

    Raises:
        ValueError: if the source's ``config`` item cannot be read (the check
            cannot run, so the graft is refused), or listing every mismatching
            key.
    """
    import orbax.checkpoint as ocp

    try:
        src_cfg = ocp.Checkpointer(ocp.JsonCheckpointHandler()).restore(
            str(Path(graft_path) / "config")
        )
    except Exception as exc:
        raise ValueError(
            f"transfer.graft_ff_checkpoint={graft_path}: could not read the "
            f"source checkpoint's config item ({exc}). Without it the graft "
            "cannot be checked against this config's "
            + ", ".join(f"{s}.{k}" for s, k, _ in _GRAFT_CONFIG_KEYS)
            + "; refusing to graft."
        ) from exc

    def _norm(v):
        return str(v).rstrip("/") if isinstance(v, str) else v

    bad = []
    for section, key, default in _GRAFT_CONFIG_KEYS:
        src_v = (src_cfg.get(section) or {}).get(key, default)
        tgt_v = (cfg_dict.get(section) or {}).get(key, default)
        if isinstance(default, float):
            src_v, tgt_v = float(src_v), float(tgt_v)
        if (section, key) == ("transfer", "prior_checkpoint_path"):
            aliases = _prior_aliases()
            src_id, tgt_id = _prior_id(src_v, aliases), _prior_id(tgt_v, aliases)
            if src_id != tgt_id:
                bad.append(
                    f"{section}.{key}: source={src_v!r} (prior id {src_id!r}) "
                    f"this config={tgt_v!r} (prior id {tgt_id!r})"
                )
        elif _norm(src_v) != _norm(tgt_v):
            bad.append(f"{section}.{key}: source={src_v!r} this config={tgt_v!r}")
    if bad:
        raise ValueError(
            f"transfer.graft_ff_checkpoint={graft_path} was trained under a "
            "different network function / frozen prior than this config:\n  "
            + "\n  ".join(bad)
            + "\nThe graft copies param VALUES only; with these keys different "
            "the grafted policy is not the source policy at step 0."
        )


GRAFT_OVERRIDE_KEY = "transfer.graft_allow_algorithm_mismatch"
RESUME_OVERRIDE_KEY = "train_config.allow_alignment_change_on_resume"


def _check_graft_alignment(graft_path: str, cfg: DMPOConfig, transfer_cfg) -> None:
    """Refuse a graft source trained with different alignment settings.

    The graft copies parameter values into a learner that then trains them
    under this config's algorithm. A source saved without the fields was
    trained by the pre-alignment learner. ``transfer.
    graft_allow_algorithm_mismatch: true`` turns the refusal into a WARNING.
    ``policy_loss_state`` is compared only between a source and a target that
    use the same learner path (both feed-forward, or both recurrent); it
    resolves differently on the two paths, so across them it cannot differ by
    choice. The other five fields are always compared.

    Raises:
        ValueError: listing every differing field, without the override.
    """
    check_alignment(
        saved_alignment(graft_path),
        cfg,
        source=f"transfer.graft_ff_checkpoint={graft_path}",
        override_key=GRAFT_OVERRIDE_KEY,
        allow=bool(transfer_cfg.get(GRAFT_OVERRIDE_KEY.split(".")[-1], False)),
        allow_path_change=True,
    )


def _restore_for_resume(ckpt_mgr, state, cfg: DMPOConfig, saved, opt_kwargs):
    """Restore the newest checkpoint, adapting across a clip-scope change.

    ``saved`` is what ``check_resume_alignment`` returned (``None`` for a fresh
    run). It can differ from ``cfg`` only when the resume override is set. The
    policy optimizer state is a ``multi_transform`` of per-block clip+Adam
    chains under ``policy_grad_clip_scope="per_block"`` and one trainable clip
    before a ``multi_transform`` of Adams under ``"global"``, so a checkpoint
    saved under the other scope is restored into a template of ITS structure
    and the policy optimizer state is then re-initialized (Adam moments at
    zero) for this config's optimizer. Every other item restores as saved.

    Returns:
        The restored ``TrainingState``, or ``None`` when there is none.
    """
    saved_scope = None if saved is None else saved["policy_grad_clip_scope"]
    if saved_scope is None or saved_scope == cfg.policy_grad_clip_scope:
        return restore_ckpt(ckpt_mgr, state_template=state)
    saved_opt, _, _ = make_kl_anchor_optimizers(
        dataclasses.replace(cfg, policy_grad_clip_scope=saved_scope), **opt_kwargs
    )
    template = state._replace(policy_opt_state=saved_opt.init(state.policy_params))
    restored = restore_ckpt(ckpt_mgr, state_template=template)
    if restored is None:
        return None
    pol_opt, _, _ = make_kl_anchor_optimizers(cfg, **opt_kwargs)
    log.warning(
        "policy_grad_clip_scope %r -> %r on resume: the policy optimizer state "
        "is re-initialized (Adam moments at zero); every other item is restored.",
        saved_scope,
        cfg.policy_grad_clip_scope,
    )
    return restored._replace(policy_opt_state=pol_opt.init(restored.policy_params))


def _fmt_paths(paths) -> str:
    return "[" + ",".join(paths) + "]"


def _apply_graft(state, graft_path: str, optimizers):
    """Graft a ``DMPONetwork_<step>`` dir into a FRESH training state.

    Policy, target policy, critic and target critic go through
    ``graft_params`` (shape-checked; widened rows zero), the normalizer through
    ``graft_normalizer``. The policy and critic optimizer states are then
    re-initialized on the grafted params (Adam moments at zero), as the
    from_state_dict warm start does for the policy. Duals stay fresh.

    Returns:
        ``(state, info)``; ``info`` = ``{"policy": report, "critic": report or
        None, "critic_mode": "grafted" | "init"}``.

    Raises:
        ValueError: if the source holds exactly one of the two critic items.
    """
    src = load_graft_source(graft_path)
    pol, rep_pol = graft_params(state.policy_params, src["policy_params"])
    tpol, _ = graft_params(state.target_policy_params, src["target_policy_params"])
    norm = graft_normalizer(state.normalizer_params, src["normalizer_params"])
    crit_items = [k for k in ("critic_params", "target_critic_params") if k in src]
    if len(crit_items) == 1:
        raise ValueError(
            f"graft source {graft_path} holds {crit_items[0]} but not its "
            "online/target partner; refusing to graft half a critic."
        )
    crit, tcrit, rep_crit = state.critic_params, state.target_critic_params, None
    if crit_items:
        crit, rep_crit = graft_params(state.critic_params, src["critic_params"])
        tcrit, _ = graft_params(state.target_critic_params, src["target_critic_params"])
    pol_opt, crit_opt, _ = optimizers
    state = state._replace(
        policy_params=pol,
        target_policy_params=tpol,
        critic_params=crit,
        target_critic_params=tcrit,
        normalizer_params=norm,
        policy_opt_state=pol_opt.init(pol),
        critic_opt_state=crit_opt.init(crit),
    )
    return state, {
        "policy": rep_pol,
        "critic": rep_crit,
        "critic_mode": "grafted" if crit_items else "init",
    }


def _graft_invariance_probe(
    nets,
    state,
    obs_template,
    action_size,
    critic_privileged_dim,
    rng,
    batch=4,
):
    """Max |output change| of the grafted nets when ONLY the new inputs move.

    The widened rows are zero, so the slow hidden (policy) and the privileged
    vector (critic) cannot reach the outputs at step 0; both numbers must be 0
    up to float noise. Together with the exact copy of every other leaf this
    is "the grafted nets equal the source nets at step 0" on THIS config's
    networks. Random obs: the invariance does not depend on obs realism.

    Returns:
        ``(policy_diff, critic_diff)``; ``None`` where there is nothing to
        probe (FF policy -- an exact copy; critic without privileged input).
    """
    k_obs, k_h, k_p = jax.random.split(rng, 3)
    leaves, treedef = jax.tree.flatten(obs_template)
    obs_keys = jax.random.split(k_obs, len(leaves))
    obs = treedef.unflatten(
        [
            jax.random.normal(k, (batch, *x.shape), jnp.float32)
            for k, x in zip(obs_keys, leaves)
        ]
    )
    pol_diff = None
    if nets.recurrent_meta is not None:
        h0 = nets.recurrent_meta.init_hidden(batch)
        h_leaves, h_def = jax.tree.flatten(h0)
        # Leaf 0 = the slow state; the clock leaves stay at a fresh phase so
        # the core ticks on the probed step (the tick path is the one the
        # rollout takes first).
        h_leaves[0] = 3.0 * jax.random.normal(k_h, h_leaves[0].shape)
        h1 = h_def.unflatten(h_leaves)

        def _pol(o, h):
            d, _ = nets.policy.apply(state.policy_params, o, h)
            return d.mean(), d.stddev()

        m0, s0 = jax.vmap(_pol)(obs, h0)
        m1, s1 = jax.vmap(_pol)(obs, h1)
        pol_diff = max(
            float(jnp.max(jnp.abs(m0 - m1))), float(jnp.max(jnp.abs(s0 - s1)))
        )
    crit_diff = None
    if critic_privileged_dim > 0:
        act = jnp.zeros((batch, action_size), jnp.float32)

        def _crit(o, a):
            return nets.critic.apply(state.critic_params, o, a).logits_parameter()

        p0 = jnp.zeros((batch, critic_privileged_dim), jnp.float32)
        p1 = 5.0 * jax.random.normal(k_p, (batch, critic_privileged_dim))
        l0 = jax.vmap(_crit)({**obs, "privileged": p0}, act)
        l1 = jax.vmap(_crit)({**obs, "privileged": p1}, act)
        crit_diff = float(jnp.max(jnp.abs(l0 - l1)))
    return pol_diff, crit_diff


# ---- Spout-maze eval (SPOUT_EVAL) ------------------------------------------
# vnl_playground.tasks.rodent.spout_eval computes the spout-maze trial
# statistics from the all-env `allenv` eval arrays. Its summary reads each
# `spout/<name>` through the mirror its SpoutEvalMetricAlias writes under
# `SPOUT_ALIAS + <name>` (an `anchor/` key), so the EVAL env (never the
# training env) is wrapped in it; the mirrored keys are stripped from the
# generic eval metrics before logging. spout_eval is imported only for a
# spout-maze env.
_SPOUT_MAZE_MODULE = "vnl_playground.tasks.rodent.spout_maze"
# ---- Full-maze eval (MAZE_EVAL) --------------------------------------------
# vnl_playground.tasks.rodent.maze_eval summarizes each env's first eval
# episode on the full maze (`MazeForageVision`, not its spout-maze subclass):
# one `MAZE_EVAL env_steps=...` line per eval, for every full-maze arm. The
# visited-cell count exists only with `env_config.maze_critic_priv`, as the env
# metric `maze/visited_cells`; the eval env of such an arm mirrors it under
# `MAZE_ALIAS + "visited_cells"` like the spout shim above. maze_eval is
# imported only for a full-maze env.
_MAZE_MODULE = "vnl_playground.tasks.rodent.maze_forage_vision"


def _innermost_env(env):
    """The env at the bottom of a wrapper stack (each wrapper has ``.env``)."""
    while hasattr(env, "env"):
        env = env.env
    return env


def _is_spout_maze_env(env) -> bool:
    """Whether the innermost env of a wrapper stack is a ``RodentSpoutMazeVision``.

    The class is looked up through ``sys.modules``, so this check imports
    nothing itself: an instance of the class implies its module is loaded.
    (The entry's own ``from vnl_playground import tasks`` already imports the
    spout-maze env modules; only ``spout_eval`` is gated on this check.)
    """
    spout_maze = sys.modules.get(_SPOUT_MAZE_MODULE)
    return spout_maze is not None and isinstance(
        _innermost_env(env), spout_maze.RodentSpoutMazeVision
    )


def _full_maze_env(env):
    """The innermost env if it is the full maze, else None.

    The full maze is a ``MazeForageVision`` that is not the spout-maze
    subclass, which has its own eval summary. The class is looked up through
    ``sys.modules``, as in ``_is_spout_maze_env``.
    """
    maze = sys.modules.get(_MAZE_MODULE)
    raw_env = _innermost_env(env)
    if maze is None or not isinstance(raw_env, maze.MazeForageVision):
        return None
    return None if _is_spout_maze_env(raw_env) else raw_env


def _check_maze_critic_priv_horizon(raw_env, episode_length: int) -> None:
    """A ``maze_critic_priv`` env must normalise its time entry by the real horizon.

    The last entry of ``info["critic_priv"]`` is ``step_count /
    env_config.episode_length`` as the env reads it, while the training
    wrapper truncates at ``episode_length`` (``env_config.episode_length``,
    else ``train_config.unroll_length``). An env config without the key keeps
    the env's own default, and the critic would see a mis-scaled time in the
    episode with no error. No-op for any env without the flag.

    Args:
        raw_env: The innermost env.
        episode_length: The truncation horizon of the training wrapper.

    Raises:
        ValueError: If the two horizons differ.
    """
    if not getattr(raw_env, "_maze_critic_priv", False):
        return
    env_horizon = int(raw_env._config.episode_length)
    if env_horizon != int(episode_length):
        raise ValueError(
            f"maze_critic_priv=True but the env's episode_length={env_horizon} "
            "differs from the training wrapper's truncation horizon "
            f"{int(episode_length)}: the critic's step_count / episode_length "
            "entry would be mis-scaled. Set env_config.episode_length to the "
            "horizon (it is the one the wrapper truncates at)."
        )


def _check_maze_critic_priv_dim(raw_env, critic_privileged_dim: int) -> None:
    """The critic's privileged width must be the width the full maze writes.

    A full-maze env writes ``info["critic_priv"]`` of ``env.critic_priv_dim``
    entries when built with ``env_config.maze_critic_priv``, and nothing
    otherwise, so ``network_config.critic_privileged_dim`` must be that width
    or 0. Any other env (the spout maze writes its own vector) is not checked
    here; the startup probe checks the shape it resets.

    Args:
        raw_env: The innermost env.
        critic_privileged_dim: ``network_config.critic_privileged_dim``.

    Raises:
        ValueError: If the widths differ.
    """
    if _full_maze_env(raw_env) is None:
        return
    if getattr(raw_env, "_maze_critic_priv", False):
        env_dim = int(raw_env.critic_priv_dim)
        if int(critic_privileged_dim) != env_dim:
            raise ValueError(
                f"network_config.critic_privileged_dim={critic_privileged_dim} "
                f"but the env writes info['critic_priv'] of width {env_dim} "
                "(env_config.maze_critic_priv=true): set "
                f"critic_privileged_dim: {env_dim}."
            )
    elif int(critic_privileged_dim) != 0:
        raise ValueError(
            f"network_config.critic_privileged_dim={critic_privileged_dim} but "
            "the full-maze env writes no info['critic_priv'] "
            "(env_config.maze_critic_priv is off): set maze_critic_priv: true "
            "or critic_privileged_dim: 0."
        )


def _make_eval_env(base_env_wrapped):
    """The env the eval rollout runs on, plus the spout eval module if it applies.

    Args:
        base_env_wrapped: The fully wrapped (training-stack) env.

    Returns:
        ``(eval_env, spout_eval)``. For a ``RodentSpoutMazeVision`` env,
        ``eval_env`` mirrors ``spout/*`` metrics and ``spout_eval`` is the
        ``vnl_playground.tasks.rodent.spout_eval`` module. For a full-maze env
        built with ``maze_critic_priv``, ``eval_env`` mirrors
        ``maze/visited_cells`` (``maze_eval.MazeEvalMetricAlias``). For any
        other env ``eval_env`` is ``base_env_wrapped`` itself. ``spout_eval``
        is None off the spout maze.
    """
    if not _is_spout_maze_env(base_env_wrapped):
        maze_env = _full_maze_env(base_env_wrapped)
        if maze_env is None:
            return base_env_wrapped, None
        if not getattr(maze_env, "_maze_critic_priv", False):
            log.info(
                "MAZE_EVAL active: one summary line per eval; no "
                "maze/visited_cells (maze_critic_priv off): visited_ep1=nan"
            )
            return base_env_wrapped, None
        from vnl_playground.tasks.rodent import maze_eval

        log.info(
            "MAZE_EVAL active: one summary line per eval; eval env mirrors "
            "maze/visited_cells to %svisited_cells (the training env is untouched)",
            maze_eval.MAZE_ALIAS,
        )
        return maze_eval.MazeEvalMetricAlias(base_env_wrapped, ("visited_cells",)), None
    from vnl_playground.tasks.rodent import spout_eval
    from vnl_playground.tasks.rodent.spout_maze import SPOUT_METRICS

    log.info(
        "SPOUT_EVAL active: eval env mirrors spout/%s to %s* (the training env "
        "is untouched)",
        "{" + ",".join(SPOUT_METRICS) + "}",
        spout_eval.SPOUT_ALIAS,
    )
    return spout_eval.SpoutEvalMetricAlias(base_env_wrapped, SPOUT_METRICS), spout_eval


def _build_env(
    hydra_cfg,
    prior_fn,
    decoder_logits_fn,
    latent_size,
    action_size,
    from_scratch=False,
):
    """Load registry env, wrap in KLAnchorPriorDecoderWrapper (or
    EndToEndWrapper on the from-scratch arm) + wrap_for_dmpo_training (the brax
    training stack plus the truncation stash) + BinocularVisionRenderWrapper.

    `from_scratch=True` is the no-prior/no-decoder control: the policy emits
    joint actions directly, so there is no frozen pipeline to anchor against
    and `prior_fn` / `decoder_logits_fn` / `latent_size` are unused. Every
    downstream wrapper is identical, so the two arms see the same env, the
    same vision pipeline and the same episode/reset semantics.
    """
    env_name = str(hydra_cfg.env_name)
    env_args = (
        OmegaConf.to_container(hydra_cfg.get("env_config", {}), resolve=True) or {}
    )
    env_args = {
        k: v for k, v in env_args.items() if k not in ("env_name", "flatten_obs")
    }
    valid_keys = set(tasks.get_default_config(env_name).keys())
    env_args = {k: v for k, v in env_args.items() if k in valid_keys}
    base_env = tasks.load(env_name, flatten_obs=False, config_overrides=env_args)
    _validate_reset_requirements(hydra_cfg, base_env, env_name)
    episode_length = int(
        hydra_cfg.env_config.get(
            "episode_length", hydra_cfg.train_config.get("unroll_length", 1000)
        )
    )
    # The full maze's optional critic vector must match the critic's input
    # width and the horizon the wrapper truncates at; both are no-ops off it.
    _check_maze_critic_priv_horizon(_innermost_env(base_env), episode_length)
    _check_maze_critic_priv_dim(
        _innermost_env(base_env),
        int(hydra_cfg.network_config.get("critic_privileged_dim", 0) or 0),
    )
    raw_env = base_env.env if hasattr(base_env, "env") else base_env
    mj_model = getattr(raw_env, "mj_model", None)
    mjx_model = getattr(raw_env, "mjx_model", None)
    if mj_model is None or mjx_model is None:
        raise RuntimeError(
            "Could not find mj_model/mjx_model on base env for vision rendering"
        )

    if from_scratch:
        # No prior, no decoder, no anchor. The `anchor/*` metrics and the
        # `anchor_mu_imit` / `anchor_log_std_imit` info keys that
        # KLAnchorPriorDecoderWrapper populates are all absent on this arm --
        # by construction, not by omission. action_size stays the env's native
        # joint control space, which is what the policy now emits directly.
        from vnl_playground.tasks.wrappers import EndToEndWrapper

        base_env = EndToEndWrapper(
            base_env,
            highlvl_obs_key=str(hydra_cfg.transfer.get("highlvl_obs_key", "task_obs")),
            lowlvl_obs_key=str(
                hydra_cfg.transfer.get("decoder_obs_key", "proprioception")
            ),
        )
    else:
        base_env = KLAnchorPriorDecoderWrapper(
            base_env,
            prior_fn=prior_fn,
            decoder_logits_fn=decoder_logits_fn,
            action_size=action_size,
            w_anchor=float(hydra_cfg.kl_anchor.w_anchor),
            alpha_anchor=float(hydra_cfg.kl_anchor.alpha_anchor),
        )

    from vnl_playground.tasks.rodent.vision_jax import BinocularVisionRenderWrapper

    action_repeat = int(hydra_cfg.env_config.get("action_repeat", 1))
    # Optional per-episode resets, read from the top-level `wrappers:` block.
    # With no such block this is the historical brax training stack with
    # full_reset=False and no info wrapper, plus the truncation stash.
    base_env = _wrap_for_brax_training(
        base_env,
        hydra_cfg,
        episode_length=episode_length,
        action_repeat=action_repeat,
    )
    base_env = BinocularVisionRenderWrapper(
        base_env,
        mj_model=mj_model,
        mjx_model=mjx_model,
        width=int(hydra_cfg.env_config.get("vision_width", 32)),
        height=int(hydra_cfg.env_config.get("vision_height", 32)),
        grayscale=bool(hydra_cfg.env_config.get("grayscale", True)),
        left_camera_name=str(
            hydra_cfg.env_config.get("left_camera_name", "eye_left-rodent")
        ),
        right_camera_name=str(
            hydra_cfg.env_config.get("right_camera_name", "eye_right-rodent")
        ),
        render_depth=False,
        use_textures=bool(hydra_cfg.env_config.get("use_textures", False)),
        use_shadows=bool(hydra_cfg.env_config.get("use_shadows", False)),
        eye_dropout_rate=float(hydra_cfg.env_config.get("eye_dropout_rate", 0.0)),
        eval_eye_mode=str(hydra_cfg.env_config.get("eval_eye_mode", "binocular")),
        # Gaussian sensor noise on the rendered image. The 0.0 default keeps
        # every config without the key bit-identical.
        vision_noise_std=float(hydra_cfg.env_config.get("vision_noise_std", 0.0)),
        # Temporal blur (first-order low-pass, motion blur). The 0.0 default
        # keeps every config without the key bit-identical. ctrl_dt turns tau
        # into a per-step retention weight, and configs do vary it.
        vision_blur_tau_ms=float(hydra_cfg.env_config.get("vision_blur_tau_ms", 0.0)),
        ctrl_dt=float(hydra_cfg.env_config.get("ctrl_dt", 0.02)),
    )
    env_adapter = _VnlPlaygroundEnvAdapter(base_env, pre_batched=True)
    vision_shape = tuple(
        getattr(
            base_env, "vision_shape", getattr(base_env.env, "vision_shape", (32, 32, 2))
        )
    )
    return env_adapter, base_env, mj_model, mjx_model, vision_shape


# ---------------------------------------------------------------------------
# Build stages shared with scripts/eval_dmpo_checkpoints.py
# ---------------------------------------------------------------------------
# main() runs these in order. The post-hoc checkpoint evaluator calls the same
# functions, so the env, the networks and the eval it builds are the ones a
# training run builds.


def _parse_slow_core_cfg(slow_core_cfg: dict) -> dict:
    """Validates ``network_config.policy_head_slow_core`` and fills its defaults.

    Keys: ``hidden_size`` and ``tick_stride`` (required ints), ``store_dtype``
    (str, default ``"float16"``), ``cell`` (``"gru"`` default, or ``"lstm"``)
    and ``forget_bias`` (float, default ``0.0``: the LSTM forget-gate bias,
    only valid with ``cell: lstm``). Every default is the slow-core factory's
    own, so a config that omits the optional keys builds the same network.

    Args:
        slow_core_cfg: The ``policy_head_slow_core`` block as a plain dict.

    Returns:
        ``{hidden_size, tick_stride, store_dtype, cell, forget_bias}``.

    Raises:
        ValueError: On an unknown or missing key, a ``cell`` outside
            ``("gru", "lstm")``, a non-finite ``forget_bias``, or a nonzero
            ``forget_bias`` with ``cell="gru"``.
    """
    keys = {"hidden_size", "tick_stride", "store_dtype", "cell", "forget_bias"}
    unknown = sorted(set(slow_core_cfg) - keys)
    missing = sorted({"hidden_size", "tick_stride"} - set(slow_core_cfg))
    if unknown or missing:
        raise ValueError(
            "network_config.policy_head_slow_core must be "
            "{hidden_size: int, tick_stride: int[, store_dtype: str]"
            "[, cell: 'gru'|'lstm'][, forget_bias: float]}; "
            f"unknown keys {unknown}, missing keys {missing}."
        )
    cell = slow_core_cfg.get("cell", "gru")
    if cell not in ("gru", "lstm"):
        raise ValueError(
            "network_config.policy_head_slow_core.cell must be 'gru' or "
            f"'lstm', got {cell!r}."
        )
    forget_bias = float(slow_core_cfg.get("forget_bias", 0.0))
    if not math.isfinite(forget_bias):
        raise ValueError(
            "network_config.policy_head_slow_core.forget_bias must be finite, "
            f"got {forget_bias}."
        )
    if forget_bias != 0.0 and cell != "lstm":
        raise ValueError(
            f"network_config.policy_head_slow_core.forget_bias={forget_bias} "
            f"only applies to cell='lstm' (got cell={cell!r}); drop it or set "
            "cell: lstm."
        )
    return {
        "hidden_size": int(slow_core_cfg["hidden_size"]),
        "tick_stride": int(slow_core_cfg["tick_stride"]),
        "store_dtype": str(slow_core_cfg.get("store_dtype", "float16")),
        "cell": cell,
        "forget_bias": forget_bias,
    }


class _RunSetup(NamedTuple):
    """The learner config and policy-head choice resolved from the hydra config.

    ``recurrent`` is true for either recurrent head (the GRU head or the slow
    core); ``gru_head`` and ``slow_core`` tell them apart. The ``slow_*`` and
    ``tick_stride`` fields are None unless ``slow_core``.
    """

    cfg: DMPOConfig
    anchor_extras: tuple
    from_scratch: bool
    recurrent: bool
    gru_head: bool
    slow_core: bool
    rnn_head_cfg: dict | None
    slow_hidden_size: int | None
    tick_stride: int | None
    slow_store_dtype: str | None
    slow_cell: str | None
    slow_forget_bias: float | None
    critic_privileged_dim: int
    priv_extras: tuple
    graft_path: str | None


class _FrozenPrior(NamedTuple):
    """The frozen prior + decoder of a grafted arm (all None/empty from scratch)."""

    prior_params: Any
    decoder_params: Any
    normalizer_params: Any
    prior_fn: Any
    decoder_logits_fn: Any
    latent_size: int
    prior_layer_sizes: tuple
    decoder_layer_sizes: tuple


class _EnvBuild(NamedTuple):
    """The training env stack and the observation/action sizes read off it."""

    env: Any
    base_env_wrapped: Any
    mj_model: Any
    mjx_model: Any
    vision_shape: tuple
    proprio_size: int
    task_obs_size: int
    action_size: int


def _resolve_run_setup(hydra_cfg) -> _RunSetup:
    """Learner config + policy-head choice, with every fail-loud config guard.

    Runs before anything is built, so a miswired arm dies in seconds.
    """
    slow_hidden_size = tick_stride = slow_store_dtype = None
    slow_cell = slow_forget_bias = None

    raw_train_cfg = OmegaConf.to_container(hydra_cfg.train_config, resolve=True)
    cfg = DMPOConfig(**_filter_dmpo_kwargs(raw_train_cfg))
    # Populate the loss-side KL-anchor coefficients on cfg so _policy_loss_fn
    # picks them up. DMPOConfig is a dataclass; mutate in place.
    cfg.kl_anchor_alpha = float(hydra_cfg.kl_anchor.alpha_anchor)
    cfg.kl_anchor_w = float(hydra_cfg.kl_anchor.w_anchor)
    cfg.kl_anchor_w_floor = float(hydra_cfg.kl_anchor.get("w_anchor_floor", 0.0))
    cfg.kl_anchor_beta_linear = float(hydra_cfg.kl_anchor.get("beta_linear", 0.0))
    cfg.kl_anchor_decay_sgd_steps = int(hydra_cfg.kl_anchor.get("decay_sgd_steps", 0))
    # Is the loss-side anchor actually switched on? This mirrors the gate the
    # learner uses to decide whether `anchor_mu_imit` is USED, which exists once
    # per learner path (feed-forward and recurrent), both as
    # `cfg.kl_anchor_alpha != 0.0 or cfg.kl_anchor_beta_linear != 0.0`. The
    # learner still READS the field unconditionally (`batch.get(...)` returns
    # None when absent); the gate decides whether the value is used.
    #
    # When the term is off we stop threading the two anchor arrays through the
    # rollout AND stop reserving room for them in the replay: at action_size 38
    # that is 2 * 38 * 4 = 304 B/transition, about 8% of a 3.7 KB feed-forward
    # vision transition.
    #
    # THE BACKSTOP IS WEAKER THAN IT LOOKS. The learner's own guard raises a
    # ValueError on `cfg.kl_anchor_alpha != 0.0 and anchor_mu is None` -- alpha
    # ONLY. `beta_linear` is not checked, so alpha == 0, beta_linear != 0 with
    # the extras dropped would silently contribute nothing. That combination
    # cannot arise from THIS predicate, which covers beta; it would arise from
    # "simplifying" this line to match the learner's guard. Do not.
    #
    # It costs NO diagnostics. `train/anchor_kl_mean` / `anchor_reward_mean` /
    # `anchor_loss_term` are still emitted every chunk, pinned at exactly 0.0
    # (a property of alpha == 0, not of dropping the extras): read a flat zero
    # there as "term is off", not as "no drift". `eval/anchor/r_anchor_mean`
    # (MSE-based) is computed by KLAnchorPriorDecoderWrapper from state.info,
    # independent of both alpha and the replay, so it keeps logging either way.
    #
    # Resume-safe: the replay is never checkpointed, it is rebuilt empty from
    # this template on every launch, after the restore.
    anchor_in_loss = cfg.kl_anchor_alpha != 0.0 or cfg.kl_anchor_beta_linear != 0.0
    anchor_extras = ("anchor_mu_imit", "anchor_log_std_imit") if anchor_in_loss else ()
    log.info(
        "kl-anchor loss term: %s (alpha=%.3g, beta_linear=%.3g) -> replay "
        "anchor extras %s",
        "ON" if anchor_in_loss else "OFF",
        cfg.kl_anchor_alpha,
        cfg.kl_anchor_beta_linear,
        "stored" if anchor_in_loss else "dropped",
    )

    # ---- from-scratch control arm: no prior, no decoder ---------------------
    # Absent/"" and "prior_decoder" both mean the grafted (prior + decoder)
    # path, so a config without `transfer.mode` resolves from_scratch=False and
    # nothing below fires. The from-scratch arm is recurrent only: the
    # recurrent policy head, its rollout hidden-state storage, the compressed
    # replay schema and the BPTT gates all live in this entry point.
    #
    # COMPATIBILITY: this entry point historically never read transfer.mode,
    # so older configs carry `kl_anchor_b_aggressive` (or `decoder_only`)
    # purely as a label. Both are accepted as aliases for the grafted path --
    # which is what they already did.
    #
    # The one thing NOT made lenient is an unrecognised value, because the
    # dangerous failure is asymmetric: a typo'd `from_scrach` that quietly
    # falls through to the grafted path would run exactly the experiment the
    # config was written to avoid, and the logs would look normal.
    _GRAFTED_MODES = ("", "prior_decoder", "kl_anchor_b_aggressive", "decoder_only")
    transfer_mode = str(hydra_cfg.get("transfer", {}).get("mode", ""))
    if transfer_mode not in (*_GRAFTED_MODES, "from_scratch"):
        raise ValueError(
            f"Unknown transfer.mode={transfer_mode!r}; expected 'from_scratch' "
            f"or one of the grafted-path values {_GRAFTED_MODES}. Refusing to "
            "guess: silently falling through to the grafted path would run a "
            "different experiment than the one this config asks for."
        )
    from_scratch = transfer_mode == "from_scratch"
    if from_scratch:
        if cfg.kl_anchor_alpha != 0.0 or cfg.kl_anchor_beta_linear != 0.0:
            raise ValueError(
                "transfer.mode=from_scratch with a non-zero kl-anchor "
                f"(alpha={cfg.kl_anchor_alpha}, "
                f"beta_linear={cfg.kl_anchor_beta_linear}). There is no frozen "
                "prior/decoder to anchor to on this arm -- set "
                "kl_anchor.alpha_anchor=0.0."
            )
        if hydra_cfg.transfer.get("warm_start_dmpo_checkpoint", None):
            raise ValueError(
                "transfer.warm_start_dmpo_checkpoint is set with "
                "transfer.mode=from_scratch. A grafted policy param tree has "
                "prior/decoder blocks this network does not."
            )
        log.info(
            "FROM-SCRATCH arm ACTIVE: no prior, no decoder, no anchor. "
            "Policy emits joint actions directly."
        )

    # ---- Recurrent (GRU) policy head: config surface + fail-loud guards ----
    # network_config.policy_head_rnn absent/null => FF path, byte-identical
    # (every FF config resolves `recurrent = False` here and no branch
    # below changes anything). All guards fire BEFORE the env is built so a
    # miswired arm dies in seconds, not after warp compilation.
    rnn_head_cfg = hydra_cfg.network_config.get("policy_head_rnn", None)
    if rnn_head_cfg is not None:
        rnn_head_cfg = OmegaConf.to_container(rnn_head_cfg, resolve=True)
    recurrent = rnn_head_cfg is not None
    # network_config.policy_head_slow_core: the slow-clock memory core
    # (networks_kl_anchor_slow). Absent/null => nothing below changes.
    slow_core_cfg = hydra_cfg.network_config.get("policy_head_slow_core", None)
    if slow_core_cfg is not None:
        slow_core_cfg = OmegaConf.to_container(slow_core_cfg, resolve=True)
    slow_core = slow_core_cfg is not None
    _transfer_cfg = hydra_cfg.get("transfer", None) or {}
    graft_path = _transfer_cfg.get("graft_ff_checkpoint", None)
    if slow_core and recurrent:
        raise ValueError(
            "network_config.policy_head_slow_core and "
            "network_config.policy_head_rnn are both set; they are two "
            "different recurrent policy heads -- set exactly one."
        )
    if recurrent:
        if graft_path:
            raise ValueError(
                "transfer.graft_ff_checkpoint is set together with "
                "network_config.policy_head_rnn. The GRU head's param tree is "
                "not a superset of the FF head's, so an FF checkpoint cannot "
                "be grafted into it; use policy_head_slow_core (whose tree is "
                "the FF head plus widened rows) or drop the graft."
            )
        rnn_L, rnn_n = int(cfg.rnn_bptt_length), int(cfg.n_step)
        if rnn_L <= 0:
            raise ValueError(
                "network_config.policy_head_rnn is set but "
                f"train_config.rnn_bptt_length={rnn_L}. The recurrent policy "
                "head requires the recurrent learner (rnn_bptt_length > 0): "
                "a GRU trained with the FF single-point loss would never "
                "learn what to remember."
            )
        if int(cfg.sequence_length) != rnn_L + rnn_n:
            raise ValueError(
                f"sequence_length={int(cfg.sequence_length)} != "
                f"rnn_bptt_length + n_step = {rnn_L} + {rnn_n} = "
                f"{rnn_L + rnn_n}. The recurrent learner needs every loss "
                "point t < L to have a full n-step window and a bootstrap "
                "state at observation[:, t + n]."
            )
        if bool(getattr(cfg, "store_next_observation", True)):
            raise ValueError(
                "policy_head_rnn requires the compressed replay schema: set "
                "train_config.store_next_observation=false (the recurrent "
                "learner bootstraps from observation[:, t + n])."
            )
        if not bool(getattr(cfg, "use_n_step", False)):
            raise ValueError(
                "policy_head_rnn requires train_config.use_n_step=true (the "
                "recurrent learner's per-point returns are n-step by "
                "construction; there is no single-step fallback)."
            )
        _ws_probe = (
            hydra_cfg.transfer.get("warm_start_dmpo_checkpoint", None)
            if "transfer" in hydra_cfg
            else None
        )
        if _ws_probe:
            raise ValueError(
                "transfer.warm_start_dmpo_checkpoint is set together with "
                "network_config.policy_head_rnn. An FF policy param tree "
                "cannot be grafted into a recurrent one (the policy_head "
                "subtree has a different structure) -- unset the warm start "
                "for RNN arms."
            )
        if float(getattr(cfg, "behavior_mix_init", 0.0)) > 0.0:
            raise ValueError(
                "train_config.behavior_mix_init > 0 with policy_head_rnn: "
                "behavior mixing is not supported with a recurrent policy "
                "(v1) -- the frozen behavior policy is FF and the rollout "
                "would need a second, per-policy hidden stream."
            )
    elif slow_core:
        _sc = _parse_slow_core_cfg(slow_core_cfg)
        slow_hidden_size = _sc["hidden_size"]
        tick_stride = _sc["tick_stride"]
        slow_store_dtype = _sc["store_dtype"]
        slow_cell = _sc["cell"]
        slow_forget_bias = _sc["forget_bias"]
        sc_L, sc_n = int(cfg.rnn_bptt_length), int(cfg.n_step)
        if sc_L <= 0:
            raise ValueError(
                "network_config.policy_head_slow_core is set but "
                f"train_config.rnn_bptt_length={sc_L}. The slow core needs the "
                "recurrent learner (rnn_bptt_length > 0 core ticks of BPTT)."
            )
        if int(cfg.sequence_length) != sc_L * tick_stride + sc_n:
            raise ValueError(
                f"sequence_length={int(cfg.sequence_length)} != "
                "rnn_bptt_length * tick_stride + n_step = "
                f"{sc_L} * {tick_stride} + {sc_n} = "
                f"{sc_L * tick_stride + sc_n}. The strided learner unrolls L "
                "core ticks (L*k env steps) and every loss point needs a full "
                "n-step window plus its bootstrap observation."
            )
        if bool(getattr(cfg, "store_next_observation", True)):
            raise ValueError(
                "policy_head_slow_core requires the compressed replay schema: "
                "set train_config.store_next_observation=false."
            )
        if not bool(getattr(cfg, "use_n_step", False)):
            raise ValueError(
                "policy_head_slow_core requires train_config.use_n_step=true "
                "(the recurrent learner has no single-step fallback)."
            )
        if _transfer_cfg.get("warm_start_dmpo_checkpoint", None):
            raise ValueError(
                "transfer.warm_start_dmpo_checkpoint is set together with "
                "network_config.policy_head_slow_core. That warm start loads "
                "via from_state_dict with no shape checks and would keep the "
                "FF checkpoint's narrow policy_head/hidden_0/kernel; use "
                "transfer.graft_ff_checkpoint instead."
            )
        if float(getattr(cfg, "behavior_mix_init", 0.0)) > 0.0:
            raise ValueError(
                "train_config.behavior_mix_init > 0 with "
                "policy_head_slow_core: behavior mixing is not supported with "
                "a recurrent policy."
            )
        if from_scratch:
            raise ValueError(
                "policy_head_slow_core with transfer.mode=from_scratch: the "
                "slow core is the kl-anchor (prior + decoder) head only."
            )
        # Fails loud on a track-mjx whose DMPOConfig has no rnn_tick_stride
        # (a missing field raises here rather than being skipped).
        cfg = _replace_cfg_fields(cfg, rnn_tick_stride=tick_stride)
    elif int(getattr(cfg, "rnn_bptt_length", 0)) > 0:
        raise ValueError(
            f"train_config.rnn_bptt_length={int(cfg.rnn_bptt_length)} but "
            "network_config.policy_head_rnn and policy_head_slow_core are "
            "absent/null. The recurrent learner needs a recurrent policy head "
            "(stored hidden + raw(obs, hidden) apply); set both or neither."
        )
    # The GRU head and the slow core share every downstream recurrent code
    # path (hidden storage, policy init adapter, probe, eval hidden); only the
    # factory and the startup log line differ. `gru_head` keeps the
    # GRU-specific bits apart.
    gru_head = recurrent
    recurrent = gru_head or slow_core

    # ---- Privileged critic input (network_config.critic_privileged_dim) ----
    # 0 (default) => no key, no replay field, no extra: the old path exactly.
    critic_privileged_dim = int(
        hydra_cfg.network_config.get("critic_privileged_dim", 0) or 0
    )
    priv_extras: tuple = ()
    if critic_privileged_dim < 0:
        raise ValueError(
            f"network_config.critic_privileged_dim={critic_privileged_dim} "
            "must be >= 0."
        )
    if critic_privileged_dim > 0:
        if from_scratch:
            raise ValueError(
                "network_config.critic_privileged_dim > 0 with "
                "transfer.mode=from_scratch: the scratch-RNN factory has no "
                "privileged critic input."
            )
        # The env writes the vector to state.info["critic_priv"]; the rollout
        # stores it from the PRE-step state via extra_state_extras, aligned
        # with observation[:, t]. Only the critic reads it.
        cfg = _replace_cfg_fields(cfg, critic_privileged_key="critic_priv")
        priv_extras = ("critic_priv",)
        log.info(
            "PRIVILEGED critic input ACTIVE: critic_privileged_dim=%d "
            "key=%r (critic only; replay stores it per step)",
            critic_privileged_dim,
            cfg.critic_privileged_key,
        )

    # ---- transfer.graft_ff_checkpoint guards (before the env is built) ----
    if graft_path:
        if _transfer_cfg.get("warm_start_dmpo_checkpoint", None):
            raise ValueError(
                "transfer.graft_ff_checkpoint and "
                "transfer.warm_start_dmpo_checkpoint are both set; they are "
                "two warm starts -- set exactly one (graft_ff_checkpoint is "
                "the shape-checked one and also grafts the critic)."
            )
        if from_scratch:
            raise ValueError(
                "transfer.graft_ff_checkpoint with transfer.mode=from_scratch: "
                "the source has prior/decoder blocks the scratch network does "
                "not."
            )
        if not (Path(str(graft_path)) / "train_state").is_dir():
            raise FileNotFoundError(
                f"transfer.graft_ff_checkpoint={graft_path} has no "
                "train_state item; expected a DMPONetwork_<step> directory."
            )
        _check_graft_source_config(
            str(graft_path), OmegaConf.to_container(hydra_cfg, resolve=True)
        )
        _check_graft_alignment(str(graft_path), cfg, _transfer_cfg)
    # -----------------------------------------------------------------------
    return _RunSetup(
        cfg=cfg,
        anchor_extras=anchor_extras,
        from_scratch=from_scratch,
        recurrent=recurrent,
        gru_head=gru_head,
        slow_core=slow_core,
        rnn_head_cfg=rnn_head_cfg,
        slow_hidden_size=slow_hidden_size,
        tick_stride=tick_stride,
        slow_store_dtype=slow_store_dtype,
        slow_cell=slow_cell,
        slow_forget_bias=slow_forget_bias,
        critic_privileged_dim=critic_privileged_dim,
        priv_extras=priv_extras,
        graft_path=graft_path,
    )


def _load_frozen_prior(hydra_cfg, from_scratch: bool) -> _FrozenPrior:
    """Loads the frozen prior + decoder a grafted arm anchors to."""

    # --- 1. Load frozen prior + decoder (grafted arms only) ---
    if from_scratch:
        # Nothing to load. These stay None/empty so the shared call sites below
        # read the same on both arms; _build_env and the network factory both
        # ignore them when from_scratch is set.
        prior_params = decoder_params = normalizer_params = prior_cfg = None
        prior_fn = decoder_logits_fn = None
        latent_size = 0
        prior_layer_sizes = decoder_layer_sizes = ()
        log.info("From-scratch arm: skipping prior checkpoint load entirely.")
    else:
        prior_ckpt_path = str(hydra_cfg.transfer.prior_checkpoint_path)
        prior_ckpt_step = hydra_cfg.transfer.get("prior_checkpoint_step", None)
        log.info("Loading prior checkpoint from: %s", prior_ckpt_path)
        (
            _enc_params,
            prior_params,
            decoder_params,
            normalizer_params,
            prior_cfg,
        ) = load_prior_checkpoint(prior_ckpt_path, prior_ckpt_step)
        latent_size = int(prior_cfg["network_config"]["intention_size"])
        prior_layer_sizes = tuple(
            prior_cfg["network_config"].get("prior_layer_sizes", [1024, 1024])
        )
        decoder_layer_sizes = tuple(prior_cfg["network_config"]["decoder_layer_sizes"])
        log.info("Prior loaded. intention_size=%d", latent_size)

        prior_fn = make_prior_inference_fn(prior_params, normalizer_params, prior_cfg)
        decoder_logits_fn = make_decoder_logits_fn(
            decoder_params, normalizer_params, prior_cfg
        )
    return _FrozenPrior(
        prior_params=prior_params,
        decoder_params=decoder_params,
        normalizer_params=normalizer_params,
        prior_fn=prior_fn,
        decoder_logits_fn=decoder_logits_fn,
        latent_size=latent_size,
        prior_layer_sizes=prior_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
    )


def _make_env(hydra_cfg, prior: _FrozenPrior, from_scratch: bool) -> _EnvBuild:
    """Builds the training env stack (``_build_env``) and reads its sizes off it."""
    prior_fn, decoder_logits_fn = prior.prior_fn, prior.decoder_logits_fn
    latent_size = prior.latent_size

    # --- 2. Build env ---
    base_env_for_size = tasks.load(str(hydra_cfg.env_name), flatten_obs=False)
    action_size_base = int(base_env_for_size.action_size)
    env, base_env_wrapped, mj_model, mjx_model, vision_shape = _build_env(
        hydra_cfg,
        prior_fn,
        decoder_logits_fn,
        latent_size,
        action_size_base,
        from_scratch=from_scratch,
    )
    obs_size_dict = dict(env.observation_size)
    proprio_size = int(obs_size_dict.get("proprioception", 0))
    task_obs_size = int(obs_size_dict.get("imitation_target", 0))
    action_size = int(env.action_size)
    log.info(
        "env_spec: action_size=%d, proprio=%d, task_obs=%d, vision_shape=%s, latent=%s",
        action_size,
        proprio_size,
        task_obs_size,
        vision_shape,
        "n/a(from_scratch)" if from_scratch else latent_size,
    )
    return _EnvBuild(
        env=env,
        base_env_wrapped=base_env_wrapped,
        mj_model=mj_model,
        mjx_model=mjx_model,
        vision_shape=vision_shape,
        proprio_size=proprio_size,
        task_obs_size=task_obs_size,
        action_size=action_size,
    )


def _env_spec(built: _EnvBuild) -> dict:
    """The unbatched observation template and action size the learner inits from."""
    return {
        "obs_template": {
            "vision": jnp.zeros(tuple(built.vision_shape), dtype=jnp.float32),
            "imitation_target": jnp.zeros((built.task_obs_size,), dtype=jnp.float32),
            "proprioception": jnp.zeros((built.proprio_size,), dtype=jnp.float32),
        },
        "action_size": built.action_size,
    }


def _make_networks(hydra_cfg, setup: _RunSetup, prior: _FrozenPrior, built: _EnvBuild):
    """Builds the DMPO networks for the configured policy head.

    A recurrent head's ``policy.init`` is wrapped to default to a zero hidden,
    so ``init_training_state``'s feed-forward call signature works for it.
    """
    cfg = setup.cfg
    from_scratch, recurrent = setup.from_scratch, setup.recurrent
    slow_core, rnn_head_cfg = setup.slow_core, setup.rnn_head_cfg
    slow_hidden_size, tick_stride = setup.slow_hidden_size, setup.tick_stride
    slow_store_dtype = setup.slow_store_dtype
    critic_privileged_dim = setup.critic_privileged_dim
    prior_params, decoder_params = prior.prior_params, prior.decoder_params
    latent_size = prior.latent_size
    prior_layer_sizes = prior.prior_layer_sizes
    decoder_layer_sizes = prior.decoder_layer_sizes
    proprio_size, task_obs_size = built.proprio_size, built.task_obs_size
    action_size, vision_shape = built.action_size, built.vision_shape

    # FF and recurrent factories share every size source; the recurrent one
    # swaps policy_head_layer_sizes for the policy_head_rnn block (pre-RNN
    # mixing MLP + stacked GRU cells + replay dtype for the stored hidden).
    _shared_net_kwargs = dict(
        proprio_size=proprio_size,
        task_obs_size=task_obs_size,
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=tuple(vision_shape),
        cfg=cfg,
        prior_layer_sizes=prior_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
        cnn_feature_size=int(hydra_cfg.network_config.get("vision_feature_size", 32)),
        cnn_channels=tuple(
            hydra_cfg.network_config.get("vision_channels", [4, 8, 16, 32])
        ),
        mono_channels=1 if hydra_cfg.env_config.get("grayscale", True) else 3,
        shared_weights=hydra_cfg.network_config.get("binocular_mode", "shared")
        == "shared",
        value_hidden_layer_sizes=tuple(
            hydra_cfg.network_config.get(
                "value_hidden_layer_sizes", [512, 512, 512, 512]
            )
        ),
        warm_start_prior_params=prior_params,
        warm_start_decoder_params=decoder_params,
        residual_mode=str(hydra_cfg.network_config.get("residual_mode", "sigma_tanh")),
        residual_scale=float(hydra_cfg.network_config.get("residual_scale", 2.0)),
        critic_use_proprio=bool(
            hydra_cfg.network_config.get("critic_use_proprio", False)
        ),
        # 0 is every factory's default, so the old path is unchanged.
        critic_privileged_dim=critic_privileged_dim,
    )
    if from_scratch:
        if not recurrent:
            raise ValueError(
                "transfer.mode=from_scratch on this entry point requires "
                "network_config.policy_head_rnn: its from-scratch network is "
                "recurrent only (there is no feed-forward from-scratch path)."
            )
        # Drop every prior/decoder-shaped kwarg; the scratch factory's
        # signature is the grafted one minus exactly these keys.
        _scratch_kwargs = {
            k: v
            for k, v in _shared_net_kwargs.items()
            if k
            not in (
                "latent_size",
                "prior_layer_sizes",
                "decoder_layer_sizes",
                "warm_start_prior_params",
                "warm_start_decoder_params",
                "residual_mode",
                "residual_scale",
                # Not a scratch-factory kwarg; > 0 is rejected for
                # from_scratch above, so this only ever drops a 0.
                "critic_privileged_dim",
            )
        }
        nets = make_dmpo_scratch_rnn_networks(
            rnn_cell=str(rnn_head_cfg.get("cell", "gru")),
            rnn_mlp_layers=tuple(rnn_head_cfg.get("mlp_layers", [512])),
            rnn_hidden_sizes=tuple(rnn_head_cfg.get("hidden_sizes", [512])),
            rnn_store_dtype=str(rnn_head_cfg.get("store_dtype", "float16")),
            **_scratch_kwargs,
        )
    elif slow_core:
        # The fast path IS the FF head (policy_head_layer_sizes), so an FF
        # checkpoint grafts in; residual_mode/scale come from network_config
        # via _shared_net_kwargs (the slow factory's own defaults differ from
        # this entry's, so they are always passed explicitly).
        nets = make_dmpo_kl_anchor_slow_networks(
            policy_head_layer_sizes=tuple(
                hydra_cfg.network_config.get("policy_head_layer_sizes", [256, 256, 256])
            ),
            slow_hidden_size=slow_hidden_size,
            tick_stride=tick_stride,
            rnn_cell=setup.slow_cell,
            lstm_forget_bias=setup.slow_forget_bias,
            rnn_store_dtype=slow_store_dtype,
            **_shared_net_kwargs,
        )
        # The startup line below reports the cell the factory built; a factory
        # that built another cell would train this arm under the wrong name.
        if nets.recurrent_meta.cell_type != setup.slow_cell:
            raise ValueError(
                f"policy_head_slow_core.cell={setup.slow_cell!r} but the factory "
                f"built cell_type={nets.recurrent_meta.cell_type!r}."
            )
    elif recurrent:
        nets = make_dmpo_kl_anchor_rnn_networks(
            rnn_cell=str(rnn_head_cfg.get("cell", "gru")),
            rnn_mlp_layers=tuple(rnn_head_cfg.get("mlp_layers", [256])),
            rnn_hidden_sizes=tuple(rnn_head_cfg.get("hidden_sizes", [256])),
            rnn_store_dtype=str(rnn_head_cfg.get("store_dtype", "float16")),
            **_shared_net_kwargs,
        )
    else:
        nets = make_dmpo_kl_anchor_networks(
            policy_head_layer_sizes=tuple(
                hydra_cfg.network_config.get("policy_head_layer_sizes", [256, 256, 256])
            ),
            **_shared_net_kwargs,
        )
    if recurrent:
        # init_training_state calls `nets.policy.init(rng, obs)` — the FF
        # signature. Adapt with a default zero hidden rather than forking
        # init_training_state: GRU param shapes do not depend on hidden
        # VALUES. On the grafted arm the zero-init residual additionally makes
        # the resulting params correct for any hidden (the step-0 invariant);
        # on the from-scratch arm there is no such invariant to preserve, but
        # the shape argument is the same. Same init-wrapping precedent as the
        # grafted factory's own warm-start splice.
        _rnn_policy_init = nets.policy.init

        def _policy_init_default_hidden(rng, obs, hidden=None):
            if hidden is None:
                hidden = nets.recurrent_meta.init_hidden()
            return _rnn_policy_init(rng, obs, hidden)

        nets.policy.init = _policy_init_default_hidden  # type: ignore[method-assign]
        # Startup validation marker (smoke gates grep for the echo).
        if slow_core:
            log.info(
                "RECURRENT policy head ACTIVE: SLOW CORE cell=%s "
                "forget_bias=%g hidden_size=%d tick_stride=%d store_dtype=%s "
                "policy_head_layer_sizes=%s | rnn_bptt_length=%d n_step=%d "
                "sequence_length=%d (= L*k + n) cfg.rnn_tick_stride=%d",
                nets.recurrent_meta.cell_type,
                setup.slow_forget_bias,
                nets.recurrent_meta.hidden_sizes[0],
                nets.recurrent_meta.tick_stride,
                nets.recurrent_meta.store_dtype,
                tuple(
                    hydra_cfg.network_config.get(
                        "policy_head_layer_sizes", [256, 256, 256]
                    )
                ),
                int(cfg.rnn_bptt_length),
                int(cfg.n_step),
                int(cfg.sequence_length),
                int(cfg.rnn_tick_stride),
            )
        else:
            log.info(
                "RECURRENT policy head ACTIVE: cell=%s mlp_layers=%s "
                "hidden_sizes=%s store_dtype=%s | rnn_bptt_length=%d n_step=%d "
                "sequence_length=%d (= L + n)",
                nets.recurrent_meta.cell_type,
                tuple(rnn_head_cfg.get("mlp_layers", [256])),
                nets.recurrent_meta.hidden_sizes,
                nets.recurrent_meta.store_dtype,
                int(cfg.rnn_bptt_length),
                int(cfg.n_step),
                int(cfg.sequence_length),
            )
    if from_scratch:
        log.info(
            "no latent residual (from-scratch: policy -> action directly) | "
            "critic_use_proprio=%s",
            bool(hydra_cfg.network_config.get("critic_use_proprio", False)),
        )
    else:
        log.info(
            "latent residual: mode=%s scale=%.3g | critic_use_proprio=%s",
            str(hydra_cfg.network_config.get("residual_mode", "sigma_tanh")),
            float(hydra_cfg.network_config.get("residual_scale", 2.0)),
            bool(hydra_cfg.network_config.get("critic_use_proprio", False)),
        )
    return nets


class _EvalResult(NamedTuple):
    """What one eval of the training entry computed (see ``_run_entry_eval``).

    ``metrics`` holds the keys the eval logs as ``eval/<key>`` (None when the
    rollout came back empty), ``spout`` the ``spout/<key>`` summary (None off the
    spout maze, or when that summary failed), ``maze`` the ``maze/<key>``
    summary (None off the full maze, or when that summary failed) and
    ``vision_sensitivity`` the ``eval/vision_sensitivity`` of a recurrent head
    (None otherwise). The other fields are what the eval video is rendered
    from.
    """

    metrics: dict | None
    spout: dict | None
    maze: dict | None
    vision_sensitivity: float | None
    rollout: list
    term_events: list
    erc: dict
    remix_key: str | None
    remix_lambda: float | None


def _run_entry_eval(
    hydra_cfg,
    cfg: DMPOConfig,
    K: int,
    nets,
    recurrent: bool,
    eval_env,
    spout_eval,
    state,
    env_steps: int,
    k_eval: jax.Array,
) -> _EvalResult:
    """One eval of ``state``: the eval rollout, its metrics and their log lines.

    The training run's eval callback calls this at every eval and then renders
    the eval video from the result. ``scripts/eval_dmpo_checkpoints.py`` calls
    it on saved checkpoints, so a post-hoc eval runs the same rollout and
    reports the same keys. Metrics go to stdout, and to wandb when a wandb run
    is active.

    On the full maze (``_full_maze_env``) it also logs one ``MAZE_EVAL ...``
    line, from ``vnl_playground.tasks.rodent.maze_eval``.

    Args:
        hydra_cfg: The entry's hydra config (``eval_render_config`` is read;
            ``env_config.action_repeat`` on the full maze).
        cfg: The resolved learner config (``_resolve_run_setup``).
        K: SGD updates per rollout, for the reward-anneal lambda.
        nets: The DMPO networks (``_make_networks``).
        recurrent: Whether the policy head is recurrent.
        eval_env: The eval env (``_make_eval_env``).
        spout_eval: The spout eval module, or None (``_make_eval_env``).
        state: A ``TrainingState``; ``policy_params``, ``normalizer_params``
            and ``steps`` are read.
        env_steps: The env step the eval is reported at.
        k_eval: PRNG key of the eval rollout.
    """
    sp = sm = rm = sens = None

    erc_raw = hydra_cfg.get("eval_render_config", {})
    erc = OmegaConf.to_container(erc_raw, resolve=True) if erc_raw else {}
    eval_episode_length = int(erc.get("episode_length", 1000))
    rollout, term_events, _batch_rm_unmixed, allenv = run_eval_rollout_envzero(
        env=eval_env,
        policy_apply=nets.policy.apply,
        params=state.policy_params,
        rng=k_eval,
        episode_length=eval_episode_length,
        num_envs=cfg.num_envs,
        normalizer_params=state.normalizer_params,
        # None on every FF arm (DMPONetworks default) => byte-identical
        # eval; a RecurrentPolicyMeta switches the eval scan to the
        # threaded-hidden recurrent apply.
        recurrent_meta=nets.recurrent_meta,
    )
    # Rebuild the reward the replay buffer actually stored. lambda comes
    # from state.steps via the SAME expression the fused training step
    # uses (`reward_anneal_lambda(env_steps_estimate(state.steps, cfg,
    # K), cfg)` in train_dmpo_step.make_fused_train_step) -- the host
    # env_steps int is a different quantity and would drift from the
    # buffer.
    remix_key = getattr(cfg, "reward_anneal_sparse_key", None)
    remix_lambda = (
        float(reward_anneal_lambda(env_steps_estimate(state.steps, cfg, K), cfg))
        if remix_key is not None
        else None
    )
    if spout_eval is not None:
        # Own handler: a failure here must not take the video with it,
        # and its absence from the log must be explained.
        try:
            sp = spout_eval.spout_eval_summary(allenv)
            log.info("%s", spout_eval.spout_eval_log_line(env_steps, sp))
            if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                wandb.log(
                    {f"spout/{k}": v for k, v in sp.items()}
                    | {"env_steps": int(env_steps)},
                    step=int(env_steps),
                    commit=False,
                )
        except Exception as exc:  # logged loudly
            log.warning("SPOUT_EVAL failed: %s", exc, exc_info=True)
    maze_env = _full_maze_env(eval_env)
    maze_alias = None
    if maze_env is not None:
        # Own handler, for the same reason as SPOUT_EVAL above.
        try:
            from vnl_playground.tasks.rodent import maze_eval

            maze_alias = maze_eval.MAZE_ALIAS
            # Seconds per eval step: the env's control dt times the repeat.
            step_dt = float(maze_env._config.ctrl_dt) * int(
                hydra_cfg.get("env_config", {}).get("action_repeat", 1)
            )
            sm = maze_eval.maze_eval_summary(allenv, step_dt)
            log.info("%s", maze_eval.maze_eval_log_line(env_steps, sm))
            if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                wandb.log(
                    {f"maze/{k}": v for k, v in sm.items()}
                    | {"env_steps": int(env_steps)},
                    step=int(env_steps),
                    commit=False,
                )
        except Exception as exc:  # logged loudly
            sm = None
            log.warning("MAZE_EVAL failed: %s", exc, exc_info=True)
    if rollout:
        rm = compute_rollout_metrics(rollout, remix_key, remix_lambda)
        # All-env estimator. `rm` starts as the legacy env-0 statistic,
        # kept so runs stay comparable to older logs; the `batch/` keys
        # are the ones to judge arms on (env 0 alone is a very noisy
        # n=1 estimate -- see compute_batch_rollout_metrics).
        rm.update(compute_batch_rollout_metrics(allenv, remix_key, remix_lambda))
        if spout_eval is not None:
            # Drop the mirrored keys (SPOUT_ALIAS + <name> and their
            # batch/ and _mean/_sum/... derivatives, which all keep the
            # full prefix); SPOUT_EVAL reports them properly.
            alias = spout_eval.SPOUT_ALIAS
            rm = {k: v for k, v in rm.items() if alias not in k}
        if maze_alias is not None:
            # Same for the full maze's mirrored keys; MAZE_EVAL reports them.
            rm = {k: v for k, v in rm.items() if maze_alias not in k}
        # Mirror eval metrics to stdout, so the run log alone answers
        # questions like "did it ever cross a gap?" without parsing
        # wandb's binary log.
        log.info(
            "eval env_steps=%d %s",
            int(env_steps),
            " ".join(
                f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in sorted(rm.items())
            ),
        )
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(
                {f"eval/{k}": v for k, v in rm.items()}
                | {
                    "env_steps": int(env_steps),
                    "num_steps_thousands": float(env_steps) / 1000.0,
                },
                step=int(env_steps),
                commit=False,
            )
    if recurrent and rollout and "vision" in rollout[0].obs:
        # Recurrent-only (FF kl-anchor arms never logged this key, so
        # adding it unguarded would break FF log-identity). Mirrors the
        # vision-sensitivity probe in scripts/train_dmpo_highlvl.py, with
        # the recurrent apply.
        mid = len(rollout) // 2
        sens = compute_vision_sensitivity(
            nets.policy.apply,
            state.policy_params,
            rollout[mid].obs,
            k_eval,
            # Zeros hidden: a step-0 approximation under recurrence (the mid-rollout hidden is not retained by the env-0 slice).
            hidden=nets.recurrent_meta.init_hidden(),
        )
        log.info(
            "eval env_steps=%d vision_sensitivity=%.6g",
            int(env_steps),
            sens,
        )
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(
                {"eval/vision_sensitivity": sens, "env_steps": int(env_steps)},
                step=int(env_steps),
                commit=False,
            )
    return _EvalResult(
        metrics=rm,
        spout=sp,
        maze=sm,
        vision_sensitivity=sens,
        rollout=rollout,
        term_events=term_events,
        erc=erc,
        remix_key=remix_key,
        remix_lambda=remix_lambda,
    )


def _report_eval_failure(env_steps: int, exc: Exception) -> None:
    """Report a failed in-run eval at ERROR and as ``eval/failed=1`` in wandb.

    Training goes on without that eval's metrics; the ERROR line and the wandb
    key make the gap visible in the log and on the dashboard.
    """
    log.error(
        "Eval failed at env_steps=%d (eval/failed=1); training continues: %s",
        int(env_steps),
        exc,
        exc_info=exc,
    )
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        try:
            wandb.log(
                {"eval/failed": 1, "env_steps": int(env_steps)},
                step=int(env_steps),
                commit=False,
            )
        except Exception as log_exc:  # the eval failure is already logged
            log.error("Could not log eval/failed to wandb: %s", log_exc)


def _eval_and_render(
    hydra_cfg,
    cfg: DMPOConfig,
    K: int,
    nets,
    recurrent: bool,
    eval_env,
    spout_eval,
    state,
    env_steps: int,
    k_eval: jax.Array,
    *,
    mj_model,
    ckpt_dir: str,
) -> _EvalResult | None:
    """One in-run eval: its metrics (``_run_entry_eval``), then its video.

    The training run's eval callback calls this at every eval. The two stages
    fail separately and neither raises, so training goes on either way:

    * a failed metric eval is logged at ERROR and as ``eval/failed=1``
      (``_report_eval_failure``), and nothing is rendered;
    * a failed render is a WARNING ("Eval render failed"); the metrics,
      already logged by then, stand.

    Args:
        hydra_cfg, cfg, K, nets, recurrent, eval_env, spout_eval, state,
        env_steps, k_eval: As for ``_run_entry_eval``.
        mj_model: The MuJoCo model the video is rendered with; None renders
            nothing.
        ckpt_dir: Directory the video is written to, as ``eval_<env_steps>.mp4``.

    Returns:
        The eval result, or None when the metric eval failed.
    """
    try:
        ev = _run_entry_eval(
            hydra_cfg,
            cfg,
            K,
            nets,
            recurrent,
            eval_env,
            spout_eval,
            state,
            env_steps,
            k_eval,
        )
    except Exception as exc:
        _report_eval_failure(env_steps, exc)
        return None
    rollout, term_events, erc = ev.rollout, ev.term_events, ev.erc
    remix_key, remix_lambda = ev.remix_key, ev.remix_lambda
    if mj_model is None or not rollout:
        return ev
    try:
        video_path = Path(ckpt_dir) / f"eval_{int(env_steps)}.mp4"
        hud_cfg = erc.get("hud") if isinstance(erc, dict) else None
        rew_cfg_raw = hydra_cfg.get("env_config", {}).get("reward_terms", None)
        rew_cfg = (
            OmegaConf.to_container(rew_cfg_raw, resolve=True)
            if rew_cfg_raw is not None
            else None
        )
        # `camera` below is a dead parameter (never forwarded --
        # render_video builds its own tracking camera). `cameras` is
        # the live one: a list of panel specs rendered side by side.
        # Absent -> the historical single camera, unchanged.
        #
        # `erc` has ALREADY been converted to a plain dict by the time
        # we get here (see the isinstance check above), so `cameras` is
        # a plain list and OmegaConf.to_container REJECTS it --
        # "Input cfg is not an OmegaConf config object (list)". That
        # raises inside the eval-render try/except, which logs a
        # warning and continues, so the eval produces no video while
        # training carries on looking healthy. Convert only when it
        # really is an OmegaConf node.
        cam_specs = erc.get("cameras", None)
        if OmegaConf.is_config(cam_specs):
            cam_specs = OmegaConf.to_container(cam_specs, resolve=True)
        render_eval_video(
            rollout,
            mj_model,
            video_path,
            fps=int(erc.get("fps", 50)),
            height=int(erc.get("height", 480)),
            width=int(erc.get("width", 640)),
            camera=str(erc.get("camera", "close_profile-rodent")),
            cameras=cam_specs,
            hud_config=hud_cfg,
            reward_config=rew_cfg,
            termination_events=term_events,
            reward_remix=(
                {"sparse_key": remix_key, "lambda": remix_lambda}
                if remix_key is not None
                else None
            ),
        )
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(
                {"videos/eval": wandb.Video(str(video_path), format="mp4")},
                step=int(env_steps),
            )
    except Exception as e:
        log.warning("Eval render failed: %s", e, exc_info=True)
    return ev


@hydra.main(
    config_path="../track_mjx/config/dmpo",
    config_name="run_gap/velocity_only_kl_anchor",
    version_base=None,
)
def main(hydra_cfg: DictConfig):
    setup = _resolve_run_setup(hydra_cfg)
    cfg = setup.cfg
    anchor_extras = setup.anchor_extras
    from_scratch = setup.from_scratch
    recurrent = setup.recurrent
    slow_core = setup.slow_core
    critic_privileged_dim = setup.critic_privileged_dim
    priv_extras = setup.priv_extras
    graft_path = setup.graft_path

    iters_per_chunk = int(hydra_cfg.train_config.get("iters_per_chunk", 32))
    # Every checkpoint records the alignment settings it was trained with.
    cfg_dict = record_alignment(OmegaConf.to_container(hydra_cfg, resolve=True), cfg)
    seed = int(hydra_cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)

    config_name = str(
        hydra_cfg.get("logging_config", {}).get("exp_name", "dmpo-kl-anchor")
    )
    git_sha = detect_git_sha(Path(__file__).resolve().parents[1])
    run_id = make_run_id(config_name, seed, git_sha)
    log.info("wandb run_id=%s", run_id)

    ckpt_dir = str(hydra_cfg.get("checkpoint_dir", "./checkpoints/dmpo_kl_anchor"))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # A resume must train with the settings the run started with. Checked
    # before the env is built and before any state is restored; keep this call
    # above every restore.
    resume_alignment = check_resume_alignment(
        ckpt_dir,
        cfg,
        override_key=RESUME_OVERRIDE_KEY,
        allow=bool(
            hydra_cfg.train_config.get(RESUME_OVERRIDE_KEY.split(".")[-1], False)
        ),
    )
    existing = load_dmpo_wandb_state(ckpt_dir)
    if _WANDB_IMPORTED:
        try:
            wandb.init(
                project=str(
                    hydra_cfg.get("logging_config", {}).get(
                        "project_name", "dmpo-rodent"
                    )
                ),
                config=cfg_dict,
                mode=os.environ.get("WANDB_MODE", "online"),
                # id keeps the sha-bearing run_id: it is the resume key and the
                # provenance record (the config commit is embedded in it).
                # name is the human-facing label and is just exp_name, so a
                # config called `sweep_noise_sig_0.1_sparse_seed_0` shows up in
                # wandb under exactly that, not with a `_seed0_g1234567` tail.
                id=existing["wandb_run_id"] if existing else run_id,
                name=config_name,
                resume="must" if existing else "allow",
                group=str(
                    hydra_cfg.get("logging_config", {}).get("group_name", "kl-anchor")
                ),
                notes=str(hydra_cfg.get("logging_config", {}).get("notes", "")),
                reinit=True,
            )
            save_dmpo_wandb_state(
                ckpt_dir, run_id if not existing else existing["wandb_run_id"]
            )
        except Exception as e:
            log.warning("wandb.init failed (%s); continuing without wandb.", e)

    # --- 1. Load frozen prior + decoder (grafted arms only) ---
    prior = _load_frozen_prior(hydra_cfg, from_scratch)
    normalizer_params = prior.normalizer_params

    # --- 2. Build env ---
    built = _make_env(hydra_cfg, prior, from_scratch)
    env, base_env_wrapped = built.env, built.base_env_wrapped
    mj_model, action_size = built.mj_model, built.action_size

    # --- 3. Build B-aggressive networks (warm-started) ---
    nets = _make_networks(hydra_cfg, setup, prior, built)

    # --- 4. Asymmetric optimizers ---
    opt_kwargs = {
        "prior_lr_mult": float(hydra_cfg.kl_anchor.get("prior_lr_mult", 0.1)),
        "decoder_lr_mult": float(hydra_cfg.kl_anchor.get("decoder_lr_mult", 1.0)),
        "policy_head_lr_mult": float(
            hydra_cfg.kl_anchor.get("policy_head_lr_mult", 1.0)
        ),
    }
    optimizers = make_kl_anchor_optimizers(cfg, **opt_kwargs)

    # --- 5. State + replay ---
    env_spec = _env_spec(built)
    # Replay schema, shaped by the compression flags (see DMPOConfig).
    obs_template = env_spec["obs_template"]
    if bool(getattr(cfg, "vision_uint8_storage", False)):
        if not (isinstance(obs_template, dict) and "vision" in obs_template):
            raise ValueError(
                "vision_uint8_storage=true but the obs template has no "
                f"'vision' key: {type(obs_template).__name__}"
            )
        obs_template = {
            **obs_template,
            "vision": jnp.zeros(obs_template["vision"].shape, dtype=jnp.uint8),
        }
    transition_template = {
        "observation": obs_template,
        "action": jnp.zeros((action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
    }
    # Post-step time-out flag, stored when the learner bootstraps through time
    # limits (the rollout writes it under the same switch).
    if cfg.bootstrap_on_truncation:
        transition_template["truncation"] = jnp.zeros((), dtype=jnp.float32)
    # Only reserved when the loss-side anchor is on -- see `anchor_in_loss`.
    for _k in anchor_extras:
        transition_template[_k] = jnp.zeros((action_size,), dtype=jnp.float32)
    # Critic-only privileged vector, stored from the PRE-step state like the
    # anchor extras (absent when critic_privileged_dim == 0). Its dtype is the
    # one the env writes: env_config.maze_critic_priv_dtype on the full maze
    # (float32 unless set; the spout maze always writes float32). The startup
    # probe below checks it against the env.
    priv_dtype = jnp.dtype(
        str(hydra_cfg.env_config.get("maze_critic_priv_dtype", "float32"))
    )
    for _k in priv_extras:
        transition_template[_k] = jnp.zeros((critic_privileged_dim,), dtype=priv_dtype)
    if bool(getattr(cfg, "store_next_observation", True)):
        transition_template["next_observation"] = obs_template
    else:
        eff_n = min(int(cfg.n_step), int(cfg.sequence_length) - 1)
        log.info(
            "Compressed replay schema: next_observation dropped "
            "(bootstrap from observation[:, n]; effective n-step = "
            "min(n_step=%d, sequence_length-1=%d) = %d), vision_uint8=%s",
            int(cfg.n_step),
            int(cfg.sequence_length) - 1,
            eff_n,
            bool(getattr(cfg, "vision_uint8_storage", False)),
        )
    if recurrent:
        # Per-step PRE-step hidden (the state the policy consumed at this
        # obs) — collect_rollout stores it, the learner unrolls BPTT from
        # it. Unbatched leaf shapes like every other template leaf;
        # store_dtype (f16) so a GRU-256 hidden costs 512 B/transition
        # instead of 1 KB. Added BEFORE the _tx_bytes line so the replay
        # sizing log below accounts for it.
        transition_template["policy_hidden"] = jax.tree.map(
            lambda h: h.astype(nets.recurrent_meta.store_dtype),
            nets.recurrent_meta.init_hidden(),
        )
    # Per-transition storage cost, for buffer sizing against GPU memory.
    _tx_bytes = sum(
        x.size * x.dtype.itemsize for x in jax.tree.leaves(transition_template)
    )
    log.info(
        "Replay transition size: %.2f KB -> max_replay_size=%d is %.2f GB "
        "(%d steps/env deep at num_envs=%d = %.1f rollouts)",
        _tx_bytes / 1024,
        cfg.max_replay_size,
        _tx_bytes * (cfg.max_replay_size // cfg.num_envs) * cfg.num_envs / 2**30,
        cfg.max_replay_size // cfg.num_envs,
        cfg.num_envs,
        (cfg.max_replay_size // cfg.num_envs) / cfg.unroll_length,
    )
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    # init_training_state builds state with the DEFAULT optimizer; replace
    # the policy_opt_state with one initialized by the kl-anchor multi_transform
    # optimizer so update() can resolve per-block inner_states.
    pol_opt_kl, _, _ = optimizers
    state = state._replace(policy_opt_state=pol_opt_kl.init(state.policy_params))
    # Seed DMPO's running-stats normalizer with the imit checkpoint's proprio
    # stats. Without this, the warm-started prior+decoder receive un-normalized
    # proprio at step 0 and produce garbage — breaking the warm-start invariant
    # that the entire B-aggressive design depends on. Done BEFORE restore_ckpt
    # so a resumed run will overwrite this with its own (already-seeded-and-
    # updated) normalizer from disk.
    if from_scratch:
        # Nothing to seed from, and nothing that needs seeding: the reason the
        # grafted arm seeds is to keep the FROZEN prior+decoder in-distribution
        # at step 0. With no frozen pipeline, DMPO's own running stats warm up
        # from the rollouts like any from-scratch run.
        log.info(
            "From-scratch arm: normalizer NOT seeded (no imit checkpoint; "
            "running stats warm up from rollouts)."
        )
    else:
        state = state._replace(
            normalizer_params=seed_proprio_from_imit(
                state.normalizer_params,
                normalizer_params,
            )
        )
        log.info(
            "Seeded DMPO normalizer with imit proprio stats: mean=%.3f±%.3f std=%.3f±%.3f",
            float(state.normalizer_params.proprioception.mean.mean()),
            float(state.normalizer_params.proprioception.mean.std()),
            float(state.normalizer_params.proprioception.std.mean()),
            float(state.normalizer_params.proprioception.std.std()),
        )

    # ---- DMPO warm start from ANOTHER run's checkpoint ------------------------
    # transfer.warm_start_dmpo_checkpoint points at a DMPONetwork_<step> dir of a
    # previous DMPO run. Grafts policy_params + target_policy_params +
    # normalizer_params into the fresh state (critic/duals/optimizers stay
    # fresh -- the critic may have a different atom count than the source run,
    # and DMPOConfig.critic_warmup_sgd_steps exists precisely so the fresh
    # critic can fit before the warm policy is allowed to move). The grafted
    # policy also becomes frozen_behavior_params for behavior mixing
    # (cfg.behavior_mix_init). Done BEFORE restore_ckpt so that resuming a
    # warm-started run restores its own state over the graft, and AFTER the
    # normalizer seeding, which it supersedes (the source run's normalizer is
    # the converged version of the seeded one).
    frozen_behavior_params = None
    ws_path = None
    if "transfer" in hydra_cfg:
        ws_path = hydra_cfg.transfer.get("warm_start_dmpo_checkpoint", None)
    if ws_path:
        import flax.serialization as _flax_ser

        from track_mjx.agent.dmpo.checkpoint import load_train_state_items_numpy

        ws = load_train_state_items_numpy(str(ws_path))
        ws_policy = _flax_ser.from_state_dict(state.policy_params, ws["policy_params"])
        ws_target = _flax_ser.from_state_dict(
            state.target_policy_params, ws["target_policy_params"]
        )
        ws_norm = _flax_ser.from_state_dict(
            state.normalizer_params, ws["normalizer_params"]
        )
        state = state._replace(
            policy_params=ws_policy,
            target_policy_params=ws_target,
            normalizer_params=ws_norm,
        )
        # Optimizer state must be re-initialized on the grafted params (Adam
        # moments start at zero for the trainable blocks, as intended).
        state = state._replace(policy_opt_state=pol_opt_kl.init(state.policy_params))
        frozen_behavior_params = ws_policy
        n_params = sum(x.size for x in jax.tree.leaves(ws_policy))
        log.info(
            "DMPO WARM START: grafted policy(+target)+normalizer from %s "
            "(%d policy params). Critic/duals/optimizers are FRESH; "
            "critic_warmup_sgd_steps=%d behavior_mix_init=%.2f",
            ws_path,
            n_params,
            int(getattr(cfg, "critic_warmup_sgd_steps", 0)),
            float(getattr(cfg, "behavior_mix_init", 0.0)),
        )
    elif float(getattr(cfg, "behavior_mix_init", 0.0)) > 0.0:
        raise ValueError(
            "train_config.behavior_mix_init > 0 requires "
            "transfer.warm_start_dmpo_checkpoint (the frozen behavior policy "
            "is the warm-start policy)."
        )

    # Retention is the `checkpoint_max_to_keep` config key. Absent or null
    # means KEEP EVERY CHECKPOINT (make_checkpointer's default): a diverged run
    # that keeps training would otherwise prune its own last healthy
    # checkpoints. With eval_every_steps=25M over a 1B budget that is 40
    # checkpoints at ~58 MB each, ~2.3 GB total.
    max_to_keep = hydra_cfg.get("checkpoint_max_to_keep", None)
    ckpt_mgr = make_checkpointer(
        ckpt_dir, max_to_keep=None if max_to_keep is None else int(max_to_keep)
    )
    log.info(
        "Checkpoint retention: %s",
        "ALL (keeping every step)"
        if max_to_keep is None
        else f"last {int(max_to_keep)}",
    )
    try:
        restored = _restore_for_resume(
            ckpt_mgr, state, cfg, resume_alignment, opt_kwargs
        )
    except Exception as exc:
        if recurrent:
            # The most likely cause: checkpoint_dir points at an FF run's
            # checkpoints and policy_head_rnn was newly enabled — orbax
            # cannot match the FF param tree to the recurrent template.
            raise ValueError(
                f"Failed to restore from checkpoint_dir={ckpt_dir} with "
                f"{'policy_head_slow_core' if slow_core else 'policy_head_rnn'} "
                "enabled. An FF run's checkpoints cannot be "
                "restored into a recurrent param tree; point checkpoint_dir "
                "at a fresh directory for RNN arms (or disable the RNN head "
                "to resume the FF run; to START from an FF checkpoint use "
                f"transfer.graft_ff_checkpoint). Original error: {exc}"
            ) from exc
        raise
    # Checkpoints are saved at `step=total_env_steps`, so the manager's latest
    # step IS the env-step count to resume the training-loop counter from. Without
    # this the loop restarted counting at 0 and a resumed run would train
    # `num_timesteps` MORE steps under colliding checkpoint names.
    start_env_steps = 0
    if restored is not None:
        start_env_steps = int(ckpt_mgr.latest_step() or 0)
        log.info(
            "Restored DMPO checkpoint: sgd_step=%d env_steps=%d (resuming counter)",
            int(restored.steps),
            start_env_steps,
        )
        state = restored

    # ---- transfer.graft_ff_checkpoint: FRESH runs only ------------------------
    # After the restore attempt, so a resume (e.g. the restart after an OOM)
    # keeps its own trained state and never re-grafts the source over it. The
    # restore template is the fresh state, whose shapes the graft preserves.
    graft_info = None
    if graft_path:
        if restored is not None:
            log.info("GRAFT skipped: resuming from %d", start_env_steps)
        else:
            state, graft_info = _apply_graft(state, str(graft_path), optimizers)
            rep_p, rep_c = graft_info["policy"], graft_info["critic"]
            widened = [f"policy:{p}" for p in rep_p["widened"]]
            new = [f"policy:{p}" for p in rep_p["new"]]
            copied = rep_p["copied"]
            if rep_c is not None:
                widened += [f"critic:{p}" for p in rep_c["widened"]]
                new += [f"critic:{p}" for p in rep_c["new"]]
                copied += rep_c["copied"]
            log.info(
                "GRAFT ok copied=%d widened=%s new=%s critic=%s",
                copied,
                _fmt_paths(widened),
                _fmt_paths(new),
                graft_info["critic_mode"],
            )
            log.info(
                "GRAFT source=%s: policy+target (copied=%d leaves each), "
                "critic+target (%s), normalizer grafted; policy/critic "
                "optimizer states re-initialized, duals fresh. copied counts "
                "the online policy + online critic leaves.",
                graft_path,
                rep_p["copied"],
                f"copied={rep_c['copied']} leaves each" if rep_c else "fresh init",
            )
            rng, k_graft = jax.random.split(rng)
            pol_diff, crit_diff = _graft_invariance_probe(
                nets,
                state,
                env_spec["obs_template"],
                action_size,
                critic_privileged_dim if rep_c is not None else 0,
                k_graft,
            )
            log.info(
                "GRAFT invariance: policy_max_abs_diff=%s critic_max_abs_diff=%s "
                "(output change when ONLY the new inputs -- slow hidden / "
                "privileged vector -- change; must be ~0)",
                "n/a(FF: exact copy)" if pol_diff is None else f"{pol_diff:.3g}",
                "n/a" if crit_diff is None else f"{crit_diff:.3g}",
            )
            if max(pol_diff or 0.0, crit_diff or 0.0) > 1e-4:
                raise RuntimeError(
                    "GRAFT invariance violated: the new inputs reach the "
                    f"outputs at step 0 (policy {pol_diff}, critic "
                    f"{crit_diff}); the widened rows are not zero."
                )

    # ---- Startup invariant probe -------------------------------------------------
    # Run a single env step with the warm-started policy and log the wrapper's
    # anchor metrics. This is a tripwire: a fresh kl-anchor pipeline with
    # working warm-start should observe r_anchor very close to 1.0 (the policy's
    # mode action equals tanh(mu_imit_pretanh) up to numerics). If r_anchor drops,
    # the prior/decoder warm-start splice (or the normalizer seeding) regressed.
    priv_probe_shape = priv_probe_dtype = None
    if from_scratch:
        # st0.info has no `anchor_mu_imit` -- EndToEndWrapper does not populate
        # it. Skipping explicitly beats letting the probe raise into its own
        # except handler and logging a warning that looks like a regression.
        log.info(
            "anchor_invariant_probe SKIPPED (from_scratch: no frozen "
            "prior/decoder to be invariant to)"
        )
    else:
        try:
            from track_mjx.agent.dmpo.kl_anchor_utils import pretanh_gaussian_kl
            from track_mjx.agent.dmpo.learner import _normalize_obs

            probe_label = (
                "anchor_invariant_probe_resumed"
                if restored is not None
                else "anchor_invariant_probe"
            )
            if ws_path and restored is None:
                # A DMPO-warm-started policy head has learned away from the
                # anchor BY DESIGN -- r_anchor < 1 here is expected, not a
                # regression of the prior/decoder warm-start splice.
                probe_label = (
                    "anchor_invariant_probe_dmpo_warmstart(r_anchor<1 expected)"
                )
            if graft_info is not None:
                # Same reason: the grafted head is a trained DMPO head, so its
                # residual is not zero. The graft's own step-0 check is the
                # "GRAFT invariance" line above, not this number.
                probe_label = "anchor_invariant_probe_graft(r_anchor<1 expected)"
            rng_probe, k_probe = jax.random.split(rng)
            keys = jax.random.split(k_probe, cfg.num_envs)
            st0 = env.reset(keys)
            if priv_extras:
                priv_probe_shape = tuple(st0.info[priv_extras[0]].shape)
                priv_probe_dtype = jnp.dtype(st0.info[priv_extras[0]].dtype)
            norm_obs = _normalize_obs(st0.obs, state.normalizer_params)
            # Online policy distribution at the spawn obs.
            if recurrent:
                # Per-env zeros hidden, matching what the first rollout will
                # consume. The invariant is hidden-AGNOSTIC by design (zero-init
                # residual), so probing at zeros loses no coverage.
                dist0, _ = jax.vmap(
                    lambda o, h: nets.policy.apply(state.policy_params, o, h)
                )(norm_obs, nets.recurrent_meta.init_hidden(cfg.num_envs))
            else:
                dist0 = jax.vmap(lambda o: nets.policy.apply(state.policy_params, o))(
                    norm_obs
                )
            mu_theta = dist0.mean()
            log_std_theta = jnp.log(dist0.stddev())
            # Anchor distribution from state.info — populated by the wrapper.
            # Both wrapper and loss compute log_std as log(softplus(raw)+1e-3) so the
            # KL on this side is in the same units as the loss-side anchor signal.
            mu_imit = st0.info["anchor_mu_imit"]
            log_std_imit = st0.info["anchor_log_std_imit"]
            kl = pretanh_gaussian_kl(mu_theta, log_std_theta, mu_imit, log_std_imit)
            # kl is per-sample shape (num_envs,); aggregate via mean(exp(-w*kl))
            # which matches the loss-side r_anchor formula.
            r_anchor = float(jnp.mean(jnp.exp(-cfg.kl_anchor_w * kl)))
            kl_mean = float(jnp.mean(kl))
            # Format keeps `action_mse=` for smoke-test regex compat (post-port
            # the wrapper's MSE diagnostic is no longer relevant for the probe;
            # we report kl_mean here as the meaningful diagnostic).
            log.info(
                "%s r_anchor=%.4f action_mse=%.4f kl_mean=%.4f",
                probe_label,
                r_anchor,
                0.0,
                kl_mean,
            )
            rng = rng_probe
        except Exception as exc:
            log.warning(
                "Startup invariant probe failed (non-fatal): %s", exc, exc_info=True
            )
    if priv_extras:
        # The env's privileged vector must match the critic's input width and
        # the replay template. Checked outside the non-fatal probe handler.
        if priv_probe_shape is None:
            log.warning(
                "critic_priv width NOT verified at startup (the probe did not "
                "reach env.reset); the replay insert will still fail on a "
                "mismatch."
            )
        elif priv_probe_shape != (cfg.num_envs, critic_privileged_dim):
            raise ValueError(
                f"env info[{priv_extras[0]!r}] has shape {priv_probe_shape}, "
                f"expected (num_envs, critic_privileged_dim) = "
                f"({cfg.num_envs}, {critic_privileged_dim})."
            )
        elif priv_probe_dtype != transition_template[priv_extras[0]].dtype:
            raise ValueError(
                f"env info[{priv_extras[0]!r}] has dtype {priv_probe_dtype}, but "
                f"the replay field is {transition_template[priv_extras[0]].dtype} "
                "(env_config.maze_critic_priv_dtype)."
            )
        else:
            log.info(
                "critic_priv width verified: env info[%r] %s %s == "
                "critic_privileged_dim %d",
                priv_extras[0],
                priv_probe_shape,
                priv_probe_dtype,
                critic_privileged_dim,
            )
    # ----------------------------------------------------------------------------

    rb = make_replay(
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    rb_state = rb.init(transition_template)

    # SGD updates per rollout. `sgd_steps_per_rollout`, when set, pins K directly;
    # otherwise fall back to the historical expression so every config without it
    # stays bit-reproducible. NOTE that historical expression DIVIDES by
    # samples_per_insert, inverting the Acme/Reverb meaning of the knob -- see the
    # long note on DMPOConfig.sgd_steps_per_rollout. Do not "fix" it in place.
    K = resolve_sgd_steps_per_rollout(cfg)
    log.info(
        "K=%d SGD updates/rollout via %s | %s "
        "(Acme/Ray reference on the gap task: realized_samples_per_insert=3.236)",
        K,
        "sgd_steps_per_rollout (explicit; samples_per_insert is unread)"
        if cfg.sgd_steps_per_rollout is not None
        else f"legacy inverted formula from samples_per_insert={cfg.samples_per_insert}",
        " ".join(f"{k}={v:.4g}" for k, v in realized_ratios(cfg, K).items()),
    )
    log.info("DMPO kl-anchor: K=%d SGD updates per rollout", K)

    # --- 6. Callbacks ---
    def wandb_log_cb(payload, env_steps):
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            # `num_steps_thousands` mirrors the track-mjx PPO convention
            # (track_mjx.agent.wandb_logging, where PPO's env_steps is ALREADY
            # in thousands) so DMPO and PPO runs can share an x-axis in wandb.
            # It is added as a METRIC, exactly as PPO does -- the wandb `step=`
            # stays raw env_steps on purpose: wandb steps must increase
            # monotonically, and a run resumed at e.g. 297.7M env steps has
            # already logged steps up to 297,676,800. Switching the step
            # counter to thousands would emit 297,676 next and wandb would
            # drop or misorder every subsequent point.
            payload = dict(payload)
            payload["num_steps_thousands"] = float(env_steps) / 1000.0
            wandb.log(payload, step=int(env_steps))
        # Also mirror the anchor metric to stdout so non-wandb runs (smoke
        # tests, offline) can verify the warm-start invariant without
        # parsing wandb's binary log.
        if "anchor/r_anchor" in payload:
            log.info(
                "anchor_metrics env_steps=%d r_anchor=%.4f action_mse=%.4f",
                int(env_steps),
                float(payload.get("anchor/r_anchor", 0.0)),
                float(payload.get("anchor/action_mse", 0.0)),
            )

    def ckpt_save_cb(state, env_steps):
        save_ckpt(ckpt_mgr, int(env_steps), state, config=cfg_dict)

    erc_enable = bool(hydra_cfg.get("eval_render_config", {}).get("enable", True))

    # Spout-maze eval statistics (SPOUT_EVAL). Any other env: eval_env IS
    # base_env_wrapped, spout_eval is None and nothing below changes.
    eval_env, spout_eval = _make_eval_env(base_env_wrapped)

    def eval_cb(state, env_steps, k_eval):
        if not erc_enable:
            return
        try:
            _eval_and_render(
                hydra_cfg,
                cfg,
                K,
                nets,
                recurrent,
                eval_env,
                spout_eval,
                state,
                env_steps,
                k_eval,
                mj_model=mj_model,
                ckpt_dir=ckpt_dir,
            )
        finally:
            import gc

            gc.collect()

    rng, k_run = jax.random.split(rng)
    state, env_state, rb_state, last_metrics = run_training_loop(
        env=env,
        nets=nets,
        optimizers=optimizers,
        rb=rb,
        cfg=cfg,
        K=K,
        iters_per_chunk=iters_per_chunk,
        rng=k_run,
        state=state,
        env_state=None,
        rb_state=rb_state,
        eval_callback=eval_cb,
        wandb_log_callback=wandb_log_cb,
        ckpt_mgr=ckpt_mgr,
        ckpt_save_callback=ckpt_save_cb,
        cfg_dict=cfg_dict,
        # () + () on every arm without the anchor term or a privileged critic.
        extra_state_extras=anchor_extras + priv_extras,
        start_env_steps=start_env_steps,
        frozen_behavior_params=frozen_behavior_params,
    )

    ckpt_mgr.wait_until_finished()
    log.info(
        "Training complete: final policy_loss=%.4g critic_loss=%.4g",
        float(last_metrics.get("policy_loss", 0.0)),
        float(last_metrics.get("critic_loss", 0.0)),
    )
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
