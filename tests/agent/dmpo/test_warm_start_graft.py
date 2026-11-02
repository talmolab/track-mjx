"""Tests for the FF -> superset warm-start graft (`warm_start_graft`).

`flax.serialization.from_state_dict` does not shape-check leaves, so a widened
kernel would silently load the OLD shape. `graft_params` is the explicit
replacement: per leaf path it copies, zero-pads a widened `kernel` along axis
0, keeps target-only leaves at their init, and raises (naming the path) on
anything else. The step-0 contract it exists for:

  * FF policy -> slow-core policy: the grafted slow net's output equals the
    FF net's output for ANY hidden state (the extra `hidden_0` rows are zero).
  * critic with privileged_dim=0 -> privileged_dim>0: the grafted critic's
    first Dense kernel gains trailing zero rows, so Q is unchanged for ANY
    privileged vector.

Tiny nets, CPU-only. The real-checkpoint tests are artifact-marked and skip
unless ``DMPO_CKPT_ROOT`` holds the reference FF maze checkpoint.
"""

import logging
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax.traverse_util import flatten_dict

from track_mjx.agent.dmpo.checkpoint import make_checkpointer, save
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks
from track_mjx.agent.dmpo.networks_kl_anchor_slow import (
    make_dmpo_kl_anchor_slow_networks,
)
from track_mjx.agent.dmpo.warm_start_graft import graft_params, load_graft_source

# ---------------------------------------------------------------------------
# graft_params on hand-built trees.
# ---------------------------------------------------------------------------


def _rand(key, *shape):
    return jax.random.normal(jax.random.PRNGKey(key), shape)


def test_graft_exact_copy():
    source = {
        "params": {
            "a": {"kernel": _rand(0, 4, 3), "bias": _rand(1, 3)},
            "b": {"scale": _rand(2, 5)},
        }
    }
    # Target has the same tree but a DIFFERENT (e.g. fresh-init) value in
    # every leaf: the graft must replace all of them with the source.
    target = jax.tree.map(lambda x: x + 100.0, source)
    out, report = graft_params(target, source)

    assert jax.tree.structure(out) == jax.tree.structure(source)
    for got, want in zip(jax.tree.leaves(out), jax.tree.leaves(source)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert got.dtype == want.dtype
    assert report == {"copied": 3, "widened": [], "new": []}


def test_graft_widened_rows_zero():
    src_k = _rand(0, 4, 3)
    # A nonzero (init-like) target so "zeroed" cannot be confused with
    # "left at the target's random init".
    target = {
        "params": {"hidden_0": {"kernel": jnp.ones((7, 3)), "bias": jnp.ones((3,))}}
    }
    source = {"params": {"hidden_0": {"kernel": src_k, "bias": _rand(1, 3)}}}
    out, report = graft_params(target, source)

    k = np.asarray(out["params"]["hidden_0"]["kernel"])
    assert k.shape == (7, 3)
    np.testing.assert_array_equal(k[:4], np.asarray(src_k))
    np.testing.assert_array_equal(k[4:], np.zeros((3, 3)))
    assert (
        out["params"]["hidden_0"]["kernel"].dtype
        == target["params"]["hidden_0"]["kernel"].dtype
    )
    np.testing.assert_array_equal(
        np.asarray(out["params"]["hidden_0"]["bias"]),
        np.asarray(source["params"]["hidden_0"]["bias"]),
    )
    assert report["widened"] == ["params/hidden_0/kernel"]
    assert report["copied"] == 1  # the bias; the widened kernel is listed apart
    assert report["new"] == []


def test_graft_keeps_new_subtree():
    core = {"gru": {"kernel": _rand(5, 6, 6), "bias": _rand(6, 6)}}
    target = {
        "params": {
            "policy_head": {"hidden_0": {"kernel": _rand(1, 4, 3)}, "slow_core": core}
        }
    }
    source = {"params": {"policy_head": {"hidden_0": {"kernel": _rand(2, 4, 3)}}}}
    out, report = graft_params(target, source)

    flat_out = flatten_dict(out["params"])
    # Target-only subtree: the target's init, untouched.
    for path, leaf in flatten_dict({"slow_core": core}).items():
        np.testing.assert_array_equal(
            np.asarray(flat_out[("policy_head", *path)]), np.asarray(leaf)
        )
    # Shared leaf: the source's value.
    np.testing.assert_array_equal(
        np.asarray(out["params"]["policy_head"]["hidden_0"]["kernel"]),
        np.asarray(source["params"]["policy_head"]["hidden_0"]["kernel"]),
    )
    assert report["copied"] == 1
    assert report["widened"] == []
    assert sorted(report["new"]) == [
        "params/policy_head/slow_core/gru/bias",
        "params/policy_head/slow_core/gru/kernel",
    ]


@pytest.mark.parametrize(
    "t_shape,s_shape,leaf",
    [
        # Trailing dim differs on a kernel: not a row widening.
        ((7, 4), (4, 3), "kernel"),
        # Kernel SHRINKS along axis 0: not a widening either.
        ((3, 3), (4, 3), "kernel"),
        # Only `kernel` may widen: a bias that grew is a real mismatch.
        ((7,), (4,), "bias"),
        # Rank change.
        ((7, 3, 1), (7, 3), "kernel"),
        # Scale vector of a norm layer: never widens.
        ((8,), (5,), "scale"),
    ],
)
def test_graft_rejects_non_row_mismatch(t_shape, s_shape, leaf):
    target = {"params": {"layer": {leaf: jnp.zeros(t_shape)}}}
    source = {"params": {"layer": {leaf: jnp.zeros(s_shape)}}}
    with pytest.raises(ValueError, match=f"params/layer/{leaf}"):
        graft_params(target, source)


def test_graft_rejects_widening_a_conv_kernel():
    """Rows-widening is a Dense-kernel notion (rank 2). A conv kernel that
    differs along axis 0 is a different spatial window, not extra inputs."""
    target = {"params": {"conv": {"kernel": jnp.zeros((5, 3, 1, 4))}}}
    source = {"params": {"conv": {"kernel": jnp.zeros((3, 3, 1, 4))}}}
    with pytest.raises(ValueError, match="params/conv/kernel"):
        graft_params(target, source)


def test_graft_rejects_source_only_path():
    target = {"params": {"a": {"kernel": jnp.zeros((2, 2))}}}
    source = {
        "params": {
            "a": {"kernel": jnp.zeros((2, 2))},
            "renamed": {"kernel": jnp.zeros((2, 2))},
        }
    }
    with pytest.raises(ValueError, match="params/renamed/kernel"):
        graft_params(target, source)


def test_graft_error_names_every_offender():
    target = {"params": {"a": {"bias": jnp.zeros((7,))}}}
    source = {
        "params": {
            "a": {"bias": jnp.zeros((4,))},
            "gone": {"kernel": jnp.zeros((2, 2))},
        }
    }
    with pytest.raises(ValueError) as exc:
        graft_params(target, source)
    msg = str(exc.value)
    assert "params/a/bias" in msg and "params/gone/kernel" in msg


def test_graft_does_not_mutate_inputs():
    target = {"params": {"a": {"kernel": jnp.ones((6, 2))}}}
    source = {"params": {"a": {"kernel": np.full((4, 2), 3.0, np.float32)}}}
    graft_params(target, source)
    np.testing.assert_array_equal(
        np.asarray(target["params"]["a"]["kernel"]), np.ones((6, 2))
    )
    np.testing.assert_array_equal(source["params"]["a"]["kernel"], np.full((4, 2), 3.0))


def test_graft_accepts_numpy_source_and_frozen_target():
    """The loader returns numpy leaves; a flax init may return a FrozenDict."""
    from flax.core import freeze

    src = {"params": {"a": {"kernel": np.arange(8, dtype=np.float32).reshape(4, 2)}}}
    tgt = freeze({"params": {"a": {"kernel": jnp.ones((6, 2))}}})
    out, report = graft_params(tgt, src)
    assert type(out) is type(tgt)
    k = out["params"]["a"]["kernel"]
    assert isinstance(k, jax.Array)
    np.testing.assert_array_equal(np.asarray(k[:4]), src["params"]["a"]["kernel"])
    np.testing.assert_array_equal(np.asarray(k[4:]), np.zeros((2, 2)))
    assert report["widened"] == ["params/a/kernel"]


# ---------------------------------------------------------------------------
# Real tiny nets: FF policy -> slow-core policy (the step-0 contract).
# ---------------------------------------------------------------------------

PROPRIO_SIZE = 10
TASK_OBS_SIZE = 6
ACTION_SIZE = 5
LATENT_SIZE = 4
VISION_SHAPE = (8, 8, 2)
FF_LAYERS = (16, 12)
CNN_FEAT = 4
SLOW_H = 6
TICK = 3
FF_IN_ROWS = 2 * CNN_FEAT + TASK_OBS_SIZE + PROPRIO_SIZE


@pytest.fixture
def cfg():
    return DMPOConfig(num_envs=4, unroll_length=4, batch_size=4, sequence_length=4)


def _net_kwargs(cfg, **overrides):
    kw = dict(
        proprio_size=PROPRIO_SIZE,
        task_obs_size=TASK_OBS_SIZE,
        action_size=ACTION_SIZE,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
        prior_layer_sizes=(16, 16),
        decoder_layer_sizes=(16, 16),
        policy_head_layer_sizes=FF_LAYERS,
        cnn_feature_size=CNN_FEAT,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        value_hidden_layer_sizes=(16,),
    )
    kw.update(overrides)
    return kw


def _random_obs(key):
    k_v, k_t, k_p = jax.random.split(key, 3)
    return {
        "vision": jax.random.uniform(k_v, VISION_SHAPE),
        "imitation_target": jax.random.normal(k_t, (TASK_OBS_SIZE,)),
        "proprioception": jax.random.normal(k_p, (PROPRIO_SIZE,)),
    }


def _perturb(tree, key, scale=0.3):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [
            x + scale * jax.random.normal(k, x.shape, x.dtype)
            for x, k in zip(leaves, keys)
        ],
    )


# The graft moves PARAMETERS only: residual_mode / residual_scale are net
# hyperparameters and must match between the source and target nets. The FF
# and slow factories have different defaults (sigma_tanh/2.0 vs
# sigma_ball/5.0), so they are always passed explicitly. ("sigma_ball", 5.0)
# is the reference FF maze checkpoint's setting.
@pytest.mark.parametrize(
    "residual_mode,residual_scale", [("sigma_ball", 5.0), ("tanh", 2.0)]
)
def test_ff_to_slow_core_output_equal(cfg, residual_mode, residual_scale):
    ff_nets = make_dmpo_kl_anchor_networks(
        **_net_kwargs(cfg, residual_mode=residual_mode, residual_scale=residual_scale)
    )
    slow_nets = make_dmpo_kl_anchor_slow_networks(
        **_net_kwargs(
            cfg,
            residual_mode=residual_mode,
            residual_scale=residual_scale,
            slow_hidden_size=SLOW_H,
            tick_stride=TICK,
            rnn_store_dtype="float16",
        )
    )
    obs = _random_obs(jax.random.PRNGKey(1))
    # Perturb the WHOLE FF tree (including the zero-init residual) so the head
    # really contributes to the output; otherwise equality would be trivial.
    ff_params = _perturb(
        ff_nets.policy.init(jax.random.PRNGKey(0), obs), jax.random.PRNGKey(8)
    )
    slow_init = slow_nets.policy.init(
        jax.random.PRNGKey(3), obs, slow_nets.recurrent_meta.init_hidden()
    )

    # Round-trip the source through host numpy, as the real loader returns it.
    ff_numpy = jax.tree.map(np.asarray, ff_params)
    grafted, report = graft_params(slow_init, ff_numpy)

    # Report: exactly one widened leaf, every new leaf under slow_core.
    assert report["widened"] == ["params/policy_head/hidden_0/kernel"]
    assert report["new"], "the slow core must show up as new leaves"
    assert all(p.startswith("params/policy_head/slow_core/") for p in report["new"])
    assert report["copied"] == len(jax.tree.leaves(ff_params)) - 1
    assert jax.tree.structure(grafted) == jax.tree.structure(slow_init)
    k = np.asarray(grafted["params"]["policy_head"]["hidden_0"]["kernel"])
    assert k.shape == (FF_IN_ROWS + SLOW_H, FF_LAYERS[0])
    np.testing.assert_array_equal(k[FF_IN_ROWS:], 0.0)
    # New slow-core leaves keep the slow net's own init.
    jax.tree.map(
        np.testing.assert_array_equal,
        grafted["params"]["policy_head"]["slow_core"],
        slow_init["params"]["policy_head"]["slow_core"],
    )

    key = jax.random.PRNGKey(10)
    for i, phase in enumerate([0, 1, 0, 2]):
        key, k_o, k_h = jax.random.split(key, 3)
        o = _random_obs(k_o)
        hidden = (
            jax.random.normal(k_h, (SLOW_H,)) * 3.0,
            jnp.full((1,), float(phase)),
        )
        ff_dist = ff_nets.policy.apply(ff_params, o)
        mu, scale, _ = slow_nets.policy.apply(grafted, o, hidden, method="raw")
        np.testing.assert_allclose(
            np.asarray(mu),
            np.asarray(ff_dist.mean()),
            atol=1e-6,
            err_msg=f"mu mismatch (case {i})",
        )
        np.testing.assert_allclose(
            np.asarray(scale),
            np.asarray(ff_dist.stddev()),
            atol=1e-6,
            err_msg=f"scale mismatch (case {i})",
        )

    # Control: the UN-grafted slow net does not match, so the check above can
    # fail.
    o = _random_obs(jax.random.PRNGKey(99))
    hidden = (jnp.ones((SLOW_H,)), jnp.zeros((1,)))
    mu_bad, _, _ = slow_nets.policy.apply(slow_init, o, hidden, method="raw")
    assert not np.allclose(
        np.asarray(mu_bad),
        np.asarray(ff_nets.policy.apply(ff_params, o).mean()),
        atol=1e-4,
    )


def test_ff_policy_to_ff_policy_is_exact_copy(cfg):
    """An FF policy paired with a privileged critic: the policy tree is identical,
    so the graft is a pure copy with nothing widened and nothing new."""
    nets = make_dmpo_kl_anchor_networks(**_net_kwargs(cfg))
    obs = _random_obs(jax.random.PRNGKey(1))
    src = _perturb(nets.policy.init(jax.random.PRNGKey(0), obs), jax.random.PRNGKey(2))
    tgt = nets.policy.init(jax.random.PRNGKey(5), obs)
    out, report = graft_params(tgt, jax.tree.map(np.asarray, src))
    assert report == {"copied": len(jax.tree.leaves(src)), "widened": [], "new": []}
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(src)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# Real tiny critics: privileged_dim 0 -> privileged_dim > 0.
# ---------------------------------------------------------------------------

PRIV = 8


def test_privileged_critic_graft_q_unchanged(cfg):
    plain = make_dmpo_kl_anchor_networks(**_net_kwargs(cfg, critic_use_proprio=True))
    priv = make_dmpo_kl_anchor_networks(
        **_net_kwargs(cfg, critic_use_proprio=True, critic_privileged_dim=PRIV)
    )
    obs = _random_obs(jax.random.PRNGKey(1))
    act = jnp.zeros((ACTION_SIZE,))
    src = _perturb(
        plain.critic.init(jax.random.PRNGKey(0), obs, act), jax.random.PRNGKey(2)
    )
    tgt = priv.critic.init(
        jax.random.PRNGKey(3), dict(obs, privileged=jnp.zeros((PRIV,))), act
    )

    out, report = graft_params(tgt, jax.tree.map(np.asarray, src))
    # The widened leaf is the first critic Dense kernel; nothing else changes.
    assert report["widened"] == ["params/Dense_0/kernel"]
    assert report["new"] == []
    assert report["copied"] == len(jax.tree.leaves(src)) - 1
    k_src = np.asarray(src["params"]["Dense_0"]["kernel"])
    k_out = np.asarray(out["params"]["Dense_0"]["kernel"])
    assert k_out.shape == (k_src.shape[0] + PRIV, k_src.shape[1])
    np.testing.assert_array_equal(k_out[: k_src.shape[0]], k_src)
    np.testing.assert_array_equal(k_out[k_src.shape[0] :], 0.0)

    # Step-0 equivalence: the grafted critic's value distribution equals the
    # source critic's for ANY privileged vector.
    q_ref = plain.critic.apply(src, obs, act)
    for i in range(3):
        p = 5.0 * jax.random.normal(jax.random.PRNGKey(20 + i), (PRIV,))
        q = priv.critic.apply(out, dict(obs, privileged=p), act)
        np.testing.assert_allclose(
            np.asarray(q.mean()), np.asarray(q_ref.mean()), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(q.logits_parameter()),
            np.asarray(q_ref.logits_parameter()),
            atol=1e-5,
        )
    # Control: un-grafted (random-init) critic does not match.
    q_bad = priv.critic.apply(tgt, dict(obs, privileged=jnp.ones((PRIV,))), act)
    assert not np.allclose(
        np.asarray(q_bad.mean()), np.asarray(q_ref.mean()), atol=1e-4
    )


# ---------------------------------------------------------------------------
# load_graft_source on synthetic checkpoints.
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_state(tmp_path, rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)
    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=7, state=state)
    mgr.wait_until_finished()
    return state, tmp_path / "DMPONetwork_7"


def test_load_graft_source_returns_all_present_items(saved_state):
    state, step_dir = saved_state
    src = load_graft_source(step_dir)
    assert set(src) == {
        "policy_params",
        "target_policy_params",
        "critic_params",
        "target_critic_params",
        "normalizer_params",
    }
    assert all(isinstance(x, np.ndarray) for x in jax.tree.leaves(src))
    for got, want in zip(
        jax.tree.leaves(src["policy_params"]), jax.tree.leaves(state.policy_params)
    ):
        np.testing.assert_array_equal(got, np.asarray(want))
    for got, want in zip(
        jax.tree.leaves(src["critic_params"]), jax.tree.leaves(state.critic_params)
    ):
        np.testing.assert_array_equal(got, np.asarray(want))
    # The nesting is the one `graft_params` compares against: a top-level
    # "params" key, exactly like a flax init.
    assert "params" in src["policy_params"]
    assert "params" in src["critic_params"]


def _write_train_state(step_dir, items):
    step_dir.mkdir(parents=True)
    ocp.PyTreeCheckpointer().save(str(step_dir / "train_state"), items)


def test_load_graft_source_tolerates_absent_critic(saved_state, tmp_path, caplog):
    state, _ = saved_state
    step_dir = tmp_path / "no_critic" / "DMPONetwork_1"
    _write_train_state(
        step_dir,
        {
            "policy_params": state.policy_params,
            "target_policy_params": state.target_policy_params,
            "normalizer_params": jax.tree.map(np.asarray, state.normalizer_params),
        },
    )
    with caplog.at_level(logging.WARNING):
        src = load_graft_source(step_dir)
    assert set(src) == {"policy_params", "target_policy_params", "normalizer_params"}
    # Reported, not silent: the critic stays at its init.
    assert "critic_params" in caplog.text
    assert "target_critic_params" in caplog.text


def test_load_graft_source_requires_policy_and_normalizer(saved_state, tmp_path):
    state, _ = saved_state
    step_dir = tmp_path / "no_norm" / "DMPONetwork_1"
    _write_train_state(
        step_dir,
        {
            "policy_params": state.policy_params,
            "target_policy_params": state.target_policy_params,
        },
    )
    with pytest.raises(KeyError, match="normalizer_params"):
        load_graft_source(step_dir)


def test_load_graft_source_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graft_source(tmp_path / "nope")


# ---------------------------------------------------------------------------
# The real reference FF maze checkpoint (artifact; skipped unless staged).
# ---------------------------------------------------------------------------

# A local artifact, not something the repo ships or CI can provision: point
# DMPO_CKPT_ROOT at a directory containing
# maze_gen1_prior_mlp/DMPONetwork_482816000 (a feed-forward kl-anchor maze
# policy with a vision critic) to exercise these tests. Unset means they skip
# visibly rather than silently passing or hard-failing.
_CKPT_ROOT = os.environ.get("DMPO_CKPT_ROOT")
_GEN1_CKPT = (
    pathlib.Path(_CKPT_ROOT) / "maze_gen1_prior_mlp" / "DMPONetwork_482816000"
    if _CKPT_ROOT
    else None
)
_SKIP_NO_GEN1 = _GEN1_CKPT is None or not _GEN1_CKPT.is_dir()
_SKIP_REASON = (
    "reference FF maze checkpoint not available: set DMPO_CKPT_ROOT to a "
    "directory containing maze_gen1_prior_mlp/DMPONetwork_482816000"
)


@pytest.mark.artifact
@pytest.mark.skipif(_SKIP_NO_GEN1, reason=_SKIP_REASON)
def test_real_gen1_checkpoint_grafts_into_widened_shapes():
    src = load_graft_source(_GEN1_CKPT)
    assert set(src) == {
        "policy_params",
        "target_policy_params",
        "critic_params",
        "target_critic_params",
        "normalizer_params",
    }
    pol = src["policy_params"]
    assert set(pol["params"]) == {"prior", "decoder", "policy_head"}
    head_k = pol["params"]["policy_head"]["hidden_0"]["kernel"]
    crit_k = src["critic_params"]["params"]["Dense_0"]["kernel"]

    # Slow-core target: hidden_0 widened by 512 trailing rows + a new subtree.
    H = 512
    tgt = jax.tree.map(jnp.asarray, pol)
    tgt["params"]["policy_head"]["hidden_0"]["kernel"] = jnp.ones(
        (head_k.shape[0] + H, head_k.shape[1]), head_k.dtype
    )
    tgt["params"]["policy_head"]["slow_core"] = {"ir": {"kernel": jnp.ones((3, 3))}}
    for name in ("policy_params", "target_policy_params"):
        out, report = graft_params(tgt, src[name])
        assert report["widened"] == ["params/policy_head/hidden_0/kernel"]
        assert report["new"] == ["params/policy_head/slow_core/ir/kernel"]
        assert report["copied"] == 55  # 56 policy leaves, one of them widened
        k = np.asarray(out["params"]["policy_head"]["hidden_0"]["kernel"])
        np.testing.assert_array_equal(
            k[: head_k.shape[0]],
            src[name]["params"]["policy_head"]["hidden_0"]["kernel"],
        )
        np.testing.assert_array_equal(k[head_k.shape[0] :], 0.0)

    # Privileged critic target: Dense_0 widened by the 8-D privileged vector.
    ctgt = jax.tree.map(jnp.asarray, src["critic_params"])
    ctgt["params"]["Dense_0"]["kernel"] = jnp.ones(
        (crit_k.shape[0] + 8, crit_k.shape[1]), crit_k.dtype
    )
    for name in ("critic_params", "target_critic_params"):
        out, report = graft_params(ctgt, src[name])
        assert report["widened"] == ["params/Dense_0/kernel"]
        assert report["new"] == []
        assert report["copied"] == 29
