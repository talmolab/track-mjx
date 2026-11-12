"""Tests for the slow-clock memory core policy head (kl-anchor DMPO).

The slow core is a GRU (or, with `rnn_cell="lstm"`, an LSTM -- see the LSTM
section at the end of this file) that ticks only when the stored phase is 0
(every `tick_stride` env steps), holds its state in between, and feeds its
state into the FF head's MLP as the LAST input rows. Two properties carry the
design and are pinned here:

  * Graftability: the slow head's param tree is the FF `_PolicyHeadModule`
    tree plus a `slow_core` subtree, with `hidden_0/kernel` widened by H
    trailing rows. Copying every FF leaf and zero-filling those rows gives a
    policy that equals the FF policy for ANY hidden state.
  * The clock: hold between ticks, tick at phase 0, phase cycles mod k, and
    an episode reset (which zeroes every hidden leaf) restarts the clock
    with a tick on the first step of the new episode.

Tiny unbatched shapes (the codebase convention is a per-env vmap of an
unbatched apply), 8x8 binocular vision as in test_networks_kl_anchor_rnn.
CPU-only.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import _reset_hidden, init_training_state
from track_mjx.agent.dmpo.networks_kl_anchor import (
    _DecoderModule,
    _PolicyHeadModule,
    _PriorModule,
    make_dmpo_kl_anchor_networks,
)
from track_mjx.agent.dmpo.networks_kl_anchor_slow import (
    SlowCoreMeta,
    _SlowCorePolicyHeadModule,
    make_dmpo_kl_anchor_slow_networks,
)
from track_mjx.agent.dmpo.optim_kl_anchor import label_param_tree
from track_mjx.agent.dmpo.rollout import _reset_hidden_where_done
from track_mjx.agent.dmpo.vision_encoders import BinocularVisionEncoder
from track_mjx.agent.dmpo.warm_start_graft import graft_params

PROPRIO_SIZE = 10
TASK_OBS_SIZE = 6
ACTION_SIZE = 5
LATENT_SIZE = 4
VISION_SHAPE = (8, 8, 2)
PRIOR_LAYERS = (16, 16)
DECODER_LAYERS = (16, 16)
FF_LAYERS = (16, 12)
CNN_FEAT = 4
CNN_CHANNELS = (2, 4, 8)
SLOW_H = 6
TICK = 3
# FF head input rows: binocular CNN features (2 eyes x CNN_FEAT) + task obs
# + proprio. The slow head appends SLOW_H rows after these.
FF_IN_ROWS = 2 * CNN_FEAT + TASK_OBS_SIZE + PROPRIO_SIZE
RESIDUAL_MODES = ("sigma_ball", "sigma_tanh", "tanh", "none")


@pytest.fixture
def cfg():
    return DMPOConfig(num_envs=4, unroll_length=4, batch_size=4, sequence_length=4)


def _shared_kwargs(cfg):
    return dict(
        proprio_size=PROPRIO_SIZE,
        task_obs_size=TASK_OBS_SIZE,
        action_size=ACTION_SIZE,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
        prior_layer_sizes=PRIOR_LAYERS,
        decoder_layer_sizes=DECODER_LAYERS,
        policy_head_layer_sizes=FF_LAYERS,
        cnn_feature_size=CNN_FEAT,
        cnn_channels=CNN_CHANNELS,
        mono_channels=1,
        shared_weights=True,
        value_hidden_layer_sizes=(16,),
    )


def _slow_nets(cfg, **overrides):
    kw = _shared_kwargs(cfg)
    kw.update(
        slow_hidden_size=SLOW_H,
        tick_stride=TICK,
        rnn_store_dtype="float16",
    )
    kw.update(overrides)
    return make_dmpo_kl_anchor_slow_networks(**kw)


def _ff_nets(cfg, **overrides):
    kw = _shared_kwargs(cfg)
    kw.update(overrides)
    return make_dmpo_kl_anchor_networks(**kw)


def _random_obs(key, batch=None):
    shape = () if batch is None else (batch,)
    k_v, k_t, k_p = jax.random.split(key, 3)
    return {
        "vision": jax.random.uniform(k_v, shape + VISION_SHAPE),
        "imitation_target": jax.random.normal(k_t, (*shape, TASK_OBS_SIZE)),
        "proprioception": jax.random.normal(k_p, (*shape, PROPRIO_SIZE)),
    }


def _hidden(slow_h, phase):
    return (jnp.asarray(slow_h, jnp.float32), jnp.full((1,), phase, jnp.float32))


def _init(nets, key=0):
    obs = _random_obs(jax.random.PRNGKey(1))
    return nets.policy.init(
        jax.random.PRNGKey(key), obs, nets.recurrent_meta.init_hidden()
    )


def _raw(nets, params, obs, hidden):
    return nets.policy.apply(params, obs, hidden, method="raw")


def _ff_input(params, obs):
    """concat([vis, task_obs, proprio]) -- the GRU input and the FF head input."""
    vis = BinocularVisionEncoder(
        feature_size=CNN_FEAT,
        channels=CNN_CHANNELS,
        mono_channels=1,
        shared_weights=True,
    ).apply(
        {"params": params["params"]["policy_head"]["BinocularVisionEncoder_0"]},
        obs["vision"],
    )
    return jnp.concatenate(
        [vis, obs["imitation_target"], obs["proprioception"]], axis=-1
    )


# ---------------------------------------------------------------------------
# Meta / hidden contract
# ---------------------------------------------------------------------------


def test_meta_and_init_hidden(cfg):
    nets = _slow_nets(cfg)
    meta = nets.recurrent_meta
    assert isinstance(meta, SlowCoreMeta)
    assert meta.cell_type == "gru"
    assert meta.hidden_sizes == (SLOW_H,)
    assert meta.store_dtype == jnp.dtype("float16")
    assert meta.tick_stride == TICK
    assert meta.phase_leaf == 1

    h = meta.init_hidden()
    assert isinstance(h, tuple) and len(h) == 2
    assert [x.shape for x in h] == [(SLOW_H,), (1,)]
    hb = meta.init_hidden(3)
    assert [x.shape for x in hb] == [(3, SLOW_H), (3, 1)]
    for x in h + hb:
        assert x.dtype == jnp.float32
        assert not np.any(np.asarray(x))
    # The phase lives at phase_leaf.
    assert hb[meta.phase_leaf].shape == (3, 1)


# ---------------------------------------------------------------------------
# Param tree: graftable from the FF head
# ---------------------------------------------------------------------------


def test_param_names_match_ff_head(cfg):
    obs = _random_obs(jax.random.PRNGKey(1))

    # The FF head's exact tree, from the module itself.
    ff_head = _PolicyHeadModule(
        layer_sizes=FF_LAYERS,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cnn_feature_size=CNN_FEAT,
        cnn_channels=CNN_CHANNELS,
        mono_channels=1,
        shared_weights=True,
    )
    ff_head_params = ff_head.init(
        jax.random.PRNGKey(0),
        obs["vision"],
        obs["imitation_target"],
        obs["proprioception"],
    )["params"]
    assert "BinocularVisionEncoder_0" in ff_head_params

    slow_head = _SlowCorePolicyHeadModule(
        layer_sizes=FF_LAYERS,
        slow_hidden_size=SLOW_H,
        tick_stride=TICK,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cnn_feature_size=CNN_FEAT,
        cnn_channels=CNN_CHANNELS,
        mono_channels=1,
        shared_weights=True,
    )
    slow_head_params = slow_head.init(
        jax.random.PRNGKey(0),
        obs["vision"],
        obs["imitation_target"],
        obs["proprioception"],
        (jnp.zeros((SLOW_H,)), jnp.zeros((1,))),
    )["params"]
    assert "slow_core" in slow_head_params

    ff_flat = flatten_dict(ff_head_params)
    slow_flat = flatten_dict(slow_head_params)
    slow_minus_core = {k: v for k, v in slow_flat.items() if k[0] != "slow_core"}
    assert set(slow_minus_core) == set(ff_flat)

    for k, v in ff_flat.items():
        if k == ("hidden_0", "kernel"):
            assert v.shape == (FF_IN_ROWS, FF_LAYERS[0])
            assert slow_minus_core[k].shape == (FF_IN_ROWS + SLOW_H, FF_LAYERS[0])
        else:
            assert slow_minus_core[k].shape == v.shape, k

    # Same story at the full-policy level, against the FF factory's policy.
    ff_params = _ff_nets(cfg).policy.init(jax.random.PRNGKey(0), obs)
    slow_params = _init(_slow_nets(cfg))
    assert set(slow_params["params"]) == {"prior", "decoder", "policy_head"}
    ff_pol = flatten_dict(ff_params["params"])
    slow_pol = {
        k: v
        for k, v in flatten_dict(slow_params["params"]).items()
        if not (k[0] == "policy_head" and k[1] == "slow_core")
    }
    assert set(slow_pol) == set(ff_pol)
    widened = [k for k in ff_pol if slow_pol[k].shape != ff_pol[k].shape]
    assert widened == [("policy_head", "hidden_0", "kernel")]

    # The slow core is trained at the policy_head LR.
    labels = label_param_tree(slow_params)
    core_labels = jax.tree_util.tree_leaves(
        labels["params"]["policy_head"]["slow_core"]
    )
    assert len(core_labels) > 0
    assert all(lbl == "policy_head" for lbl in core_labels)


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


def test_hold_between_ticks(cfg):
    nets = _slow_nets(cfg)
    params = _init(nets)
    obs = _random_obs(jax.random.PRNGKey(2))
    slow_h = jax.random.normal(jax.random.PRNGKey(3), (SLOW_H,)) * 2.0
    for phase in range(1, TICK):
        _, _, (new_slow, new_phase) = _raw(nets, params, obs, _hidden(slow_h, phase))
        np.testing.assert_array_equal(np.asarray(new_slow), np.asarray(slow_h))
        np.testing.assert_array_equal(
            np.asarray(new_phase), np.asarray([(phase + 1) % TICK], np.float32)
        )


def test_tick_updates(cfg):
    nets = _slow_nets(cfg)
    params = _init(nets)
    obs = _random_obs(jax.random.PRNGKey(2))
    slow_h = jax.random.normal(jax.random.PRNGKey(3), (SLOW_H,))
    _, _, (new_slow, new_phase) = _raw(nets, params, obs, _hidden(slow_h, 0))

    assert not np.allclose(np.asarray(new_slow), np.asarray(slow_h))
    # Exactly one GRU step on the FF input x = concat([vis, task, proprio]).
    expected, _ = nn.GRUCell(features=SLOW_H).apply(
        {"params": params["params"]["policy_head"]["slow_core"]},
        slow_h,
        _ff_input(params, obs),
    )
    np.testing.assert_allclose(
        np.asarray(new_slow), np.asarray(expected), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(new_phase), np.asarray([1.0]))


def test_head_forward_matches_reference_flow(cfg):
    """Independent re-implementation of the head's flow: the MLP reads
    concat([x, slow_h]) with the POST-tick slow state appended last."""
    nets = _slow_nets(cfg)
    params = _perturb(_init(nets), jax.random.PRNGKey(14))
    head_p = params["params"]["policy_head"]
    head = _SlowCorePolicyHeadModule(
        layer_sizes=FF_LAYERS,
        slow_hidden_size=SLOW_H,
        tick_stride=TICK,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cnn_feature_size=CNN_FEAT,
        cnn_channels=CNN_CHANNELS,
        mono_channels=1,
        shared_weights=True,
    )
    obs = _random_obs(jax.random.PRNGKey(15))
    x = _ff_input(params, obs)
    slow_h = jax.random.normal(jax.random.PRNGKey(16), (SLOW_H,))
    for phase in (0, 1):
        s = slow_h
        if phase == 0:
            s, _ = nn.GRUCell(features=SLOW_H).apply(
                {"params": head_p["slow_core"]}, slow_h, x
            )
        h = jnp.concatenate([x, s], axis=-1)
        for i in range(len(FF_LAYERS)):
            d = head_p[f"hidden_{i}"]
            h = h @ d["kernel"] + d["bias"]
            h = nn.LayerNorm().apply({"params": head_p[f"LayerNorm_{i}"]}, h)
            h = nn.silu(h)
        ref = h @ head_p["residual"]["kernel"] + head_p["residual"]["bias"]

        out, (new_slow, _) = head.apply(
            {"params": head_p},
            obs["vision"],
            obs["imitation_target"],
            obs["proprioception"],
            _hidden(slow_h, phase),
        )
        assert np.any(np.asarray(ref) != 0.0)
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-5)
        np.testing.assert_allclose(np.asarray(new_slow), np.asarray(s), atol=1e-6)


def test_phase_cycles_mod_k(cfg):
    nets = _slow_nets(cfg)
    params = _init(nets)
    hidden = nets.recurrent_meta.init_hidden()
    key = jax.random.PRNGKey(4)
    n_steps = 2 * TICK + 2
    phases, ticked = [], []
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        slow_before, phase_before = hidden
        _, _, hidden = _raw(nets, params, _random_obs(k), hidden)
        phases.append(float(phase_before[0]))
        ticked.append(
            not np.array_equal(np.asarray(hidden[0]), np.asarray(slow_before))
        )
    assert phases == [float(t % TICK) for t in range(n_steps)]
    assert ticked == [t % TICK == 0 for t in range(n_steps)]


def test_k1_ticks_every_step(cfg):
    nets = _slow_nets(cfg, tick_stride=1)
    assert nets.recurrent_meta.tick_stride == 1
    params = _init(nets)
    hidden = nets.recurrent_meta.init_hidden()
    key = jax.random.PRNGKey(5)
    for _ in range(4):
        key, k = jax.random.split(key)
        slow_before = hidden[0]
        _, _, hidden = _raw(nets, params, _random_obs(k), hidden)
        assert not np.array_equal(np.asarray(hidden[0]), np.asarray(slow_before))
        np.testing.assert_array_equal(np.asarray(hidden[1]), np.asarray([0.0]))


# ---------------------------------------------------------------------------
# Step-0 equivalence with the FF policy (the graft contract)
# ---------------------------------------------------------------------------


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


def _graft_ff_into_slow(ff_params, slow_params, slow_rows):
    """Copy every FF leaf; hidden_0 kernel = FF rows then `slow_rows`."""
    ff_flat = flatten_dict(ff_params["params"])
    out = dict(flatten_dict(slow_params["params"]))
    for k, v in ff_flat.items():
        if k == ("policy_head", "hidden_0", "kernel"):
            out[k] = jnp.concatenate([v, slow_rows], axis=0)
        else:
            out[k] = v
    return {"params": unflatten_dict(out)}


@pytest.mark.parametrize("mode", RESIDUAL_MODES)
def test_zero_slow_rows_equals_ff(cfg, mode):
    ff_nets = _ff_nets(cfg, residual_mode=mode, residual_scale=2.0)
    slow_nets = _slow_nets(cfg, residual_mode=mode, residual_scale=2.0)
    obs = _random_obs(jax.random.PRNGKey(1))
    # Perturb the WHOLE FF tree (including the zero-init residual) so the head
    # really contributes to the output -- otherwise equality is trivial.
    ff_params = _perturb(
        ff_nets.policy.init(jax.random.PRNGKey(0), obs), jax.random.PRNGKey(8)
    )
    slow_init = _init(slow_nets, key=3)
    grafted = _graft_ff_into_slow(
        ff_params, slow_init, jnp.zeros((SLOW_H, FF_LAYERS[0]))
    )

    key = jax.random.PRNGKey(10)
    for i, phase in enumerate([0, 1, 0, 2]):
        key, k_o, k_h = jax.random.split(key, 3)
        o = _random_obs(k_o)
        hidden = _hidden(jax.random.normal(k_h, (SLOW_H,)) * 3.0, phase)
        ff_dist = ff_nets.policy.apply(ff_params, o)
        mu, scale, _ = _raw(slow_nets, grafted, o, hidden)
        np.testing.assert_allclose(
            np.asarray(mu),
            np.asarray(ff_dist.mean()),
            atol=1e-6,
            err_msg=f"mu mismatch (mode={mode}, case {i})",
        )
        np.testing.assert_allclose(
            np.asarray(scale),
            np.asarray(ff_dist.stddev()),
            atol=1e-6,
            err_msg=f"scale mismatch (mode={mode}, case {i})",
        )

    # Control: nonzero slow rows DO change the output, so the check above can
    # fail (the slow state is wired into hidden_0).
    bad = _graft_ff_into_slow(
        ff_params,
        slow_init,
        jax.random.normal(jax.random.PRNGKey(12), (SLOW_H, FF_LAYERS[0])),
    )
    hidden = _hidden(jnp.ones((SLOW_H,)), 1)
    mu_bad, _, _ = _raw(slow_nets, bad, obs, hidden)
    assert not np.allclose(
        np.asarray(mu_bad),
        np.asarray(ff_nets.policy.apply(ff_params, obs).mean()),
        atol=1e-4,
    )


def test_warm_start_splice_and_step0_invariant(cfg):
    """Spliced prior+decoder + fresh zero-init residual => the frozen
    prior->decoder pipeline, for any hidden and any phase."""
    donor = _ff_nets(cfg)
    obs = _random_obs(jax.random.PRNGKey(1))
    donor_params = donor.policy.init(jax.random.PRNGKey(7), obs)
    prior_p = donor_params["params"]["prior"]
    decoder_p = donor_params["params"]["decoder"]

    nets = _slow_nets(
        cfg, warm_start_prior_params=prior_p, warm_start_decoder_params=decoder_p
    )
    params = _init(nets, key=11)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, params["params"]["prior"], prior_p
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, params["params"]["decoder"], decoder_p
    )

    proprio = obs["proprioception"]
    z_prior, _ = _PriorModule(layer_sizes=PRIOR_LAYERS, latents=LATENT_SIZE).apply(
        {"params": prior_p}, proprio
    )
    dec = _DecoderModule(layer_sizes=(*DECODER_LAYERS, 2 * ACTION_SIZE)).apply(
        {"params": decoder_p}, jnp.concatenate([z_prior, proprio], axis=-1)
    )
    mu_ref = dec[..., :ACTION_SIZE]
    scale_ref = jax.nn.softplus(dec[..., ACTION_SIZE:]) + 1e-3
    for i, phase in enumerate([0, 2]):
        hidden = _hidden(
            jax.random.normal(jax.random.PRNGKey(20 + i), (SLOW_H,)) * 3.0, phase
        )
        dist, _ = nets.policy.apply(params, obs, hidden)
        np.testing.assert_allclose(
            np.asarray(dist.mean()), np.asarray(mu_ref), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(dist.stddev()), np.asarray(scale_ref), atol=1e-6
        )


def test_raw_call_consistent(cfg):
    nets = _slow_nets(cfg)
    params = _perturb(_init(nets), jax.random.PRNGKey(13))
    obs = _random_obs(jax.random.PRNGKey(2))
    for phase in (0, 1):
        hidden = _hidden(jax.random.normal(jax.random.PRNGKey(6), (SLOW_H,)), phase)
        mu, scale, nh_raw = _raw(nets, params, obs, hidden)
        dist, nh_call = nets.policy.apply(params, obs, hidden)
        np.testing.assert_allclose(np.asarray(dist.mean()), np.asarray(mu), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(dist.stddev()), np.asarray(scale), atol=1e-6
        )
        for a, b in zip(nh_raw, nh_call):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# Episode reset restarts the clock (rollout + learner reset helpers)
# ---------------------------------------------------------------------------


def test_reset_hidden_where_done_resets_phase(cfg):
    nets = _slow_nets(cfg)
    meta = nets.recurrent_meta
    params = _init(nets)
    B = 3
    slow_h = jax.random.normal(jax.random.PRNGKey(7), (B, SLOW_H)) + 1.0
    phase = jnp.asarray([[2.0], [1.0], [2.0]])
    hidden = (slow_h, phase)
    done = jnp.asarray([1.0, 0.0, 0.0])

    reset = _reset_hidden_where_done(hidden, done)
    # The learner's mirror of the rollout reset must agree leaf-for-leaf.
    reset_learner = _reset_hidden(hidden, done > 0)
    for a, b in zip(reset, reset_learner):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    np.testing.assert_array_equal(np.asarray(reset[0][0]), np.zeros(SLOW_H))
    np.testing.assert_array_equal(np.asarray(reset[1]), [[0.0], [1.0], [2.0]])
    np.testing.assert_array_equal(np.asarray(reset[0][1:]), np.asarray(slow_h[1:]))

    # Next step (per-env vmap, the rollout convention): env 0 ticks from the
    # fresh zero state; env 1 (phase 1) and env 2 (phase 2) hold.
    obs_b = _random_obs(jax.random.PRNGKey(8), batch=B)
    _, _, (new_slow, new_phase) = jax.vmap(lambda o, h: _raw(nets, params, o, h))(
        obs_b, reset
    )
    np.testing.assert_array_equal(np.asarray(new_phase), [[1.0], [2.0], [0.0]])
    expected0, _ = nn.GRUCell(features=SLOW_H).apply(
        {"params": params["params"]["policy_head"]["slow_core"]},
        jnp.zeros((SLOW_H,)),
        _ff_input(params, jax.tree_util.tree_map(lambda x: x[0], obs_b)),
    )
    np.testing.assert_allclose(
        np.asarray(new_slow[0]), np.asarray(expected0), rtol=1e-6, atol=1e-6
    )
    assert np.any(np.asarray(new_slow[0]) != 0.0)
    np.testing.assert_array_equal(np.asarray(new_slow[1:]), np.asarray(slow_h[1:]))

    # The phase survives the replay store dtype exactly.
    stored = jax.tree_util.tree_map(lambda h: h.astype(meta.store_dtype), reset)
    np.testing.assert_array_equal(
        np.asarray(stored[meta.phase_leaf].astype(jnp.float32)), np.asarray(reset[1])
    )


# ---------------------------------------------------------------------------
# Factory: critic pass-through and guards
# ---------------------------------------------------------------------------


def test_critic_privileged_dim_passthrough(cfg):
    assert _slow_nets(cfg).critic.privileged_dim == 0
    nets = _slow_nets(cfg, critic_privileged_dim=8)
    assert nets.critic.privileged_dim == 8

    # init_training_state with the vnl entry point's default-hidden adapter.
    orig = nets.policy.init

    def _init_default_hidden(rng, obs, hidden=None):
        if hidden is None:
            hidden = nets.recurrent_meta.init_hidden()
        return orig(rng, obs, hidden)

    nets.policy.init = _init_default_hidden  # type: ignore[method-assign]
    template = {
        "vision": jnp.zeros(VISION_SHAPE),
        "imitation_target": jnp.zeros((TASK_OBS_SIZE,)),
        "proprioception": jnp.zeros((PROPRIO_SIZE,)),
    }
    state = init_training_state(
        jax.random.PRNGKey(0),
        nets,
        {"obs_template": template, "action_size": ACTION_SIZE},
        cfg,
    )
    assert "slow_core" in state.policy_params["params"]["policy_head"]
    crit_k = state.critic_params["params"]["Dense_0"]["kernel"]
    assert crit_k.shape[0] == 2 * CNN_FEAT + TASK_OBS_SIZE + ACTION_SIZE + 8


def test_factory_guards(cfg):
    # 'lstm' is supported (see the LSTM section below); any other cell raises,
    # and the message names the supported ones.
    with pytest.raises(ValueError, match="gru"):
        _slow_nets(cfg, rnn_cell="simple")
    with pytest.raises(ValueError, match="lstm"):
        _slow_nets(cfg, rnn_cell="rnn")
    with pytest.raises(ValueError, match="tick_stride"):
        _slow_nets(cfg, tick_stride=0)
    # float16 holds integers exactly only up to 2048: a longer cycle would
    # store a corrupted phase in replay.
    with pytest.raises(ValueError, match="tick_stride"):
        _slow_nets(cfg, tick_stride=4097, rnn_store_dtype="float16")
    _slow_nets(cfg, tick_stride=4097, rnn_store_dtype="float32")


# ---------------------------------------------------------------------------
# LSTM slow core (rnn_cell="lstm")
#
# The hidden stays the 2-leaf tuple (slow_h, phase) with phase_leaf == 1, so
# the rollout / replay / strided learner / eval are untouched. For an LSTM,
# slow_h is [.., 2H] = concat([c, h], -1) (cell state FIRST, output second).
# The FF head still reads only the H-wide output h, appended LAST, so an FF
# checkpoint grafts with zero-filled rows exactly as for the GRU.
# ---------------------------------------------------------------------------

LSTM_ROWS = 2 * SLOW_H


def _lstm_nets(cfg, **overrides):
    return _slow_nets(cfg, rnn_cell="lstm", **overrides)


def _lstm_cell_ref(params, slow_h, x):
    """One LSTM step from the packed [c, h] leaf: (new [c, h] leaf, new h)."""
    c, h = slow_h[..., :SLOW_H], slow_h[..., SLOW_H:]
    (nc, nh), _ = nn.LSTMCell(features=SLOW_H).apply(
        {"params": params["params"]["policy_head"]["slow_core"]}, (c, h), x
    )
    return jnp.concatenate([nc, nh], axis=-1), nh


def _lstm_hidden(rng_key, phase, scale=2.0):
    return (
        jax.random.normal(rng_key, (LSTM_ROWS,)) * scale,
        jnp.full((1,), float(phase), jnp.float32),
    )


def test_lstm_meta_and_init_hidden(cfg):
    nets = _lstm_nets(cfg)
    meta = nets.recurrent_meta
    assert isinstance(meta, SlowCoreMeta)
    assert meta.cell_type == "lstm"
    # hidden_sizes keeps the core width H; the packed leaf is 2H.
    assert meta.hidden_sizes == (SLOW_H,)
    assert meta.state_size == LSTM_ROWS
    assert meta.store_dtype == jnp.dtype("float16")
    assert meta.tick_stride == TICK
    assert meta.phase_leaf == 1

    h = meta.init_hidden()
    assert isinstance(h, tuple) and len(h) == 2
    assert [x.shape for x in h] == [(LSTM_ROWS,), (1,)]
    hb = meta.init_hidden(3)
    assert [x.shape for x in hb] == [(3, LSTM_ROWS), (3, 1)]
    for x in h + hb:
        assert x.dtype == jnp.float32
        assert not np.any(np.asarray(x))
    assert hb[meta.phase_leaf].shape == (3, 1)

    # The GRU meta is untouched: width H.
    gru = _slow_nets(cfg).recurrent_meta
    assert gru.cell_type == "gru"
    assert gru.state_size == SLOW_H
    assert [x.shape for x in gru.init_hidden(2)] == [(2, SLOW_H), (2, 1)]


def test_lstm_init_apply_shapes_unbatched_and_batched(cfg):
    nets = _lstm_nets(cfg)
    params = _init(nets)
    obs = _random_obs(jax.random.PRNGKey(2))

    mu, scale, (new_slow, new_phase) = _raw(
        nets, params, obs, nets.recurrent_meta.init_hidden()
    )
    assert mu.shape == (ACTION_SIZE,) and scale.shape == (ACTION_SIZE,)
    assert new_slow.shape == (LSTM_ROWS,) and new_phase.shape == (1,)
    assert new_slow.dtype == jnp.float32
    # A fresh episode ticks (phase == 0): the packed state moved off zero.
    assert np.any(np.asarray(new_slow) != 0.0)

    B = 3
    obs_b = _random_obs(jax.random.PRNGKey(3), batch=B)
    mu_b, scale_b, (slow_b, phase_b) = jax.vmap(lambda o, h: _raw(nets, params, o, h))(
        obs_b, nets.recurrent_meta.init_hidden(B)
    )
    assert mu_b.shape == (B, ACTION_SIZE) and scale_b.shape == (B, ACTION_SIZE)
    assert slow_b.shape == (B, LSTM_ROWS) and phase_b.shape == (B, 1)

    # tfd-returning __call__ agrees with raw().
    dist, nh = nets.policy.apply(params, obs, nets.recurrent_meta.init_hidden())
    np.testing.assert_allclose(np.asarray(dist.mean()), np.asarray(mu), atol=1e-6)
    assert nh[0].shape == (LSTM_ROWS,)


def test_lstm_param_names_graftable(cfg):
    """LSTM slow head = FF head tree + a `slow_core` LSTM subtree, with
    hidden_0/kernel widened by H (the OUTPUT width, not 2H) trailing rows."""
    ff_params = _ff_nets(cfg).policy.init(
        jax.random.PRNGKey(0), _random_obs(jax.random.PRNGKey(1))
    )
    lstm_params = _init(_lstm_nets(cfg))
    assert set(lstm_params["params"]) == {"prior", "decoder", "policy_head"}

    ff_pol = flatten_dict(ff_params["params"])
    lstm_flat = flatten_dict(lstm_params["params"])
    core = {k: v for k, v in lstm_flat.items() if k[:2] == ("policy_head", "slow_core")}
    rest = {k: v for k, v in lstm_flat.items() if k not in core}
    assert set(rest) == set(ff_pol)
    widened = [k for k in ff_pol if rest[k].shape != ff_pol[k].shape]
    assert widened == [("policy_head", "hidden_0", "kernel")]
    assert rest[("policy_head", "hidden_0", "kernel")].shape == (
        FF_IN_ROWS + SLOW_H,
        FF_LAYERS[0],
    )
    # nn.LSTMCell param names: input kernels i{i,f,g,o}, recurrent h{i,f,g,o}.
    assert {k[2] for k in core} == {"ii", "if", "ig", "io", "hi", "hf", "hg", "ho"}
    # The GRU core keeps its own names (the GRU path is unchanged).
    gru_flat = flatten_dict(_init(_slow_nets(cfg))["params"])
    assert {k[2] for k in gru_flat if k[:2] == ("policy_head", "slow_core")} == {
        "ir",
        "iz",
        "in",
        "hr",
        "hz",
        "hn",
    }

    labels = label_param_tree(lstm_params)
    core_labels = jax.tree_util.tree_leaves(
        labels["params"]["policy_head"]["slow_core"]
    )
    assert len(core_labels) > 0
    assert all(lbl == "policy_head" for lbl in core_labels)


def test_lstm_tick_updates_and_head_reads_h_last(cfg):
    """Tick = exactly one LSTM step on x = concat([vis, task, proprio]) from
    the packed [c, h]; the MLP input is concat([x, h_new]) -- the H-wide
    output h only (never c), appended LAST."""
    nets = _lstm_nets(cfg)
    params = _perturb(_init(nets), jax.random.PRNGKey(14))
    head_p = params["params"]["policy_head"]
    head = _SlowCorePolicyHeadModule(
        layer_sizes=FF_LAYERS,
        slow_hidden_size=SLOW_H,
        tick_stride=TICK,
        latent_size=LATENT_SIZE,
        vision_shape=VISION_SHAPE,
        cnn_feature_size=CNN_FEAT,
        cnn_channels=CNN_CHANNELS,
        mono_channels=1,
        shared_weights=True,
        cell_type="lstm",
    )
    obs = _random_obs(jax.random.PRNGKey(15))
    x = _ff_input(params, obs)
    packed = jax.random.normal(jax.random.PRNGKey(16), (LSTM_ROWS,))
    for phase in (0, 1):
        if phase == 0:
            new_packed, new_h = _lstm_cell_ref(params, packed, x)
        else:
            new_packed, new_h = packed, packed[SLOW_H:]
        h = jnp.concatenate([x, new_h], axis=-1)
        for i in range(len(FF_LAYERS)):
            d = head_p[f"hidden_{i}"]
            h = h @ d["kernel"] + d["bias"]
            h = nn.LayerNorm().apply({"params": head_p[f"LayerNorm_{i}"]}, h)
            h = nn.silu(h)
        ref = h @ head_p["residual"]["kernel"] + head_p["residual"]["bias"]

        out, (new_slow, new_phase) = head.apply(
            {"params": head_p},
            obs["vision"],
            obs["imitation_target"],
            obs["proprioception"],
            (packed, jnp.full((1,), float(phase))),
        )
        assert np.any(np.asarray(ref) != 0.0)
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(new_slow), np.asarray(new_packed), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            np.asarray(new_phase), np.asarray([(phase + 1) % TICK], np.float32)
        )
    # The tick moved BOTH halves (c and h) of a nonzero state.
    moved = _lstm_cell_ref(params, packed, x)[0]
    assert not np.allclose(np.asarray(moved[:SLOW_H]), np.asarray(packed[:SLOW_H]))
    assert not np.allclose(np.asarray(moved[SLOW_H:]), np.asarray(packed[SLOW_H:]))


def test_lstm_hold_semantics_over_a_stride(cfg):
    """Over a k-step stride c AND h change only on the tick step (phase == 0);
    both are held exactly on the k-1 non-tick steps; the phase cycles mod k."""
    nets = _lstm_nets(cfg)
    params = _perturb(_init(nets), jax.random.PRNGKey(21))
    hidden = (
        jax.random.normal(jax.random.PRNGKey(22), (LSTM_ROWS,)),
        jnp.zeros((1,)),
    )
    key = jax.random.PRNGKey(23)
    n_steps = 2 * TICK + 1
    for t in range(n_steps):
        key, k = jax.random.split(key)
        before = np.asarray(hidden[0])
        phase_before = float(hidden[1][0])
        _, _, hidden = _raw(nets, params, _random_obs(k), hidden)
        after = np.asarray(hidden[0])
        c_changed = not np.array_equal(after[:SLOW_H], before[:SLOW_H])
        h_changed = not np.array_equal(after[SLOW_H:], before[SLOW_H:])
        assert phase_before == float(t % TICK)
        if t % TICK == 0:
            assert c_changed and h_changed, f"tick step {t} must update c and h"
        else:
            assert not c_changed and not h_changed, f"step {t} must hold c and h"
        np.testing.assert_array_equal(
            np.asarray(hidden[1]), np.asarray([(t + 1) % TICK], np.float32)
        )


def test_lstm_k1_ticks_every_step(cfg):
    nets = _lstm_nets(cfg, tick_stride=1)
    params = _perturb(_init(nets), jax.random.PRNGKey(24))
    hidden = nets.recurrent_meta.init_hidden()
    key = jax.random.PRNGKey(25)
    for _ in range(4):
        key, k = jax.random.split(key)
        before = np.asarray(hidden[0])
        _, _, hidden = _raw(nets, params, _random_obs(k), hidden)
        after = np.asarray(hidden[0])
        assert not np.array_equal(after[:SLOW_H], before[:SLOW_H])
        assert not np.array_equal(after[SLOW_H:], before[SLOW_H:])
        np.testing.assert_array_equal(np.asarray(hidden[1]), np.asarray([0.0]))


@pytest.mark.parametrize("mode", RESIDUAL_MODES)
def test_lstm_graft_invariance_with_graft_params(cfg, mode):
    """FF params grafted (via `warm_start_graft.graft_params`) into the LSTM
    slow-core net give the FF policy's outputs for ANY packed [c, h] state:
    the state reaches the output only through the zero-filled hidden_0 rows.
    (GRU counterpart: test_warm_start_graft.test_ff_to_slow_core_output_equal.)
    """
    ff_nets = _ff_nets(cfg, residual_mode=mode, residual_scale=2.0)
    lstm_nets = _lstm_nets(cfg, residual_mode=mode, residual_scale=2.0)
    obs = _random_obs(jax.random.PRNGKey(1))
    # Perturb the whole FF tree (incl. the zero-init residual) so the head
    # really contributes -- otherwise equality is trivial.
    ff_params = _perturb(
        ff_nets.policy.init(jax.random.PRNGKey(0), obs), jax.random.PRNGKey(8)
    )
    lstm_init = _init(lstm_nets, key=3)
    grafted, report = graft_params(lstm_init, jax.tree.map(np.asarray, ff_params))

    assert report["widened"] == ["params/policy_head/hidden_0/kernel"]
    assert report["new"] and all(
        p.startswith("params/policy_head/slow_core/") for p in report["new"]
    )
    assert report["copied"] == len(jax.tree.leaves(ff_params)) - 1
    assert jax.tree.structure(grafted) == jax.tree.structure(lstm_init)
    k = np.asarray(grafted["params"]["policy_head"]["hidden_0"]["kernel"])
    assert k.shape == (FF_IN_ROWS + SLOW_H, FF_LAYERS[0])
    np.testing.assert_array_equal(k[FF_IN_ROWS:], 0.0)
    jax.tree.map(
        np.testing.assert_array_equal,
        grafted["params"]["policy_head"]["slow_core"],
        lstm_init["params"]["policy_head"]["slow_core"],
    )

    key = jax.random.PRNGKey(10)
    for i, phase in enumerate([0, 1, 0, 2]):
        key, k_o, k_h = jax.random.split(key, 3)
        o = _random_obs(k_o)
        hidden = _lstm_hidden(k_h, phase, scale=3.0)
        ff_dist = ff_nets.policy.apply(ff_params, o)
        mu, scale, _ = _raw(lstm_nets, grafted, o, hidden)
        np.testing.assert_allclose(
            np.asarray(mu),
            np.asarray(ff_dist.mean()),
            atol=1e-6,
            err_msg=f"mu mismatch (mode={mode}, case {i})",
        )
        np.testing.assert_allclose(
            np.asarray(scale),
            np.asarray(ff_dist.stddev()),
            atol=1e-6,
            err_msg=f"scale mismatch (mode={mode}, case {i})",
        )

    # Control: the UN-grafted LSTM net does not match, and nonzero rows DO
    # make the output depend on the state (so the check above can fail).
    ff_mu = np.asarray(ff_nets.policy.apply(ff_params, obs).mean())
    mu_bad, _, _ = _raw(
        lstm_nets, lstm_init, obs, _lstm_hidden(jax.random.PRNGKey(9), 1)
    )
    assert not np.allclose(np.asarray(mu_bad), ff_mu, atol=1e-4)
    bad = _graft_ff_into_slow(
        ff_params,
        lstm_init,
        jax.random.normal(jax.random.PRNGKey(12), (SLOW_H, FF_LAYERS[0])),
    )
    h_ones = (jnp.ones((LSTM_ROWS,)), jnp.ones((1,)))
    mu_nz, _, _ = _raw(lstm_nets, bad, obs, h_ones)
    assert not np.allclose(np.asarray(mu_nz), ff_mu, atol=1e-4)


def test_lstm_graft_zero_rows_see_only_h_not_c(cfg):
    """The widened hidden_0 rows multiply the H-wide OUTPUT h only. With
    NONZERO rows on a held step (phase=1: the cell does not run), perturbing
    the cell state c leaves the policy unchanged -- the head never reads c --
    while perturbing h changes it."""
    lstm_nets = _lstm_nets(cfg)
    ff_nets = _ff_nets(cfg)
    obs = _random_obs(jax.random.PRNGKey(1))
    ff_params = _perturb(
        ff_nets.policy.init(jax.random.PRNGKey(0), obs), jax.random.PRNGKey(8)
    )
    grafted = _graft_ff_into_slow(
        ff_params,
        _init(lstm_nets, key=3),
        jax.random.normal(jax.random.PRNGKey(12), (SLOW_H, FF_LAYERS[0])),
    )
    base = jax.random.normal(jax.random.PRNGKey(30), (LSTM_ROWS,))
    phase_hold = jnp.full((1,), 1.0)
    mu0, _, _ = _raw(lstm_nets, grafted, obs, (base, phase_hold))
    mu_c, _, _ = _raw(lstm_nets, grafted, obs, (base.at[:SLOW_H].add(5.0), phase_hold))
    mu_h, _, _ = _raw(lstm_nets, grafted, obs, (base.at[SLOW_H:].add(5.0), phase_hold))
    np.testing.assert_array_equal(np.asarray(mu0), np.asarray(mu_c))
    assert not np.allclose(np.asarray(mu0), np.asarray(mu_h), atol=1e-4)


def test_lstm_reset_zeroes_whole_packed_leaf(cfg):
    """A done zeroes the whole 2H leaf (c and h) and the phase: the rollout
    helper and the learner helper agree; the next step restarts the clock
    with an LSTM tick from the all-zero state."""
    nets = _lstm_nets(cfg)
    meta = nets.recurrent_meta
    params = _perturb(_init(nets), jax.random.PRNGKey(31))
    B = 3
    slow_h = jax.random.normal(jax.random.PRNGKey(7), (B, LSTM_ROWS)) + 1.0
    phase = jnp.asarray([[2.0], [1.0], [2.0]])
    hidden = (slow_h, phase)
    done = jnp.asarray([1.0, 0.0, 0.0])

    reset = _reset_hidden_where_done(hidden, done)
    reset_learner = _reset_hidden(hidden, done > 0)
    for a, b in zip(reset, reset_learner):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert reset[0].shape == (B, LSTM_ROWS)
    np.testing.assert_array_equal(np.asarray(reset[0][0]), np.zeros(LSTM_ROWS))
    np.testing.assert_array_equal(np.asarray(reset[1]), [[0.0], [1.0], [2.0]])
    np.testing.assert_array_equal(np.asarray(reset[0][1:]), np.asarray(slow_h[1:]))

    obs_b = _random_obs(jax.random.PRNGKey(8), batch=B)
    _, _, (new_slow, new_phase) = jax.vmap(lambda o, h: _raw(nets, params, o, h))(
        obs_b, reset
    )
    np.testing.assert_array_equal(np.asarray(new_phase), [[1.0], [2.0], [0.0]])
    expected0, _ = _lstm_cell_ref(
        params,
        jnp.zeros((LSTM_ROWS,)),
        _ff_input(params, jax.tree_util.tree_map(lambda x: x[0], obs_b)),
    )
    np.testing.assert_allclose(
        np.asarray(new_slow[0]), np.asarray(expected0), rtol=1e-6, atol=1e-6
    )
    assert np.any(np.asarray(new_slow[0][:SLOW_H]) != 0.0)
    assert np.any(np.asarray(new_slow[0][SLOW_H:]) != 0.0)
    np.testing.assert_array_equal(np.asarray(new_slow[1:]), np.asarray(slow_h[1:]))

    # The replay store dtype (float16) keeps the packed leaf shape and holds
    # the phase exactly (c is stored at f16 too, so it is range-limited).
    stored = jax.tree_util.tree_map(lambda h: h.astype(meta.store_dtype), reset)
    assert stored[0].shape == (B, LSTM_ROWS) and stored[0].dtype == jnp.float16
    np.testing.assert_array_equal(
        np.asarray(stored[meta.phase_leaf].astype(jnp.float32)), np.asarray(reset[1])
    )
