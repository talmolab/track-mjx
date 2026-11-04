"""Optional privileged critic input (asymmetric actor-critic).

The critic may see a task-state vector the policy never sees (which maze arm
is baited, the current choice, ...). The env writes it to
``state.info[cfg.critic_privileged_key]``; the rollout's
``extra_state_extras`` side channel stores it from the PRE-step state, so
``batch[key][:, t]`` is aligned with ``observation[:, t]``. The learner
attaches it as ``obs["privileged"]`` to critic inputs ONLY, gathered at the
index of the obs it accompanies:

  * FF path: t0 for the critic loss and the policy-loss Q; the bootstrap
    index n for the target critic (compressed schema obs[:, n], or
    next_observation[:, n-1], which is the same state).
  * Recurrent path: [:, :L] for the loss points, [:, n:n+L] for bootstrap.

Defaults (``critic_privileged_dim=0``, ``critic_privileged_key=""``) leave
every old path untouched: no new params, same ops.

Tiny real kl-anchor nets (8x8 binocular vision, as in test_fused_rnn) so the
production critic class is what is exercised. CPU-only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    _critic_obs,
    init_training_state,
    make_optimizers,
    sgd_step,
)
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks
from track_mjx.agent.dmpo.networks_kl_anchor_rnn import (
    make_dmpo_kl_anchor_rnn_networks,
)

VISION_SHAPE = (8, 8, 2)
TASK_OBS = 4
PROPRIO = 5
ACTION = 4
LATENT = 3
PRIV = 8
CNN_FEAT = 4
RNN_HIDDEN = (6,)
KEY = "critic_priv"
B = 4

# Critic first-Dense input rows without the privileged block: binocular CNN
# features (2 eyes x CNN_FEAT) + task obs + proprio (critic_use_proprio=True)
# + action.
_BASE_ROWS = 2 * CNN_FEAT + TASK_OBS + PROPRIO + ACTION


def _shared_net_kwargs(cfg):
    return dict(
        proprio_size=PROPRIO,
        task_obs_size=TASK_OBS,
        action_size=ACTION,
        latent_size=LATENT,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
        prior_layer_sizes=(8,),
        decoder_layer_sizes=(8,),
        cnn_feature_size=CNN_FEAT,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        value_hidden_layer_sizes=(16,),
        critic_use_proprio=True,
    )


def _ff_nets(cfg, priv_dim=PRIV):
    kw = _shared_net_kwargs(cfg)
    if priv_dim is not None:
        kw["critic_privileged_dim"] = priv_dim
    return make_dmpo_kl_anchor_networks(policy_head_layer_sizes=(8,), **kw)


def _rnn_nets(cfg, priv_dim=PRIV):
    kw = _shared_net_kwargs(cfg)
    if priv_dim is not None:
        kw["critic_privileged_dim"] = priv_dim
    nets = make_dmpo_kl_anchor_rnn_networks(
        rnn_cell="gru",
        rnn_mlp_layers=(8,),
        rnn_hidden_sizes=RNN_HIDDEN,
        rnn_store_dtype="float16",
        **kw,
    )
    # Same default-hidden init adapter the vnl entry point installs so that
    # init_training_state (FF-shaped policy.init call) serves recurrent nets.
    _orig = nets.policy.init

    def _init_default_hidden(rng, obs, hidden=None):
        if hidden is None:
            hidden = nets.recurrent_meta.init_hidden()
        return _orig(rng, obs, hidden)

    nets.policy.init = _init_default_hidden  # type: ignore[method-assign]
    return nets


def _obs_template():
    return {
        "vision": jnp.zeros(VISION_SHAPE, jnp.float32),
        "imitation_target": jnp.zeros((TASK_OBS,), jnp.float32),
        "proprioception": jnp.zeros((PROPRIO,), jnp.float32),
    }


def _env_spec():
    return {"obs_template": _obs_template(), "action_size": ACTION}


def _ff_cfg(**overrides):
    kw = dict(
        batch_size=B,
        sequence_length=6,
        n_step=3,
        use_n_step=True,
        store_next_observation=False,
        num_samples=3,
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        discount=0.9,
        critic_privileged_key=KEY,
    )
    kw.update(overrides)
    return DMPOConfig(**kw)


def _rnn_cfg(L=3, n=4, **overrides):
    kw = dict(
        batch_size=B,
        sequence_length=L + n,
        n_step=n,
        rnn_bptt_length=L,
        use_n_step=True,
        store_next_observation=False,
        num_samples=3,
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        discount=0.9,
        critic_privileged_key=KEY,
    )
    kw.update(overrides)
    return DMPOConfig(**kw)


def _obs_batch(key, T):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "vision": jax.random.uniform(k1, (B, T, *VISION_SHAPE)),
        "imitation_target": jax.random.normal(k2, (B, T, TASK_OBS)),
        "proprioception": jax.random.normal(k3, (B, T, PROPRIO)),
    }


def _batch(cfg, seed=1, next_obs=False, recurrent=False, priv=True):
    T = cfg.sequence_length
    k = jax.random.split(jax.random.PRNGKey(seed), 6)
    batch = {
        "observation": _obs_batch(k[0], T),
        "action": jax.random.uniform(k[1], (B, T, ACTION), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(k[2], (B, T)),
        "discount": jnp.ones((B, T)),
        "truncation": jnp.zeros((B, T)),
    }
    if next_obs:
        batch["next_observation"] = _obs_batch(k[3], T)
    if priv:
        # Nonzero everywhere: a zero input row has an identically-zero
        # kernel gradient, which would mask a wiring bug in the grad test.
        batch[KEY] = jax.random.normal(k[4], (B, T, PRIV)) + 2.0
    if recurrent:
        # Stored per-step hidden (values are DATA to the learner; staleness
        # is irrelevant here).
        batch["policy_hidden"] = tuple(
            (0.1 * jax.random.normal(k[5], (B, T, h))).astype(jnp.float16)
            for h in RNN_HIDDEN
        )
    return batch


def _single_obs(seed, priv=None):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    obs = {
        "vision": jax.random.uniform(k[0], VISION_SHAPE),
        "imitation_target": jax.random.normal(k[1], (TASK_OBS,)),
        "proprioception": jax.random.normal(k[2], (PROPRIO,)),
    }
    if priv is not None:
        obs["privileged"] = priv
    return obs


def _perturb(batch, t, delta=3.0):
    out = dict(batch)
    out[KEY] = batch[KEY].at[:, t].add(delta)
    return out


def _jit_step(nets, cfg):
    optimizers = make_optimizers(cfg)
    return jax.jit(lambda st, b: sgd_step(st, b, nets, optimizers, cfg))


# ---------------------------------------------------------------------------
# Network side.
# ---------------------------------------------------------------------------


def test_privileged_changes_q():
    cfg = _ff_cfg()
    nets = _ff_nets(cfg)
    assert nets.critic.privileged_dim == PRIV
    p0 = jnp.zeros((PRIV,))
    obs = _single_obs(0, priv=p0)
    act = jnp.zeros((ACTION,))
    params = nets.critic.init(jax.random.PRNGKey(0), obs, act)

    # Privileged block appended LAST on the first Dense's input rows.
    k0 = params["params"]["Dense_0"]["kernel"]
    assert k0.shape[0] == _BASE_ROWS + PRIV

    q_a = nets.critic.apply(params, obs, act).mean()
    obs_b = dict(obs, privileged=jnp.linspace(-2.0, 2.0, PRIV))
    q_b = nets.critic.apply(params, obs_b, act).mean()
    assert np.isfinite(float(q_a)) and np.isfinite(float(q_b))
    assert float(q_a) != float(q_b)

    # Fail loud on a missing / wrong-width privileged input.
    with pytest.raises(KeyError):
        nets.critic.apply(params, _single_obs(0), act)
    with pytest.raises(ValueError, match="privileged"):
        nets.critic.apply(params, dict(obs, privileged=jnp.zeros((PRIV + 1,))), act)


def test_policy_invariant_to_privileged():
    cfg = _ff_cfg()
    nets = _ff_nets(cfg)
    obs = _single_obs(3)
    obs_p = _single_obs(3, priv=jnp.arange(PRIV, dtype=jnp.float32))
    params = nets.policy.init(jax.random.PRNGKey(1), obs)
    d0 = nets.policy.apply(params, obs)
    d1 = nets.policy.apply(params, obs_p)
    np.testing.assert_array_equal(np.asarray(d0.mean()), np.asarray(d1.mean()))
    np.testing.assert_array_equal(np.asarray(d0.stddev()), np.asarray(d1.stddev()))

    rcfg = _rnn_cfg()
    rnets = _rnn_nets(rcfg)
    h = rnets.recurrent_meta.init_hidden()
    rparams = rnets.policy.init(jax.random.PRNGKey(2), obs)
    mu0, sc0, h0 = rnets.policy.apply(rparams, obs, h, method="raw")
    mu1, sc1, h1 = rnets.policy.apply(rparams, obs_p, h, method="raw")
    np.testing.assert_array_equal(np.asarray(mu0), np.asarray(mu1))
    np.testing.assert_array_equal(np.asarray(sc0), np.asarray(sc1))
    for a, b in zip(h0, h1):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("recurrent", [False, True])
def test_sgd_step_never_feeds_privileged_to_policy(recurrent):
    """The learner must hand obs["privileged"] to critic applies only."""
    cfg = _rnn_cfg() if recurrent else _ff_cfg()
    nets = (_rnn_nets if recurrent else _ff_nets)(cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    seen = []
    orig_apply = nets.policy.apply

    def _spy_apply(params, obs, *args, **kwargs):
        seen.append(sorted(obs.keys()))
        assert "privileged" not in obs, "policy received the privileged key"
        return orig_apply(params, obs, *args, **kwargs)

    nets.policy.apply = _spy_apply  # type: ignore[method-assign]
    batch = _batch(cfg, recurrent=recurrent)
    _, metrics = sgd_step(state, batch, nets, make_optimizers(cfg), cfg)
    assert seen, "spy never called"
    assert np.isfinite(float(metrics["critic_loss"]))


# ---------------------------------------------------------------------------
# Learner helper.
# ---------------------------------------------------------------------------


def test_critic_obs_gather_modes():
    T = 7
    priv = jnp.arange(B * T * PRIV, dtype=jnp.float32).reshape(B, T, PRIV)
    batch = {KEY: priv}
    obs_b = {"imitation_target": jnp.zeros((B, TASK_OBS))}
    obs_bl = {"imitation_target": jnp.zeros((B, 3, TASK_OBS))}

    # Off: the very same object back (no new ops on the default path).
    assert _critic_obs(obs_b, batch, "", 0) is obs_b

    out = _critic_obs(obs_b, batch, KEY, 2)
    np.testing.assert_array_equal(out["privileged"], priv[:, 2])
    assert "privileged" not in obs_b  # input not mutated
    assert out["imitation_target"] is obs_b["imitation_target"]

    out = _critic_obs(obs_bl, batch, KEY, slice(4, 7))
    np.testing.assert_array_equal(out["privileged"], priv[:, 4:7])

    # Per-row gather indices: [B] and [B, L] (strided tick learner).
    idx_b = jnp.array([0, 3, 6, 1])
    out = _critic_obs(obs_b, batch, KEY, idx_b)
    np.testing.assert_array_equal(
        out["privileged"], np.stack([priv[i, idx_b[i]] for i in range(B)])
    )
    idx_bl = jnp.array([[0, 1, 2], [3, 4, 5], [6, 6, 6], [0, 5, 2]])
    out = _critic_obs(obs_bl, batch, KEY, idx_bl)
    expect = np.stack([np.stack([priv[i, j] for j in idx_bl[i]]) for i in range(B)])
    np.testing.assert_array_equal(out["privileged"], expect)

    with pytest.raises(ValueError, match=KEY):
        _critic_obs(obs_b, {}, KEY, 0)
    with pytest.raises(ValueError, match="dict"):
        _critic_obs(jnp.zeros((B, 5)), batch, KEY, 0)


# ---------------------------------------------------------------------------
# FF learner path.
# ---------------------------------------------------------------------------


def _priv_rows_delta(new_state, state):
    k_new = new_state.critic_params["params"]["Dense_0"]["kernel"]
    k_old = state.critic_params["params"]["Dense_0"]["kernel"]
    assert k_new.shape[0] == _BASE_ROWS + PRIV
    return np.asarray(jnp.abs(k_new[-PRIV:] - k_old[-PRIV:]))


def test_ff_sgd_step_with_privileged_jit():
    cfg = _ff_cfg()
    nets = _ff_nets(cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    assert (
        state.critic_params["params"]["Dense_0"]["kernel"].shape[0] == _BASE_ROWS + PRIV
    )
    step = _jit_step(nets, cfg)
    new_state, metrics = step(state, _batch(cfg))
    for k, v in metrics.items():
        assert np.isfinite(np.asarray(v)).all(), f"non-finite metric {k}"
    # First Adam step: a parameter moves iff its gradient is nonzero. Every
    # privileged input row of the critic's first kernel must have moved.
    delta = _priv_rows_delta(new_state, state)
    assert (delta.max(axis=1) > 0).all(), delta.max(axis=1)

    # Control: an all-zero privileged input gives those rows exactly zero
    # gradient -- proves the rows above moved because of the input wiring.
    zero_batch = _batch(cfg)
    zero_batch[KEY] = jnp.zeros_like(zero_batch[KEY])
    new_state0, _ = step(state, zero_batch)
    assert _priv_rows_delta(new_state0, state).max() == 0.0


@pytest.mark.parametrize(
    "schema,used_q,used_boot,unused",
    [
        # Compressed schema, n=3: critic loss + policy Q at t0, bootstrap
        # observation[:, 3].
        ("compressed", 0, 3, (1, 2, 4, 5)),
        # Legacy next_observation schema, single-step TD: bootstrap is
        # next_observation[:, 0] == the state at t=1.
        ("next_obs", 0, 1, (2, 3, 4, 5)),
    ],
)
def test_ff_privileged_alignment(schema, used_q, used_boot, unused):
    if schema == "compressed":
        cfg = _ff_cfg()
    else:
        cfg = _ff_cfg(use_n_step=False, store_next_observation=True)
    nets = _ff_nets(cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    step = _jit_step(nets, cfg)
    batch = _batch(cfg, next_obs=(schema == "next_obs"))
    _, base = step(state, batch)

    def run(t):
        return step(state, _perturb(batch, t))[1]

    # t0: online critic loss AND the policy-loss Q (target critic at t0).
    m = run(used_q)
    assert float(m["critic_loss"]) != float(base["critic_loss"])
    assert float(m["q_max"]) != float(base["q_max"])
    # Bootstrap index: target only -- the policy-loss Q is untouched.
    m = run(used_boot)
    assert float(m["critic_loss"]) != float(base["critic_loss"])
    for k in ("q_min", "q_max", "policy_loss"):
        assert float(m[k]) == float(base[k]), k
    # Every other index is never read.
    for t in unused:
        m = run(t)
        for k, v in base.items():
            np.testing.assert_array_equal(
                np.asarray(m[k]), np.asarray(v), err_msg=f"t={t} {k}"
            )


def test_ff_next_obs_schema_full_horizon_fails_loud():
    """With next_observation and n == T there is no stored privileged
    vector for the bootstrap state; refuse instead of misaligning."""
    cfg = _ff_cfg(store_next_observation=True, n_step=6, sequence_length=6)
    nets = _ff_nets(cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    with pytest.raises(ValueError, match="privileged"):
        sgd_step(state, _batch(cfg, next_obs=True), nets, make_optimizers(cfg), cfg)


def test_privileged_config_mismatch_fails_loud():
    # Key set, critic built without privileged input: the critic would
    # silently ignore the vector.
    cfg = _ff_cfg()
    nets0 = _ff_nets(cfg, priv_dim=0)
    st0 = init_training_state(jax.random.PRNGKey(0), nets0, _env_spec(), cfg)
    with pytest.raises(ValueError, match="critic_privileged"):
        sgd_step(st0, _batch(cfg), nets0, make_optimizers(cfg), cfg)
    # Critic expects privileged input, key empty.
    cfg_off = _ff_cfg(critic_privileged_key="")
    nets = _ff_nets(cfg_off)
    st = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg_off)
    with pytest.raises(ValueError, match="critic_privileged"):
        sgd_step(st, _batch(cfg_off), nets, make_optimizers(cfg_off), cfg_off)
    # Key set but the batch lacks it (extra_state_extras not wired).
    nets = _ff_nets(cfg)
    st = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    with pytest.raises(ValueError, match=KEY):
        sgd_step(st, _batch(cfg, priv=False), nets, make_optimizers(cfg), cfg)


# ---------------------------------------------------------------------------
# Recurrent learner path.
# ---------------------------------------------------------------------------


def test_rnn_sgd_step_with_privileged_jit():
    cfg = _rnn_cfg()
    nets = _rnn_nets(cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    step = _jit_step(nets, cfg)
    new_state, metrics = step(state, _batch(cfg, recurrent=True))
    for k, v in metrics.items():
        assert np.isfinite(np.asarray(v)).all(), f"non-finite metric {k}"
    delta = _priv_rows_delta(new_state, state)
    assert (delta.max(axis=1) > 0).all(), delta.max(axis=1)


def test_rnn_privileged_alignment():
    """L=3, n=4, T=7: loss points read [0, 3), bootstrap reads [4, 7);
    index 3 is never read."""
    L, n = 3, 4
    cfg = _rnn_cfg(L=L, n=n)
    nets = _rnn_nets(cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    step = _jit_step(nets, cfg)
    batch = _batch(cfg, recurrent=True)
    _, base = step(state, batch)

    def run(t):
        return step(state, _perturb(batch, t))[1]

    for t in range(L):
        m = run(t)
        assert float(m["critic_loss"]) != float(base["critic_loss"]), t
        assert float(m["q_max"]) != float(base["q_max"]) or float(m["q_min"]) != float(
            base["q_min"]
        ), t
    for t in range(n, n + L):
        m = run(t)
        assert float(m["critic_loss"]) != float(base["critic_loss"]), t
        for k in ("q_min", "q_max", "policy_loss"):
            assert float(m[k]) == float(base[k]), (t, k)
    m = run(L)  # L < n: the gap index is read by nothing
    for k, v in base.items():
        np.testing.assert_array_equal(np.asarray(m[k]), np.asarray(v), err_msg=k)


# ---------------------------------------------------------------------------
# Defaults leave the old param tree untouched.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recurrent", [False, True])
def test_default_dim_zero_param_tree_unchanged(recurrent):
    cfg = (
        _rnn_cfg(critic_privileged_key="")
        if recurrent
        else _ff_cfg(critic_privileged_key="")
    )
    assert DMPOConfig().critic_privileged_key == ""
    make = _rnn_nets if recurrent else _ff_nets
    nets_default = make(cfg, priv_dim=None)  # kwarg omitted
    nets_zero = make(cfg, priv_dim=0)
    assert nets_default.critic.privileged_dim == 0
    assert nets_zero.critic == nets_default.critic

    s_def = init_training_state(jax.random.PRNGKey(0), nets_default, _env_spec(), cfg)
    s_zero = init_training_state(jax.random.PRNGKey(0), nets_zero, _env_spec(), cfg)
    assert jax.tree.structure(s_def.critic_params) == jax.tree.structure(
        s_zero.critic_params
    )
    for a, b in zip(
        jax.tree.leaves(s_def.critic_params), jax.tree.leaves(s_zero.critic_params)
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # No privileged rows: the first critic kernel has the historical width.
    assert s_def.critic_params["params"]["Dense_0"]["kernel"].shape[0] == _BASE_ROWS
    # Same critic tree as a direct init on the historical 3-key obs.
    direct = nets_default.critic.init(
        jax.random.PRNGKey(0), _obs_template(), jnp.zeros((ACTION,))
    )
    assert jax.tree.structure(direct) == jax.tree.structure(s_def.critic_params)
    # And an untouched batch (no privileged key) trains as before.
    new_state, metrics = sgd_step(
        s_def,
        _batch(cfg, recurrent=recurrent, priv=False),
        nets_default,
        make_optimizers(cfg),
        cfg,
    )
    assert np.isfinite(float(metrics["critic_loss"]))
