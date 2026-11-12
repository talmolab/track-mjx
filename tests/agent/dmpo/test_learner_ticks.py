"""Strided ("tick") BPTT in the recurrent learner, for slow-core heads.

A slow-core policy (networks_kl_anchor_slow) carries hidden = (slow_h, phase)
and updates slow_h only at TICKS (stored phase < 0.5), every k env steps,
holding it in between. With ``cfg.rnn_tick_stride = k`` and
``rnn_bptt_length = L`` the learner samples T = L*k + n steps and backprops
over the first L ticks of the W = L*k window: frames, actions and loss points
are gathered at the tick indices, the unroll forces the phase leaf to 0 (so
every unrolled step IS a tick), the hidden is reset where a done falls
between consecutive ticks, the n-step returns use every step and are gathered
at the ticks, and each tick bootstraps at idx + n with the STORED hidden.

Coverage:
  * test_tick_indices_with_resets: tick positions from phase rows containing
    episode resets, against hand-computed indices; fill_frac 0 (and > 0 for
    a corrupted row, so the metric can fire).
  * test_stride1_bit_identical_to_old_path: a k=1 phase-leaf head through
    the tick path reproduces the legacy (no phase leaf) slicing path
    exactly -- losses, metrics and updated params.
  * test_forced_phase_unroll_updates_every_tick
  * test_done_between_ticks_resets
  * test_bptt_grad_reaches_slow_core_kernel_across_ticks
  * test_sgd_step_ticks_smoke_jit (k=10, L=4, n=3, privileged critic on,
    B == L so the per-row [B, L] privileged gather is exercised in the one
    shape where a 1-D index would be ambiguous)
  * test_guard_sequence_length (+ the stride / phase-leaf guards)
plus: the privileged alignment at idx / idx + n; the critic loss against an
independent reference (brute-force returns, bootstrap at idx + n with the
stored, unforced hidden); rnn/tick_fill_frac on corrupted phases; and the
phase-leaf exclusion from the rnn/ diagnostics. The per-step path itself is
pinned by test_learner_rnn (unchanged). CPU-only, tiny real networks.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.agent.dmpo.test_learner_rnn import (
    _HIDDEN,
    _StubRecurrentPolicy,
)
from tests.agent.dmpo.test_learner_rnn import _make_batch as _make_legacy_batch
from tests.agent.dmpo.test_learner_rnn import _setup as _setup_legacy
from track_mjx.agent.dmpo.config import LEGACY_ALIGNMENT, DMPOConfig
from track_mjx.agent.dmpo.learner import (
    _critic_loss_fn,
    _done_between_ticks,
    _normalize_obs,
    _policy_loss_fn_rnn,
    _reset_hidden,
    _sgd_step_rnn,
    _take_steps,
    _unroll_policy_raw,
    compute_categorical_target,
    init_training_state,
    make_optimizers,
    sgd_step,
    tick_indices,
)
from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.networks_kl_anchor_slow import (
    SlowCoreMeta,
    make_dmpo_kl_anchor_slow_networks,
)

VISION_SHAPE = (8, 8, 2)
TASK_OBS = 4
PROPRIO = 5
ACTION = 4
LATENT = 3
SLOW_H = 6
PRIV = 8
KEY = "critic_priv"
B = 4
PHASE = 1  # SlowCoreMeta.phase_leaf


# ---------------------------------------------------------------------------
# Fixtures: tiny real slow-core nets + a hand rollout that stores exactly what
# collect_rollout stores (the PRE-step hidden, cast to store_dtype, all leaves
# zeroed after a done).
# ---------------------------------------------------------------------------


def _cfg(L=4, n=3, k=10, priv=False, **overrides):
    kw = dict(
        batch_size=B,
        sequence_length=L * k + n,
        n_step=n,
        rnn_bptt_length=L,
        rnn_tick_stride=k,
        use_n_step=True,
        store_next_observation=False,
        num_samples=3,
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        discount=0.9,
        critic_privileged_key=KEY if priv else "",
    )
    kw.update(overrides)
    return DMPOConfig(**kw)


def _slow_nets(cfg, k, priv_dim=0, cell="gru"):
    nets = make_dmpo_kl_anchor_slow_networks(
        proprio_size=PROPRIO,
        task_obs_size=TASK_OBS,
        action_size=ACTION,
        latent_size=LATENT,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
        prior_layer_sizes=(8,),
        decoder_layer_sizes=(8,),
        policy_head_layer_sizes=(8,),
        slow_hidden_size=SLOW_H,
        tick_stride=k,
        rnn_cell=cell,
        rnn_store_dtype="float16",
        cnn_feature_size=4,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        value_hidden_layer_sizes=(16,),
        critic_use_proprio=True,
        critic_privileged_dim=priv_dim,
    )
    # Default-hidden init adapter (as the vnl entry point installs) so the
    # FF-shaped init_training_state serves the recurrent policy.
    _orig = nets.policy.init

    def _init(rng, obs, hidden=None):
        if hidden is None:
            hidden = nets.recurrent_meta.init_hidden()
        return _orig(rng, obs, hidden)

    nets.policy.init = _init  # type: ignore[method-assign]
    return nets


def _env_spec():
    return {
        "obs_template": {
            "vision": jnp.zeros(VISION_SHAPE, jnp.float32),
            "imitation_target": jnp.zeros((TASK_OBS,), jnp.float32),
            "proprioception": jnp.zeros((PROPRIO,), jnp.float32),
        },
        "action_size": ACTION,
    }


def _perturb(tree, key, scale=0.3):
    """Perturb every leaf. The head's residual Dense is zero-init, which makes
    the policy independent of slow_h (no gradient into the core); a perturbed
    tree makes the slow state actually reach the output."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [
            x + scale * jax.random.normal(kk, x.shape, x.dtype)
            for x, kk in zip(leaves, keys)
        ],
    )


def _state(nets, cfg, perturb=True):
    state = init_training_state(jax.random.PRNGKey(0), nets, _env_spec(), cfg)
    if perturb:
        p = _perturb(state.policy_params, jax.random.PRNGKey(5))
        pol_opt = make_optimizers(cfg)[0]
        state = state._replace(
            policy_params=p,
            target_policy_params=p,
            policy_opt_state=pol_opt.init(p),
        )
    return state


def _hand_rollout(nets, params, normalizer, obs, discount, h_init):
    """Python-loop reference of the recurrent rollout's hidden storage.

    Returns (stored hidden tuple of [B, T, ...] in store_dtype, mu [B, T, A]
    the policy mean at every step from the live f32 hidden)."""
    T = discount.shape[1]
    store = nets.recurrent_meta.store_dtype
    obs_norm = _normalize_obs(obs, normalizer)
    step = jax.jit(jax.vmap(lambda o, h: nets.policy.apply(params, o, h, method="raw")))
    h = h_init
    stored, mus = [], []
    for t in range(T):
        stored.append(tuple(x.astype(store) for x in h))
        obs_t = jax.tree.map(lambda x: x[:, t], obs_norm)  # noqa: B023 (applied eagerly, this iteration)
        mu, _, h_new = step(obs_t, h)
        mus.append(mu)
        h = _reset_hidden(h_new, discount[:, t] == 0)
    hidden = tuple(
        jnp.stack([s[i] for s in stored], axis=1) for i in range(len(h_init))
    )
    return hidden, jnp.stack(mus, axis=1)


def _make_batch(
    nets,
    state,
    cfg,
    seed=1,
    phase0=(0, 0, 0, 0),
    slow0=0.5,
    done_at=(),
    priv=False,
    with_mu=False,
):
    """[B, T] batch whose stored hidden is a hand rollout of the CURRENT
    policy entering the window mid-episode: per-env clock phase ``phase0``
    and carried slow state ``slow0 * N(0, 1)`` (0.0 = fresh episode).
    ``with_mu`` also returns the rollout's per-step policy mean."""
    T = cfg.sequence_length
    ks = jax.random.split(jax.random.PRNGKey(seed), 7)
    obs = {
        "vision": jax.random.uniform(ks[0], (B, T, *VISION_SHAPE)),
        "imitation_target": jax.random.normal(ks[1], (B, T, TASK_OBS)),
        "proprioception": jax.random.normal(ks[2], (B, T, PROPRIO)),
    }
    discount = jnp.ones((B, T))
    for b, t in done_at:
        discount = discount.at[b, t].set(0.0)
    batch = {
        "observation": obs,
        "action": jax.random.uniform(ks[3], (B, T, ACTION), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(ks[4], (B, T)),
        "discount": discount,
        "truncation": jnp.zeros((B, T)),
    }
    if priv:
        # Nonzero everywhere (a zero input row has a zero kernel gradient).
        batch[KEY] = jax.random.normal(ks[5], (B, T, PRIV)) + 2.0
    # Width of the packed slow_h leaf: H for a GRU, 2H ([c, h]) for an LSTM.
    slow_w = nets.recurrent_meta.init_hidden()[0].shape[-1]
    h_init = (
        slow0 * jax.random.normal(ks[6], (B, slow_w)),
        jnp.asarray(phase0, jnp.float32)[:, None],
    )
    batch["policy_hidden"], mu = _hand_rollout(
        nets,
        state.policy_params,
        state.normalizer_params,
        obs,
        discount,
        h_init,
    )
    return (batch, mu) if with_mu else batch


def _tick_window(state, batch, cfg):
    """The tick path's window prep, mirrored from ``_sgd_step_rnn`` for
    helper-level tests (as test_learner_rnn mirrors the per-step path)."""
    L, k = cfg.rnn_bptt_length, cfg.rnn_tick_stride
    W = L * k
    h_all = jax.tree.map(lambda h: h.astype(jnp.float32), batch["policy_hidden"])
    idx, fill = tick_indices(h_all[PHASE][:, :W, 0], L)
    obs_L = _normalize_obs(
        jax.tree.map(lambda x: _take_steps(x, idx), batch["observation"]),
        state.normalizer_params,
    )
    h0 = jax.tree.map(lambda h: _take_steps(h, idx[:, :1])[:, 0], h_all)
    h_stored_tm = jax.tree.map(lambda h: jnp.swapaxes(_take_steps(h, idx), 0, 1), h_all)
    obs_tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), obs_L)
    done_tm = jnp.swapaxes(_done_between_ticks(batch["discount"], idx, W), 0, 1)
    nb = batch["reward"].shape[0]
    obs_flat = jax.tree.map(lambda x: x.reshape((nb * L, *x.shape[2:])), obs_L)
    return dict(
        idx=idx,
        fill=fill,
        obs_tm=obs_tm,
        done_tm=done_tm,
        h0=h0,
        h_stored_tm=h_stored_tm,
        obs_flat=obs_flat,
    )


# ---------------------------------------------------------------------------
# 1. Tick indices.
# ---------------------------------------------------------------------------


def test_tick_indices_with_resets():
    """k=3, L=4, W=12. A done at step j zeroes the hidden entering j+1, so the
    phase restarts at 0 there and j+1 is a tick. Rows are the stored
    PRE-step phases; expected indices are the first L positions with
    phase 0, hand-computed."""
    L = 4
    phases = np.array(
        [
            # steady clock from phase 0: ticks every 3 steps.
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            # window entered mid-cycle at phase 2.
            [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
            # done at step 3 (phase 1 -> reset): step 4 restarts at 0.
            [1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1],
            # two resets (dones at steps 1 and 2): 5 ticks in W, first 4 kept.
            [0, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        ],
        np.float32,
    )
    expected = np.array(
        [[0, 3, 6, 9], [1, 4, 7, 10], [2, 4, 7, 10], [0, 2, 3, 6]], np.int32
    )
    for dtype in (jnp.float32, jnp.float16):  # f16 = the replay store dtype
        idx, fill = jax.jit(tick_indices, static_argnums=1)(
            jnp.asarray(phases, dtype), L
        )
        assert idx.shape == (4, L)
        np.testing.assert_array_equal(np.asarray(idx), expected)
        assert float(fill) == 0.0

    # Brute-force cross-check on a simulated clock with random resets.
    rng = np.random.default_rng(0)
    k, Lr = 5, 6
    W = Lr * k
    rows, refs = [], []
    for _ in range(16):
        p = int(rng.integers(k))
        row = []
        for _t in range(W):
            row.append(p)
            p = 0 if rng.random() < 0.05 else (p + 1) % k
        rows.append(row)
        refs.append([t for t, ph in enumerate(row) if ph == 0][:Lr])
    idx, fill = tick_indices(jnp.asarray(rows, jnp.float16), Lr)
    np.testing.assert_array_equal(np.asarray(idx), np.asarray(refs))
    assert float(fill) == 0.0

    # The fill metric can fire: a corrupted row with one tick in W pads with
    # W-1 (3 of 4 slots filled) -> fill_frac = 3 / (rows * L).
    bad = phases.copy()
    bad[1] = 1.0
    bad[1, 5] = 0.0
    idx, fill = tick_indices(jnp.asarray(bad), L)
    np.testing.assert_array_equal(np.asarray(idx[1]), [5, 11, 11, 11])
    np.testing.assert_allclose(float(fill), 3.0 / (4 * L))


# ---------------------------------------------------------------------------
# 2. Stride 1: the tick path reproduces the legacy path exactly.
# ---------------------------------------------------------------------------


class _StubPhasePolicy(_StubRecurrentPolicy):
    """test_learner_rnn's stub GRU plus a k=1 phase leaf: hidden (h, phase),
    phase always 0 (ticks every step). Same param tree as the stub."""

    def raw(self, obs, hidden):
        h, phase = hidden
        mu, scale, (new_h,) = super().raw(obs, (h,))
        return mu, scale, (new_h, jnp.mod(phase + 1.0, 1.0))


@pytest.mark.parametrize("aligned", [False, True], ids=["legacy", "aligned"])
def test_stride1_bit_identical_to_old_path(aligned):
    """Same batch, same params: the legacy net (no phase leaf -> the old
    slicing path) vs the same cell with a k=1 phase leaf (-> the tick path:
    tick_indices, gathers, forced phase, cumsum done_between, W-length returns
    gathered at idx). Every loss/metric and every updated param must be
    bit-identical; dones sit at a loss point, the last loss point, and in the
    n-step tail.

    With the acme-aligned defaults the two paths compute the same values
    through differently fused weighted means, so they agree to float32
    rounding instead of bit for bit; two of the dones are time-outs there."""
    s = _setup_legacy() if aligned else _setup_legacy(**LEGACY_ALIGNMENT)
    cfg, state = s["cfg"], s["state"]
    L, n = cfg.rnn_bptt_length, cfg.n_step
    assert cfg.rnn_tick_stride == 1
    batch = _make_legacy_batch(
        jax.random.PRNGKey(9),
        state,
        s["nets"],
        T=L + n,
        done_at=((1, 1), (2, 0), (3, L - 1), (0, L + 1)),
    )
    if aligned:
        batch["truncation"] = (
            batch["truncation"].at[1, 1].set(1.0).at[0, L + 1].set(1.0)
        )
    else:
        del batch["truncation"]
    nets_phase = DMPONetworks(
        policy=_StubPhasePolicy(action_size=s["nets"].policy.action_size),
        critic=s["nets"].critic,
        recurrent_meta=SlowCoreMeta(
            cell_type="gru",
            hidden_sizes=(_HIDDEN,),
            store_dtype=jnp.float32,
            tick_stride=1,
        ),
    )
    (h,) = batch["policy_hidden"]
    batch_phase = dict(batch, policy_hidden=(h, jnp.zeros((*h.shape[:2], 1), h.dtype)))

    def run(nets, b):
        return jax.jit(lambda st, bb: sgd_step(st, bb, nets, s["optimizers"], cfg))(
            state, b
        )

    old_state, old_m = run(s["nets"], batch)
    new_state, new_m = run(nets_phase, batch_phase)

    assert set(new_m) == set(old_m) | {"rnn/tick_fill_frac"}
    assert float(new_m["rnn/tick_fill_frac"]) == 0.0
    tol = {"rtol": 1e-5, "atol": 1e-6} if aligned else {"rtol": 0, "atol": 0}
    for key, v in old_m.items():
        np.testing.assert_allclose(
            np.asarray(new_m[key]), np.asarray(v), err_msg=key, **tol
        )
    for a, b in zip(jax.tree.leaves(old_state), jax.tree.leaves(new_state)):
        np.testing.assert_allclose(_leaf_bits(a), _leaf_bits(b), **tol)


def _leaf_bits(x):
    """A state leaf as a numpy array; a typed PRNG key (the state's rng) by
    its key data, which numpy cannot read directly."""
    if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        return np.asarray(jax.random.key_data(x))
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 3. Forced-phase unroll.
# ---------------------------------------------------------------------------


def test_forced_phase_unroll_updates_every_tick():
    """k=10: unrolled one gathered tick per scan step, the stored clock would
    hold slow_h after the first step (phase 1, 2, 3 < k). Forcing the phase
    leaf to 0 makes EVERY scan step a GRU tick, matching a hand chain of
    phase-0 applies."""
    k, Lt = 10, 4
    cfg = _cfg(L=Lt, k=k)
    nets = _slow_nets(cfg, k)
    state = _state(nets, cfg)
    params = state.policy_params
    ks = jax.random.split(jax.random.PRNGKey(3), 4)
    obs_tm = {
        "vision": jax.random.uniform(ks[0], (Lt, B, *VISION_SHAPE)),
        "imitation_target": jax.random.normal(ks[1], (Lt, B, TASK_OBS)),
        "proprioception": jax.random.normal(ks[2], (Lt, B, PROPRIO)),
    }
    done_tm = jnp.zeros((Lt, B), bool)
    h0 = (jax.random.normal(ks[3], (B, SLOW_H)), jnp.zeros((B, 1)))

    mu, _, h_pre = _unroll_policy_raw(
        params, nets, obs_tm, done_tm, h0, force_phase_leaf=PHASE
    )
    slow = np.asarray(h_pre[0])  # [L, B, H] entering t
    for t in range(1, Lt):
        assert np.all(np.any(slow[t] != slow[t - 1], axis=-1)), t
    np.testing.assert_array_equal(np.asarray(h_pre[PHASE]), 0.0)

    # Hand chain: one phase-0 raw apply per tick.
    apply = jax.vmap(lambda o, h: nets.policy.apply(params, o, h, method="raw"))
    h = h0
    for t in range(Lt):
        np.testing.assert_allclose(slow[t], np.asarray(h[0]), atol=1e-6)
        obs_t = jax.tree.map(lambda x: x[t], obs_tm)  # noqa: B023 (applied eagerly, this iteration)
        mu_t, _, (s_new, _) = apply(obs_t, (h[0], jnp.zeros((B, 1))))
        np.testing.assert_allclose(np.asarray(mu[t]), np.asarray(mu_t), atol=1e-6)
        h = (s_new, jnp.zeros((B, 1)))

    # Control: without forcing, the clock advances 0 -> 1 -> 2 -> 3 and the
    # core holds after its first tick.
    _, _, h_free = _unroll_policy_raw(params, nets, obs_tm, done_tm, h0)
    free = np.asarray(h_free[0])
    assert np.any(free[1] != free[0])
    for t in range(2, Lt):
        np.testing.assert_array_equal(free[t], free[1])
    np.testing.assert_array_equal(
        np.asarray(h_free[PHASE])[:, :, 0], np.arange(Lt)[:, None] * np.ones(B)
    )


# ---------------------------------------------------------------------------
# 4. Dones between ticks.
# ---------------------------------------------------------------------------


def test_done_between_ticks_resets():
    # (a) The cumsum helper vs a brute-force any() over [idx_i, idx_{i+1}),
    # the last interval running to the window end W.
    rng = np.random.default_rng(1)
    Bq, T, W, Lq = 5, 30, 24, 4
    disc = (rng.random((Bq, T)) > 0.15).astype(np.float32)
    idx = np.sort(
        np.stack([rng.choice(W, Lq, replace=False) for _ in range(Bq)]), axis=1
    ).astype(np.int32)
    got = np.asarray(_done_between_ticks(jnp.asarray(disc), jnp.asarray(idx), W))
    for b in range(Bq):
        for i in range(Lq):
            hi = idx[b, i + 1] if i + 1 < Lq else W
            assert got[b, i] == bool(np.any(disc[b, idx[b, i] : hi] == 0)), (b, i)

    # (b) Integration on a real k=3 slow core: a done between ticks i and i+1
    # zeroes the hidden entering tick i+1 (= the step after the done), and
    # the forced-phase unroll with done_between reproduces the stored hidden
    # at every tick.
    k, L, n = 3, 4, 2
    cfg = _cfg(L=L, n=n, k=k)
    nets = _slow_nets(cfg, k)
    state = _state(nets, cfg)
    # env 0: ticks 0,3,... ; done at 4 (between ticks 3 and 6) -> tick at 5.
    # env 1: enters at phase 2 (ticks 1,4,...); done at 1 (a tick itself).
    # env 2: enters at phase 1 carrying memory; done at 0, BEFORE its first
    #   tick (1): h0 must be the reset zeros stored there, not the carried
    #   state at column 0.  env 3: done at 7 (phase 1 there).
    batch = _make_batch(
        nets,
        state,
        cfg,
        phase0=(0, 2, 1, 0),
        done_at=((0, 4), (1, 1), (2, 0), (3, 7)),
    )
    w = _tick_window(state, batch, cfg)
    np.testing.assert_array_equal(
        np.asarray(w["idx"]),
        [[0, 3, 5, 8], [1, 2, 5, 8], [1, 4, 7, 10], [0, 3, 6, 8]],
    )
    np.testing.assert_array_equal(
        np.asarray(w["done_tm"]).T,
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
    )
    assert np.max(np.abs(np.asarray(batch["policy_hidden"][0][2, 0]))) > 0.1
    stored_slow = np.asarray(w["h_stored_tm"][0])
    zero_at = ((2, 0), (1, 1), (0, 2), (3, 3))
    for t, b in zero_at:
        assert np.all(stored_slow[t, b] == 0.0), (t, b)

    _, _, h_pre = _unroll_policy_raw(
        state.policy_params,
        nets,
        w["obs_tm"],
        w["done_tm"],
        w["h0"],
        force_phase_leaf=PHASE,
    )
    pre_slow = np.asarray(h_pre[0])
    for t, b in zero_at:
        assert np.all(pre_slow[t, b] == 0.0), (t, b)
    # Stored hiddens are f16: recomputed == stored up to the storage rounding.
    np.testing.assert_allclose(pre_slow, stored_slow, rtol=2e-3, atol=2e-3)

    # Control: dropping the reset leaves memory flowing across the done.
    _, _, h_nores = _unroll_policy_raw(
        state.policy_params,
        nets,
        w["obs_tm"],
        jnp.zeros_like(w["done_tm"]),
        w["h0"],
        force_phase_leaf=PHASE,
    )
    assert np.max(np.abs(np.asarray(h_nores[0])[2, 0])) > 1e-2

    # Full step: fresh stored hidden -> ~zero staleness through the reset.
    _, m = _sgd_step_rnn(state, batch, nets, make_optimizers(cfg), cfg)
    assert float(m["rnn/hidden_staleness"]) < 1e-4
    assert float(m["rnn/tick_fill_frac"]) == 0.0


# ---------------------------------------------------------------------------
# 5. BPTT across ticks.
# ---------------------------------------------------------------------------


def _slow_core_grads(L, k=10, n=3, cell="gru"):
    cfg = _cfg(L=L, n=n, k=k)
    nets = _slow_nets(cfg, k, cell=cell)
    state = _state(nets, cfg)
    # Fresh episode at phase 0 everywhere: tick 0 consumes slow_h == 0.
    batch = _make_batch(nets, state, cfg, phase0=(0, 0, 0, 0), slow0=0.0)
    w = _tick_window(state, batch, cfg)
    assert np.all(np.asarray(w["h0"][0]) == 0.0)
    tgt_mu, tgt_scale, _ = _unroll_policy_raw(
        state.target_policy_params,
        nets,
        w["obs_tm"],
        w["done_tm"],
        w["h0"],
        force_phase_leaf=PHASE,
    )
    A = tgt_mu.shape[-1]
    tgt_mu = jnp.swapaxes(tgt_mu, 0, 1).reshape(B * L, A)
    tgt_scale = jnp.swapaxes(tgt_scale, 0, 1).reshape(B * L, A)

    def loss_of(p):
        return _policy_loss_fn_rnn(
            p,
            state.dual_params,
            nets,
            w["obs_tm"],
            w["done_tm"],
            w["h0"],
            w["h_stored_tm"],
            w["obs_flat"],
            tgt_mu,
            tgt_scale,
            state.target_critic_params,
            cfg,
            jax.random.PRNGKey(7),
            phase_leaf=PHASE,
        )[0]

    g = jax.grad(loss_of)(state.policy_params)
    core = g["params"]["policy_head"]["slow_core"]
    # Recurrent (h->h) and input (x->gates) kernel names per flax cell.
    rec_names, inp_names = {
        "gru": (("hr", "hz", "hn"), ("ir", "iz", "in")),
        "lstm": (("hi", "hf", "hg", "ho"), ("ii", "if", "ig", "io")),
    }[cell]
    rec = jnp.concatenate([core[c]["kernel"].ravel() for c in rec_names])
    inp = jnp.concatenate([core[c]["kernel"].ravel() for c in inp_names])
    return np.asarray(rec), np.asarray(inp)


def test_bptt_grad_reaches_slow_core_kernel_across_ticks():
    """The recurrent (h->h) GRU kernels get gradient only through TIME: at
    L=1 tick from a zero slow state the single application consumes h=0 and
    their gradient is exactly zero; at L=4 ticks (40 env steps apart from
    the first) BPTT carries it back across the held gaps."""
    rec4, inp4 = _slow_core_grads(L=4)
    assert np.max(np.abs(rec4)) > 0.0
    assert np.max(np.abs(inp4)) > 0.0

    rec1, inp1 = _slow_core_grads(L=1)
    np.testing.assert_array_equal(rec1, 0.0)
    assert np.max(np.abs(inp1)) > 0.0  # the core IS trained at L=1


# ---------------------------------------------------------------------------
# 6. Full jitted step (k=10, L=4, n=3, privileged critic).
# ---------------------------------------------------------------------------


def _priv_rows_delta(new_state, state):
    k0 = state.critic_params["params"]["Dense_0"]["kernel"]
    k1 = new_state.critic_params["params"]["Dense_0"]["kernel"]
    return np.abs(np.asarray(k1[-PRIV:] - k0[-PRIV:]))


def _smoke_setup():
    k, L, n = 10, 4, 3
    cfg = _cfg(L=L, n=n, k=k, priv=True)
    nets = _slow_nets(cfg, k, priv_dim=PRIV)
    state = _state(nets, cfg)
    batch = _make_batch(
        nets,
        state,
        cfg,
        phase0=(0, 3, 7, 9),
        done_at=((1, 12), (2, 30)),
        priv=True,
    )
    step = jax.jit(lambda st, b: sgd_step(st, b, nets, make_optimizers(cfg), cfg))
    return cfg, nets, state, batch, step


def test_sgd_step_ticks_smoke_jit():
    cfg, nets, state, batch, step = _smoke_setup()
    assert cfg.rnn_bptt_length == 4 == B  # B == L: per-row gather must hold
    new_state, metrics = step(state, batch)

    for key, v in metrics.items():
        assert np.isfinite(np.asarray(v)).all(), f"non-finite metric {key}"
    assert float(metrics["rnn/tick_fill_frac"]) == 0.0
    assert float(metrics["rnn/hidden_staleness"]) < 1e-4
    assert int(new_state.steps) == 1
    # Metric keys: the per-step recurrent path's keys + the fill fraction.
    s = _setup_legacy()
    _, legacy_m = _sgd_step_rnn(
        s["state"], s["batch"], s["nets"], s["optimizers"], s["cfg"]
    )
    assert set(metrics) == set(legacy_m) | {"rnn/tick_fill_frac"}

    # The slow core is trained, and the privileged critic rows moved.
    core_delta = jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))),
        new_state.policy_params["params"]["policy_head"]["slow_core"],
        state.policy_params["params"]["policy_head"]["slow_core"],
    )
    assert max(jax.tree.leaves(core_delta)) > 0.0
    assert (_priv_rows_delta(new_state, state).max(axis=1) > 0).all()


def test_tick_privileged_alignment():
    """The critic's privileged input is gathered per row at the ticks (loss
    points) and at idx + n (bootstrap); a step that is neither is never read.
    Rows enter at different phases, so each row has its own indices."""
    cfg, nets, state, batch, step = _smoke_setup()
    n, k, L = cfg.n_step, cfg.rnn_tick_stride, cfg.rnn_bptt_length
    h_phase = batch["policy_hidden"][PHASE].astype(jnp.float32)
    idx, _ = tick_indices(h_phase[:, : L * k, 0], L)
    idx = np.asarray(idx)
    _, base = step(state, batch)

    def run(cols):
        pert = batch[KEY].at[np.arange(B), cols].add(3.0)
        return step(state, dict(batch, **{KEY: pert}))[1]

    m = run(idx[:, 2])  # a loss point
    assert float(m["critic_loss"]) != float(base["critic_loss"])
    assert (float(m["q_min"]), float(m["q_max"])) != (
        float(base["q_min"]),
        float(base["q_max"]),
    )
    m = run(idx[:, 2] + n)  # its bootstrap state: target only
    assert float(m["critic_loss"]) != float(base["critic_loss"])
    for key in ("q_min", "q_max", "policy_loss"):
        assert float(m[key]) == float(base[key]), key
    m = run(idx[:, 2] + 1)  # neither a tick (k=10) nor a bootstrap (n=3)
    for key, v in base.items():
        np.testing.assert_array_equal(np.asarray(m[key]), np.asarray(v), key)


def test_tick_critic_target_matches_reference():
    """Critic side of a tick step vs an independent reference: python-loop
    tick positions from the stored phase, brute-force n-step returns over
    EVERY step after each tick (one window crosses a done), actions and
    privileged inputs at the ticks, and the bootstrap action = the target
    policy at idx + n with the STORED hidden there, phase NOT forced -- the
    hidden the rollout actually held at idx + n, which reproduces the
    rollout's own action to f16 storage precision (checked), while a forced
    phase would tick the core at a hold step (checked to differ).

    The target critic is sharpened (params x5) so the bootstrap action is
    visible in the loss: a bootstrap hidden read at idx (or forced) moves
    the loss by ~6e-5 / 8e-5 relative, against an exact match here.

    The reference is the single mean-action target on the stored raw action,
    so the step runs with those two switches (the sampled mixture target and
    the executed-action input have their own tests)."""
    k, L, n = 10, 4, 3
    cfg = _cfg(
        L=L,
        n=n,
        k=k,
        priv=True,
        critic_target_mode="mean_action",
        critic_action_input="raw",
    )
    nets = _slow_nets(cfg, k, priv_dim=PRIV)
    state = _state(nets, cfg)
    state = state._replace(
        target_critic_params=jax.tree.map(lambda x: 5.0 * x, state.target_critic_params)
    )
    # env 0: ticks 0, 10, 20, then a done at 21 -> tick 22; tick 20's
    # n-step window [20, 23) crosses that done.
    batch, mu_live = _make_batch(
        nets,
        state,
        cfg,
        phase0=(0, 3, 7, 9),
        done_at=((0, 21), (1, 12), (2, 30)),
        priv=True,
        with_mu=True,
    )
    phase = np.asarray(batch["policy_hidden"][PHASE][:, :, 0], np.float32)
    ticks = np.array(
        [[t for t in range(L * k) if phase[b, t] == 0.0][:L] for b in range(B)]
    )
    np.testing.assert_array_equal(ticks[0], [0, 10, 20, 22])
    r, d = np.asarray(batch["reward"]), np.asarray(batch["discount"])
    R, D = np.zeros((B, L)), np.zeros((B, L))
    for b in range(B):
        for i in range(L):
            m = 1.0
            for j in range(n):
                R[b, i] += cfg.discount**j * m * r[b, ticks[b, i] + j]
                m *= d[b, ticks[b, i] + j]
            D[b, i] = cfg.discount ** (n - 1) * m
    assert D[0, 2] == 0.0

    rows = np.arange(B)[:, None]
    obs = _normalize_obs(batch["observation"], state.normalizer_params)

    def at(cols, priv=True):
        o = jax.tree.map(lambda x: x[rows, cols], obs)
        return {**o, "privileged": batch[KEY][rows, cols]} if priv else o

    # Bootstrap action from the stored hidden at idx + n.
    h_boot = tuple(
        x.astype(jnp.float32)[rows, ticks + n] for x in batch["policy_hidden"]
    )
    target_mu = jax.vmap(
        jax.vmap(
            lambda o, h: nets.policy.apply(
                state.target_policy_params, o, h, method="raw"
            )[0]
        )
    )
    next_act = target_mu(at(ticks + n, priv=False), h_boot)
    live = np.asarray(mu_live[rows, ticks + n])
    np.testing.assert_allclose(np.asarray(next_act), live, atol=1e-3)
    forced = target_mu(at(ticks + n, priv=False), (h_boot[0], 0 * h_boot[1]))
    assert np.max(np.abs(np.asarray(forced) - live)) > 1e-2

    target = compute_categorical_target(
        nets,
        state.target_critic_params,
        at(ticks + n),
        next_act,
        jnp.asarray(R, jnp.float32),
        jnp.asarray(D, jnp.float32),
        cfg,
    ).reshape(B * L, cfg.num_atoms)
    obs_q = jax.tree.map(lambda x: x.reshape((B * L, *x.shape[2:])), at(ticks))
    act_q = batch["action"][rows, ticks].reshape(B * L, ACTION)
    ref = _critic_loss_fn(state.critic_params, nets, obs_q, act_q, target)

    _, m = _sgd_step_rnn(state, batch, nets, make_optimizers(cfg), cfg)
    np.testing.assert_allclose(float(m["critic_loss"]), float(ref), rtol=1e-6)


def test_tick_fill_frac_reports_missing_ticks():
    """Ticks are searched in the window [0, L*k) only. A row whose stored
    phase (corrupted) holds 1 tick there -- and another in the n-step tail
    -- pads 3 of its L slots: fill_frac = 3 / (B*L), never hidden by a tick
    found past the window."""
    k, L, n = 10, 4, 3
    cfg = _cfg(L=L, n=n, k=k)
    nets = _slow_nets(cfg, k)
    state = _state(nets, cfg, perturb=False)
    batch = _make_batch(nets, state, cfg)
    slow, phase = batch["policy_hidden"]
    bad = jnp.ones_like(phase).at[0, 5].set(0.0).at[0, L * k + 1].set(0.0)
    phase = phase.at[0].set(bad[0])
    _, m = _sgd_step_rnn(
        state,
        dict(batch, policy_hidden=(slow, phase)),
        nets,
        make_optimizers(cfg),
        cfg,
    )
    assert float(m["rnn/tick_fill_frac"]) == 3.0 / (B * L)
    for key, v in m.items():
        assert np.isfinite(np.asarray(v)).all(), key


# ---------------------------------------------------------------------------
# 7. Diagnostics exclude the phase leaf.
# ---------------------------------------------------------------------------


def test_rnn_metrics_exclude_phase_leaf():
    """hidden_abs_mean / hidden_staleness are over the slow state only: the
    forced phase leaf (all 0) would halve abs_mean and dilute staleness."""
    k = 10
    cfg = _cfg(L=4, n=3, k=k)
    nets = _slow_nets(cfg, k)
    state = _state(nets, cfg)
    batch = _make_batch(nets, state, cfg, phase0=(0, 3, 7, 9))
    opt = make_optimizers(cfg)
    _, m = _sgd_step_rnn(state, batch, nets, opt, cfg)
    w = _tick_window(state, batch, cfg)
    slow_abs = float(jnp.mean(jnp.abs(w["h_stored_tm"][0])))
    np.testing.assert_allclose(float(m["rnn/hidden_abs_mean"]), slow_abs, rtol=1e-2)
    assert float(m["rnn/hidden_staleness"]) < 1e-4

    # Corrupt the stored slow state at ticks 1..L-1 only (tick 0 seeds the
    # unroll): staleness fires at the full per-leaf scale, undiluted.
    slow, phase = batch["policy_hidden"]
    rows = np.arange(B)[:, None]
    cols = np.asarray(w["idx"])[:, 1:]
    bad = slow.astype(jnp.float32).at[rows, cols].add(1.0).astype(slow.dtype)
    _, m2 = _sgd_step_rnn(
        state, dict(batch, policy_hidden=(bad, phase)), nets, opt, cfg
    )
    st = jnp.swapaxes(_take_steps(bad.astype(jnp.float32), w["idx"]), 0, 1)
    re = w["h_stored_tm"][0]
    ref = jnp.mean(jnp.sum((re - st) ** 2, -1) / (jnp.sum(st**2, -1) + 1e-6))
    np.testing.assert_allclose(float(m2["rnn/hidden_staleness"]), float(ref), rtol=5e-2)


# ---------------------------------------------------------------------------
# 8. Guards.
# ---------------------------------------------------------------------------


def test_guard_sequence_length():
    k, L, n = 10, 4, 3
    cfg = _cfg(L=L, n=n, k=k)
    nets = _slow_nets(cfg, k)
    state = _state(nets, cfg, perturb=False)
    batch = _make_batch(nets, state, cfg)
    opt = make_optimizers(cfg)

    def run(c, b=batch):
        return sgd_step(state, b, nets, opt, c)

    # T must be L*k + n (here 43); the per-step length L + n is rejected.
    short = jax.tree.map(lambda x: x[:, : L + n], batch)
    with pytest.raises(ValueError, match=r"rnn_bptt_length \* rnn_tick_stride"):
        run(cfg, b=short)
    with pytest.raises(ValueError, match="rnn_tick_stride"):
        run(_cfg(L=L, n=n + 1, k=k, sequence_length=L * k + n))
    run(cfg)  # the exact length passes

    # Stride 1 keeps the historical message (test_learner_rnn matches it).
    s = _setup_legacy()

    def run_legacy(**cfg_overrides):
        return sgd_step(
            s["state"],
            s["batch"],
            s["nets"],
            s["optimizers"],
            dataclasses.replace(s["cfg"], **cfg_overrides),
        )

    with pytest.raises(ValueError, match=r"!= rnn_bptt_length \+ n_step = 3 \+ 5 = 8"):
        run_legacy(n_step=5)

    # The net's clock must match the learner's stride (net k=10, cfg k=1).
    with pytest.raises(ValueError, match="tick_stride=10"):
        run(_cfg(L=L, n=n, k=1, sequence_length=L * k + n))
    # A stride > 1 needs a phase leaf to locate the ticks.
    with pytest.raises(ValueError, match="phase leaf"):
        run_legacy(rnn_tick_stride=2)
    with pytest.raises(ValueError, match="rnn_tick_stride must be >= 1"):
        run(_cfg(L=L, n=n, k=0, sequence_length=L * k + n))


# ---------------------------------------------------------------------------
# 9. LSTM slow core (rnn_cell="lstm"): the hidden is still (slow_h, phase) with
# phase_leaf == 1; slow_h is [.., 2H] = concat([c, h]). The tick path is
# leaf-shape-agnostic, so the same learner code must run unchanged. The cell
# type is a network choice, independent of the acme-alignment fields: the
# smoke step runs under both the aligned defaults and the legacy preset.
# ---------------------------------------------------------------------------

LSTM_W = 2 * SLOW_H


def test_lstm_bptt_grad_reaches_slow_core_kernel_across_ticks():
    """As the GRU test: the LSTM's recurrent (h->h) kernels get gradient only
    through TIME -- exactly zero at L=1 from a zero state, nonzero at L=4."""
    rec4, inp4 = _slow_core_grads(L=4, cell="lstm")
    assert np.max(np.abs(rec4)) > 0.0
    assert np.max(np.abs(inp4)) > 0.0

    rec1, inp1 = _slow_core_grads(L=1, cell="lstm")
    np.testing.assert_array_equal(rec1, 0.0)
    assert np.max(np.abs(inp1)) > 0.0


@pytest.mark.parametrize("aligned", [True, False], ids=["aligned", "legacy"])
def test_lstm_sgd_step_ticks_smoke_jit(aligned):
    """One strided learner step (k=10, L=4, n=3, privileged critic, dones
    mid-window) with an LSTM core on tiny CPU nets: runs, finite losses and
    metrics, rnn/tick_fill_frac == 0, the LSTM core is trained."""
    k, L, n = 10, 4, 3
    cfg = _cfg(L=L, n=n, k=k, priv=True, **({} if aligned else LEGACY_ALIGNMENT))
    nets = _slow_nets(cfg, k, priv_dim=PRIV, cell="lstm")
    assert nets.recurrent_meta.cell_type == "lstm"
    assert nets.recurrent_meta.phase_leaf == PHASE
    state = _state(nets, cfg)
    batch = _make_batch(
        nets,
        state,
        cfg,
        phase0=(0, 3, 7, 9),
        done_at=((1, 12), (2, 30)),
        priv=True,
    )

    # Stored hidden: packed 2H slow leaf + 1-wide phase leaf, f16 in replay.
    slow, phase = batch["policy_hidden"]
    T = cfg.sequence_length
    assert slow.shape == (B, T, LSTM_W) and phase.shape == (B, T, 1)
    assert slow.dtype == jnp.float16 and phase.dtype == jnp.float16
    # The hand rollout resets with the learner's `_reset_hidden` (the
    # rollout's own `_reset_hidden_where_done` is covered in
    # test_networks_kl_anchor_slow::test_lstm_reset_zeroes_whole_packed_leaf).
    # It zeroes the WHOLE 2H leaf (c and h) after a done: the step after
    # done_at (1, 12) / (2, 30) enters with an all-zero state and phase == 0.
    for b, t in ((1, 12), (2, 30)):
        np.testing.assert_array_equal(np.asarray(slow[b, t + 1]), 0.0)
        assert float(phase[b, t + 1, 0]) == 0.0
        assert np.any(np.asarray(slow[b, t]) != 0.0)  # the PRE-done step is live

    step = jax.jit(lambda st, b: sgd_step(st, b, nets, make_optimizers(cfg), cfg))
    new_state, metrics = step(state, batch)
    for key, v in metrics.items():
        assert np.isfinite(np.asarray(v)).all(), f"non-finite metric {key}"
    assert float(metrics["rnn/tick_fill_frac"]) == 0.0
    assert float(metrics["rnn/hidden_staleness"]) < 1e-4
    assert float(metrics["rnn/hidden_abs_mean"]) > 0.0
    assert int(new_state.steps) == 1
    assert np.isfinite(float(metrics["policy_loss"]))
    assert np.isfinite(float(metrics["critic_loss"]))

    # Same metric keys as the GRU tick path.
    s = _setup_legacy()
    _, legacy_m = _sgd_step_rnn(
        s["state"], s["batch"], s["nets"], s["optimizers"], s["cfg"]
    )
    assert set(metrics) == set(legacy_m) | {"rnn/tick_fill_frac"}

    core_delta = jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))),
        new_state.policy_params["params"]["policy_head"]["slow_core"],
        state.policy_params["params"]["policy_head"]["slow_core"],
    )
    assert set(core_delta) == {"ii", "if", "ig", "io", "hi", "hf", "hg", "ho"}
    assert max(jax.tree.leaves(core_delta)) > 0.0
    assert (_priv_rows_delta(new_state, state).max(axis=1) > 0).all()


def test_lstm_rnn_metrics_cover_the_whole_packed_leaf():
    """The rnn/ diagnostics skip only the phase leaf; for an LSTM they cover
    the whole 2H [c, h] leaf (a mean over c AND h)."""
    k = 10
    cfg = _cfg(L=4, n=3, k=k)
    nets = _slow_nets(cfg, k, cell="lstm")
    state = _state(nets, cfg)
    batch = _make_batch(nets, state, cfg, phase0=(0, 3, 7, 9))
    _, m = _sgd_step_rnn(state, batch, nets, make_optimizers(cfg), cfg)
    w = _tick_window(state, batch, cfg)
    assert w["h_stored_tm"][0].shape[-1] == LSTM_W
    slow_abs = float(jnp.mean(jnp.abs(w["h_stored_tm"][0])))
    np.testing.assert_allclose(float(m["rnn/hidden_abs_mean"]), slow_abs, rtol=1e-2)
