"""Bootstrapping through time limits (``bootstrap_on_truncation``).

A time-out at step k ends the episode without making s_k terminal. The true
final state s_T is swapped out by the auto-reset, so a window reaching k ends
one step early: rewards before k, bootstrap from s_k with gamma^k * alive, and
the window starting at k itself is excluded (weight 0). Real terminations,
including a fall on the final step, behave as before.

Covered on hand-built arrays (gamma 0.9, n 4), on the feed-forward and both
recurrent learner paths (the critic loss against an independent reference),
and at the rollout, which must refuse an env with no readable time-out flag.
Every learner-level case runs with only this switch on, so the reference is
the mean-action target on the raw action evaluated at the current state.
"""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.agent.dmpo import test_learner_rnn as rnn_t
from tests.agent.dmpo import test_learner_ticks as tick_t
from track_mjx.agent.dmpo.config import LEGACY_ALIGNMENT, DMPOConfig
from track_mjx.agent.dmpo.learner import (
    _critic_loss_fn,
    _normalize_obs,
    _per_point_nstep_returns,
    _sgd_step_rnn,
    compute_categorical_target,
    init_training_state,
    make_optimizers,
    sgd_step,
    window_returns,
)
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.rollout import collect_rollout

GAMMA, N = 0.9, 4

# Only the time-limit switch on: the other alignment switches stay at their
# historical values, so the reference below needs nothing else.
TRUNC_ONLY = {**LEGACY_ALIGNMENT, "bootstrap_on_truncation": True}


def _arrays(T=6, trunc_at=None, done_at=None, rewards=None):
    r = np.arange(1.0, T + 1.0) if rewards is None else np.asarray(rewards, float)
    d = np.ones(T)
    tr = np.zeros(T)
    if trunc_at is not None:
        d[trunc_at] = 0.0
        tr[trunc_at] = 1.0
    if done_at is not None:
        d[done_at] = 0.0
    as_b = lambda x: jnp.asarray(x, jnp.float32)[None]  # noqa: E731
    return as_b(r), as_b(d), as_b(tr)


def _ref_window(r, d, tr, gamma, n):
    """Python loop over one row: (R, disc, boot, weight)."""
    k = next((j for j in range(n) if tr[j] > 0.5), n)
    R, m = 0.0, 1.0
    for j in range(k):
        R += gamma**j * m * r[j]
        m *= d[j]
    return R, gamma ** (max(k, 1) - 1) * m, k, float(k > 0)


def test_timeout_window_ends_one_step_early():
    r, d, tr = _arrays(trunc_at=2)
    R, disc, boot, w = window_returns(r, d, tr, GAMMA, N)
    np.testing.assert_allclose(R, [1.0 + GAMMA * 2.0], rtol=1e-6)
    # The target applies one more gamma: gamma * disc = gamma^2.
    np.testing.assert_allclose(GAMMA * disc, [GAMMA**2], rtol=1e-6)
    np.testing.assert_allclose(disc, [GAMMA], rtol=1e-6)
    assert boot.dtype == jnp.int32 and int(boot[0]) == 2
    assert float(w[0]) == 1.0


def test_window_starting_at_timeout_excluded():
    r, d, tr = _arrays(trunc_at=0)
    R, disc, boot, w = window_returns(r, d, tr, GAMMA, N)
    assert float(w[0]) == 0.0
    assert int(boot[0]) == 0
    assert float(R[0]) == 0.0 and np.isfinite(float(disc[0]))


def test_fall_on_final_step_is_termination():
    """done=1, truncation=0 at index 2: the n-step values, r2 included."""
    r, d, tr = _arrays(done_at=2)
    R, disc, boot, w = window_returns(r, d, tr, GAMMA, N)
    np.testing.assert_allclose(R, [1.0 + GAMMA * 2.0 + GAMMA**2 * 3.0], rtol=1e-6)
    assert float(disc[0]) == 0.0
    assert int(boot[0]) == N and float(w[0]) == 1.0


def test_no_cut_matches_plain_n_step():
    r, d, tr = _arrays()
    R, disc, boot, w = window_returns(r, d, tr, GAMMA, N)
    np.testing.assert_allclose(
        R, [sum(GAMMA**i * (i + 1.0) for i in range(N))], rtol=1e-6
    )
    np.testing.assert_allclose(disc, [GAMMA ** (N - 1)], rtol=1e-6)
    assert int(boot[0]) == N and float(w[0]) == 1.0


@pytest.mark.parametrize("garbage", [1e6, np.inf, np.nan])
def test_window_crossing_reset_masks_next_episode(garbage):
    """episode_length 3 < n 4: the time-out at 2 cuts the window, and nothing
    of the next episode (huge, inf or NaN rewards from 3 on) leaks into R."""
    rewards = [1.0, 2.0, 3.0, garbage, garbage, garbage]
    r, d, tr = _arrays(trunc_at=2, rewards=rewards)
    R, disc, boot, w = window_returns(r, d, tr, GAMMA, N)
    np.testing.assert_allclose(R, [1.0 + GAMMA * 2.0], rtol=1e-6)
    np.testing.assert_allclose(disc, [GAMMA], rtol=1e-6)
    assert int(boot[0]) == 2 and float(w[0]) == 1.0


def test_termination_before_later_timeout_has_zero_bootstrap():
    """A fall at 1 then the next episode's time-out at 3: the fall decides."""
    r = jnp.asarray([[1.0, 2.0, 1e6, 1e6, 1e6]])
    d = jnp.asarray([[1.0, 0.0, 1.0, 0.0, 1.0]])
    tr = jnp.asarray([[0.0, 0.0, 0.0, 1.0, 0.0]])
    R, disc, _, w = window_returns(r, d, tr, GAMMA, N)
    np.testing.assert_allclose(R, [1.0 + GAMMA * 2.0], rtol=1e-6)
    assert float(disc[0]) == 0.0 and float(w[0]) == 1.0


def test_rnn_per_point_matches_ff_at_each_t():
    """Per-point values at t == window_returns on the window starting at t."""
    rng = np.random.default_rng(0)
    B, L, n = 5, 3, N
    T = L + n
    r = rng.normal(size=(B, T)).astype(np.float32)
    done = rng.uniform(size=(B, T)) < 0.3
    done[0, 1] = True  # the point starting at (0, 1) is a time-out: excluded
    tr = (done & (rng.uniform(size=(B, T)) < 0.6)).astype(np.float32)
    tr[0, 1] = 1.0
    d = (~done).astype(np.float32)
    r, d, tr = map(jnp.asarray, (r, d, tr))
    R, disc, off, w = _per_point_nstep_returns(r, d, GAMMA, L, n, truncation=tr)
    assert R.shape == disc.shape == off.shape == w.shape == (B, L)
    assert float(jnp.sum(1.0 - w)) > 0.0  # the draw holds an excluded point
    for t in range(L):
        Rt, dt, bt, wt = window_returns(r[:, t:], d[:, t:], tr[:, t:], GAMMA, n)
        np.testing.assert_allclose(R[:, t], Rt, rtol=1e-6)
        np.testing.assert_allclose(disc[:, t], dt, rtol=1e-6)
        np.testing.assert_array_equal(off[:, t], bt)
        np.testing.assert_array_equal(w[:, t], wt)
        for b in range(B):
            ref = _ref_window(r[b, t:], d[b, t:], tr[b, t:], GAMMA, n)
            np.testing.assert_allclose(
                [R[b, t], disc[b, t], off[b, t], w[b, t]], ref, rtol=1e-5
            )


def test_per_point_without_truncation_reports_full_windows():
    r = jnp.ones((2, 5))
    d = jnp.ones((2, 5))
    _, _, off, w = _per_point_nstep_returns(r, d, GAMMA, 2, 3)
    np.testing.assert_array_equal(off, 3)
    np.testing.assert_array_equal(w, 1.0)


# ---------------------------------------------------------------------------
# Feed-forward learner: the critic loss against an independent reference.
# ---------------------------------------------------------------------------


def _ff_setup(next_obs):
    cfg = DMPOConfig(
        policy_layer_sizes=(16,),
        critic_layer_sizes=(16,),
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        num_samples=4,
        n_step=N,
        use_n_step=True,
        discount=GAMMA,
        batch_size=4,
        sequence_length=N if next_obs else N + 1,
        store_next_observation=next_obs,
        **TRUNC_ONLY,
    )
    nets = make_dmpo_networks(5, 2, cfg)
    state = init_training_state(
        jax.random.key(0), nets, {"obs_size": 5, "action_size": 2}, cfg
    )
    # Sharpen the target critic so the bootstrap state is visible in the loss.
    state = state._replace(
        target_critic_params=jax.tree.map(lambda x: 4.0 * x, state.critic_params)
    )
    return cfg, nets, state


def _ff_batch(next_obs):
    """Rows: mid-episode, time-out at 2, time-out at 0 (excluded), fall at 1."""
    B = 4
    ks = jax.random.split(jax.random.key(3), 3)
    o = jax.random.normal(ks[0], (B, N + 1, 5))
    tr = np.zeros((B, N + 1), np.float32)
    d = np.ones((B, N + 1), np.float32)
    for b, t in ((1, 2), (2, 0)):
        tr[b, t], d[b, t] = 1.0, 0.0
    d[3, 1] = 0.0
    batch = {
        "action": jax.random.uniform(ks[1], (B, N + 1, 2), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(ks[2], (B, N + 1)),
        "discount": jnp.asarray(d),
        "truncation": jnp.asarray(tr),
    }
    if next_obs:
        batch = {k: v[:, :N] for k, v in batch.items()}
        batch["observation"] = o[:, :N]
        batch["next_observation"] = o[:, 1:]
    else:
        batch["observation"] = o
    return batch, o


@pytest.mark.parametrize("next_obs", [False, True], ids=["compressed", "next_obs"])
def test_ff_sgd_step_bootstraps_from_timeout_state(next_obs):
    cfg, nets, state = _ff_setup(next_obs)
    batch, o = _ff_batch(next_obs)
    r, d, tr = (np.asarray(batch[k]) for k in ("reward", "discount", "truncation"))
    refs = [_ref_window(r[b], d[b], tr[b], GAMMA, N) for b in range(4)]
    R, D, boot, w = (np.asarray(x, np.float32) for x in zip(*refs))
    assert list(boot) == [N, 2, 0, N] and list(w) == [1, 1, 0, 1]

    norm = lambda x: _normalize_obs(x, state.normalizer_params)  # noqa: E731
    s_boot = norm(o[np.arange(4), boot.astype(int)])
    a_boot = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        state.target_policy_params, s_boot
    ).mode()
    target = compute_categorical_target(
        nets,
        state.target_critic_params,
        s_boot[:, None],
        a_boot[:, None],
        jnp.asarray(R)[:, None],
        jnp.asarray(D)[:, None],
        cfg,
    )[:, 0]
    ref = _critic_loss_fn(
        state.critic_params,
        nets,
        norm(o[:, 0]),
        batch["action"][:, 0],
        target,
        jnp.asarray(w),
    )
    _, m = jax.jit(lambda s, b: sgd_step(s, b, nets, make_optimizers(cfg), cfg))(
        state, batch
    )
    np.testing.assert_allclose(float(m["critic_loss"]), float(ref), rtol=1e-6)
    for v in m.values():
        assert np.isfinite(np.asarray(v)).all()


def test_ff_batch_without_truncation_field_fails_loud():
    cfg, nets, state = _ff_setup(next_obs=False)
    batch, _ = _ff_batch(next_obs=False)
    del batch["truncation"]
    with pytest.raises(ValueError, match="'truncation'"):
        sgd_step(state, batch, nets, make_optimizers(cfg), cfg)


def test_all_points_excluded_is_finite_with_zero_grad():
    """Every row starts at a time-out: zero weight everywhere, finite metrics,
    and the critic does not move."""
    cfg, nets, state = _ff_setup(next_obs=False)
    batch, _ = _ff_batch(next_obs=False)
    batch["truncation"] = batch["truncation"].at[:, 0].set(1.0)
    new_state, m = sgd_step(state, batch, nets, make_optimizers(cfg), cfg)
    for k, v in m.items():
        assert np.isfinite(np.asarray(v)).all(), k
    for a, b in zip(
        jax.tree.leaves(new_state.critic_params), jax.tree.leaves(state.critic_params)
    ):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Recurrent learner, per-step and strided.
# ---------------------------------------------------------------------------


def _ref_points(batch, starts, n, gamma):
    """Per-point (R, D, boot, w) by python loop at the given [B, L] starts."""
    r, d, tr = (np.asarray(batch[k]) for k in ("reward", "discount", "truncation"))
    out = np.zeros((4, *starts.shape))
    for b in range(starts.shape[0]):
        for i in range(starts.shape[1]):
            t = int(starts[b, i])
            R, D, k, w = _ref_window(r[b, t:], d[b, t:], tr[b, t:], gamma, n)
            out[:, b, i] = (R, D, t + k, w)
    R, D, boot, w = out
    return R.astype(np.float32), D.astype(np.float32), boot.astype(int), w


def _rnn_ref_loss(state, nets, batch, cfg, starts, priv_key=""):
    """Critic loss with bootstrap = target policy at the cut, fed the STORED
    hidden there, against hand-computed per-point returns."""
    B, L = starts.shape
    R, D, boot, w = _ref_points(batch, starts, cfg.n_step, cfg.discount)
    rows = np.arange(B)[:, None]
    obs = _normalize_obs(batch["observation"], state.normalizer_params)

    def at(cols, priv):
        o = jax.tree.map(lambda x: x[rows, cols], obs)
        return {**o, "privileged": batch[priv_key][rows, cols]} if priv else o

    h_boot = tuple(x.astype(jnp.float32)[rows, boot] for x in batch["policy_hidden"])
    a_boot = jax.vmap(
        jax.vmap(
            lambda o, h: nets.policy.apply(
                state.target_policy_params, o, h, method="raw"
            )[0]
        )
    )(at(boot, False), h_boot)
    target = compute_categorical_target(
        nets,
        state.target_critic_params,
        at(boot, bool(priv_key)),
        a_boot,
        jnp.asarray(R),
        jnp.asarray(D),
        cfg,
    ).reshape(B * L, cfg.num_atoms)
    obs_q = jax.tree.map(
        lambda x: x.reshape((B * L, *x.shape[2:])), at(starts, bool(priv_key))
    )
    act_q = batch["action"][rows, starts].reshape(B * L, -1)
    loss = _critic_loss_fn(
        state.critic_params, nets, obs_q, act_q, target, jnp.asarray(w.reshape(-1))
    )
    return loss, boot, w


def _with_timeouts(batch, at):
    """Mark the given (b, t) steps as time-outs (their discount is already 0)."""
    tr = batch["truncation"]
    for b, t in at:
        assert float(batch["discount"][b, t]) == 0.0
        tr = tr.at[b, t].set(1.0)
    return dict(batch, truncation=tr)


def test_rnn_per_step_path_cuts_windows():
    L, n = 3, N
    s = rnn_t._setup(L=L, n=n, **TRUNC_ONLY)
    state, nets, cfg = s["state"], s["nets"], s["cfg"]
    # Time-outs at a loss point (env 0, t=1: the point starting there is
    # excluded and t=0 cuts after one step) and in the tail (env 2, t=4).
    done_at = ((0, 1), (1, 2), (2, 4))
    batch = rnn_t._make_batch(jax.random.key(4), state, nets, L + n, done_at)
    batch = _with_timeouts(batch, ((0, 1), (2, 4)))
    state = state._replace(
        target_critic_params=jax.tree.map(lambda x: 4.0 * x, state.critic_params)
    )
    starts = np.tile(np.arange(L), (rnn_t._B, 1))
    ref, boot, w = _rnn_ref_loss(state, nets, batch, cfg, starts)
    assert boot[0, 0] == 1 and w[0, 1] == 0.0 and boot[2, 0] == 4
    _, m = jax.jit(lambda st, b: _sgd_step_rnn(st, b, nets, s["optimizers"], cfg))(
        state, batch
    )
    np.testing.assert_allclose(float(m["critic_loss"]), float(ref), rtol=1e-6)


def test_tick_path_uses_window_cut():
    """rnn_tick_stride=10: the tick points' targets are the per-point values
    gathered at the ticks, and each bootstraps at idx + boot_off with the
    stored (unforced) hidden there. The target critic is sharpened so a
    wrong bootstrap state or hidden moves the loss."""
    k, L, n = 10, 4, 3
    cfg = tick_t._cfg(L=L, n=n, k=k, priv=True, **TRUNC_ONLY)
    nets = tick_t._slow_nets(cfg, k, priv_dim=tick_t.PRIV)
    state = tick_t._state(nets, cfg)
    state = state._replace(
        target_critic_params=jax.tree.map(lambda x: 5.0 * x, state.target_critic_params)
    )
    # env 0: ticks 0, 10, 20, 22 with a time-out at 21 (tick 20 cuts after
    # one step); env 2: ticks 3, 13, 23, ... with a time-out AT tick 23
    # (excluded); env 1: a real termination at 12.
    batch = tick_t._make_batch(
        nets,
        state,
        cfg,
        phase0=(0, 3, 7, 9),
        done_at=((0, 21), (1, 12), (2, 23)),
        priv=True,
    )
    batch = _with_timeouts(batch, ((0, 21), (2, 23)))
    phase = np.asarray(batch["policy_hidden"][tick_t.PHASE][:, :, 0], np.float32)
    ticks = np.array(
        [[t for t in range(L * k) if phase[b, t] == 0.0][:L] for b in range(tick_t.B)]
    )
    np.testing.assert_array_equal(ticks[0], [0, 10, 20, 22])
    ref, boot, w = _rnn_ref_loss(state, nets, batch, cfg, ticks, priv_key=tick_t.KEY)
    assert boot[0, 2] == 21 and (boot != ticks + n).any()
    assert w.min() == 0.0

    _, m = _sgd_step_rnn(state, batch, nets, make_optimizers(cfg), cfg)
    np.testing.assert_allclose(float(m["critic_loss"]), float(ref), rtol=1e-6)


def test_rnn_batch_without_truncation_field_fails_loud():
    s = rnn_t._setup(**TRUNC_ONLY)
    batch = dict(s["batch"])
    del batch["truncation"]
    with pytest.raises(ValueError, match="'truncation'"):
        _sgd_step_rnn(s["state"], batch, s["nets"], s["optimizers"], s["cfg"])


# ---------------------------------------------------------------------------
# Rollout: the flag is stored, and an env without one fails loud.
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _State:
    obs: jnp.ndarray
    done: jnp.ndarray
    steps: jnp.ndarray
    info: dict

    def tree_flatten(self):
        return (self.obs, self.done, self.steps, self.info), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class _TimeLimitEnv:
    """Episodes of 3 steps ending in a time-out, flagged like the stash does
    (``stash=False`` leaves the flag out, as a plain brax stack would)."""

    pre_batched = False
    episode_length = 3

    def __init__(self, stash=True):
        self.stash = stash

    def _info(self, trunc):
        if not self.stash:
            return {}
        return {"AutoResetWrapper_preserve_info": {"truncation": trunc}}

    def reset(self, rng):
        del rng
        z = jnp.zeros((), jnp.float32)
        return _State(jnp.zeros(2), z, z, self._info(z))

    def step(self, state, action):
        steps = state.steps + 1
        timeout = (steps >= self.episode_length).astype(jnp.float32)
        steps = jnp.where(timeout > 0, 0.0, steps)
        new = _State(state.obs + action[:2], timeout, steps, self._info(timeout))
        return new, jnp.float32(1.0)


def _policy_apply(params, obs):
    from tensorflow_probability.substrates import jax as tfp

    del params
    return tfp.distributions.MultivariateNormalDiag(
        loc=jnp.zeros(2) + 0.0 * obs.sum(), scale_diag=jnp.ones(2)
    )


def _flat_normalizer():
    from brax.training.acme import running_statistics, specs

    return running_statistics.init_state(specs.Array((2,), jnp.float32))


def test_rollout_stores_truncation_and_keeps_discount():
    traj, _, _ = collect_rollout(
        _TimeLimitEnv(),
        _policy_apply,
        None,
        _flat_normalizer(),
        jax.random.key(0),
        num_envs=2,
        num_steps=7,
        store_truncation=True,
    )
    expected = [0, 0, 1, 0, 0, 1, 0]
    np.testing.assert_array_equal(traj["truncation"][0], expected)
    assert traj["truncation"].dtype == jnp.float32
    # The stored discount stays 1 - done: a time-out is still an episode end.
    np.testing.assert_array_equal(traj["discount"][0], 1.0 - np.asarray(expected))


def test_rollout_schema_unchanged_without_the_flag():
    traj, _, _ = collect_rollout(
        _TimeLimitEnv(stash=False),
        _policy_apply,
        None,
        _flat_normalizer(),
        jax.random.key(0),
        num_envs=2,
        num_steps=3,
    )
    assert "truncation" not in traj


def test_missing_truncation_signal_raises():
    """An env not composed through wrap_for_dmpo_training, with the switch on,
    fails at trace time instead of silently storing zeros."""
    with pytest.raises(KeyError, match="wrap_for_dmpo_training"):
        jax.jit(
            lambda key: collect_rollout(
                _TimeLimitEnv(stash=False),
                _policy_apply,
                None,
                _flat_normalizer(),
                key,
                num_envs=2,
                num_steps=3,
                store_truncation=True,
            )
        )(jax.random.key(0))


# ---------------------------------------------------------------------------
# Legacy: the flag is ignored even when the batch carries time-outs.
# ---------------------------------------------------------------------------


def _load_golden_maker():
    path = pathlib.Path(__file__).parent / "golden" / "make_pre_alignment_goldens.py"
    spec = importlib.util.spec_from_file_location("make_pre_alignment_goldens", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="CPU-bitwise goldens")
@pytest.mark.parametrize("group", ["ff", "rnn", "tick"])
def test_legacy_ignores_truncation(group, monkeypatch):
    """Under LEGACY_ALIGNMENT every learner path matches the pre-alignment
    golden bit for bit even with time-outs flagged all over the batch."""
    maker = _load_golden_maker()

    def _inject(fn):
        def wrapped(state, batch, *args, **kwargs):
            tr = jnp.zeros_like(batch["reward"]).at[:, ::2].set(1.0)
            return fn(state, dict(batch, truncation=tr), *args, **kwargs)

        return wrapped

    monkeypatch.setattr(maker, "sgd_step", _inject(maker.sgd_step))
    monkeypatch.setattr(maker, "_sgd_step_rnn", _inject(maker._sgd_step_rnn))
    got = maker.compute_group(group, dict(LEGACY_ALIGNMENT))
    with np.load(pathlib.Path(maker.GOLDEN_PATH)) as z:
        want = {k: z[k] for k in z.files if k.startswith(f"{group}/")}
    assert set(got) == set(want)
    bad = [k for k in want if not np.array_equal(got[k], want[k])]
    assert not bad, bad[:5]
