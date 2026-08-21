"""Recurrent (short-BPTT) learner path: ``_sgd_step_rnn`` and its helpers.

Uses a STUB recurrent policy — a minimal GRU module honoring the pinned
interface contract (``__call__(obs, hidden) -> (dist, new_hidden)`` and
``raw(obs, hidden) -> (mu, scale, new_hidden)``, hidden = tuple of per-layer
arrays) — plus a tiny dict-obs critic. Deliberately decoupled from the real
``networks_kl_anchor_rnn`` module: the learner's only contract with the
networks is the raw-apply signature and the tuple-of-[B, H] hidden pytree,
and this suite must keep passing even if the production net changes shape.

Coverage (per PLAN_rnn_policy_head.md section 9.3):
  1. Per-point n-step returns vs a python triple-loop reference, including
     windows crossing a done; length=1 reduces bit-for-bit to the FF
     ``sgd_step`` expressions.
  2. Unroll done-reset: the hidden entering t+1 after a done at t is exactly
     zeros; recomputed pre-step hiddens match a hand rollout (staleness ~ 0),
     and corrupting the stored hidden makes the staleness metric fire.
  3. BPTT reality: grad of the policy loss w.r.t. the GRU's recurrent (h->h)
     kernels is nonzero at L > 1 and exactly zero at L = 1 with a zero
     window-start hidden (the only h the single cell application consumes).
  4. MPO flattening: the loss on a flattened [B*L] batch equals the mean of
     per-t losses given the SAME sampled actions/q per column (exact sample
     alignment by construction — a shared fixed sample tensor).
  5. Full ``_sgd_step_rnn`` smoke under jit (finite metrics incl. both rnn/
     keys, metric-key parity with the FF path) + fail-loud schema checks +
     the rnn_bptt_length=0 dispatch still running the FF body.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    TrainingState,
    _build_loss,
    _normalize_obs,
    _per_point_nstep_returns,
    _policy_loss_fn_rnn,
    _reset_hidden,
    _sgd_step_rnn,
    _unroll_policy_raw,
    init_training_state,
    make_optimizers,
    sgd_step,
)
from track_mjx.agent.dmpo.networks import (
    CategoricalCriticHead,
    DMPONetworks,
    make_dmpo_networks,
)
from track_mjx.agent.observation_utils import init_dict_normalizer

tfd = tfp.distributions

_IMIT, _PROP, _ACT, _HIDDEN = 5, 7, 3, 8
_B, _L, _N = 4, 3, 4


class _StubRecurrentPolicy(nn.Module):
    """Minimal recurrent policy per the pinned interface contract.

    Shape-agnostic (Dense/GRUCell handle unbatched input), so the learner can
    vmap it over per-env slices — the codebase convention. ``scale`` is a
    free parameter broadcast to mu's shape so the dist is well-formed while
    keeping the module tiny.
    """

    action_size: int
    hidden_size: int = _HIDDEN

    def setup(self):
        self.mix = nn.Dense(self.hidden_size, name="mix")
        self.cell = nn.GRUCell(features=self.hidden_size, name="cell")
        self.loc = nn.Dense(self.action_size, name="loc")
        self.log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_size,)
        )

    def raw(self, obs, hidden):
        x = jnp.concatenate(
            [obs["imitation_target"], obs["proprioception"]], axis=-1
        )
        x = jnp.tanh(self.mix(x))
        (h,) = hidden
        new_h, y = self.cell(h, x)
        mu = self.loc(y)
        scale = jax.nn.softplus(self.log_std) + 1e-3
        return mu, jnp.broadcast_to(scale, mu.shape), (new_h,)

    def __call__(self, obs, hidden):
        mu, scale, new_hidden = self.raw(obs, hidden)
        return (
            tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale),
            new_hidden,
        )


class _StubCritic(nn.Module):
    """Tiny dict-obs C51 critic (feed-forward, as in the real design)."""

    num_atoms: int
    vmin: float
    vmax: float

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate(
            [obs["imitation_target"], obs["proprioception"], action], axis=-1
        )
        h = jnp.tanh(nn.Dense(16)(x))
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


def _make_cfg(L=_L, n=_N, **overrides):
    kw = dict(
        n_step=n,
        rnn_bptt_length=L,
        sequence_length=L + n,
        use_n_step=True,
        store_next_observation=False,
        num_samples=5,
        discount=0.9,
        vmin=-10.0,
        vmax=10.0,
        num_atoms=11,
        batch_size=_B,
    )
    kw.update(overrides)
    return DMPOConfig(**kw)


def _obs_template():
    return {
        "imitation_target": jnp.zeros((_IMIT,), jnp.float32),
        "proprioception": jnp.zeros((_PROP,), jnp.float32),
    }


def _hand_rollout_hidden(state, nets, batch):
    """Recreate what the recurrent rollout stores: the PRE-step hidden the
    policy consumed at each t, zero-reset after every done. A plain python
    loop — the independent reference the learner's scan is checked against.
    """
    T = batch["reward"].shape[1]
    B = batch["reward"].shape[0]
    obs_norm = _normalize_obs(batch["observation"], state.normalizer_params)
    h = jnp.zeros((B, _HIDDEN), jnp.float32)
    stored = []
    for t in range(T):
        stored.append(h)
        obs_t = jax.tree.map(lambda x: x[:, t], obs_norm)
        _, _, (h_new,) = jax.vmap(
            lambda o, hh: nets.policy.apply(
                state.policy_params, o, hh, method="raw"
            )
        )(obs_t, (h,))
        done_t = batch["discount"][:, t] == 0
        h = jnp.where(done_t[:, None], 0.0, h_new)
    return (jnp.stack(stored, axis=1),)  # tuple of [B, T, H]


def _make_batch(key, state, nets, T, done_at=((1, 1),)):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = {
        "imitation_target": jax.random.normal(k1, (_B, T, _IMIT)),
        "proprioception": jax.random.normal(k2, (_B, T, _PROP)),
    }
    action = jax.random.uniform(k3, (_B, T, _ACT), minval=-0.9, maxval=0.9)
    reward = jax.random.normal(k4, (_B, T))
    discount = jnp.ones((_B, T))
    for b, t in done_at:
        discount = discount.at[b, t].set(0.0)
    batch = {
        "observation": obs,
        "action": action,
        "reward": reward,
        "discount": discount,
    }
    batch["policy_hidden"] = _hand_rollout_hidden(state, nets, batch)
    return batch


def _setup(seed=0, L=_L, n=_N, **cfg_overrides):
    """Stub nets + hand-built TrainingState + synthetic [B, T] batch whose
    stored hidden mirrors what the recurrent rollout would store (so hidden
    staleness is ~0 with the current params — the clean-parity regime)."""
    cfg = _make_cfg(L=L, n=n, **cfg_overrides)
    policy = _StubRecurrentPolicy(action_size=_ACT)
    critic = _StubCritic(num_atoms=cfg.num_atoms, vmin=cfg.vmin, vmax=cfg.vmax)
    nets = DMPONetworks(policy=policy, critic=critic)

    rng = jax.random.PRNGKey(seed)
    rng, k_pol, k_crit, k_batch = jax.random.split(rng, 4)
    obs_dummy = _obs_template()
    hidden_dummy = (jnp.zeros((_HIDDEN,), jnp.float32),)
    policy_params = policy.init(k_pol, obs_dummy, hidden_dummy)
    critic_params = critic.init(
        k_crit, obs_dummy, jnp.zeros((_ACT,), jnp.float32)
    )
    dual_params = _build_loss(cfg).init_params(_ACT, jnp.float32)
    optimizers = make_optimizers(cfg)
    pol_opt, crit_opt, dual_opt = optimizers
    state = TrainingState(
        policy_params=policy_params,
        critic_params=critic_params,
        target_policy_params=policy_params,
        target_critic_params=critic_params,
        dual_params=dual_params,
        policy_opt_state=pol_opt.init(policy_params),
        critic_opt_state=crit_opt.init(critic_params),
        dual_opt_state=dual_opt.init(dual_params),
        normalizer_params=init_dict_normalizer(obs_dummy),
        steps=jnp.zeros((), jnp.int32),
        rng=rng,
    )
    batch = _make_batch(k_batch, state, nets, T=L + n)
    return dict(
        cfg=cfg, nets=nets, optimizers=optimizers, state=state, batch=batch
    )


def _prep_policy_loss_args(state, nets, batch, cfg):
    """Replicate ``_sgd_step_rnn``'s window prep for direct helper-level tests."""
    L, n = cfg.rnn_bptt_length, cfg.n_step
    B = batch["reward"].shape[0]
    obs_norm = _normalize_obs(batch["observation"], state.normalizer_params)
    h_all = jax.tree.map(
        lambda h: h.astype(jnp.float32), batch["policy_hidden"]
    )
    h0 = jax.tree.map(lambda h: h[:, 0], h_all)
    h_stored_tm = jax.tree.map(
        lambda h: jnp.swapaxes(h[:, :L], 0, 1), h_all
    )
    obs_L = jax.tree.map(lambda x: x[:, :L], obs_norm)
    obs_tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), obs_L)
    done_tm = jnp.swapaxes(batch["discount"][:, :L] == 0, 0, 1)
    obs_flat = jax.tree.map(
        lambda x: x.reshape((B * L,) + x.shape[2:]), obs_L
    )
    tgt_mu, tgt_scale, _ = _unroll_policy_raw(
        state.target_policy_params, nets, obs_tm, done_tm, h0
    )
    A = tgt_mu.shape[-1]
    tgt_mu = jnp.swapaxes(tgt_mu, 0, 1).reshape(B * L, A)
    tgt_scale = jnp.swapaxes(tgt_scale, 0, 1).reshape(B * L, A)
    return dict(
        obs_tm=obs_tm,
        done_tm=done_tm,
        h0=h0,
        h_stored_tm=h_stored_tm,
        obs_flat=obs_flat,
        tgt_mu=tgt_mu,
        tgt_scale=tgt_scale,
    )


# ---------------------------------------------------------------------------
# 1. Per-point n-step returns.
# ---------------------------------------------------------------------------


def test_per_point_nstep_returns_matches_bruteforce():
    """Vectorized idx-gather returns vs a python triple loop, with dones
    scattered mid-window (the alive mask must stop both the reward sum and
    the bootstrap coefficient at the first done)."""
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    B, L, n, gamma = 3, 4, 5, 0.87
    T = L + n
    rewards = jax.random.normal(k1, (B, T))
    discounts = (jax.random.uniform(k2, (B, T)) > 0.3).astype(jnp.float32)
    # Ensure at least one done inside a window regardless of the draw.
    discounts = discounts.at[0, 2].set(0.0)

    R, D = _per_point_nstep_returns(rewards, discounts, gamma, L, n)
    assert R.shape == (B, L) and D.shape == (B, L)

    r, d = np.asarray(rewards), np.asarray(discounts)
    for b in range(B):
        for t in range(L):
            m, ref_R = 1.0, 0.0
            for j in range(n):
                ref_R += (gamma**j) * m * r[b, t + j]
                m *= d[b, t + j]
            ref_D = (gamma ** (n - 1)) * m
            np.testing.assert_allclose(R[b, t], ref_R, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(D[b, t], ref_D, rtol=1e-5, atol=1e-6)


def test_per_point_nstep_returns_length1_equals_ff_formula():
    """length=1 must reduce to the FF ``sgd_step`` n-step expressions —
    the bit-identity anchor between the two learner paths."""
    key = jax.random.PRNGKey(4)
    k1, k2 = jax.random.split(key)
    B, n, gamma = 5, 6, 0.95
    rewards = jax.random.normal(k1, (B, n + 1))
    discounts = (jax.random.uniform(k2, (B, n + 1)) > 0.4).astype(jnp.float32)
    R, D = _per_point_nstep_returns(rewards, discounts, gamma, 1, n)

    # FF sgd_step expressions, copied verbatim from learner.py.
    _d = discounts[:, :n]
    _r = rewards[:, :n]
    _alive = jnp.cumprod(_d, axis=1)
    _m = jnp.concatenate([jnp.ones_like(_d[:, :1]), _alive[:, :-1]], axis=1)
    _g = gamma ** jnp.arange(n, dtype=_r.dtype)
    rew_t0 = jnp.sum(_g[None, :] * _m * _r, axis=1)
    disc_t0 = (gamma ** (n - 1)) * _alive[:, -1]

    np.testing.assert_allclose(
        np.asarray(R[:, 0]), np.asarray(rew_t0), rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(D[:, 0]), np.asarray(disc_t0), rtol=1e-6, atol=0
    )


# ---------------------------------------------------------------------------
# 2. Unroll done-reset + staleness metric.
# ---------------------------------------------------------------------------


def test_reset_hidden_zeros_only_done_rows():
    h = (jnp.ones((3, 4)), jnp.full((3, 2), 2.0))
    done = jnp.array([False, True, False])
    out = _reset_hidden(h, done)
    for layer in out:
        assert bool(jnp.all(layer[1] == 0.0))
        assert bool(jnp.all(layer[0] != 0.0))
        assert bool(jnp.all(layer[2] != 0.0))


def test_unroll_resets_hidden_after_done_and_matches_hand_rollout():
    s = _setup()
    state, nets, batch, cfg = s["state"], s["nets"], s["batch"], s["cfg"]
    args = _prep_policy_loss_args(state, nets, batch, cfg)
    _, _, h_pre = _unroll_policy_raw(
        state.policy_params, nets, args["obs_tm"], args["done_tm"], args["h0"]
    )
    # _make_batch forces done at (env 1, t=1): the hidden ENTERING t=2 must be
    # exactly zeros for env 1 (reset), nonzero for an env that did not reset.
    assert bool(jnp.all(h_pre[0][2, 1] == 0.0))
    assert float(jnp.max(jnp.abs(h_pre[0][2, 0]))) > 0.0
    # Recomputed pre-step hiddens == hand-rolled stored hiddens (same params,
    # same ops), for every t including across the done.
    np.testing.assert_allclose(
        np.asarray(h_pre[0]),
        np.asarray(args["h_stored_tm"][0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_staleness_metric_zero_fresh_and_positive_when_corrupted():
    s = _setup()
    _, metrics = _sgd_step_rnn(
        s["state"], s["batch"], s["nets"], s["optimizers"], s["cfg"]
    )
    # Stored hidden was generated with the CURRENT params -> zero staleness.
    assert float(metrics["rnn/hidden_staleness"]) < 1e-6

    # Corrupt the stored hidden at t in [1, L) only. t=0 must stay intact:
    # it seeds the recomputation, so corrupting it moves recomputed and
    # stored together and staleness would stay 0 by construction. The
    # bootstrap slice [n, n+L) is untouched, so only the metric can move.
    bad = s["batch"]["policy_hidden"][0].at[:, 1:_L].add(1.0)
    batch2 = dict(s["batch"])
    batch2["policy_hidden"] = (bad,)
    _, metrics2 = _sgd_step_rnn(
        s["state"], batch2, s["nets"], s["optimizers"], s["cfg"]
    )
    assert float(metrics2["rnn/hidden_staleness"]) > 1e-3


# ---------------------------------------------------------------------------
# 3. BPTT reality.
# ---------------------------------------------------------------------------


def test_bptt_gradient_reaches_recurrent_kernel():
    """Grad w.r.t. the GRU's h->h kernels is nonzero iff gradients flow
    through TIME: at L=1 with the zero window-start hidden, the single cell
    application consumes h=0 and the recurrent-kernel grad is exactly zero
    (dW_h = delta (x) h = 0); at L>1 the t>=1 hiddens are nonzero and BPTT
    reaches the kernels."""

    def _recurrent_kernel_grad(s):
        state, nets, cfg = s["state"], s["nets"], s["cfg"]
        args = _prep_policy_loss_args(state, nets, s["batch"], cfg)

        def loss_of(p):
            return _policy_loss_fn_rnn(
                p,
                state.dual_params,
                nets,
                args["obs_tm"],
                args["done_tm"],
                args["h0"],
                args["h_stored_tm"],
                args["obs_flat"],
                args["tgt_mu"],
                args["tgt_scale"],
                state.target_critic_params,
                cfg,
                jax.random.PRNGKey(7),
            )[0]

        grads = jax.grad(loss_of)(state.policy_params)
        return jnp.concatenate(
            [
                grads["params"]["cell"][k]["kernel"].ravel()
                for k in ("hr", "hz", "hn")
            ]
        )

    rec_L3 = _recurrent_kernel_grad(_setup(L=3))
    assert float(jnp.max(jnp.abs(rec_L3))) > 0.0

    rec_L1 = _recurrent_kernel_grad(_setup(L=1))
    np.testing.assert_array_equal(np.asarray(rec_L1), 0.0)


# ---------------------------------------------------------------------------
# 4. MPO flattening.
# ---------------------------------------------------------------------------


def test_mpo_loss_on_flattened_bl_equals_mean_of_per_t_losses():
    """The recurrent branch folds L loss points into the batch axis and calls
    the MPO loss ONCE on [B*L]. That is valid iff the module is batch-shape
    agnostic: every reduction is a mean over — or linear in a mean over —
    the batch axis, so the flattened loss equals the mean of per-t losses
    given the SAME sampled actions and q per column. Sample alignment is
    exact by construction here (one shared fixed sample tensor sliced both
    ways), so this is the strict-equality variant of check 4."""
    key = jax.random.PRNGKey(11)
    ks = jax.random.split(key, 6)
    B, L, A, N = 4, 3, 5, 7
    mu_o = jax.random.normal(ks[0], (B, L, A))
    sc_o = jax.nn.softplus(jax.random.normal(ks[1], (B, L, A))) + 1e-2
    mu_t = jax.random.normal(ks[2], (B, L, A))
    sc_t = jax.nn.softplus(jax.random.normal(ks[3], (B, L, A))) + 1e-2
    sampled = jax.random.normal(ks[4], (N, B, L, A))
    q = jax.random.normal(ks[5], (N, B, L))

    cfg = _make_cfg()
    loss_module = _build_loss(cfg)
    params = loss_module.init_params(A, jnp.float32)

    flat_loss, _ = loss_module(
        params=params,
        online_action_distribution=tfd.MultivariateNormalDiag(
            loc=mu_o.reshape(B * L, A), scale_diag=sc_o.reshape(B * L, A)
        ),
        target_action_distribution=tfd.MultivariateNormalDiag(
            loc=mu_t.reshape(B * L, A), scale_diag=sc_t.reshape(B * L, A)
        ),
        actions=sampled.reshape(N, B * L, A),
        q_values=q.reshape(N, B * L),
    )

    per_t = []
    for t in range(L):
        loss_t, _ = loss_module(
            params=params,
            online_action_distribution=tfd.MultivariateNormalDiag(
                loc=mu_o[:, t], scale_diag=sc_o[:, t]
            ),
            target_action_distribution=tfd.MultivariateNormalDiag(
                loc=mu_t[:, t], scale_diag=sc_t[:, t]
            ),
            actions=sampled[:, :, t],
            q_values=q[:, :, t],
        )
        per_t.append(float(np.squeeze(np.asarray(loss_t))))

    np.testing.assert_allclose(
        float(np.squeeze(np.asarray(flat_loss))),
        float(np.mean(per_t)),
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 5. Full smoke + dispatch + fail-loud schema checks.
# ---------------------------------------------------------------------------


def _ff_smoke_metrics():
    """Run one FF sgd_step (rnn_bptt_length=0 default) on the flat-obs mock
    setup; returns its metrics dict for key-parity and dispatch checks."""
    cfg = DMPOConfig(num_envs=4, batch_size=_B, sequence_length=6)
    env_spec = {"obs_size": 7, "action_size": 3}
    nets = make_dmpo_networks(7, 3, cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, env_spec, cfg)
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    batch = {
        "observation": jax.random.normal(k1, (_B, 6, 7)),
        "action": jax.random.uniform(k2, (_B, 6, 3), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(k3, (_B, 6)),
        "discount": jnp.ones((_B, 6)),
        "next_observation": jax.random.normal(k4, (_B, 6, 7)),
    }
    _, metrics = sgd_step(state, batch, nets, make_optimizers(cfg), cfg)
    return metrics


def test_sgd_step_rnn_smoke_under_jit():
    s = _setup()
    nets, optimizers, cfg = s["nets"], s["optimizers"], s["cfg"]
    step_fn = jax.jit(lambda st, b: sgd_step(st, b, nets, optimizers, cfg))
    new_state, metrics = step_fn(s["state"], s["batch"])

    assert "rnn/hidden_staleness" in metrics
    assert "rnn/hidden_abs_mean" in metrics
    for k, v in metrics.items():
        assert np.isfinite(np.asarray(v)).all(), f"non-finite metric {k}"
    assert int(new_state.steps) == 1
    # Policy params actually moved (the update is not a no-op).
    deltas = jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))),
        new_state.policy_params,
        s["state"].policy_params,
    )
    assert max(jax.tree.leaves(deltas)) > 0.0

    # Metric-key parity with the FF path: same keys plus exactly the two
    # rnn/ diagnostics (so downstream logging needs no per-path key lists).
    ff_metrics = _ff_smoke_metrics()
    assert set(metrics.keys()) == set(ff_metrics.keys()) | {
        "rnn/hidden_staleness",
        "rnn/hidden_abs_mean",
    }
    for v in ff_metrics.values():
        assert np.isfinite(np.asarray(v)).all()


def test_rnn_bptt_length_zero_runs_ff_path():
    """rnn_bptt_length=0 (default) must dispatch to the untouched FF body —
    no rnn/ keys, no recurrent schema requirements."""
    metrics = _ff_smoke_metrics()
    assert not any(k.startswith("rnn/") for k in metrics)


def test_sgd_step_rnn_fail_loud_schema_checks():
    s = _setup()

    def run(batch=None, cfg=None):
        return sgd_step(
            s["state"],
            s["batch"] if batch is None else batch,
            s["nets"],
            s["optimizers"],
            s["cfg"] if cfg is None else cfg,
        )

    # Missing stored hidden.
    b = dict(s["batch"])
    del b["policy_hidden"]
    with pytest.raises(ValueError, match="policy_hidden"):
        run(batch=b)

    # Window-length mismatch: batch T = L + _N but cfg expects L + _N + 1.
    with pytest.raises(ValueError, match="n_step"):
        run(cfg=_make_cfg(n=_N + 1, sequence_length=_L + _N))

    # use_n_step must be on.
    with pytest.raises(ValueError, match="use_n_step"):
        run(cfg=_make_cfg(use_n_step=False))

    # Compressed schema required: next_observation must be absent.
    b2 = dict(s["batch"])
    b2["next_observation"] = s["batch"]["observation"]
    with pytest.raises(ValueError, match="next_observation"):
        run(batch=b2)

    # KL-anchor opted in but anchor keys absent (guard mirrored from FF).
    with pytest.raises(ValueError, match="anchor_mu_imit"):
        run(cfg=_make_cfg(kl_anchor_alpha=0.5))
