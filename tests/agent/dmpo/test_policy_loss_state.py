"""Policy loss state (``policy_loss_state``).

acme's DMPO evaluates the online and target policies, the N samples and the
E-step Q-values at the n-step arrival state, and reuses one sample set for the
critic target and the E-step. With "next" the feed-forward learner does the
same at s_p, p = min(bootstrap index, first termination): s_n mid-episode,
s_k for a window cut at a time-out at k, s_j for a termination at j. A policy
whose mean is the time index of its observation shows which state was used.
The recurrent learner keeps "current" ("auto" resolves to it).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from tests.agent.dmpo import test_learner_rnn as rnn_t
from track_mjx.agent.dmpo import learner
from track_mjx.agent.dmpo.config import DMPOConfig, validate_alignment
from track_mjx.agent.dmpo.learner import (
    _sgd_step_rnn,
    init_training_state,
    make_optimizers,
    sgd_step,
)
from track_mjx.agent.dmpo.networks import CategoricalCriticHead, DMPONetworks

tfd = tfp.distributions

OBS, ACT, B, N_STEP = 3, 2, 4, 3
SCALE = 0.5


class _IndexPolicy(nn.Module):
    """Mean = feature 0 of the obs (the time index), fixed scale."""

    action_size: int

    @nn.compact
    def __call__(self, obs):
        bias = self.param("bias", nn.initializers.zeros, (self.action_size,))
        loc = obs[..., :1] * jnp.ones((self.action_size,)) + bias
        return tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=SCALE * jnp.ones_like(loc)
        )


class _Critic(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        h = jnp.tanh(nn.Dense(8)(jnp.concatenate([obs, action], -1)))
        return CategoricalCriticHead(num_atoms=11, vmin=-5.0, vmax=5.0)(h)


class _Recorder:
    def __init__(self):
        self.seen = []

    def __call__(self, x):
        self.seen.append(np.asarray(x))

    def get(self):
        jax.effects_barrier()
        return self.seen


def _setup(policy_state, target_mode="sample_mixture", **kw):
    fields = dict(
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        num_samples=4,
        n_step=N_STEP,
        use_n_step=True,
        discount=0.9,
        batch_size=B,
        sequence_length=N_STEP + 1,
        store_next_observation=False,
        policy_loss_state=policy_state,
        critic_target_mode=target_mode,
    )
    fields.update(kw)
    cfg = DMPOConfig(**fields)
    nets = DMPONetworks(policy=_IndexPolicy(action_size=ACT), critic=_Critic())
    state = init_training_state(
        jax.random.key(0), nets, {"obs_size": OBS, "action_size": ACT}, cfg
    )
    return cfg, nets, state


def _batch():
    """Rows: mid-episode, time-out at 1, termination at 2, time-out at 0."""
    T = N_STEP + 1
    t = jnp.arange(T, dtype=jnp.float32)
    obs = jnp.zeros((B, T, OBS)).at[..., 0].set(t)
    tr = jnp.zeros((B, T)).at[1, 1].set(1.0).at[3, 0].set(1.0)
    d = (1.0 - tr).at[2, 2].set(0.0)
    ks = jax.random.split(jax.random.key(1), 2)
    return {
        "observation": obs,
        "action": jax.random.uniform(ks[0], (B, T, ACT), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(ks[1], (B, T)),
        "discount": d,
        "truncation": tr,
    }


def _spy_mpo(monkeypatch, rec_mean, rec_actions):
    real_build = learner._build_loss

    class _SpyMPO:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, **kw):
            jax.debug.callback(rec_mean, kw["online_action_distribution"].mean())
            jax.debug.callback(rec_actions, kw["actions"])
            return self.inner(**kw)

    monkeypatch.setattr(learner, "_build_loss", lambda c: _SpyMPO(real_build(c)))


@pytest.mark.parametrize("target_mode", ["sample_mixture", "mean_action"])
@pytest.mark.parametrize(
    "policy_state,expected",
    [("next", [N_STEP, 1, 2, 0]), ("current", [0, 0, 0, 0])],
)
def test_next_evaluates_policy_at_bootstrap_obs(
    policy_state, expected, target_mode, monkeypatch
):
    cfg, nets, state = _setup(policy_state, target_mode)
    rec_mean, rec_actions = _Recorder(), _Recorder()
    _spy_mpo(monkeypatch, rec_mean, rec_actions)
    jax.block_until_ready(sgd_step(state, _batch(), nets, make_optimizers(cfg), cfg))
    (mean,) = rec_mean.get()
    # The fresh normalizer is the identity on these small indices.
    np.testing.assert_allclose(mean[:, 0], expected, atol=1e-6)


def test_next_shares_one_sample_set(monkeypatch):
    """With "next" + the mixture target, bootstrap_samples runs once per step
    and the E-step receives exactly its samples."""
    cfg, nets, state = _setup("next")
    calls, rec_boot = [], _Recorder()
    real = learner.bootstrap_samples

    def counting(*args, **kwargs):
        calls.append(1)
        actions, logits = real(*args, **kwargs)
        jax.debug.callback(rec_boot, actions)
        return actions, logits

    monkeypatch.setattr(learner, "bootstrap_samples", counting)
    rec_mean, rec_actions = _Recorder(), _Recorder()
    _spy_mpo(monkeypatch, rec_mean, rec_actions)
    jax.block_until_ready(
        jax.jit(lambda s, b: sgd_step(s, b, nets, make_optimizers(cfg), cfg))(
            state, _batch()
        )
    )
    assert len(calls) == 1
    (boot,), (mpo,) = rec_boot.get(), rec_actions.get()
    np.testing.assert_array_equal(boot, mpo)


def test_current_draws_separate_target_samples(monkeypatch):
    cfg, nets, state = _setup("current")
    calls, rec_boot = [], _Recorder()
    real = learner.bootstrap_samples

    def counting(*args, **kwargs):
        calls.append(1)
        actions, logits = real(*args, **kwargs)
        jax.debug.callback(rec_boot, actions)
        return actions, logits

    monkeypatch.setattr(learner, "bootstrap_samples", counting)
    rec_mean, rec_actions = _Recorder(), _Recorder()
    _spy_mpo(monkeypatch, rec_mean, rec_actions)
    jax.block_until_ready(sgd_step(state, _batch(), nets, make_optimizers(cfg), cfg))
    assert len(calls) == 1  # the target only; the E-step samples at s_0
    (boot,), (mpo,) = rec_boot.get(), rec_actions.get()
    assert not np.allclose(boot, mpo)


def test_shared_q_values_match_the_critic_mean():
    """The E-step Q-values taken from the shared logits equal the critic
    head's own mean, so "next" rates the samples as "current" would."""
    cfg, nets, state = _setup("next")
    obs = jax.random.normal(jax.random.key(3), (B, OBS))
    actions, logits = learner.bootstrap_samples(
        nets,
        state.target_policy_params,
        state.target_critic_params,
        obs,
        obs,
        cfg,
        jax.random.key(4),
    )
    want = jax.vmap(
        lambda a: jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            state.target_critic_params, obs, learner.critic_action(a, cfg)
        ).mean()
    )(actions)
    np.testing.assert_allclose(learner._expected_value(logits, cfg), want, rtol=1e-6)


def test_next_gathers_the_kl_anchor_at_the_policy_state():
    """The loss-side anchor is read at the same state as the policy: an anchor
    equal to the policy there gives zero KL."""
    cfg, nets, state = _setup("next", kl_anchor_alpha=1.0)
    batch = _batch()
    t = batch["observation"][..., :1]
    batch["anchor_mu_imit"] = t * jnp.ones((ACT,))
    batch["anchor_log_std_imit"] = jnp.full_like(batch["action"], np.log(SCALE))
    _, m = sgd_step(state, batch, nets, make_optimizers(cfg), cfg)
    assert abs(float(m["anchor_kl_mean"])) < 1e-6
    cfg_cur, _, _ = _setup("current", kl_anchor_alpha=1.0)
    _, m_cur = sgd_step(state, batch, nets, make_optimizers(cfg_cur), cfg_cur)
    assert abs(float(m_cur["anchor_kl_mean"])) < 1e-6


def test_next_with_anchor_needs_a_stored_arrival_state():
    """next_observation schema with n == T: no anchor is stored for s_n."""
    cfg, nets, state = _setup(
        "next", kl_anchor_alpha=1.0, sequence_length=N_STEP, store_next_observation=True
    )
    batch = {k: v[:, :N_STEP] for k, v in _batch().items()}
    batch["next_observation"] = _batch()["observation"][:, 1:]
    batch["anchor_mu_imit"] = jnp.zeros_like(batch["action"])
    batch["anchor_log_std_imit"] = jnp.zeros_like(batch["action"])
    with pytest.raises(ValueError, match="anchor"):
        sgd_step(state, batch, nets, make_optimizers(cfg), cfg)


def test_recurrent_auto_is_current():
    assert validate_alignment(DMPOConfig(), recurrent=True).policy_loss_state == (
        "current"
    )
    assert validate_alignment(DMPOConfig(), recurrent=False).policy_loss_state == (
        "next"
    )
    s = rnn_t._setup()  # the defaults: "auto"
    assert s["cfg"].policy_loss_state == "auto"
    _, m = _sgd_step_rnn(s["state"], s["batch"], s["nets"], s["optimizers"], s["cfg"])
    assert np.isfinite(float(m["policy_loss"]))
    bad = rnn_t._make_cfg(policy_loss_state="next")
    with pytest.raises(ValueError, match="recurrent"):
        _sgd_step_rnn(s["state"], s["batch"], s["nets"], s["optimizers"], bad)
