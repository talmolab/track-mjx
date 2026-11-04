"""N-sample mixture critic target (``critic_target_mode="sample_mixture"``).

acme's DMPO bootstraps from the mean, in probability space, of the target
critic's distributions over N actions sampled from the target policy at the
bootstrap state. Covered: the mixture against a brute-force per-sample
projection, N=1 with zero noise against the mean-action target, the
feed-forward and recurrent steps against a hand reference built from injected
noise (the recurrent one fed the stored hidden at the bootstrap step), and a
privileged critic paired with the vector of the bootstrap state.
"""

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.agent.dmpo import test_learner_rnn as rnn_t
from tests.agent.dmpo.test_truncation_bootstrap import _ref_window
from track_mjx.agent.dmpo.categorical import categorical_l2_project
from track_mjx.agent.dmpo.config import LEGACY_ALIGNMENT, DMPOConfig
from track_mjx.agent.dmpo.learner import (
    _critic_loss_fn,
    _normalize_obs,
    _per_point_nstep_returns,
    _sgd_step_rnn,
    bootstrap_samples,
    init_training_state,
    make_optimizers,
    mixture_categorical_target,
    sgd_step,
)
from track_mjx.agent.dmpo.networks import (
    CategoricalCriticHead,
    DMPONetworks,
    GaussianPolicyHead,
    make_dmpo_networks,
)

MIX_ONLY = {**LEGACY_ALIGNMENT, "critic_target_mode": "sample_mixture"}
OBS, ACT = 5, 2


def _flat_cfg(**kw):
    base = dict(
        policy_layer_sizes=(16,),
        critic_layer_sizes=(16,),
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        num_samples=4,
        n_step=3,
        use_n_step=True,
        discount=0.9,
        batch_size=4,
        sequence_length=4,
        store_next_observation=False,
    )
    base.update(kw)
    return DMPOConfig(**base)


def _flat_setup(cfg):
    nets = make_dmpo_networks(OBS, ACT, cfg)
    state = init_training_state(
        jax.random.key(0), nets, {"obs_size": OBS, "action_size": ACT}, cfg
    )
    # Spread the policy and sharpen the target critic so the sampled actions
    # actually move the bootstrap distribution.
    leaves, tdef = jax.tree_util.tree_flatten(state.policy_params)
    keys = jax.random.split(jax.random.key(7), len(leaves))
    pol = jax.tree_util.tree_unflatten(
        tdef, [x + 0.5 * jax.random.normal(k, x.shape) for x, k in zip(leaves, keys)]
    )
    state = state._replace(
        policy_params=pol,
        target_policy_params=pol,
        policy_opt_state=make_optimizers(cfg)[0].init(pol),
        target_critic_params=jax.tree.map(lambda x: 4.0 * x, state.critic_params),
    )
    return nets, state


def _flat_batch(B=4, T=4, seed=3):
    ks = jax.random.split(jax.random.key(seed), 4)
    return {
        "observation": jax.random.normal(ks[0], (B, T, OBS)),
        "action": jax.random.uniform(ks[1], (B, T, ACT), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(ks[2], (B, T)),
        "discount": jnp.ones((B, T)).at[2, 1].set(0.0),
        "truncation": jnp.zeros((B, T)),
    }


def test_mixture_equals_bruteforce_per_sample_projection():
    cfg = _flat_cfg()
    N, B = 6, 5
    ks = jax.random.split(jax.random.key(0), 3)
    logits = 2.0 * jax.random.normal(ks[0], (N, B, cfg.num_atoms))
    rewards = jax.random.normal(ks[1], (B,))
    discounts = jnp.array([1.0, 0.5, 0.0, 0.81, 0.9])
    got = mixture_categorical_target(logits, rewards, discounts, cfg)

    atoms = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_atoms)
    project = jax.vmap(functools.partial(categorical_l2_project, z_q=atoms))
    z = rewards[:, None] + cfg.discount * discounts[:, None] * atoms
    per_sample = [project(z, jax.nn.softmax(logits[i], -1)) for i in range(N)]
    want = jnp.mean(jnp.stack(per_sample), axis=0)
    np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(jnp.sum(got, -1), 1.0, atol=1e-6)


def test_mixture_N1_with_zero_eps_equals_mean_action():
    """One sample at zero noise is the mean action: the whole step matches
    the mean-action step (the policy loss keeps its own key)."""
    nets, state = _flat_setup(_flat_cfg(num_samples=1, **LEGACY_ALIGNMENT))
    batch = _flat_batch()
    out = {}
    for mode in ("mean_action", "sample_mixture"):
        cfg = _flat_cfg(
            num_samples=1, **{**LEGACY_ALIGNMENT, "critic_target_mode": mode}
        )
        eps = jnp.zeros((1, 4, ACT))
        out[mode] = jax.jit(
            lambda s, b, e, cfg=cfg: sgd_step(
                s, b, nets, make_optimizers(cfg), cfg, sample_eps=e
            )
        )(state, batch, eps)
    (s_a, m_a), (s_b, m_b) = out["mean_action"], out["sample_mixture"]
    for k in m_a:
        np.testing.assert_allclose(m_b[k], m_a[k], rtol=1e-6, atol=1e-7, err_msg=k)
    for a, b in zip(
        jax.tree.leaves(s_a.critic_params), jax.tree.leaves(s_b.critic_params)
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_ff_mixture_matches_hand_reference():
    cfg = _flat_cfg(**MIX_ONLY)
    nets, state = _flat_setup(cfg)
    batch = _flat_batch()
    n = cfg.n_step
    eps = jax.random.normal(jax.random.key(11), (cfg.num_samples, 4, ACT))
    norm = lambda x: _normalize_obs(x, state.normalizer_params)  # noqa: E731

    r, d = np.asarray(batch["reward"]), np.asarray(batch["discount"])
    R = np.array(
        [sum(0.9**j * np.prod(d[b, :j]) * r[b, j] for j in range(n)) for b in range(4)]
    )
    D = 0.9 ** (n - 1) * np.prod(d[:, :n], axis=1)
    s_n = norm(batch["observation"][:, n])
    dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        state.target_policy_params, s_n
    )
    samples = dist.mean()[None] + dist.stddev()[None] * eps
    logits = jnp.stack(
        [
            jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
                state.target_critic_params, s_n, samples[i]
            ).logits_parameter()
            for i in range(cfg.num_samples)
        ]
    )
    target = mixture_categorical_target(
        logits, jnp.asarray(R, jnp.float32), jnp.asarray(D, jnp.float32), cfg
    )
    ref = _critic_loss_fn(
        state.critic_params,
        nets,
        norm(batch["observation"][:, 0]),
        batch["action"][:, 0],
        target,
    )
    _, m = jax.jit(
        lambda s, b, e: sgd_step(s, b, nets, make_optimizers(cfg), cfg, sample_eps=e)
    )(state, batch, eps)
    np.testing.assert_allclose(float(m["critic_loss"]), float(ref), rtol=1e-6)
    # And the injected noise matters: a different draw moves the loss.
    _, m2 = sgd_step(state, batch, nets, make_optimizers(cfg), cfg, sample_eps=-eps)
    assert abs(float(m2["critic_loss"]) - float(ref)) > 1e-6


def test_bootstrap_samples_shapes_and_vision_map():
    cfg = _flat_cfg(num_samples=3)
    nets, state = _flat_setup(cfg)
    obs = jax.random.normal(jax.random.key(1), (4, OBS))
    actions, logits = bootstrap_samples(
        nets,
        state.target_policy_params,
        state.target_critic_params,
        obs,
        obs,
        cfg,
        jax.random.key(2),
    )
    assert actions.shape == (3, 4, ACT)
    assert logits.shape == (3, 4, cfg.num_atoms)


# ---------------------------------------------------------------------------
# Privileged critic: each sample is rated with the vector of the BOOTSTRAP
# state, never the loss-point one.
# ---------------------------------------------------------------------------

PRIV, KEY = 6, "critic_priv"


class _DictPolicy(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, obs):
        x = jnp.concatenate([obs["imitation_target"], obs["proprioception"]], -1)
        return GaussianPolicyHead(action_size=self.action_size)(nn.Dense(8)(x))


class _PrivOnlyCritic(nn.Module):
    """Output depends on obs["privileged"] alone."""

    num_atoms: int
    vmin: float
    vmax: float
    privileged_dim: int = PRIV

    @nn.compact
    def __call__(self, obs, action):
        del action
        h = jnp.tanh(nn.Dense(8)(obs["privileged"]))
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_privileged_paired_at_bootstrap_index(n):
    """Default alignment switches (mixture target, time-out cuts). Row 1 times
    out at step 1, so for n >= 2 its bootstrap index (1) differs from the
    others' (n): the target must use the vector stored at that index."""
    T, B = 4, 4
    cfg = DMPOConfig(
        num_atoms=11,
        vmin=-5.0,
        vmax=5.0,
        num_samples=3,
        n_step=n,
        use_n_step=True,
        discount=0.9,
        batch_size=B,
        sequence_length=T,
        store_next_observation=False,
        critic_privileged_key=KEY,
    )
    nets = DMPONetworks(
        policy=_DictPolicy(action_size=ACT),
        critic=_PrivOnlyCritic(num_atoms=11, vmin=-5.0, vmax=5.0),
    )
    spec = {
        "obs_template": {
            "imitation_target": jnp.zeros((3,)),
            "proprioception": jnp.zeros((4,)),
        },
        "action_size": ACT,
    }
    state = init_training_state(jax.random.key(0), nets, spec, cfg)
    state = state._replace(
        target_critic_params=jax.tree.map(lambda x: 4.0 * x, state.critic_params)
    )
    ks = jax.random.split(jax.random.key(5), 5)
    tr = jnp.zeros((B, T)).at[1, 1].set(1.0)
    batch = {
        "observation": {
            "imitation_target": jax.random.normal(ks[0], (B, T, 3)),
            "proprioception": jax.random.normal(ks[1], (B, T, 4)),
        },
        "action": jax.random.uniform(ks[2], (B, T, ACT), minval=-0.9, maxval=0.9),
        "reward": jax.random.normal(ks[3], (B, T)),
        "discount": 1.0 - tr,
        "truncation": tr,
        KEY: 3.0 * jax.random.normal(ks[4], (B, T, PRIV)),
    }
    r, d, t_ = (np.asarray(batch[k]) for k in ("reward", "discount", "truncation"))
    refs = [_ref_window(r[b], d[b], t_[b], cfg.discount, n) for b in range(B)]
    R, D, boot, w = (np.asarray(x, np.float32) for x in zip(*refs))
    boot = boot.astype(int)
    assert boot[1] == (1 if n >= 2 else n)

    rows = np.arange(B)

    def critic_logits(params, idx):
        o = {"privileged": batch[KEY][rows, idx]}
        return jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            params, o, jnp.zeros((B, ACT))
        ).logits_parameter()

    target = mixture_categorical_target(
        critic_logits(state.target_critic_params, boot)[None],
        jnp.asarray(R),
        jnp.asarray(D),
        cfg,
    )
    online = jax.nn.log_softmax(critic_logits(state.critic_params, np.zeros(B, int)))
    per_row = jnp.sum(target * online, -1)
    ref = -jnp.sum(w * per_row) / max(float(w.sum()), 1.0)

    _, m = jax.jit(lambda s, b: sgd_step(s, b, nets, make_optimizers(cfg), cfg))(
        state, batch
    )
    np.testing.assert_allclose(float(m["critic_loss"]), float(ref), rtol=1e-6)


# ---------------------------------------------------------------------------
# Recurrent: samples drawn from the target policy given the STORED hidden at
# the bootstrap step.
# ---------------------------------------------------------------------------


def test_rnn_mixture_uses_stored_boot_hidden():
    s = rnn_t._setup(**MIX_ONLY)
    state, nets, cfg, batch = s["state"], s["nets"], s["cfg"], s["batch"]
    state = state._replace(
        target_critic_params=jax.tree.map(lambda x: 4.0 * x, state.critic_params)
    )
    L, n, N = cfg.rnn_bptt_length, cfg.n_step, cfg.num_samples
    B = rnn_t._B
    eps = jax.random.normal(jax.random.key(9), (N, B * L, rnn_t._ACT))

    R, D, _, _ = _per_point_nstep_returns(
        batch["reward"], batch["discount"], cfg.discount, L, n
    )
    rows = np.arange(B)[:, None]
    starts = np.tile(np.arange(L), (B, 1))
    obs = _normalize_obs(batch["observation"], state.normalizer_params)

    def at(cols):
        return jax.tree.map(lambda x: x[rows, cols], obs)

    h_boot = tuple(
        x.astype(jnp.float32)[rows, starts + n] for x in batch["policy_hidden"]
    )

    def raw(params, o, h):
        return nets.policy.apply(params, o, h, method="raw")

    mu, scale, _ = jax.vmap(
        jax.vmap(lambda o, h: raw(state.target_policy_params, o, h))
    )(at(starts + n), h_boot)
    samples = mu[None] + scale[None] * eps.reshape(N, B, L, -1)
    rate = jax.vmap(
        jax.vmap(
            lambda o, a: nets.critic.apply(
                state.target_critic_params, o, a
            ).logits_parameter()
        )
    )
    logits = jnp.stack([rate(at(starts + n), samples[i]) for i in range(N)])
    target = mixture_categorical_target(
        logits.reshape(N, B * L, -1), R.reshape(-1), D.reshape(-1), cfg
    )
    obs_q = jax.tree.map(lambda x: x.reshape((B * L, *x.shape[2:])), at(starts))
    act_q = batch["action"][rows, starts].reshape(B * L, -1)
    ref = _critic_loss_fn(state.critic_params, nets, obs_q, act_q, target)

    step = jax.jit(
        lambda st, b, e: _sgd_step_rnn(st, b, nets, s["optimizers"], cfg, sample_eps=e)
    )
    _, m = step(state, batch, eps)
    np.testing.assert_allclose(float(m["critic_loss"]), float(ref), rtol=1e-6)

    # A zeroed bootstrap hidden gives a different target: the stored one is used.
    zero_h = jax.tree.map(jnp.zeros_like, batch["policy_hidden"])
    _, m0 = step(state, dict(batch, policy_hidden=zero_h), eps)
    assert abs(float(m0["critic_loss"]) - float(ref)) > 1e-6
