"""Per-point weights in the MPO loss and the critic loss.

``batch_weights=None`` is the historical unweighted path. With weights ``w``
(nonnegative, ``[B]``) every mean over the batch is ``sum(w*x)/max(sum(w), 1)``,
constants inside a batch mean (the epsilon and log N terms of the dual losses)
included, so that

* all-ones weights reproduce the unweighted result bit for bit,
* a zero-weight row is exactly a dropped row, and
* a batch in which every point is excluded gives a finite loss with ZERO
  gradient (not NaN, and not the constant a bare epsilon term would leave).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from tests.agent.dmpo.test_losses import REF_OUTPUT, TOL
from track_mjx.agent.dmpo.learner import _critic_loss_fn
from track_mjx.agent.dmpo.losses import MPO
from track_mjx.agent.dmpo.networks import CategoricalCriticHead

tfd = tfp.distributions

N, B, D, OBS = 6, 5, 3, 4


def _mpo(penalty: bool, per_dim: bool = True) -> MPO:
    return MPO(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-5,
        epsilon_penalty=0.1,
        init_log_temperature=2.0,
        init_log_alpha_mean=2.0,
        init_log_alpha_stddev=3.0,
        per_dim_constraining=per_dim,
        action_penalization=penalty,
    )


def _inputs(batch=B, seed=0):
    ks = jax.random.split(jax.random.key(seed), 7)
    return {
        "obs": jax.random.normal(ks[0], (batch, OBS)),
        "w_mu": 0.3 * jax.random.normal(ks[1], (OBS, D)),
        "w_sd": 0.3 * jax.random.normal(ks[2], (OBS, D)),
        "target_loc": jax.random.normal(ks[3], (batch, D)),
        "target_scale": jax.nn.softplus(jax.random.normal(ks[4], (batch, D))) + 0.1,
        "actions": 1.5 * jax.random.normal(ks[5], (N, batch, D)),
        "q_values": jax.random.normal(ks[6], (N, batch)),
    }


def _run(mpo, policy_params, dual_params, data, weights):
    """Loss and stats with a small linear policy whose params get gradients."""
    w_mu, w_sd = policy_params
    online = tfd.MultivariateNormalDiag(
        loc=data["obs"] @ w_mu,
        scale_diag=jax.nn.softplus(data["obs"] @ w_sd) + 0.1,
    )
    target = tfd.MultivariateNormalDiag(
        loc=data["target_loc"], scale_diag=data["target_scale"]
    )
    loss, stats = mpo(
        dual_params, online, target, data["actions"], data["q_values"], weights
    )
    return jnp.sum(loss), (loss, stats)


def _loss_stats_grads(mpo, data, weights):
    dual = mpo.init_params(D, jnp.float32)
    policy = (data["w_mu"], data["w_sd"])
    (_, (loss, stats)), grads = jax.value_and_grad(_run, argnums=(1, 2), has_aux=True)(
        mpo, policy, dual, data, weights
    )
    return loss, stats, grads


def _drop(data, keep):
    out = dict(data)
    out["obs"] = data["obs"][keep]
    out["target_loc"] = data["target_loc"][keep]
    out["target_scale"] = data["target_scale"][keep]
    out["actions"] = data["actions"][:, keep]
    out["q_values"] = data["q_values"][:, keep]
    return out


def _assert_trees(fn, a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        fn(np.asarray(x), np.asarray(y))


def _assert_same(jit, x, y, **kw):
    """Bitwise when run eagerly; within a few ulp under jit, where XLA fuses the
    weighted and unweighted programs differently (the unweighted program itself
    is unchanged and is pinned bitwise by test_legacy_equivalence)."""
    if jit:
        np.testing.assert_allclose(x, y, rtol=2e-6, atol=1e-6, **kw)
    else:
        np.testing.assert_array_equal(x, y, **kw)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("penalty", [True, False], ids=["penalty", "no_penalty"])
@pytest.mark.parametrize("per_dim", [True, False], ids=["per_dim", "joint"])
def test_weights_none_equals_ones(penalty, per_dim, jit, seed):
    mpo = _mpo(penalty, per_dim)
    data = _inputs(seed=seed)
    run = jax.jit(_loss_stats_grads, static_argnums=0) if jit else _loss_stats_grads
    loss0, stats0, grads0 = run(mpo, data, None)
    loss1, stats1, grads1 = run(mpo, data, jnp.ones((B,)))
    _assert_same(jit, np.asarray(loss0), np.asarray(loss1))
    for name, v0 in stats0._asdict().items():
        v1 = getattr(stats1, name)
        if v0 is None:
            assert v1 is None
            continue
        _assert_same(jit, np.asarray(v0), np.asarray(v1), err_msg=name)
    _assert_trees(lambda x, y: _assert_same(jit, x, y), grads0, grads1)


@pytest.mark.parametrize("penalty", [True, False], ids=["penalty", "no_penalty"])
def test_zero_weight_row_equals_dropping_row(penalty):
    mpo = _mpo(penalty)
    data = _inputs()
    weights = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0])
    keep = np.array([0, 1, 3, 4])

    loss_w, stats_w, (g_pol_w, g_dual_w) = _loss_stats_grads(mpo, data, weights)
    loss_d, stats_d, (g_pol_d, g_dual_d) = _loss_stats_grads(
        mpo, _drop(data, keep), None
    )
    np.testing.assert_allclose(np.asarray(loss_w), np.asarray(loss_d), atol=1e-6)
    for name, v_w in stats_w._asdict().items():
        if v_w is None:
            continue
        np.testing.assert_allclose(
            np.asarray(v_w),
            np.asarray(getattr(stats_d, name)),
            rtol=1e-5,
            atol=1e-6,
            err_msg=name,
        )
    # The policy gradient is w.r.t. the shared linear weights, so it must agree
    # with the dropped-row run; the dual gradients likewise.
    close = lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)  # noqa: E731
    _assert_trees(close, g_pol_w, g_pol_d)
    _assert_trees(close, g_dual_w, g_dual_d)


@pytest.mark.parametrize("penalty", [True, False], ids=["penalty", "no_penalty"])
@pytest.mark.parametrize("per_dim", [True, False], ids=["per_dim", "joint"])
def test_all_zero_weights_finite_zero_grad(penalty, per_dim):
    """Review focus: every loss point excluded (e.g. a tiny batch right at a
    time-out) must not produce NaN, nor a constant-driven dual update."""
    mpo = _mpo(penalty, per_dim)
    data = _inputs()
    loss, stats, (g_pol, g_dual) = _loss_stats_grads(mpo, data, jnp.zeros((B,)))
    assert np.all(np.isfinite(np.asarray(loss)))
    for name, v in stats._asdict().items():
        if v is not None:
            assert np.all(np.isfinite(np.asarray(v))), name
    for g in jax.tree.leaves((g_pol, g_dual)):
        assert np.all(np.asarray(g) == 0.0)
    assert float(jnp.sum(loss)) == 0.0


def test_weights_summing_below_one_use_a_unit_denominator():
    """max(sum(w), 1): a fractional total is not renormalised up to 1."""
    mpo = _mpo(penalty=False)
    data = _inputs()
    half = jnp.array([0.5, 0.0, 0.0, 0.0, 0.0])
    one = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    loss_half, *_ = _loss_stats_grads(mpo, data, half)
    loss_one, *_ = _loss_stats_grads(mpo, data, one)
    # Row 0 alone, weighted 1, is the "dropped to one row" loss; weighted 0.5 the
    # state-dependent part is halved and so is the constant part.
    np.testing.assert_allclose(
        np.asarray(loss_half), 0.5 * np.asarray(loss_one), rtol=1e-5, atol=1e-6
    )


def test_vnl_ray_parity_unchanged():
    """The vnl-ray reference still matches, unweighted and with all-ones weights.

    ``test_losses.test_mpo_loss_parity_with_vnl_ray`` covers the unweighted call
    through its own fixture; this pins the same reference through the new
    argument as well.
    """
    import json

    if not REF_OUTPUT.exists():
        pytest.skip(f"missing {REF_OUTPUT}")
    ref = json.loads(REF_OUTPUT.read_text())
    inp = ref["inputs"]
    online = tfd.MultivariateNormalDiag(
        loc=jnp.array(inp["online_loc"]), scale_diag=jnp.array(inp["online_scale"])
    )
    target = tfd.MultivariateNormalDiag(
        loc=jnp.array(inp["target_loc"]), scale_diag=jnp.array(inp["target_scale"])
    )
    actions = jnp.array(inp["sampled_actions"])
    q_values = jnp.array(inp["q_values"])
    mpo = MPO(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-7,
        epsilon_penalty=0.1,
        init_log_temperature=10.0,
        init_log_alpha_mean=10.0,
        init_log_alpha_stddev=1000.0,
        per_dim_constraining=True,
        action_penalization=True,
    )
    params = mpo.init_params(action_dim=actions.shape[-1], dtype=jnp.float32)
    batch = actions.shape[1]
    ref_loss = float(ref["outputs"]["loss"])
    for weights in (None, jnp.ones((batch,))):
        loss, _ = mpo(params, online, target, actions, q_values, weights)
        assert abs(float(np.squeeze(np.asarray(loss))) - ref_loss) < TOL


# --- critic loss -------------------------------------------------------------


class _Critic:
    """Stand-in exposing the ``apply`` the critic loss vmaps over rows."""

    def __init__(self, num_atoms=7):
        self._head = CategoricalCriticHead(num_atoms=num_atoms, vmin=-3.0, vmax=3.0)
        self.num_atoms = num_atoms

    def init(self, key, obs, action):
        return self._head.init(key, jnp.concatenate([obs, action]))

    def apply(self, params, obs, action):
        return self._head.apply(params, jnp.concatenate([obs, action]))


class _Nets:
    def __init__(self):
        self.critic = _Critic()


def _critic_setup(rows=B):
    nets = _Nets()
    ks = jax.random.split(jax.random.key(3), 4)
    obs = jax.random.normal(ks[0], (rows, OBS))
    act = jax.random.normal(ks[1], (rows, D))
    params = nets.critic.init(ks[2], obs[0], act[0])
    target = jax.nn.softmax(jax.random.normal(ks[3], (rows, nets.critic.num_atoms)))
    return nets, params, obs, act, target


def test_critic_loss_weights_none_equals_ones():
    nets, params, obs, act, target = _critic_setup()
    f = jax.value_and_grad(_critic_loss_fn)
    l0, g0 = f(params, nets, obs, act, target)
    l1, g1 = f(params, nets, obs, act, target, jnp.ones((B,)))
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l1))
    _assert_trees(np.testing.assert_array_equal, g0, g1)


def test_critic_loss_zero_weight_row_equals_dropping_row():
    nets, params, obs, act, target = _critic_setup()
    f = jax.value_and_grad(_critic_loss_fn)
    keep = np.array([0, 1, 3, 4])
    lw, gw = f(params, nets, obs, act, target, jnp.array([1.0, 1.0, 0.0, 1.0, 1.0]))
    ld, gd = f(params, nets, obs[keep], act[keep], target[keep])
    np.testing.assert_allclose(np.asarray(lw), np.asarray(ld), atol=1e-6)
    _assert_trees(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6), gw, gd
    )


def test_critic_loss_all_zero_weights_finite_zero_grad():
    nets, params, obs, act, target = _critic_setup()
    loss, grads = jax.value_and_grad(_critic_loss_fn)(
        params, nets, obs, act, target, jnp.zeros((B,))
    )
    assert np.isfinite(float(loss)) and float(loss) == 0.0
    for g in jax.tree.leaves(grads):
        assert np.all(np.asarray(g) == 0.0)
