"""acme's action-penalty formula and its single global policy gradient clip.

``action_penalty_formula="acme"`` penalizes only the out-of-bound part of a
sampled action, ``-||a - clip(a, -1, 1)||``. ``policy_grad_clip_scope="global"``
clips the whole trainable policy gradient by one norm before the per-block
Adam, instead of each block by its own; frozen blocks stay out of that norm.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import LEGACY_ALIGNMENT, DMPOConfig
from track_mjx.agent.dmpo.learner import _build_loss
from track_mjx.agent.dmpo.losses import MPO
from track_mjx.agent.dmpo.optim_kl_anchor import (
    clip_by_trainable_global_norm,
    label_param_tree,
    make_kl_anchor_optimizers,
)

tfd = tfp.distributions


class _CostSpy:
    """Captures the penalty cost the loss hands its temperature solve."""

    def __init__(self, monkeypatch):
        from track_mjx.agent.dmpo import losses

        self.costs = []
        real = losses.compute_weights_and_temperature_loss

        def spy(q, eps, temp, w=None):
            self.costs.append(q)
            return real(q, eps, temp, w)

        monkeypatch.setattr(losses, "compute_weights_and_temperature_loss", spy)


def _mpo(formula):
    return MPO(
        epsilon=0.1,
        epsilon_mean=0.01,
        epsilon_stddev=1e-4,
        init_log_temperature=1.0,
        init_log_alpha_mean=1.0,
        init_log_alpha_stddev=1.0,
        action_penalization=True,
        action_penalty_formula=formula,
    )


def _penalty_cost(formula, actions, monkeypatch):
    spy = _CostSpy(monkeypatch)
    mpo = _mpo(formula)
    N, B, D = actions.shape
    dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros((B, D)), scale_diag=jnp.ones((B, D))
    )
    mpo(mpo.init_params(D), dist, dist, actions, jnp.zeros((N, B)))
    q_cost, pen_cost = spy.costs  # Q-values first, then the penalty cost
    del q_cost
    return np.asarray(pen_cost)


def test_acme_penalty_formula(monkeypatch):
    actions = jnp.array(
        [[[0.0, 0.0], [1.0, -1.0], [0.3, -0.9], [1.5, 0.0], [3.0, -5.0]]]
    )
    cost = _penalty_cost("acme", actions, monkeypatch)
    np.testing.assert_allclose(cost[0, :3], 0.0)  # |a| <= 1: no cost
    np.testing.assert_allclose(cost[0, 3], -0.5, rtol=1e-6)
    np.testing.assert_allclose(cost[0, 4], -np.hypot(2.0, 4.0), rtol=1e-6)


def test_vnl_ray_penalty_formula_unchanged(monkeypatch):
    actions = jnp.array([[[0.3, -0.4], [1.5, 0.0]]])
    cost = _penalty_cost("vnl_ray", actions, monkeypatch)
    np.testing.assert_allclose(cost[0], [-0.5, -1.5], rtol=1e-6)


def test_default_and_config_select_the_formula():
    assert _mpo("acme")._action_penalty_formula == "acme"
    assert MPO(0.1, 0.1, 0.1, 1.0, 1.0, 1.0)._action_penalty_formula == "acme"
    legacy = DMPOConfig(**LEGACY_ALIGNMENT)
    assert _build_loss(legacy)._action_penalty_formula == "vnl_ray"
    assert _build_loss(DMPOConfig())._action_penalty_formula == "acme"
    with pytest.raises(ValueError, match="action_penalty_formula"):
        _mpo("l2")


# ---------------------------------------------------------------------------
# Global clip over trainable policy parameters.
# ---------------------------------------------------------------------------


def _tree(prior, decoder, head):
    return {
        "params": {
            "prior": {"k": jnp.asarray(prior, jnp.float32)},
            "decoder": {"k": jnp.asarray(decoder, jnp.float32)},
            "policy_head": {"k": jnp.asarray(head, jnp.float32)},
        }
    }


def _all_trainable(tree):
    return jax.tree.map(lambda _: True, tree)


def test_global_clip_two_blocks_uses_joint_norm():
    """Block norms 30 and 40 -> joint norm 50 > 40: both scaled by 40/50."""
    grads = _tree([30.0, 0.0], [0.0, 40.0], [0.0, 0.0])
    clip = clip_by_trainable_global_norm(40.0, _all_trainable)
    out, _ = clip.update(grads, clip.init(grads))
    np.testing.assert_allclose(out["params"]["prior"]["k"], [24.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(out["params"]["decoder"]["k"], [0.0, 32.0], rtol=1e-6)


def test_global_clip_below_the_norm_is_identity():
    grads = _tree([3.0], [4.0], [0.0])
    clip = clip_by_trainable_global_norm(40.0, _all_trainable)
    out, _ = clip.update(grads, clip.init(grads))
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(grads)):
        np.testing.assert_array_equal(a, b)


def _cfg(scope):
    return DMPOConfig(policy_lr=1e-2, grad_clip=40.0, policy_grad_clip_scope=scope)


def _run(pol_opt, params, grads_seq):
    state = pol_opt.init(params)
    outs = []
    for g in grads_seq:
        updates, state = pol_opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
        outs.append(updates)
    return params, outs


def test_global_clip_single_trainable_block_equals_per_block():
    """Only the head trains (prior and decoder frozen): one block, so the
    global norm is that block's norm and both scopes agree."""
    params = _tree(np.ones(3), np.ones(4), np.ones(5))
    rng = np.random.default_rng(0)
    grads_seq = [
        _tree(*(30.0 * rng.normal(size=s) for s in (3, 4, 5))) for _ in range(4)
    ]
    kw = dict(prior_lr_mult=0.0, decoder_lr_mult=0.0, policy_head_lr_mult=1.0)
    p_glob, u_glob = _run(
        make_kl_anchor_optimizers(_cfg("global"), **kw)[0], params, grads_seq
    )
    p_blk, u_blk = _run(
        make_kl_anchor_optimizers(_cfg("per_block"), **kw)[0], params, grads_seq
    )
    for a, b in zip(jax.tree.leaves(p_glob), jax.tree.leaves(p_blk)):
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)
    for ug, ub in zip(u_glob, u_blk):
        for a, b in zip(jax.tree.leaves(ug), jax.tree.leaves(ub)):
            np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)


def test_global_clip_differs_from_per_block_with_two_trainable_blocks():
    params = _tree(np.ones(2), np.ones(2), np.ones(2))
    grads = [_tree([30.0, 0.0], [0.0, 40.0], [0.0, 0.0])]
    kw = dict(prior_lr_mult=1.0, decoder_lr_mult=1.0)
    clip_g = clip_by_trainable_global_norm(
        40.0,
        lambda t: jax.tree.map(lambda lbl: lbl != "other", label_param_tree(t)),
    )
    joint, _ = clip_g.update(grads[0], clip_g.init(grads[0]))
    np.testing.assert_allclose(joint["params"]["prior"]["k"], [24.0, 0.0], rtol=1e-6)
    # Through the optimizers: Adam's first step is scale-free, so compare
    # the second-moment state instead (it sees the clipped gradient).
    st_g = make_kl_anchor_optimizers(_cfg("global"), **kw)[0]
    st_b = make_kl_anchor_optimizers(_cfg("per_block"), **kw)[0]
    _, s_g = st_g.update(grads[0], st_g.init(params), params)
    _, s_b = st_b.update(grads[0], st_b.init(params), params)
    nu = [x for x in jax.tree.leaves(s_g) if x.shape == (2,)]
    nu_b = [x for x in jax.tree.leaves(s_b) if x.shape == (2,)]
    assert not all(np.allclose(a, b) for a, b in zip(nu, nu_b))


def test_frozen_block_nan_does_not_poison():
    """A frozen block's NaN gradient stays out of the global norm: the
    trainable update is finite and the frozen block's update is exactly 0."""
    pol_opt, _, _ = make_kl_anchor_optimizers(
        _cfg("global"), prior_lr_mult=0.0, decoder_lr_mult=1.0
    )
    params = _tree(np.ones(2), np.ones(2), np.ones(2))
    grads = _tree([np.nan, 1.0], [50.0, 0.0], [0.0, 30.0])
    updates, _ = pol_opt.update(grads, pol_opt.init(params), params)
    np.testing.assert_array_equal(updates["params"]["prior"]["k"], 0.0)
    for block in ("decoder", "policy_head"):
        assert np.isfinite(np.asarray(updates["params"][block]["k"])).all(), block
