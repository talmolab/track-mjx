"""Latent push cost (train_config.push_cost_beta): loss += beta * 0.5 * mean ||raw||^2 on the PRE-clip push."""

import jax
import jax.numpy as jnp
import numpy as np

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks

A, L, P, T, B = 6, 4, 10, 5, 3


def _nets(cfg):
    return make_dmpo_kl_anchor_networks(
        proprio_size=P,
        task_obs_size=T,
        action_size=A,
        latent_size=L,
        vision_shape=(8, 8, 2),
        cfg=cfg,
        prior_layer_sizes=(16,),
        decoder_layer_sizes=(16,),
        policy_head_layer_sizes=(16,),
        cnn_feature_size=4,
        cnn_channels=(2,),
        mono_channels=1,
        shared_weights=True,
        value_hidden_layer_sizes=(16,),
        residual_mode="sigma_ball",
        residual_scale=3.0,
    )


def _obs(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "vision": jax.random.uniform(k1, (B, 8, 8, 2)),
        "imitation_target": jax.random.normal(k2, (B, T)),
        "proprioception": jax.random.normal(k3, (B, P)),
    }


def _setup(beta):
    cfg = DMPOConfig(num_envs=4, batch_size=B, sequence_length=4, push_cost_beta=beta)
    nets = _nets(cfg)
    obs = _obs(jax.random.key(1))
    single = jax.tree.map(lambda x: x[:1], obs)
    params = nets.policy.init(jax.random.key(0), single)
    # make the zero-init residual head non-zero (a policy that has learned to push), large enough to pass the cap
    head = params["params"]["policy_head"]["residual"]
    head = {
        "kernel": jax.random.normal(jax.random.key(2), head["kernel"].shape) * 2.0,
        "bias": jnp.full(head["bias"].shape, 3.0),
    }
    params = {
        "params": {
            **params["params"],
            "policy_head": {**params["params"]["policy_head"], "residual": head},
        }
    }
    critic = nets.critic.init(jax.random.key(3), single, jnp.zeros((1, A)))
    from track_mjx.agent.dmpo.learner import _build_loss

    dual = _build_loss(cfg).init_params(action_dim=A)
    return cfg, nets, obs, params, critic, dual


def _loss(beta, params=None):
    from track_mjx.agent.dmpo.learner import _policy_loss_fn

    cfg, nets, obs, p0, critic, dual = _setup(beta)
    params = p0 if params is None else params
    return _policy_loss_fn(
        params, dual, nets, obs, p0, critic, cfg, jax.random.key(4)
    ), (nets, obs, p0)


def test_latent_push_is_the_pre_clip_head_output_and_zero_at_init():
    cfg = DMPOConfig(num_envs=4, batch_size=B, sequence_length=4)
    nets = _nets(cfg)
    obs = _obs(jax.random.key(1))
    p = nets.policy.init(jax.random.key(0), jax.tree.map(lambda x: x[:1], obs))
    raw = jax.vmap(lambda o: nets.policy.apply(p, o, method="latent_push"))(obs)
    assert raw.shape == (B, L)
    np.testing.assert_allclose(np.asarray(raw), 0.0, atol=1e-7)


def test_push_cost_adds_beta_half_mean_squared_pre_clip_norm():
    (l0, s0), _ = _loss(0.0)
    (l1, s1), (nets, obs, p) = _loss(0.3)
    raw = jax.vmap(lambda o: nets.policy.apply(p, o, method="latent_push"))(obs)
    want = 0.3 * 0.5 * float(jnp.mean(jnp.sum(raw**2, axis=-1)))
    assert (
        float(jnp.mean(jnp.linalg.norm(raw, axis=-1))) > 3.0
    )  # past the r = 3 cap: the clip alone has no gradient
    np.testing.assert_allclose(float(l1) - float(l0), want, rtol=1e-4)
    np.testing.assert_allclose(float(s1.push_cost), want, rtol=1e-4)
    assert not hasattr(s0, "push_cost")  # beta = 0 returns the MPO stats unchanged (bit-identical configs)
    np.testing.assert_allclose(
        float(s1.push_raw_norm),
        float(jnp.mean(jnp.linalg.norm(raw, axis=-1))),
        rtol=1e-4,
    )


def test_push_cost_gradient_pulls_the_head_inward_even_past_the_cap():
    from track_mjx.agent.dmpo.learner import _policy_loss_fn

    cfg0, nets, obs, p, critic, dual = _setup(0.0)
    cfg1 = DMPOConfig(num_envs=4, batch_size=B, sequence_length=4, push_cost_beta=1.0)

    def f(params, cfg):
        return _policy_loss_fn(
            params, dual, nets, obs, p, critic, cfg, jax.random.key(4)
        )[0]

    g0 = jax.grad(f)(p, cfg0)["params"]["policy_head"]["residual"]["bias"]
    g1 = jax.grad(f)(p, cfg1)["params"]["policy_head"]["residual"]["bias"]
    raw = jax.vmap(lambda o: nets.policy.apply(p, o, method="latent_push"))(obs)
    # d/dbias of 0.5 * mean ||raw||^2 = mean(raw): the cost's own gradient, independent of the clip
    np.testing.assert_allclose(
        np.asarray(g1 - g0), np.asarray(jnp.mean(raw, axis=0)), rtol=1e-3, atol=1e-5
    )
