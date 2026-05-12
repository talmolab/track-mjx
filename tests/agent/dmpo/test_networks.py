import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks import (
    CategoricalCriticHead,
    DMPONetworks,
    GaussianPolicyHead,
    make_dmpo_networks,
)


def test_gaussian_policy_head_init_and_call(rng, env_spec):
    head = GaussianPolicyHead(action_size=env_spec["action_size"])
    obs = jnp.zeros((env_spec["obs_size"],))
    params = head.init(rng, obs)
    dist = head.apply(params, obs)

    assert dist.loc.shape == (env_spec["action_size"],)
    # TFP's MultivariateNormalDiag exposes scale_diag via .parameters / .stddev();
    # use stddev() as the stable public API for byte-for-byte parity tests.
    scale_diag = dist.stddev()
    assert scale_diag.shape == (env_spec["action_size"],)
    # init_scale=0.7 default -> scale ~ 0.7
    assert jnp.allclose(scale_diag, 0.7, atol=1e-3)


def test_gaussian_policy_head_batched(rng, env_spec):
    head = GaussianPolicyHead(action_size=env_spec["action_size"])
    obs = jnp.zeros((4, env_spec["obs_size"]))
    params = head.init(rng, obs[0])
    dist = head.apply(params, obs)
    samples = dist.sample(seed=rng)
    assert samples.shape == (4, env_spec["action_size"])
    assert jnp.all(jnp.isfinite(samples))


def test_categorical_critic_head(rng, env_spec):
    head = CategoricalCriticHead(num_atoms=51, vmin=-150.0, vmax=150.0)
    embedding = jnp.zeros((env_spec["obs_size"] + env_spec["action_size"],))
    params = head.init(rng, embedding)
    dist = head.apply(params, embedding)

    # Logits over num_atoms classes.
    logits = dist.logits_parameter()
    assert logits.shape == (51,)

    # Atom support is exposed for the Bellman projection (Task 10).
    assert head.values.shape == (51,)
    assert jnp.isclose(head.values[0], -150.0)
    assert jnp.isclose(head.values[-1], 150.0)

    # Mean must lie inside the support.
    probs = jax.nn.softmax(logits)
    mean = (probs * head.values).sum()
    assert -150.0 <= float(mean) <= 150.0


def test_categorical_critic_head_batched(rng, env_spec):
    head = CategoricalCriticHead(num_atoms=51, vmin=-150.0, vmax=150.0)
    emb_dim = env_spec["obs_size"] + env_spec["action_size"]
    embedding = jnp.zeros((4, emb_dim))
    params = head.init(rng, embedding[0])
    dist = head.apply(params, embedding)
    logits = dist.logits_parameter()
    assert logits.shape == (4, 51)


def test_make_dmpo_networks(rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    assert isinstance(nets, DMPONetworks)

    obs = jnp.zeros((env_spec["obs_size"],))
    act = jnp.zeros((env_spec["action_size"],))

    pol_params = nets.policy.init(rng, obs)
    crit_params = nets.critic.init(rng, obs, act)

    dist = nets.policy.apply(pol_params, obs)
    q_dist = nets.critic.apply(crit_params, obs, act)
    assert dist.loc.shape == (env_spec["action_size"],)
    assert q_dist.logits_parameter().shape == (cfg.num_atoms,)


def test_make_dmpo_networks_batched(rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    obs = jnp.zeros((4, env_spec["obs_size"]))
    act = jnp.zeros((4, env_spec["action_size"]))
    pol_params = nets.policy.init(rng, obs[0])
    crit_params = nets.critic.init(rng, obs[0], act[0])
    dist = nets.policy.apply(pol_params, obs)
    q_dist = nets.critic.apply(crit_params, obs, act)
    assert dist.loc.shape == (4, env_spec["action_size"])
    assert q_dist.logits_parameter().shape == (4, cfg.num_atoms)


def test_policy_net_block_order_is_dense_layernorm_activation():
    """Each torso block in _PolicyNet must apply Dense -> LayerNorm -> SiLU,
    in that order (matches networks_vision.py:_VisionPolicyNet and Acme's
    LayerNormMLP). Verify by hand-computing the first-layer output under the
    desired ordering and asserting the network's actual output matches.
    """
    cfg = dataclasses.replace(
        DMPOConfig(),
        policy_layer_sizes=(8,),  # one torso layer so we can fully reconstruct.
    )
    nets = make_dmpo_networks(obs_size=5, action_size=3, cfg=cfg)

    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (5,))
    params = nets.policy.init(rng, obs)
    p = params["params"]

    # Hand-compute Dense -> LayerNorm -> SiLU.
    dense = p["Dense_0"]
    pre = obs @ dense["kernel"] + dense["bias"]

    ln = p["LayerNorm_0"]
    mean = pre.mean(axis=-1, keepdims=True)
    var = ((pre - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (pre - mean) / jnp.sqrt(var + 1e-6)  # flax LayerNorm default eps
    normalized = normalized * ln["scale"] + ln["bias"]
    h_expected = jax.nn.silu(normalized)

    # Apply the head's loc Dense to get the final loc.
    head_loc = p["GaussianPolicyHead_0"]["loc"]
    expected_loc = h_expected @ head_loc["kernel"] + head_loc["bias"]

    actual_loc = nets.policy.apply(params, obs).loc
    np.testing.assert_allclose(
        np.asarray(actual_loc),
        np.asarray(expected_loc),
        atol=1e-5,
        err_msg=(
            "_PolicyNet block order should be Dense -> LayerNorm -> SiLU "
            "(matches networks_vision.py)."
        ),
    )


def test_critic_net_block_order_is_dense_layernorm_activation():
    """Each torso block in _CriticNet must apply Dense -> LayerNorm -> SiLU,
    in that order. Mirrors the _PolicyNet test with one critic layer.
    """
    cfg = dataclasses.replace(
        DMPOConfig(),
        critic_layer_sizes=(8,),
    )
    nets = make_dmpo_networks(obs_size=5, action_size=3, cfg=cfg)

    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (5,))
    action = jax.random.normal(jax.random.PRNGKey(1), (3,))
    params = nets.critic.init(rng, obs, action)
    p = params["params"]

    # _CriticNet concatenates [obs, action] before the torso.
    h_in = jnp.concatenate([obs, action], axis=-1)

    dense = p["Dense_0"]
    pre = h_in @ dense["kernel"] + dense["bias"]

    ln = p["LayerNorm_0"]
    mean = pre.mean(axis=-1, keepdims=True)
    var = ((pre - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (pre - mean) / jnp.sqrt(var + 1e-6)
    normalized = normalized * ln["scale"] + ln["bias"]
    h_expected = jax.nn.silu(normalized)

    head_logits = p["CategoricalCriticHead_0"]["logits"]
    expected_logits = h_expected @ head_logits["kernel"] + head_logits["bias"]

    actual_dist = nets.critic.apply(params, obs, action)
    actual_logits = actual_dist.logits_parameter()
    np.testing.assert_allclose(
        np.asarray(actual_logits),
        np.asarray(expected_logits),
        atol=1e-5,
        err_msg=(
            "_CriticNet block order should be Dense -> LayerNorm -> SiLU "
            "(matches networks_vision.py)."
        ),
    )
