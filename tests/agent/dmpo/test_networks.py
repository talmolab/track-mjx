import jax
import jax.numpy as jnp
from track_mjx.agent.dmpo.networks import GaussianPolicyHead


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
