import jax
import jax.numpy as jnp
from track_mjx.agent.dmpo.networks import CategoricalCriticHead, GaussianPolicyHead


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
