"""collect_rollout reads + updates the normalizer."""
import jax
import jax.numpy as jnp
import flax.linen as nn

from track_mjx.agent.dmpo.rollout import collect_rollout


class _StubEnv:
    """Toy env: state.obs is a flat 4-vec, action is 2-vec, reward = 1."""
    action_size = 2
    observation_size = 4

    class _State:
        def __init__(self, obs, done):
            self.obs = obs
            self.done = done

    def reset(self, rng):
        return self._State(obs=jnp.ones((4,)) * 5.0, done=jnp.array(0.0))

    def step(self, state, action):
        new_obs = state.obs + 0.1
        new_state = self._State(obs=new_obs, done=jnp.array(0.0))
        return new_state, jnp.array(1.0)


# Register _State as a pytree so jax.vmap / lax.scan can treat it as such.
import jax.tree_util as jtu
jtu.register_pytree_node(
    _StubEnv._State,
    lambda s: ((s.obs, s.done), None),
    lambda _, c: _StubEnv._State(*c),
)


class _StubPolicy(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, obs):
        # Return a degenerate Gaussian regardless of obs.
        from tensorflow_probability.substrates import jax as tfp
        loc = jnp.zeros((self.action_size,))
        scale = jnp.ones((self.action_size,))
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)


def test_collect_rollout_returns_updated_normalizer():
    """After a rollout, normalizer_params reflects observed stats."""
    from brax.training.acme import running_statistics, specs

    env = _StubEnv()
    policy = _StubPolicy(action_size=2)
    rng = jax.random.PRNGKey(0)
    policy_params = policy.init(rng, jnp.zeros((4,)))
    normalizer_params = running_statistics.init_state(
        specs.Array((4,), jnp.float32)
    )
    assert int(normalizer_params.count) == 0

    traj, final_state, new_normalizer = collect_rollout(
        env=env,
        policy_apply=policy.apply,
        policy_params=policy_params,
        normalizer_params=normalizer_params,
        rng=rng,
        num_envs=2,
        num_steps=5,
        init_state=None,
    )
    # Stub observation is constant 5.0 → mean ~5.0 after update.
    assert int(new_normalizer.count) == 2 * 5
    assert jnp.allclose(new_normalizer.mean, 5.0, atol=0.5)
    # Trajectory stored raw obs (not normalized).
    assert traj["observation"].shape == (2, 5, 4)
