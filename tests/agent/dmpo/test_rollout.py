import dataclasses
import jax
import jax.numpy as jnp
from track_mjx.agent.dmpo.rollout import collect_rollout


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class MockEnvState:
    obs: jnp.ndarray
    done: jnp.ndarray

    def tree_flatten(self):
        return (self.obs, self.done), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obs, done = children
        return cls(obs=obs, done=done)


class MockEnv:
    obs_size = 8
    action_size = 4

    def reset(self, rng):
        return MockEnvState(obs=jnp.zeros(self.obs_size), done=jnp.zeros(()))

    def step(self, state, action):
        # New obs = old obs + action[:obs_size]; reward = ||action||; never-done.
        new_obs = state.obs + jnp.pad(action, (0, max(0, self.obs_size - action.size)))[:self.obs_size]
        new_state = MockEnvState(obs=new_obs, done=state.done)
        reward = jnp.linalg.norm(action)
        return new_state, reward


class _GaussianDist:
    """Tiny stand-in for a tfd distribution exposing only ``.sample(seed=...)``."""

    def __init__(self, action_size: int):
        self._action_size = action_size

    def sample(self, seed):
        return jax.random.normal(seed, (self._action_size,))


class _ConstantDist:
    """Distribution that returns a fixed value regardless of seed."""

    def __init__(self, value: jnp.ndarray):
        self._value = value

    def sample(self, seed):
        return self._value


def test_collect_rollout_shapes():
    env = MockEnv()
    rng = jax.random.PRNGKey(0)

    def policy_apply(params, obs):
        return _GaussianDist(env.action_size)

    traj, _ = collect_rollout(
        env, policy_apply, None, rng, num_envs=8, num_steps=4
    )

    assert traj["observation"].shape == (8, 4, env.obs_size)
    assert traj["action"].shape == (8, 4, env.action_size)
    assert traj["reward"].shape == (8, 4)
    assert traj["discount"].shape == (8, 4)
    assert traj["next_observation"].shape == (8, 4, env.obs_size)


def test_collect_rollout_stores_raw_pretanh_action():
    """The action stored in the trajectory must be the raw (unbounded) Gaussian
    sample, NOT the post-tanh action that was sent to the env."""
    env = MockEnv()
    rng = jax.random.PRNGKey(0)

    LARGE = 100.0  # Big enough that tanh saturates.

    def policy_apply(params, obs):
        return _ConstantDist(LARGE * jnp.ones((env.action_size,)))

    traj, _ = collect_rollout(
        env, policy_apply, None, rng, num_envs=2, num_steps=3
    )
    # Stored action ~ LARGE (not 1.0).
    assert jnp.all(traj["action"] > 1.0)
    assert jnp.allclose(traj["action"], LARGE, atol=1e-3)


def test_collect_rollout_is_jittable():
    env = MockEnv()

    def policy_apply(params, obs):
        return _GaussianDist(env.action_size)

    @jax.jit
    def go(rng):
        return collect_rollout(
            env, policy_apply, None, rng, num_envs=4, num_steps=3
        )

    traj, _ = go(jax.random.PRNGKey(0))
    assert traj["observation"].shape == (4, 3, env.obs_size)
