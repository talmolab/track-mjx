import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs
from track_mjx.agent.dmpo.rollout import collect_rollout


def _flat_normalizer(obs_size: int):
    return running_statistics.init_state(specs.Array((obs_size,), jnp.float32))


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

    traj, _, _norm = collect_rollout(
        env, policy_apply, None, _flat_normalizer(env.obs_size), rng,
        num_envs=8, num_steps=4,
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

    traj, _, _norm = collect_rollout(
        env, policy_apply, None, _flat_normalizer(env.obs_size), rng,
        num_envs=2, num_steps=3,
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
            env, policy_apply, None, _flat_normalizer(env.obs_size), rng,
            num_envs=4, num_steps=3,
        )

    traj, _, _norm = go(jax.random.PRNGKey(0))
    assert traj["observation"].shape == (4, 3, env.obs_size)


def test_collect_rollout_resumes_from_init_state_without_resetting():
    """When init_state is provided, collect_rollout must NOT call env.reset
    and must instead resume from the provided state. Verified by checking
    that the FIRST observation of the second rollout equals the obs in
    final_state from the first rollout (i.e. continuous state).
    """
    env = MockEnv()
    rng = jax.random.PRNGKey(0)

    def policy_apply(params, obs):
        return _GaussianDist(env.action_size)

    # First call: produces final_state.
    traj1, final_state1, _norm1 = collect_rollout(
        env, policy_apply, None, _flat_normalizer(env.obs_size), rng,
        num_envs=4, num_steps=3,
    )

    # Second call with init_state=final_state1 must resume from there.
    rng2 = jax.random.PRNGKey(1)
    traj2, final_state2, _norm2 = collect_rollout(
        env, policy_apply, None, _flat_normalizer(env.obs_size), rng2,
        num_envs=4, num_steps=3,
        init_state=final_state1,
    )

    # The first observation of traj2 must equal final_state1.obs.
    # For MockEnv, reset() returns zeros while final_state1.obs is non-zero
    # (advanced by 3 steps of random actions), so this assertion would fail
    # if reset() had been called.
    np.testing.assert_array_equal(
        np.asarray(traj2["observation"][:, 0]),
        np.asarray(final_state1.obs),
    )


def test_collect_rollout_default_still_resets():
    """When init_state is omitted (default), collect_rollout must reset.
    Verified by checking that the first observation of the rollout equals
    the env's reset obs (which is jnp.zeros for MockEnv).
    """
    env = MockEnv()
    rng = jax.random.PRNGKey(0)

    def policy_apply(params, obs):
        return _GaussianDist(env.action_size)

    traj, _, _norm = collect_rollout(
        env, policy_apply, None, _flat_normalizer(env.obs_size), rng,
        num_envs=4, num_steps=3,
    )
    # MockEnv.reset returns obs=jnp.zeros(obs_size).
    expected = jnp.zeros((4, env.obs_size))
    np.testing.assert_array_equal(np.asarray(traj["observation"][:, 0]), np.asarray(expected))
