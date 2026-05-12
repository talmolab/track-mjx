"""Verify collect_rollout's extra_state_extras parameter threads info keys through."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class _State:
    obs: jax.Array
    info: dict
    done: jax.Array
    reward: jax.Array


class _AnchorStubEnv:
    """Stub env that injects fake anchor_mu_imit / anchor_log_std_imit in state.info."""

    pre_batched = True
    action_size = 4

    def reset(self, keys):
        n = keys.shape[0]
        return _State(
            obs=jnp.zeros((n, 6)),
            info={
                "anchor_mu_imit": jnp.full((n, 4), 0.5),
                "anchor_log_std_imit": jnp.full((n, 4), -1.0),
            },
            done=jnp.zeros((n,)),
            reward=jnp.zeros((n,)),
        )

    def step(self, st, action):
        return st, st.reward


def test_rollout_extra_state_extras_threads_info_keys():
    from brax.training.acme import running_statistics, specs
    from track_mjx.agent.dmpo.rollout import collect_rollout

    def policy_apply(_p, obs):
        class _D:
            def sample(self, seed):
                return jnp.zeros((4,))
        return _D()

    norm = running_statistics.init_state(specs.Array((6,), jnp.float32))
    traj, _final, _new_norm = collect_rollout(
        env=_AnchorStubEnv(),
        policy_apply=policy_apply,
        policy_params=None,
        normalizer_params=norm,
        rng=jax.random.PRNGKey(0),
        num_envs=2,
        num_steps=3,
        extra_state_extras=("anchor_mu_imit", "anchor_log_std_imit"),
    )
    # Trajectory must contain the new keys.
    assert "anchor_mu_imit" in traj
    assert "anchor_log_std_imit" in traj
    # Shape: [num_envs=2, num_steps=3, 4].
    assert traj["anchor_mu_imit"].shape == (2, 3, 4), traj["anchor_mu_imit"].shape
    np.testing.assert_allclose(np.asarray(traj["anchor_mu_imit"]), 0.5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(traj["anchor_log_std_imit"]), -1.0, atol=1e-6)


def test_rollout_default_no_extras_unchanged():
    """Default empty extra_state_extras preserves the original transition schema."""
    from brax.training.acme import running_statistics, specs
    from track_mjx.agent.dmpo.rollout import collect_rollout

    def policy_apply(_p, obs):
        class _D:
            def sample(self, seed):
                return jnp.zeros((4,))
        return _D()

    norm = running_statistics.init_state(specs.Array((6,), jnp.float32))
    traj, _final, _new_norm = collect_rollout(
        env=_AnchorStubEnv(),
        policy_apply=policy_apply,
        policy_params=None,
        normalizer_params=norm,
        rng=jax.random.PRNGKey(0),
        num_envs=2,
        num_steps=3,
    )
    assert "anchor_mu_imit" not in traj
    assert "anchor_log_std_imit" not in traj
    # Original keys still present.
    assert "observation" in traj
    assert "action" in traj
    assert "reward" in traj
