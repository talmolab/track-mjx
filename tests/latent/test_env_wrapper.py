"""Tests for LatentMimicEnvWrapper using a stubbed inner env."""
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

V8_BEST = Path(os.path.expandvars(
    "$HOME/Desktop/SalkResearch/track-mjx/checkpoints/latent_prior_v8/best"
))


class _StubData:
    def __init__(self, n_joints=67):
        self.qpos = jnp.zeros((7 + n_joints,))
        self.qvel = jnp.zeros((6 + n_joints,))


class _StubState:
    def __init__(self, data, obs, reward, done, info, metrics):
        self.data = data
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info
        self.metrics = metrics

    def replace(self, **kwargs):
        out = _StubState(self.data, self.obs, self.reward, self.done,
                         self.info, self.metrics)
        for k, v in kwargs.items():
            setattr(out, k, v)
        return out


class _StubEnv:
    """Mimics RodentImitation's surface for the wrapper tests."""
    def __init__(self, n_joints=67):
        self.n_joints = n_joints

    @property
    def action_size(self):
        return self.n_joints

    def _make_state(self, qpos, qvel):
        # mimic the nested obs the real env returns
        prop = {
            "joint_angles": jnp.zeros((self.n_joints,)),
            "joint_ang_vels": jnp.zeros((self.n_joints,)),
            "prev_action": jnp.zeros((self.n_joints,)),
        }
        obs = {"state": {"task_obs": jnp.zeros((4,)), "proprioception": prop}}
        info = {
            "start_frame": 0,
            "reference_clip": 0,
            "prev_action": jnp.zeros((self.n_joints,)),
        }
        return _StubState(_StubData(self.n_joints), obs, jnp.float32(0.0),
                          jnp.float32(0.0), info, {})

    def reset(self, rng):
        return self._make_state(None, None)

    def step(self, state, action):
        # Pretend qpos drifts by 1e-3 per step so windows aren't all identical.
        # Also rebuild obs from scratch like the real env does (the wrapper
        # replaced state.obs with a flat dict, so we must not propagate that).
        new_data = _StubData(self.n_joints)
        new_data.qpos = state.data.qpos + 1e-3
        new_data.qvel = state.data.qvel + 1e-3
        prop = {
            "joint_angles": jnp.zeros((self.n_joints,)),
            "joint_ang_vels": jnp.zeros((self.n_joints,)),
            "prev_action": action,
        }
        new_obs = {"state": {"task_obs": jnp.zeros((4,)), "proprioception": prop}}
        return state.replace(data=new_data, obs=new_obs)


@pytest.fixture(scope="module")
def wrapped():
    if not V8_BEST.exists():
        pytest.skip(f"v8 best ckpt not present at {V8_BEST}")
    from track_mjx.agent.latent_ppo.env_wrapper import LatentMimicEnvWrapper
    inner = _StubEnv(n_joints=67)
    return LatentMimicEnvWrapper(
        env=inner, prior_dir=str(V8_BEST), n_joints=67,
        w_r=0.01, history_len=5,
    )


def test_reset_exposes_three_obs_keys(wrapped):
    state = wrapped.reset(jax.random.PRNGKey(0))
    assert set(state.obs.keys()) == {"proprioception", "o_history", "z_target"}
    assert state.obs["z_target"].shape == (60,)
    assert state.obs["o_history"].shape == (5 * 3 * 67,)
    assert state.obs["proprioception"].shape == (3 * 67,)
    # initial reward should be 1.0 (perfect mimic by construction)
    assert float(state.reward) == 1.0


def test_step_replaces_reward_with_r_mimic_in_unit_interval(wrapped):
    state = wrapped.reset(jax.random.PRNGKey(0))
    action = jnp.zeros((67,))
    state = wrapped.step(state, action)
    r = float(state.reward)
    assert 0.0 < r <= 1.0, f"r_mimic out of range: {r}"
    assert "r_mimic" in state.info
    assert "mimic_kl" in state.info
    assert state.obs["z_target"].shape == (60,)


def test_buffers_propagate_across_steps(wrapped):
    state = wrapped.reset(jax.random.PRNGKey(0))
    action = jnp.zeros((67,))
    s1 = wrapped.step(state, action)
    s2 = wrapped.step(s1, action)
    # motion_window's first row should be slightly different across steps
    buf1 = s1.info["latent_buf"]
    buf2 = s2.info["latent_buf"]
    # NOT identical
    assert not bool(jnp.allclose(buf1.motion_window[-1], buf2.motion_window[-1]))


def test_action_size_passthrough(wrapped):
    assert wrapped.action_size == 67
