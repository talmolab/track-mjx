"""Replay-buffer compression: drop next_observation, store vision as uint8.

The buffer lives on-device inside the fused JIT, so depth is GPU-memory bound:
~19.5 KB/transition uncompressed (two f32 copies of the 32x32x2 vision obs)
gave only ~4 rollouts of history at 400k transitions. The two flags cut this
~4.8x. Correctness rests on two facts these tests pin:

  1. observation[t+1] IS next_observation[t] bit-for-bit in flashbax
     trajectory storage (auto-reset swaps obs on the terminal step itself;
     the time axis is continuous across rollout adds) -- so the learner can
     bootstrap from observation[:, n] and get IDENTICAL numbers.
  2. The renderer unpacks 8-bit channels to f32, so uint8 storage plus
     normalize_dict_obs's /255 dequantization loses at most 1/510 of range.

Run CPU-only (as the whole suite): JAX_PLATFORMS=cpu.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.learner import sgd_step
from track_mjx.agent.dmpo.rollout import collect_rollout
from track_mjx.agent.observation_utils import (
    init_dict_normalizer,
    normalize_dict_obs,
)

from tests.agent.dmpo.test_train_dmpo_fused import _setup


# ---------------------------------------------------------------------------
# Rollout schema
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _DictObsState:
    obs: dict
    done: jnp.ndarray
    t: jnp.ndarray

    def tree_flatten(self):
        return (self.obs, self.done, self.t), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obs, done, t = children
        return cls(obs=children[0], done=children[1], t=children[2])


class _DictObsEnv:
    """Env with run_gap-shaped dict obs: proprio + imitation_target + vision.

    Vision pixels vary deterministically with time so quantization and the
    obs[t+1] == next_obs[t] identity are both checkable on real values.
    """

    P, I, HWC = 5, 3, (4, 4, 2)
    action_size = 4

    def _obs(self, t):
        v = (jnp.arange(np.prod(self.HWC), dtype=jnp.float32).reshape(self.HWC)
             + t * 0.377) % 1.0
        return {
            "proprioception": jnp.full((self.P,), t, jnp.float32),
            "imitation_target": jnp.full((self.I,), -t, jnp.float32),
            "vision": v,
        }

    def reset(self, rng):
        return _DictObsState(obs=self._obs(0.0), done=jnp.zeros(()), t=jnp.zeros(()))

    def step(self, state, action):
        t = state.t + 1.0
        new = _DictObsState(obs=self._obs(t), done=state.done, t=t)
        return new, jnp.asarray(1.0)


def _dict_normalizer():
    env = _DictObsEnv()
    return init_dict_normalizer(env._obs(0.0))


def _const_policy(env):
    class _Dist:
        def sample(self, seed):
            return jnp.zeros((env.action_size,))

    return lambda params, obs: _Dist()


def _roll(store_next, vision_u8, num_steps=3):
    env = _DictObsEnv()
    traj, _, _ = collect_rollout(
        env, _const_policy(env), None, _dict_normalizer(), jax.random.PRNGKey(0),
        num_envs=2, num_steps=num_steps,
        store_next_observation=store_next, vision_uint8=vision_u8,
    )
    return traj


def test_default_schema_is_unchanged():
    traj = _roll(store_next=True, vision_u8=False)
    assert "next_observation" in traj
    assert traj["observation"]["vision"].dtype == jnp.float32


def test_next_observation_is_dropped_when_disabled():
    traj = _roll(store_next=False, vision_u8=False)
    assert "next_observation" not in traj
    assert set(traj) == {"observation", "action", "reward", "discount"}


def test_vision_is_quantized_and_other_keys_stay_f32():
    full = _roll(store_next=True, vision_u8=False)
    q = _roll(store_next=True, vision_u8=True)
    assert q["observation"]["vision"].dtype == jnp.uint8
    assert q["next_observation"]["vision"].dtype == jnp.uint8
    assert q["observation"]["proprioception"].dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(q["observation"]["vision"]),
        np.round(np.clip(np.asarray(full["observation"]["vision"]), 0, 1) * 255),
    )


def test_obs_shift_identity_the_whole_scheme_rests_on():
    """observation[:, t+1] must equal next_observation[:, t] exactly."""
    traj = _roll(store_next=True, vision_u8=False, num_steps=4)
    for key in ("proprioception", "imitation_target", "vision"):
        np.testing.assert_array_equal(
            np.asarray(traj["observation"][key][:, 1:]),
            np.asarray(traj["next_observation"][key][:, :-1]),
            err_msg=f"obs[t+1] != next_obs[t] for {key}",
        )


def test_uint8_roundtrip_error_is_bounded():
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 1, size=(4, 4, 2)).astype(np.float32)
    obs = {
        "proprioception": jnp.zeros((5,)),
        "imitation_target": jnp.zeros((3,)),
        "vision": jnp.round(jnp.asarray(v) * 255).astype(jnp.uint8),
    }
    out = normalize_dict_obs(obs, _dict_normalizer())
    assert out["vision"].dtype == jnp.float32
    err = np.abs(np.asarray(out["vision"]) - v)
    assert err.max() <= 1.0 / 510 + 1e-7, f"max quantization error {err.max()}"


# ---------------------------------------------------------------------------
# Learner: bootstrap from observation[:, n] must reproduce the
# next_observation path EXACTLY
# ---------------------------------------------------------------------------

def _matched_batches(B=8, n=3, obs_size=6, act_size=3, seed=0):
    """One underlying trajectory, expressed in both schemas.

    Full schema:       T = n,     observation[k] = o_k, next_observation[k] = o_{k+1}
    Compressed schema: T = n + 1, observation[k] = o_k
    """
    rng = np.random.default_rng(seed)
    o = rng.normal(size=(B, n + 1, obs_size)).astype(np.float32)
    a = rng.normal(size=(B, n + 1, act_size)).astype(np.float32)
    r = rng.normal(size=(B, n + 1)).astype(np.float32)
    d = (rng.uniform(size=(B, n + 1)) > 0.2).astype(np.float32)  # some dones
    full = {
        "observation": jnp.asarray(o[:, :n]),
        "action": jnp.asarray(a[:, :n]),
        "reward": jnp.asarray(r[:, :n]),
        "discount": jnp.asarray(d[:, :n]),
        "next_observation": jnp.asarray(o[:, 1 : n + 1]),
    }
    compressed = {
        "observation": jnp.asarray(o),
        "action": jnp.asarray(a),
        "reward": jnp.asarray(r),
        "discount": jnp.asarray(d),
    }
    return full, compressed


def _leaves_equal(x, y):
    lx, ly = jax.tree.leaves(x), jax.tree.leaves(y)
    return all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(lx, ly))


@pytest.mark.parametrize("use_n_step,n_step", [(True, 3), (False, 1)])
def test_compressed_schema_reproduces_the_full_schema_exactly(use_n_step, n_step):
    s = _setup()
    cfg = dataclasses.replace(s["cfg"], use_n_step=use_n_step, n_step=n_step)
    env = s["env"]
    full, compressed = _matched_batches(
        n=max(n_step, 1), obs_size=env.obs_size, act_size=env.action_size
    )
    state_a, metrics_a = sgd_step(s["state"], full, s["nets"], s["optimizers"], cfg)
    state_b, metrics_b = sgd_step(s["state"], compressed, s["nets"], s["optimizers"], cfg)
    for k in metrics_a:
        np.testing.assert_allclose(
            np.asarray(metrics_a[k]), np.asarray(metrics_b[k]), rtol=0, atol=0,
            err_msg=f"metric {k} differs between schemas",
        )
    assert _leaves_equal(state_a.critic_params, state_b.critic_params)
    assert _leaves_equal(state_a.policy_params, state_b.policy_params)
    assert _leaves_equal(state_a.dual_params, state_b.dual_params)


def test_compressed_schema_caps_n_at_T_minus_1():
    """seq == n_step (mis-sized config): n silently caps at T-1, no crash."""
    s = _setup()
    cfg = dataclasses.replace(s["cfg"], use_n_step=True, n_step=4)
    env = s["env"]
    _, compressed = _matched_batches(
        n=3, obs_size=env.obs_size, act_size=env.action_size
    )  # T = 4 < n_step + 1
    sgd_step(s["state"], compressed, s["nets"], s["optimizers"], cfg)  # no raise


def test_sequence_of_one_fails_loudly_without_next_observation():
    s = _setup()
    cfg = s["cfg"]
    env = s["env"]
    _, compressed = _matched_batches(
        n=1, obs_size=env.obs_size, act_size=env.action_size
    )
    tiny = jax.tree.map(lambda x: x[:, :1], compressed)  # T = 1
    with pytest.raises(ValueError, match="sequence_length"):
        sgd_step(s["state"], tiny, s["nets"], s["optimizers"], cfg)
