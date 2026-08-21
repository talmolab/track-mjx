"""Verify collect_rollout's recurrent branch (recurrent_meta/policy_hidden).

Uses a stub recurrent policy -- a plain function
``policy_apply(params, obs, hidden) -> (tfd.MultivariateNormalDiag, new_hidden)``
with the deterministic update ``nh_l = 0.5 * h_l + mean(obs)`` -- and the
pre_batched flax.struct stub-env pattern from test_rollout_extra_state_extras.
``scale_diag=0`` makes sampling deterministic (sample == loc), so chaining
equality does not depend on the rng chain (which legitimately differs between
one long call and two chained calls).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from typing import Any, NamedTuple

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.rollout import collect_rollout

tfd = tfp.distributions

OBS = 6
ACT = 3
HIDDEN_SIZES = (5, 4)


@flax.struct.dataclass
class _State:
    obs: jax.Array
    t: jax.Array
    done: jax.Array
    reward: jax.Array
    info: dict


def _obs_at(t):
    """Deterministic per-env, per-step obs: distinct rows so hidden updates
    are nontrivial. t is the per-env step counter [n]."""
    n = t.shape[0]
    base = 0.1 * t.astype(jnp.float32) + 0.01 * jnp.arange(n, dtype=jnp.float32)
    return base[:, None] + 0.001 * jnp.arange(OBS, dtype=jnp.float32)[None, :]


class _CountingEnv:
    """Deterministic counting env; optionally fires done on one known step.

    ``done_step=s`` marks done on the transition at scan index ``s`` (the
    step whose pre-step counter equals ``s``), so the rollout must store an
    exactly-zero hidden at index ``s + 1``. No auto-reset of the counter --
    this stub only exercises the rollout's own done handling.
    """

    pre_batched = True
    action_size = ACT

    def __init__(self, done_step=None):
        self.done_step = done_step

    def reset(self, keys):
        n = keys.shape[0]
        t = jnp.zeros((n,), jnp.int32)
        return _State(
            obs=_obs_at(t),
            t=t,
            done=jnp.zeros((n,)),
            reward=jnp.zeros((n,)),
            info={"aux": jnp.full((n,), 7.0)},
        )

    def step(self, st, action):
        t = st.t + 1
        if self.done_step is None:
            done = jnp.zeros_like(st.done)
        else:
            done = (st.t == self.done_step).astype(jnp.float32)
        reward = jnp.linalg.norm(action, axis=-1)
        return _State(obs=_obs_at(t), t=t, done=done, reward=reward,
                      info=st.info), reward


class _StubMeta(NamedTuple):
    """Duck-typed RecurrentPolicyMeta: rollout only touches init_hidden and
    store_dtype, so the test does not depend on networks_kl_anchor_rnn."""

    cell_type: str = "gru"
    hidden_sizes: tuple = HIDDEN_SIZES
    store_dtype: Any = jnp.float16

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            return tuple(jnp.zeros((h,)) for h in self.hidden_sizes)
        return tuple(jnp.zeros((batch_size, h)) for h in self.hidden_sizes)


def _recurrent_policy_apply(_params, obs, hidden):
    """Unbatched stub: obs [OBS], hidden tuple of [H_l]. Deterministic."""
    m = jnp.mean(obs)
    new_hidden = tuple(0.5 * h + m for h in hidden)
    mu = (m + jnp.mean(new_hidden[0]) - jnp.mean(new_hidden[1])) * jnp.ones((ACT,))
    return tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.zeros((ACT,))), new_hidden


def _flat_normalizer():
    return running_statistics.init_state(specs.Array((OBS,), jnp.float32))


def _expected_hiddens(num_envs, num_steps, done_step):
    """Hand-recompute the stored pre-step hiddens with numpy.

    Mirrors the rollout semantics: store h, then h <- 0.5*h + mean(obs_t)
    per layer, then zero h where done fired. The fresh normalizer is
    identity (mean 0, std 1), so norm_obs == raw obs.
    """
    h = [np.zeros((num_envs, hl), np.float32) for hl in HIDDEN_SIZES]
    stored = [[] for _ in HIDDEN_SIZES]
    for s in range(num_steps):
        for l, hl in enumerate(h):
            stored[l].append(hl.copy())
        obs = np.asarray(_obs_at(jnp.full((num_envs,), s, jnp.int32)))
        m = obs.mean(axis=-1, keepdims=True).astype(np.float32)
        h = [(0.5 * hl + m).astype(np.float32) for hl in h]
        if done_step is not None and s == done_step:
            h = [np.zeros_like(hl) for hl in h]
    return [np.stack(sl, axis=1) for sl in stored], h  # [n, T, H_l] each


def test_policy_hidden_shapes_dtype_and_feature_parity():
    """Stored hidden leaves are [n, T, H_l] in store_dtype; the recurrent
    branch shares extra_state_extras / store_next_observation handling."""
    n, T = 3, 4
    meta = _StubMeta()

    @jax.jit  # the fused train step jits this path; prove it traces clean
    def go(rng):
        return collect_rollout(
            env=_CountingEnv(),
            policy_apply=_recurrent_policy_apply,
            policy_params=None,
            normalizer_params=_flat_normalizer(),
            rng=rng,
            num_envs=n,
            num_steps=T,
            extra_state_extras=("aux",),
            store_next_observation=False,
            recurrent_meta=meta,
        )

    out = go(jax.random.PRNGKey(0))
    assert len(out) == 4
    traj, _final_state, _norm, final_hidden = out
    assert isinstance(traj["policy_hidden"], tuple)
    for leaf, hl in zip(traj["policy_hidden"], HIDDEN_SIZES):
        assert leaf.shape == (n, T, hl), leaf.shape
        assert leaf.dtype == meta.store_dtype, leaf.dtype
    for leaf, hl in zip(final_hidden, HIDDEN_SIZES):
        assert leaf.shape == (n, hl)
        assert leaf.dtype == jnp.float32  # live hidden stays compute dtype
    assert "aux" in traj and traj["aux"].shape == (n, T)
    assert "next_observation" not in traj


def test_stored_hidden_matches_hand_recomputed_prestep():
    n, T, done_step = 3, 6, 2
    traj, _final_state, _norm, final_hidden = collect_rollout(
        env=_CountingEnv(done_step=done_step),
        policy_apply=_recurrent_policy_apply,
        policy_params=None,
        normalizer_params=_flat_normalizer(),
        rng=jax.random.PRNGKey(1),
        num_envs=n,
        num_steps=T,
        recurrent_meta=_StubMeta(),
    )
    expected_stored, expected_final = _expected_hiddens(n, T, done_step)
    for leaf, exp in zip(traj["policy_hidden"], expected_stored):
        # f16 storage rounding dominates the tolerance (eps ~ 1e-3 rel).
        np.testing.assert_allclose(
            np.asarray(leaf, np.float32), exp, atol=2e-3, rtol=1e-3
        )
    for leaf, exp in zip(final_hidden, expected_final):
        np.testing.assert_allclose(np.asarray(leaf), exp, rtol=1e-5, atol=1e-6)


def test_hidden_reset_exactly_zero_after_done():
    n, T, done_step = 2, 5, 2
    traj, *_ = collect_rollout(
        env=_CountingEnv(done_step=done_step),
        policy_apply=_recurrent_policy_apply,
        policy_params=None,
        normalizer_params=_flat_normalizer(),
        rng=jax.random.PRNGKey(2),
        num_envs=n,
        num_steps=T,
        recurrent_meta=_StubMeta(),
    )
    assert np.all(np.asarray(traj["discount"])[:, done_step] == 0.0)
    for leaf in traj["policy_hidden"]:
        arr = np.asarray(leaf, np.float32)
        # Pre-step hidden right after the terminal transition is EXACTLY 0
        # (not merely small): the where-mask writes zeros, f16 keeps them.
        np.testing.assert_array_equal(arr[:, done_step + 1], 0.0)
        # ...and it was nonzero going in (the reset did something).
        assert np.any(arr[:, done_step] != 0.0)


def test_chained_rollouts_equal_one_long_rollout():
    n, N, done_step = 3, 3, 1  # done inside the first chunk
    meta = _StubMeta()
    kwargs = dict(
        policy_apply=_recurrent_policy_apply,
        policy_params=None,
        normalizer_params=_flat_normalizer(),
        num_envs=n,
        recurrent_meta=meta,
    )
    env = _CountingEnv(done_step=done_step)
    traj_long, _fs, _norm, fh_long = collect_rollout(
        env=env, rng=jax.random.PRNGKey(3), num_steps=2 * N, **kwargs
    )
    traj_a, fs_a, _norm_a, fh_a = collect_rollout(
        env=env, rng=jax.random.PRNGKey(3), num_steps=N, **kwargs
    )
    # Same fresh normalizer on purpose: normalizer_params are read-only
    # inside the scan, so the long call normalizes all 2N steps with the
    # initial stats; the chained call must too for bit-equality.
    traj_b, _fs_b, _norm_b, fh_b = collect_rollout(
        env=env, rng=jax.random.PRNGKey(4), num_steps=N,
        init_state=fs_a, policy_hidden=fh_a, **kwargs
    )

    chained = jax.tree.map(
        lambda a, b: jnp.concatenate([a, b], axis=1), traj_a, traj_b
    )
    assert set(traj_long.keys()) == set(chained.keys())
    for (path_l, leaf_l), (_path_c, leaf_c) in zip(
        jax.tree_util.tree_flatten_with_path(traj_long)[0],
        jax.tree_util.tree_flatten_with_path(chained)[0],
    ):
        np.testing.assert_array_equal(
            np.asarray(leaf_l), np.asarray(leaf_c), err_msg=str(path_l)
        )
    for leaf_l, leaf_b in zip(fh_long, fh_b):
        np.testing.assert_array_equal(np.asarray(leaf_l), np.asarray(leaf_b))


def _reference_ff_collect_rollout(
    env, policy_apply, policy_params, normalizer_params, rng, num_envs, num_steps
):
    """Frozen verbatim copy of the PRE-change collect_rollout body
    (specialized to: pre_batched env, flat obs, no extras/mixing/remix,
    store_next_observation=True). Scan-based on purpose -- a python-loop
    reference differs from the scanned computation by 1 ulp under XLA
    fusion; only an identical scan body pins true bit-identity of the FF
    branch (rng layout, normalize-for-policy-only, raw obs stored,
    discount=1-done)."""
    rng, k_reset = jax.random.split(rng)
    reset_keys = jax.random.split(k_reset, num_envs)
    state = env.reset(reset_keys)

    def step_fn(carry, _):
        state, rng = carry
        rng, k_act = jax.random.split(rng)
        keys = jax.random.split(k_act, num_envs)
        norm_obs = running_statistics.normalize(state.obs, normalizer_params)
        raw_action = jax.vmap(
            lambda o, k: policy_apply(policy_params, o).sample(seed=k)
        )(norm_obs, keys)
        bound_action = jnp.clip(jnp.tanh(raw_action), -1.0 + 1e-6, 1.0 - 1e-6)
        new_state, reward = env.step(state, bound_action)
        transition = {
            "observation": state.obs,
            "action": raw_action,
            "reward": reward,
            "discount": (1.0 - new_state.done).astype(jnp.float32),
            "next_observation": new_state.obs,
        }
        return (new_state, rng), transition

    (final_state, _), traj = jax.lax.scan(
        step_fn, (state, rng), None, length=num_steps,
    )
    traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
    new_norm = running_statistics.update(normalizer_params, traj["observation"])
    return traj, final_state, new_norm


def test_ff_path_bit_identical_and_no_hidden_key():
    """recurrent_meta=None: 3-tuple, no policy_hidden key, outputs equal the
    frozen pre-change implementation bit-for-bit."""
    n, T = 2, 4
    env = _CountingEnv()
    norm = _flat_normalizer()

    def ff_policy_apply(_p, obs):
        # Stochastic on purpose: exercises the per-env-key sampling layout.
        mu = jnp.mean(obs) * jnp.ones((ACT,))
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.ones((ACT,)))

    out = collect_rollout(
        env=env,
        policy_apply=ff_policy_apply,
        policy_params=None,
        normalizer_params=norm,
        rng=jax.random.PRNGKey(5),
        num_envs=n,
        num_steps=T,
    )
    assert len(out) == 3
    traj, final_state, new_norm = out
    assert "policy_hidden" not in traj

    ref_traj, ref_final_state, ref_norm = _reference_ff_collect_rollout(
        env, ff_policy_apply, None, norm, jax.random.PRNGKey(5), n, T
    )
    assert set(traj.keys()) == set(ref_traj.keys())
    for key in ref_traj:
        np.testing.assert_array_equal(
            np.asarray(traj[key]), np.asarray(ref_traj[key]), err_msg=key
        )
    np.testing.assert_array_equal(
        np.asarray(final_state.obs), np.asarray(ref_final_state.obs)
    )
    np.testing.assert_array_equal(
        np.asarray(final_state.t), np.asarray(ref_final_state.t)
    )
    for leaf, ref_leaf in zip(
        jax.tree_util.tree_leaves(new_norm), jax.tree_util.tree_leaves(ref_norm)
    ):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(ref_leaf))


def test_frozen_params_with_recurrent_raises():
    with pytest.raises(NotImplementedError, match="behavior mixing"):
        collect_rollout(
            env=_CountingEnv(),
            policy_apply=_recurrent_policy_apply,
            policy_params=None,
            normalizer_params=_flat_normalizer(),
            rng=jax.random.PRNGKey(0),
            num_envs=2,
            num_steps=2,
            frozen_policy_params={"w": jnp.zeros((1,))},
            behavior_mix_frac=jnp.float32(0.5),
            recurrent_meta=_StubMeta(),
        )


def test_policy_hidden_without_meta_raises():
    with pytest.raises(ValueError, match="policy_hidden"):
        collect_rollout(
            env=_CountingEnv(),
            policy_apply=_recurrent_policy_apply,
            policy_params=None,
            normalizer_params=_flat_normalizer(),
            rng=jax.random.PRNGKey(0),
            num_envs=2,
            num_steps=2,
            policy_hidden=_StubMeta().init_hidden(2),
        )
