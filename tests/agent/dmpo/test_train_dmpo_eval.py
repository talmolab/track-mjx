# tests/agent/dmpo/test_train_dmpo_eval.py
import os

# MUST stay module-level (not moved into the render test / a fixture). MuJoCo
# reads MUJOCO_GL at *import time*, not lazily at GLContext/Renderer creation:
# `mujoco/__init__.py` unconditionally does
# `from mujoco.rendering.classic.gl_context import *`, and that submodule's
# top-level code snapshots `os.environ.get("MUJOCO_GL", ...)` into a module
# global the first time it is imported (mujoco/rendering/classic/gl_context.py,
# the `_MUJOCO_GL = os.environ.get(...)` line near the top of the file). Once
# any code anywhere in the process has done `import mujoco`, that snapshot is
# fixed for the rest of the interpreter session, so a `monkeypatch.setenv`
# inside the render test would be too late whenever some other test module
# imports mujoco first.
os.environ.setdefault("MUJOCO_GL", "egl")

from typing import Any

import jax
import jax.numpy as jnp
import pytest


def test_vision_sensitivity_zero_for_blind_policy():
    """If the policy ignores vision entirely, sensitivity should be ~0."""
    from track_mjx.agent.dmpo.train_dmpo_eval import compute_vision_sensitivity

    def blind_policy(_params, obs):
        # Ignores obs["vision"]; mode() depends only on shape.
        action_dim = obs["proprioception"].shape[-1]

        class _Dist:
            def mode(self):
                return jnp.zeros(action_dim)

        return _Dist()

    obs = {
        "vision": jnp.ones((32, 32, 2), dtype=jnp.float32),
        "proprioception": jnp.zeros((10,), dtype=jnp.float32),
        "imitation_target": jnp.zeros((4,), dtype=jnp.float32),
    }
    sens = compute_vision_sensitivity(
        blind_policy, params=None, obs=obs, rng=jax.random.key(0)
    )
    assert float(sens) < 1e-6, (
        f"blind policy should not differ on blank vision, got {sens}"
    )


def test_run_eval_rollout_envzero_returns_expected_length():
    """Pre-batched env: jit'd lax.scan eval rollout over N steps yields N+1 states.

    The new scan-based implementation requires a jittable env, so the
    stub uses ``flax.struct.dataclass`` for State / Obs / Data so JAX
    treats them as registered pytrees.
    """
    import flax.struct

    from track_mjx.agent.dmpo.train_dmpo_eval import run_eval_rollout_envzero

    @flax.struct.dataclass
    class _Obs:
        vision: jax.Array
        proprioception: jax.Array
        imitation_target: jax.Array

    @flax.struct.dataclass
    class _Data:
        qpos: jax.Array
        qvel: jax.Array

    @flax.struct.dataclass
    class _State:
        obs: _Obs
        data: _Data
        done: jax.Array
        reward: jax.Array

    class _StubEnv:
        action_size = 3

        def reset(self, keys):
            n = keys.shape[0]
            return _State(
                obs=_Obs(
                    vision=jnp.zeros((n, 32, 32, 2)),
                    proprioception=jnp.zeros((n, 4)),
                    imitation_target=jnp.zeros((n, 2)),
                ),
                data=_Data(qpos=jnp.zeros((n, 30)), qvel=jnp.zeros((n, 29))),
                done=jnp.zeros((n,)),
                reward=jnp.zeros((n,)),
            )

        def step(self, st, _action):
            # No-op step: returns the same state. Real envs advance physics.
            return st

    def policy_apply(_params, obs):
        class _D:
            def mode(self):
                return jnp.zeros((3,))

        return _D()

    rollout, term_events, _batch, _allenv = run_eval_rollout_envzero(
        env=_StubEnv(),
        policy_apply=policy_apply,
        params=None,
        rng=jax.random.key(0),
        episode_length=10,
        num_envs=4,
    )
    assert len(rollout) == 11  # initial + 10 steps
    assert isinstance(term_events, list)


@pytest.mark.gpu
def test_render_eval_video_writes_file(tmp_path):
    """Smoke: render 5 frames using an inline model and check the MP4 is created."""
    import mujoco

    from track_mjx.agent.dmpo.train_dmpo_eval import render_eval_video

    # Use a simple fixed body (no joints) so qpos/qvel are length 0.
    xml = """<mujoco><worldbody><body name="b"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
    mj_model = mujoco.MjModel.from_xml_string(xml)

    # Build 5 stub states — qpos/qvel of the right shape.
    class _D:
        pass

    class _S:
        def __init__(self, q, v):
            self.data = _D()
            self.data.qpos = q
            self.data.qvel = v
            self.obs = {"vision": jnp.zeros((4, 4, 2))}

    rollout = [
        _S(jnp.zeros((mj_model.nq,)), jnp.zeros((mj_model.nv,))) for _ in range(5)
    ]
    render_eval_video(
        rollout,
        mj_model,
        tmp_path / "smoke.mp4",
        fps=10,
        height=64,
        width=64,
        camera="missing_cam",
    )
    assert (tmp_path / "smoke.mp4").exists()
    assert (tmp_path / "smoke.mp4").stat().st_size > 0


def test_run_eval_rollout_envzero_applies_normalizer_when_given():
    """run_eval_rollout_envzero must apply the DMPO normalizer to obs
    before calling policy_apply when ``normalizer_params`` is provided;
    with ``normalizer_params=None`` it must pass obs through raw.

    The check is load-bearing: we drive the rollout twice and verify
    the *bound action* recorded into the env state, which directly
    encodes the obs the policy saw. Running under @jax.jit / vmap means
    we can't capture obs in a Python list at trace time, so instead the
    stub policy returns ``obs["proprioception"]`` as its action and the
    stub env stores the bound action in ``data.qpos`` so we can read it
    out from the rollout. With raw ones, ``bind(1)=tanh(1)≈0.7616``;
    with normalizer (mean=1, std=1), proprio normalises to zero so
    ``bind(0)=0``.
    """
    import flax.struct
    import numpy as np
    from brax.training.acme import running_statistics

    from track_mjx.agent.dmpo.obs_utils import DictRunningStatisticsState
    from track_mjx.agent.dmpo.train_dmpo_eval import run_eval_rollout_envzero

    proprio_size = 4
    task_obs_size = 2

    @flax.struct.dataclass
    class _Data:
        qpos: jax.Array
        qvel: jax.Array

    @flax.struct.dataclass
    class _State:
        # Use a generic ``obs`` field so the stub env can return a plain
        # dict — this is what ``flatten_obs_dict`` (called inside
        # ``normalize_dict_obs``) expects (it does ``obs["proprioception"]``
        # which a flax-struct dataclass does not support).
        obs: Any
        data: _Data
        done: jax.Array
        reward: jax.Array

    class _StubDictEnv:
        # Match action_size to proprio_size so the policy can echo obs.
        action_size = proprio_size

        def reset(self, keys):
            n = keys.shape[0]
            return _State(
                obs={
                    "vision": jnp.zeros((n, 8, 8, 2)),
                    # Non-zero proprio so normalisation is observable.
                    "proprioception": jnp.ones((n, proprio_size)),
                    "imitation_target": jnp.zeros((n, task_obs_size)),
                },
                data=_Data(
                    qpos=jnp.zeros((n, proprio_size)),
                    qvel=jnp.zeros((n, 3)),
                ),
                done=jnp.zeros((n,)),
                reward=jnp.zeros((n,)),
            )

        def step(self, st, action):
            # Persist the bound action into ``data.qpos`` so the post-scan
            # rollout exposes the value the policy emitted (i.e. the obs
            # the policy actually saw, after bind()).
            return _State(
                obs=st.obs,
                data=_Data(qpos=action, qvel=st.data.qvel),
                done=st.done,
                reward=st.reward,
            )

    def policy_apply(_params, obs):
        # Echo proprioception so the bound action reflects what the
        # policy received. Under vmap this sees a single-env slice of
        # shape (proprio_size,).
        p = obs["proprioception"]

        class _D:
            def mode(self):
                return p

        return _D()

    # mean=1, std=1, count=100 → normalize(ones)=zeros for proprio.
    dict_norm = DictRunningStatisticsState(
        imitation_target=running_statistics.RunningStatisticsState(
            mean=jnp.zeros((task_obs_size,)),
            std=jnp.ones((task_obs_size,)),
            count=jnp.array(100.0),
            summed_variance=jnp.zeros((task_obs_size,)),
            std_eps=1e-6,
            mode=running_statistics.NormalizationMode.WELFORD,
        ),
        proprioception=running_statistics.RunningStatisticsState(
            mean=jnp.ones((proprio_size,)),
            std=jnp.ones((proprio_size,)),
            count=jnp.array(100.0),
            summed_variance=jnp.zeros((proprio_size,)),
            std_eps=1e-6,
            mode=running_statistics.NormalizationMode.WELFORD,
        ),
    )

    # 1. Without normalizer: policy sees raw ones → bind(1)=tanh(1)≈0.7616.
    rollout_raw, _, _, _ = run_eval_rollout_envzero(
        env=_StubDictEnv(),
        policy_apply=policy_apply,
        params=None,
        rng=jax.random.key(0),
        episode_length=2,
        num_envs=2,
        normalizer_params=None,
    )
    qpos_raw = np.asarray(rollout_raw[1].data.qpos)
    expected_raw = np.tanh(np.ones(proprio_size, dtype=np.float32))
    # bind() also clips off the very edges (1 - eps), but tanh(1) is
    # nowhere near the boundary so the clip is a no-op here.
    np.testing.assert_allclose(qpos_raw, expected_raw, atol=1e-5)

    # 2. With normalizer (mean=ones, std=ones for proprio): policy sees
    #    zeros → bind(0)=tanh(0)=0.
    rollout_norm, _, _, _ = run_eval_rollout_envzero(
        env=_StubDictEnv(),
        policy_apply=policy_apply,
        params=None,
        rng=jax.random.key(0),
        episode_length=2,
        num_envs=2,
        normalizer_params=dict_norm,
    )
    qpos_norm = np.asarray(rollout_norm[1].data.qpos)
    np.testing.assert_allclose(
        qpos_norm, np.zeros(proprio_size, dtype=np.float32), atol=1e-6
    )


def test_run_eval_rollout_envzero_recurrent_threads_and_resets_hidden():
    """Recurrent eval: same 4-tuple structure, finite metrics, hidden reset on done.

    The hidden lives inside the jitted scan, so it is observed through the
    action channel: the stub policy's ``mode()`` echoes a slice of the
    PRE-step hidden it consumed, and the stub env persists the bound action
    into ``data.qpos`` (same trick as the normalizer test above). The stub
    cell increments the hidden by 1 per step and the env forces done=1 at
    step 3, which makes the reset semantics distinguishable frame by frame:

        step 1 consumes h=0 -> qpos=tanh(0);  step 2 consumes h=1 -> tanh(1)
        step 3 consumes h=2, done fires -> hidden must be zeroed AFTER env.step
        step 4 consumes h=0 -> tanh(0)   (tanh(3)~0.995 if the reset leaked)
        step 5 consumes h=1 -> tanh(1)   (recurrence resumed accumulating)

    Frame 3 itself is not asserted: the termination splice deliberately
    replaces the done frame's data with the previous frame's for rendering.
    """
    import flax.struct
    import numpy as np

    from track_mjx.agent.dmpo.networks_kl_anchor_rnn import RecurrentPolicyMeta
    from track_mjx.agent.dmpo.train_dmpo_eval import run_eval_rollout_envzero

    hidden_size = 4
    act_dim = 3
    num_envs = 2

    @flax.struct.dataclass
    class _Data:
        qpos: jax.Array
        qvel: jax.Array

    @flax.struct.dataclass
    class _State:
        obs: Any
        data: _Data
        done: jax.Array
        reward: jax.Array
        t: jax.Array  # per-env step counter driving the forced done

    class _StubRecEnv:
        action_size = act_dim

        def reset(self, keys):
            n = keys.shape[0]
            return _State(
                obs={
                    "vision": jnp.zeros((n, 8, 8, 2)),
                    "proprioception": jnp.zeros((n, 5)),
                    "imitation_target": jnp.zeros((n, 2)),
                },
                data=_Data(
                    qpos=jnp.zeros((n, act_dim)),
                    qvel=jnp.zeros((n, 3)),
                ),
                done=jnp.zeros((n,)),
                reward=jnp.zeros((n,)),
                t=jnp.zeros((n,), dtype=jnp.int32),
            )

        def step(self, st, action):
            t_new = st.t + 1
            return _State(
                obs=st.obs,
                # Persist the bound action so the rollout exposes the hidden
                # the policy consumed this step.
                data=_Data(qpos=action, qvel=st.data.qvel),
                done=(t_new == 3).astype(jnp.float32),
                reward=jnp.ones_like(st.reward),
                t=t_new,
            )

    def policy_apply(_params, obs, hidden):
        # Recurrent signature: (params, obs, hidden) -> (dist, new_hidden).
        # Under the per-env vmap this sees unbatched slices: hidden[0] is
        # [hidden_size]. mode() echoes the PRE-step hidden; the "cell" adds 1.
        h0 = hidden[0]

        class _D:
            def mode(self):
                return h0[:act_dim]

        return _D(), (h0 + 1.0,)

    meta = RecurrentPolicyMeta(
        cell_type="gru", hidden_sizes=(hidden_size,), store_dtype=jnp.float16
    )
    rollout, term_events, batch, allenv = run_eval_rollout_envzero(
        env=_StubRecEnv(),
        policy_apply=policy_apply,
        params=None,
        rng=jax.random.key(0),
        episode_length=5,
        num_envs=num_envs,
        recurrent_meta=meta,
    )

    # Same 4-tuple structure as the FF path.
    assert len(rollout) == 6  # initial + 5 steps
    assert term_events == [(3, "done")]
    assert isinstance(batch, dict) and isinstance(allenv, dict)
    assert allenv["reward"].shape == (5, num_envs)
    assert allenv["done"].shape == (5, num_envs)
    for k, v in batch.items():
        assert np.isfinite(v), f"non-finite eval metric {k}={v}"

    t1 = np.tanh(1.0)
    # Step 2: hidden was carried (0 -> 1) across the scan.
    np.testing.assert_allclose(
        np.asarray(rollout[2].data.qpos), np.full(act_dim, t1), atol=1e-5
    )
    # Step 4: the done at step 3 zeroed the hidden AFTER env.step; a leak
    # would echo tanh(3) ~ 0.995 here instead of tanh(0) = 0.
    np.testing.assert_allclose(
        np.asarray(rollout[4].data.qpos), np.zeros(act_dim), atol=1e-6
    )
    # Step 5: post-reset the recurrence accumulates again from zero.
    np.testing.assert_allclose(
        np.asarray(rollout[5].data.qpos), np.full(act_dim, t1), atol=1e-5
    )


def test_allenv_metric_filter_keeps_spout_prefix():
    """Per-step env metrics under ``spout/`` reach the all-env eval aggregate.

    The all-env filter keeps only known metric-key prefixes; a task that
    reports its own ``spout/*`` per-env scalars must not have them dropped,
    while an unrelated key must still be filtered out.
    """
    import flax.struct
    import numpy as np

    from track_mjx.agent.dmpo.train_dmpo_eval import run_eval_rollout_envzero

    @flax.struct.dataclass
    class _Obs:
        vision: jax.Array
        proprioception: jax.Array
        imitation_target: jax.Array

    @flax.struct.dataclass
    class _Data:
        qpos: jax.Array
        qvel: jax.Array

    @flax.struct.dataclass
    class _State:
        obs: _Obs
        data: _Data
        done: jax.Array
        reward: jax.Array
        metrics: Any

    class _StubEnv:
        action_size = 3

        def reset(self, keys):
            n = keys.shape[0]
            return _State(
                obs=_Obs(
                    vision=jnp.zeros((n, 8, 8, 2)),
                    proprioception=jnp.zeros((n, 4)),
                    imitation_target=jnp.zeros((n, 2)),
                ),
                data=_Data(qpos=jnp.zeros((n, 5)), qvel=jnp.zeros((n, 4))),
                done=jnp.zeros((n,)),
                reward=jnp.zeros((n,)),
                metrics={
                    "spout/hit": jnp.ones((n,)),
                    "rewards/forward": jnp.ones((n,)),
                    "other/ignored": jnp.ones((n,)),
                },
            )

        def step(self, st, _action):
            return st

    def policy_apply(_params, obs):
        class _D:
            def mode(self):
                return jnp.zeros((3,))

        return _D()

    num_envs, episode_length = 4, 6
    _, _, _, allenv = run_eval_rollout_envzero(
        env=_StubEnv(),
        policy_apply=policy_apply,
        params=None,
        rng=jax.random.key(0),
        episode_length=episode_length,
        num_envs=num_envs,
    )
    assert "spout/hit" in allenv, f"spout/ metric dropped by filter: {sorted(allenv)}"
    assert allenv["spout/hit"].shape == (episode_length, num_envs)
    np.testing.assert_allclose(np.asarray(allenv["spout/hit"]), 1.0)
    assert "rewards/forward" in allenv  # existing prefixes still kept
    assert "other/ignored" not in allenv  # unknown prefixes still filtered


def test_index_env_slices_only_env_batched_leaves():
    """Only leaves with leading dim == num_envs are sliced; the rest pass through.

    A batched mujoco_warp ``Data`` carries leaves that are not batched over the
    envs: a shared contact pool, a ``(1,)`` counter, and solver scratch with a
    leading dim of 0. ``x[0]`` on that last kind raises IndexError.
    """
    import numpy as np

    from track_mjx.agent.dmpo.train_dmpo_eval import index_env

    n = 3
    tree = {
        "qpos": jnp.arange(n * 5, dtype=jnp.float32).reshape(n, 5),
        "pool": jnp.ones((7, 3)),
        "count": jnp.array([4], dtype=jnp.int32),
        "scratch": jnp.zeros((0, 0, 0)),
        "scalar": jnp.float32(2.0),
    }
    with pytest.raises(IndexError):
        tree["scratch"][0]
    out = index_env(tree, n)
    np.testing.assert_array_equal(np.asarray(out["qpos"]), np.arange(5))
    np.testing.assert_array_equal(
        np.asarray(index_env(tree, n, i=2)["qpos"]), np.arange(10, 15)
    )
    assert out["pool"].shape == (7, 3)
    assert out["count"].shape == (1,)
    assert out["scratch"].shape == (0, 0, 0)
    assert out["scalar"].shape == ()


def test_run_eval_rollout_envzero_handles_unbatched_and_empty_leaves():
    """The eval rollout survives a state with zero-size and shared leaves.

    Env 0's batched leaves are recorded per frame; the shared and empty leaves
    keep their shapes and the reset-state value.
    """
    import flax.struct
    import numpy as np

    from track_mjx.agent.dmpo.train_dmpo_eval import run_eval_rollout_envzero

    @flax.struct.dataclass
    class _Impl:
        contact_pool: jax.Array
        nacon: jax.Array
        cJ: jax.Array

    @flax.struct.dataclass
    class _Data:
        qpos: jax.Array
        impl: _Impl

    @flax.struct.dataclass
    class _State:
        obs: dict
        data: _Data
        done: jax.Array
        reward: jax.Array

    num_envs, episode_length = 4, 5

    class _StubEnv:
        action_size = 3

        def reset(self, keys):
            n = keys.shape[0]
            return _State(
                obs={"proprioception": jnp.zeros((n, 4))},
                data=_Data(
                    # Env e starts at qpos 10*e; each step adds 1.
                    qpos=jnp.tile(10.0 * jnp.arange(n, dtype=jnp.float32)[:, None], 2),
                    impl=_Impl(
                        contact_pool=jnp.zeros((9, 3)),
                        nacon=jnp.zeros((1,), dtype=jnp.int32),
                        cJ=jnp.zeros((0, 0, 0)),
                    ),
                ),
                done=jnp.zeros((n,)),
                reward=jnp.zeros((n,)),
            )

        def step(self, st, _action):
            impl = st.data.impl
            return st.replace(
                data=st.data.replace(
                    qpos=st.data.qpos + 1.0,
                    impl=impl.replace(
                        contact_pool=impl.contact_pool + 1.0, nacon=impl.nacon + 1
                    ),
                ),
                reward=st.reward + 1.0,
            )

    def policy_apply(_params, obs):
        class _D:
            def mode(self):
                return jnp.zeros((3,))

        return _D()

    rollout, _, _, allenv = run_eval_rollout_envzero(
        env=_StubEnv(),
        policy_apply=policy_apply,
        params=None,
        rng=jax.random.key(0),
        episode_length=episode_length,
        num_envs=num_envs,
    )
    assert len(rollout) == episode_length + 1
    for t, st in enumerate(rollout):
        np.testing.assert_allclose(np.asarray(st.data.qpos), [float(t)] * 2)
        assert np.asarray(st.reward).shape == ()
        assert st.data.impl.contact_pool.shape == (9, 3)
        assert st.data.impl.nacon.shape == (1,)
        assert st.data.impl.cJ.shape == (0, 0, 0)
        # Shared leaves carry the reset state's value in every frame.
        np.testing.assert_allclose(np.asarray(st.data.impl.contact_pool), 0.0)
    assert allenv["reward"].shape == (episode_length, num_envs)


def test_run_eval_rollout_envzero_refuses_a_num_envs_other_than_the_env_batch():
    """Every leaf would count as unbatched, and env 0's frames would all be the
    reset state, with no error."""
    import flax.struct

    from track_mjx.agent.dmpo.train_dmpo_eval import run_eval_rollout_envzero

    @flax.struct.dataclass
    class _State:
        obs: dict
        done: jax.Array
        reward: jax.Array

    class _FixedBatchEnv:
        """Resets three envs, however many keys it is given."""

        action_size = 3

        def reset(self, keys):
            return _State(
                obs={"proprioception": jnp.zeros((3, 4))},
                done=jnp.zeros((3,)),
                reward=jnp.zeros((3,)),
            )

        def step(self, st, _action):
            return st

    with pytest.raises(ValueError, match=r"num_envs=4.*\(3,\)"):
        run_eval_rollout_envzero(
            env=_FixedBatchEnv(),
            policy_apply=lambda params, obs: None,
            params=None,
            rng=jax.random.key(0),
            episode_length=5,
            num_envs=4,
        )
