# tests/agent/dmpo/test_train_dmpo_eval.py
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from typing import Any

import jax
import jax.numpy as jnp


def test_vision_sensitivity_zero_for_blind_policy():
    """If the policy ignores vision entirely, sensitivity should be ~0."""
    from track_mjx.agent.dmpo.train_dmpo_eval import compute_vision_sensitivity

    def blind_policy(_params, obs):
        # Ignores obs["vision"]; mode() depends only on shape.
        action_dim = obs["proprioception"].shape[-1]
        class _Dist:
            def mode(self): return jnp.zeros(action_dim)
        return _Dist()

    obs = {
        "vision": jnp.ones((32, 32, 2), dtype=jnp.float32),
        "proprioception": jnp.zeros((10,), dtype=jnp.float32),
        "imitation_target": jnp.zeros((4,), dtype=jnp.float32),
    }
    sens = compute_vision_sensitivity(
        blind_policy, params=None, obs=obs, rng=jax.random.PRNGKey(0)
    )
    assert float(sens) < 1e-6, f"blind policy should not differ on blank vision, got {sens}"


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
        rng=jax.random.PRNGKey(0),
        episode_length=10,
        num_envs=4,
    )
    assert len(rollout) == 11  # initial + 10 steps
    assert isinstance(term_events, list)


def test_render_eval_video_writes_file(tmp_path):
    """Smoke: render 5 frames using an inline model and check the MP4 is created."""
    import mujoco
    from track_mjx.agent.dmpo.train_dmpo_eval import render_eval_video

    # Use a simple fixed body (no joints) so qpos/qvel are length 0.
    xml = """<mujoco><worldbody><body name="b"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
    mj_model = mujoco.MjModel.from_xml_string(xml)

    # Build 5 stub states — qpos/qvel of the right shape.
    class _D: pass
    class _S:
        def __init__(self, q, v):
            self.data = _D()
            self.data.qpos = q
            self.data.qvel = v
            self.obs = {"vision": jnp.zeros((4, 4, 2))}

    rollout = [
        _S(jnp.zeros((mj_model.nq,)), jnp.zeros((mj_model.nv,))) for _ in range(5)
    ]
    out = render_eval_video(
        rollout, mj_model, tmp_path / "smoke.mp4",
        fps=10, height=64, width=64, camera="missing_cam",
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

    from track_mjx.agent.dmpo.train_dmpo_eval import run_eval_rollout_envzero
    from track_mjx.agent.observation_utils import DictRunningStatisticsState

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
        rng=jax.random.PRNGKey(0),
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
        rng=jax.random.PRNGKey(0),
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
        rng=jax.random.PRNGKey(0),
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
