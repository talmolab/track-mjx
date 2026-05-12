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

    rollout, term_events = run_eval_rollout_envzero(
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
    rollout_raw, _ = run_eval_rollout_envzero(
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
    rollout_norm, _ = run_eval_rollout_envzero(
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
