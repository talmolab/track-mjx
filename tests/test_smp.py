from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest
from mujoco_playground._src import mjx_env

from track_mjx.agent.smp.checkpointing import load_prior, save_prior
from track_mjx.agent.smp.features import (
    SMPFeatureSpec,
    compute_smp_obs,
    sample_reference_smp_obs,
)
from track_mjx.agent.smp.reward import (
    DiffNormalizer,
    SMPRewardConfig,
    compute_smp_reward,
)
from track_mjx.agent.smp.tinymdm import (
    EMAState,
    SMPNormalizer,
    TinyMDMConfig,
    denoising_loss,
    init_denoiser_params,
    normalizer_from_samples,
    update_ema,
)
from track_mjx.agent.smp.wrappers import SMPRewardWrapper


class _FakeReferenceClips:
    def __init__(self, n_clips=4, n_frames=16, n_joints=67):
        self._n_clips = n_clips
        self._n_frames = n_frames
        self._n_joints = n_joints
        shape = (n_clips, n_frames)
        base = jnp.arange(n_clips * n_frames, dtype=jnp.float32).reshape(shape)
        self.root_position = jnp.stack([base, base + 1.0, base + 2.0], axis=-1) / 100.0
        self.root_quaternion = jnp.zeros(shape + (4,), dtype=jnp.float32)
        self.root_quaternion = self.root_quaternion.at[..., 0].set(1.0)
        self.velocity = jnp.ones(shape + (3,), dtype=jnp.float32) * 0.1
        self.angular_velocity = jnp.ones(shape + (3,), dtype=jnp.float32) * 0.01
        self.joints = jnp.zeros(shape + (n_joints,), dtype=jnp.float32)
        self.joints_velocity = jnp.zeros(shape + (n_joints,), dtype=jnp.float32)
        self._body_positions = {
            name: jnp.ones(shape + (3,), dtype=jnp.float32) * (i + 1) / 10.0
            for i, name in enumerate(SMPFeatureSpec().key_body_names)
        }
        self.joint_names = [f"joint_{i}" for i in range(n_joints)]
        self.body_names = list(self._body_positions)

    @property
    def qpos(self):
        return jnp.concatenate(
            [self.root_position, self.root_quaternion, self.joints], axis=-1
        )

    @property
    def qvel(self):
        return jnp.concatenate(
            [self.velocity, self.angular_velocity, self.joints_velocity], axis=-1
        )

    def body_xpos(self, name):
        return self._body_positions[name]


def _small_config(input_dim):
    return TinyMDMConfig(
        input_dim=input_dim,
        num_history_steps=SMPFeatureSpec().num_history_steps,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
    )


def test_reference_sampling_shape_is_stable():
    clips = _FakeReferenceClips()
    spec = SMPFeatureSpec()
    obs = sample_reference_smp_obs(clips, jax.random.PRNGKey(0), 5, spec)
    assert obs.shape == (5, spec.input_dim)
    assert jnp.all(jnp.isfinite(obs))


def test_compute_smp_obs_uses_wxyz_quaternions():
    spec = SMPFeatureSpec()
    obs = compute_smp_obs(
        root_pos=jnp.zeros((2, spec.num_history_steps, 3)),
        root_quat=jnp.tile(
            jnp.array([1.0, 0.0, 0.0, 0.0]), (2, spec.num_history_steps, 1)
        ),
        root_vel=jnp.zeros((2, spec.num_history_steps, 3)),
        root_ang_vel=jnp.zeros((2, spec.num_history_steps, 3)),
        joints=jnp.zeros((2, spec.num_history_steps, 67)),
        key_body_pos=jnp.zeros(
            (2, spec.num_history_steps, len(spec.key_body_names), 3)
        ),
    )
    assert obs.shape == (2, spec.input_dim)
    first_rot = obs[0, 3:9]
    assert jnp.allclose(first_rot, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_tinymdm_loss_and_smp_reward_are_finite():
    spec = SMPFeatureSpec()
    cfg = _small_config(spec.input_dim)
    params = init_denoiser_params(jax.random.PRNGKey(1), cfg)
    x = jnp.zeros((3, spec.input_dim), dtype=jnp.float32)
    normalizer = normalizer_from_samples(x, spec.num_history_steps)

    def loss_fn(p):
        return denoising_loss(
            p,
            normalizer.normalize(x, spec.num_history_steps),
            jax.random.PRNGKey(2),
            cfg,
        )[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert jnp.isfinite(loss)
    assert optax.global_norm(grads) > 0.0

    reward, metrics = compute_smp_reward(
        params,
        normalizer,
        DiffNormalizer.identity(3),
        x,
        jax.random.PRNGKey(3),
        cfg,
    )
    assert reward.shape == (3,)
    assert jnp.all(jnp.isfinite(reward))
    assert jnp.isfinite(metrics["sds_loss_mean"])


def test_ema_and_prior_checkpoint_round_trip(tmp_path):
    spec = SMPFeatureSpec()
    cfg = _small_config(spec.input_dim)
    params = init_denoiser_params(jax.random.PRNGKey(4), cfg)
    ema = update_ema(EMAState(params=params, step=jnp.array(0)), params)
    normalizer = SMPNormalizer(
        mean=jnp.zeros((spec.per_frame_dim,), dtype=jnp.float32),
        std=jnp.ones((spec.per_frame_dim,), dtype=jnp.float32),
    )
    save_prior(
        tmp_path,
        params=params,
        ema_params=ema.params,
        normalizer=normalizer,
        diff_normalizer=DiffNormalizer.identity(3),
        model_config=cfg,
        feature_spec=spec,
        reward_config=SMPRewardConfig(),
        metadata={
            "feature_spec": spec.to_dict(),
            "joint_names": [],
            "scale_factor": 0.9,
        },
    )
    loaded = load_prior(tmp_path)
    assert loaded["feature_spec"] == spec
    assert loaded["model_config"].input_dim == cfg.input_dim
    assert loaded["normalizer"].mean.shape == (spec.per_frame_dim,)


class _FakeTaskEnv:
    def __init__(self):
        self._config = SimpleNamespace(ctrl_dt=0.01, rescale_factor=0.9)
        self._mjx_model = None
        self._key_body_pos = {
            name: jnp.ones(3, dtype=jnp.float32) * (i + 1)
            for i, name in enumerate(SMPFeatureSpec().key_body_names)
        }

    @property
    def action_size(self):
        return 2

    @property
    def dt(self):
        return 0.01

    def get_joint_names(self):
        return [f"joint_{i}" for i in range(67)]

    def root_body(self, data):
        return SimpleNamespace(
            xpos=data.root_pos,
            xquat=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        )

    def _get_bodies_pos(self, data, flatten=True):
        del data, flatten
        return self._key_body_pos

    def _get_joint_angles(self, data):
        del data
        return jnp.zeros(67, dtype=jnp.float32)

    def reset(self, rng, **kwargs):
        del rng, kwargs
        data = SimpleNamespace(
            root_pos=jnp.zeros(3, dtype=jnp.float32),
            qvel=jnp.zeros(73, dtype=jnp.float32),
        )
        return mjx_env.State(
            data=data,
            obs={"state": jnp.zeros(1, dtype=jnp.float32)},
            reward=jnp.array(1.0, dtype=jnp.float32),
            done=jnp.array(0.0, dtype=jnp.float32),
            metrics={},
            info={},
        )

    def step(self, state, action):
        del action
        data = SimpleNamespace(
            root_pos=state.data.root_pos + jnp.array([0.1, 0.0, 0.0]),
            qvel=jnp.ones(73, dtype=jnp.float32) * 0.1,
        )
        return state.replace(data=data, reward=jnp.array(2.0, dtype=jnp.float32))


def test_smp_reward_wrapper_replaces_reward_and_decimates_history():
    spec = SMPFeatureSpec()
    cfg = TinyMDMConfig(
        input_dim=spec.input_dim,
        num_history_steps=spec.num_history_steps,
        num_layers=0,
        num_attention_heads=1,
        attention_head_dim=8,
    )
    params = init_denoiser_params(jax.random.PRNGKey(6), cfg)
    normalizer = SMPNormalizer(
        mean=jnp.zeros((spec.per_frame_dim,), dtype=jnp.float32),
        std=jnp.ones((spec.per_frame_dim,), dtype=jnp.float32),
    )
    wrapper = SMPRewardWrapper(
        _FakeTaskEnv(),
        prior_params=params,
        prior_normalizer=normalizer,
        diff_normalizer=DiffNormalizer.identity(3),
        model_config=cfg,
        feature_spec=spec,
        metadata={
            "joint_names": [f"joint_{i}" for i in range(67)],
            "feature_spec": spec.to_dict(),
            "scale_factor": 0.9,
        },
    )

    state = wrapper.reset(jax.random.PRNGKey(7))
    assert state.reward == 1.0
    assert state.info["smp_root_pos_history"].shape == (spec.num_history_steps, 3)

    state = wrapper.step(state, jnp.zeros(wrapper.action_size))
    assert jnp.isfinite(state.reward)
    assert "smp/reward" in state.metrics
    assert jnp.allclose(state.info["smp_root_pos_history"][-1], jnp.zeros(3))

    state = wrapper.step(state, jnp.zeros(wrapper.action_size))
    assert jnp.allclose(
        state.info["smp_root_pos_history"][-1],
        jnp.array([0.2, 0.0, 0.0], dtype=jnp.float32),
    )


def test_real_rodent_hdf_sampling_if_available():
    data_path = Path("data/rodent/rodent_reference_clips.h5")
    if not data_path.exists():
        pytest.skip("local rodent reference clips are not present")
    from vnl_playground.tasks.reference_clips import ReferenceClips

    clips = ReferenceClips(str(data_path), n_frames_per_clip=250)
    obs = sample_reference_smp_obs(clips, jax.random.PRNGKey(5), 2, SMPFeatureSpec())
    assert obs.shape == (2, SMPFeatureSpec().input_dim)
