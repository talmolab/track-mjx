"""The imitation entry restarts the reference clock when an episode ends.

The imitation env reads its reference frame from ``info["reference_step"]``,
and the training stack auto-resets with ``full_reset=False``, which swaps the
physics back to the cached first reset state on done but keeps ``state.info``.
Unless the entry also resets the per-episode info keys, the reference runs on
from the previous episode while the body starts over, and every step after an
env's first done ends an episode.

These tests drive a stub env with the same clock through the entry's own
training stack (``_wrap_train_env``) on CPU, and check the keys the entry
resets against the info the real env writes at reset.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground._src import mjx_env
from vnl_playground import registry
from vnl_playground.tasks.rodent import imitation

from tests.scripts.test_train_dmpo_clip_loading import ENV_NAME, _write_clips

wrappers_truncation = pytest.importorskip(
    "vnl_playground.tasks.wrappers_truncation",
    reason="needs the truncation wrapper, which the pinned vnl-playground lacks",
)

# The stub's reference ends this many steps into its clip.
CLIP_STEPS = 4
ACTION_SIZE = 2
BATCH = 3
# Three clips' worth of steps, so each env ends several episodes.
N_STEPS = 3 * CLIP_STEPS


class _ClockStubEnv:
    """The imitation env's reference clock, without physics.

    ``data`` counts the physics steps since reset; the auto-reset swaps it back
    to 0 on done. ``info`` holds the keys the imitation env writes at reset. The
    stub tracks its reference only while its two clocks agree: a step whose
    ``reference_step`` differs from the physics step terminates, as a tracking
    termination does once the reference has moved away from the reset pose.
    The reference truncates ``CLIP_STEPS`` steps into the clip.
    """

    action_size = ACTION_SIZE
    observation_size = 1

    def reset(self, rng):
        del rng
        info = {
            "start_frame": jnp.int32(0),
            "reference_clip": jnp.int32(0),
            "reference_step": jnp.int32(0),
            "truncated": jnp.bool_(False),
            "prev_action": jnp.zeros(ACTION_SIZE),
            "action": jnp.zeros(ACTION_SIZE),
        }
        return mjx_env.State(
            data=jnp.int32(0),
            obs=jnp.zeros(1),
            reward=jnp.float32(0),
            done=jnp.float32(0),
            metrics={},
            info=info,
        )

    def step(self, state, action):
        info = dict(state.info)
        physics_step = state.data + 1
        info["reference_step"] = info["reference_step"] + 1
        info["truncated"] = info["reference_step"] >= CLIP_STEPS
        info["prev_action"] = info["action"]
        info["action"] = action
        terminated = info["reference_step"] != physics_step
        done = jnp.logical_or(terminated, info["truncated"])
        return state.replace(
            data=physics_step,
            obs=jnp.reshape(physics_step, (1,)).astype(jnp.float32),
            reward=jnp.float32(1),
            done=done.astype(jnp.float32),
            info=info,
        )


def _rollout(env):
    """Steps ``env`` for ``N_STEPS``; action ``i + 1`` on step ``i``."""
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), BATCH))
    step = jax.jit(env.step)
    trace = {"reference_step": [], "prev_action": [], "done": []}
    for i in range(N_STEPS):
        state, _ = step(state, jnp.full((BATCH, ACTION_SIZE), i + 1.0))
        trace["reference_step"].append(np.asarray(state.info["reference_step"]))
        trace["prev_action"].append(np.asarray(state.info["prev_action"]))
        trace["done"].append(np.asarray(state.done) > 0.5)
    return state, {k: np.stack(v) for k, v in trace.items()}


def _episode_lengths(done):
    """Lengths of the completed episodes of one env, from its done flags."""
    ends = np.flatnonzero(done) + 1
    return np.diff(np.concatenate([[0], ends])).tolist()


def _entry_stack(entry, episode_length):
    return entry["_wrap_train_env"](
        _ClockStubEnv(), episode_length=episode_length, action_repeat=1
    )


def _stack_without_info_reset(entry, episode_length):
    """The training stack as the entry built it before the info reset."""
    wrapped = wrappers_truncation.wrap_for_dmpo_training(
        _ClockStubEnv(), episode_length=episode_length, full_reset=False
    )
    return entry["_VnlPlaygroundEnvAdapter"](wrapped, pre_batched=True)


# episode_length well past the clip: the reference truncation ends episodes.
# episode_length 3: the step limit ends them before the reference does.
@pytest.mark.parametrize("episode_length", [100, 3])
def test_reference_clock_restarts_after_every_done(imitation_entry, episode_length):
    _, trace = _rollout(_entry_stack(imitation_entry, episode_length))
    length = min(CLIP_STEPS, episode_length)
    for env in range(BATCH):
        done = trace["done"][:, env]
        assert _episode_lengths(done) == [length] * (N_STEPS // length)
        clock = trace["reference_step"][:, env]
        # 0 on the done step (the next episode starts from the reset frame),
        # then 1 on the first step of the next episode.
        assert clock[done].tolist() == [0] * int(done.sum())
        after_done = np.flatnonzero(done[:-1]) + 1
        assert clock[after_done].tolist() == [1] * len(after_done)


def test_last_action_does_not_reach_the_next_episode(imitation_entry):
    _, trace = _rollout(_entry_stack(imitation_entry, episode_length=100))
    done = trace["done"][:, 0]
    after_done = np.flatnonzero(done[:-1]) + 1
    assert len(after_done) > 0
    # The first step of an episode sees the null action as its previous one.
    np.testing.assert_array_equal(trace["prev_action"][after_done], 0.0)


def test_episode_stats_report_a_healthy_clock(imitation_entry):
    state, _ = _rollout(_entry_stack(imitation_entry, episode_length=100))
    stats = imitation_entry["_training_episode_stats"](state, N_STEPS * BATCH)
    assert stats == {
        "envs": BATCH,
        "dones": BATCH * (N_STEPS // CLIP_STEPS),
        "env_steps_per_done": float(CLIP_STEPS),
        "reference_clock_mismatch": 0,
    }


def test_episode_stats_flag_a_clock_that_runs_on(imitation_entry):
    """Without the info reset, every step after the first done ends an episode."""
    state, trace = _rollout(_stack_without_info_reset(imitation_entry, 100))
    assert _episode_lengths(trace["done"][:, 0]) == [CLIP_STEPS] + [1] * (
        N_STEPS - CLIP_STEPS
    )
    stats = imitation_entry["_training_episode_stats"](state, N_STEPS * BATCH)
    assert stats["reference_clock_mismatch"] == BATCH
    assert stats["env_steps_per_done"] < 2


def test_entry_resets_every_info_key_the_env_advances(imitation_entry, tmp_path):
    """The real env writes no per-episode info key the entry leaves alone.

    Traced, not run: ``jax.eval_shape`` of the env's reset gives its info keys.
    ``reference_clip`` and ``start_frame`` are written at reset only.
    """
    path = tmp_path / "clips.h5"
    n_frames = 20
    _write_clips(path, n_clips=2, n_frames=n_frames)
    clips = registry.load_reference_clips(
        ENV_NAME, data_path=str(path), n_frames_per_clip=n_frames
    )
    config = imitation.default_config()
    config.clip_length = n_frames
    config.start_frame_range = [0, 2]
    env = registry.load(ENV_NAME, config=config, clips=clips, flatten_obs=False)
    reset_info = jax.eval_shape(env.reset, jax.random.PRNGKey(0)).info

    keys = set(imitation_entry["IMITATION_EPISODE_INFO_KEYS"])
    assert set(reset_info) == keys | {"reference_clip", "start_frame"}
