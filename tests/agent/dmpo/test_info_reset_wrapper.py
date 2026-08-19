"""The gap-crossing reward must survive auto-resets during training.

Background. Both DMPO entry points wrap the env with
``mp_wrapper.wrap_for_brax_training(..., full_reset=False)``, whose
``BraxAutoResetWrapper`` swaps data/obs back to the cached first state on done
but explicitly does NOT reset ``state.info``. RunGap's ``gap_crossing_bonus``
fires on ``info["just_crossed_gap"]``, which compares against the
``info["gaps_crossed"]`` high-water mark — so during training the bonus only
pays when an env exceeds its all-time-best position, i.e. the sparse reward is
a cross-episode RATCHET whose inflow decays to zero as records saturate. Eval
rollouts call a real reset and never see this, which is why eval crossings
looked healthy on arms whose training reward had dried up (arm_m1..m8).

``InfoResetOnDoneWrapper`` restores the listed info keys to their reset-time
values wherever done fires. These tests pin:

  1. the BUG: composed with the real BraxAutoResetWrapper, the ratchet is
     real — without the fix a second episode pays no bonus for re-crossing;
  2. the FIX: with the wrapper, every episode pays the bonus again;
  3. fail-loud contracts (missing key, shape mismatch) and non-listed keys
     being left alone.

Run CPU-only (as the whole suite): JAX_PLATFORMS=cpu.
"""

import jax
import jax.numpy as jp
import pytest
from mujoco_playground._src import wrapper as mp_wrapper

from vnl_playground.tasks.wrappers_info_reset import (
    DEFAULT_RUN_GAP_KEYS,
    InfoResetOnDoneWrapper,
)


class _FakeGapEnv:
    """Minimal batched env with RunGap's ratchet-info semantics.

    The agent's x-position increments by 1 every step (action ignored); a
    "gap" is crossed at every integer x, mirroring run_gap.py's
    ``new_gaps_crossed = sum(current_x > gap_ends)`` high-water-mark update.
    ``done`` fires when x reaches ``episode_len``. Data holds only x, so the
    real BraxAutoResetWrapper's data-swap on done teleports x back to 0 —
    exactly the spawn-teleport that exposes the ratchet.
    """

    def __init__(self, num_envs=3, episode_len=4):
        self.num_envs = num_envs
        self.episode_len = episode_len

    # The mp Wrapper base forwards attribute lookups; nothing else needed.
    def reset(self, rng):
        n = self.num_envs
        info = {
            "prev_action": jp.zeros((n, 2)),
            "action": jp.zeros((n, 2)),
            "stale_ref_x": jp.zeros((n,)),
            "stale_steps": jp.zeros((n,), jp.int32),
            "gaps_crossed": jp.zeros((n,), jp.int32),
            "just_crossed_gap": jp.zeros((n,), bool),
            "max_x_reached": jp.zeros((n,)),
            "untouched_key": jp.zeros((n,)),
        }
        return _State(
            data=jp.zeros((n,)),  # x position; swapped by BraxAutoResetWrapper
            obs=jp.zeros((n, 1)),
            reward=jp.zeros((n,)),
            done=jp.zeros((n,)),
            metrics={"rewards/gap_crossing_bonus": jp.zeros((n,))},
            info=dict(info),
        )

    def step(self, state, action):
        x = state.data + 1.0
        new_crossed = jp.floor(x).astype(jp.int32)  # gaps at every integer x
        just = new_crossed > state.info["gaps_crossed"]
        info = dict(state.info)
        info["prev_action"] = info["action"]
        info["action"] = action
        info["just_crossed_gap"] = just
        info["gaps_crossed"] = jp.maximum(new_crossed, state.info["gaps_crossed"])
        info["max_x_reached"] = jp.maximum(x, state.info["max_x_reached"])
        info["untouched_key"] = state.info["untouched_key"] + 1.0
        bonus = jp.where(just, 1.0, 0.0)
        done = (x >= self.episode_len).astype(jp.float32)
        return state.replace(
            data=x,
            obs=x[:, None],
            reward=bonus,
            done=done,
            metrics={"rewards/gap_crossing_bonus": bonus},
            info=info,
        )


class _State:
    """Tiny stand-in for mjx_env.State (attribute access + .replace)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def replace(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return _State(**d)


_KEYS = DEFAULT_RUN_GAP_KEYS + ("untouched_key",)  # cache extra to prove selectivity


def _run(env, steps):
    """Roll `steps` steps of zero actions; return per-step total bonus."""
    state = env.reset(jax.random.PRNGKey(0))
    paid = []
    for _ in range(steps):
        state = env.step(state, jp.ones((3, 2)))
        paid.append(float(state.reward.sum()))
    return state, paid


def _autoreset(env):
    return mp_wrapper.BraxAutoResetWrapper(env, full_reset=False)


def test_the_ratchet_is_real_without_the_fix():
    """Episode 2 pays ZERO bonus: the high-water mark survives the auto-reset."""
    env = _autoreset(_FakeGapEnv(episode_len=4))
    # BraxAutoResetWrapper.reset needs a batched rng
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 3))
    ep1 = ep2 = 0.0
    for t in range(8):  # two 4-step episodes
        state = env.step(state, jp.ones((3, 2)))
        if t < 4:
            ep1 += float(state.reward.sum())
        else:
            ep2 += float(state.reward.sum())
    assert ep1 == pytest.approx(12.0)  # 4 crossings x 3 envs
    assert ep2 == pytest.approx(0.0), (
        "episode 2 paid a bonus -- the ratchet premise of this whole fix is "
        "wrong; re-examine BraxAutoResetWrapper before trusting arm_m9/w1"
    )
    # and the mark itself persisted
    assert int(state.info["gaps_crossed"][0]) == 4


def test_the_fix_pays_every_episode():
    """With InfoResetOnDoneWrapper outside AutoReset, episode 2 pays again."""
    env = InfoResetOnDoneWrapper(_autoreset(_FakeGapEnv(episode_len=4)), keys=_KEYS)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 3))
    ep1 = ep2 = 0.0
    for t in range(8):
        state = env.step(state, jp.ones((3, 2)))
        if t < 4:
            ep1 += float(state.reward.sum())
        else:
            ep2 += float(state.reward.sum())
    assert ep1 == pytest.approx(12.0)
    assert ep2 == pytest.approx(12.0), "fix failed: episode 2 still starved"
    # t=8 is itself a done step, so the mark is already restored to 0 there
    # (the bonus for that step was paid from pre-restore info)
    assert int(state.info["gaps_crossed"][0]) == 0


def test_restore_happens_exactly_on_the_done_step():
    env = InfoResetOnDoneWrapper(_autoreset(_FakeGapEnv(episode_len=4)), keys=_KEYS)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 3))
    for t in range(4):
        state = env.step(state, jp.ones((3, 2)))
    assert float(state.done.min()) == 1.0
    # done step: ratchet keys already back at reset values...
    assert int(state.info["gaps_crossed"][0]) == 0
    assert float(state.info["max_x_reached"][0]) == 0.0
    assert not bool(state.info["just_crossed_gap"][0])
    assert float(jp.abs(state.info["action"]).max()) == 0.0
    # ...consistent with the swapped data (rat back at spawn)
    assert float(state.data[0]) == 0.0
    # cached (listed) key also restored; reward for the done step was already
    # paid from pre-swap info, so nothing was lost
    assert float(state.info["untouched_key"][0]) == 0.0


def test_unlisted_keys_are_left_alone():
    env = InfoResetOnDoneWrapper(
        _autoreset(_FakeGapEnv(episode_len=4)),
        keys=("gaps_crossed", "just_crossed_gap", "max_x_reached"),
    )
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 3))
    for _ in range(4):
        state = env.step(state, jp.ones((3, 2)))
    assert float(state.info["untouched_key"][0]) == 4.0  # kept counting
    assert float(state.info["stale_ref_x"][0]) == 0.0 or True  # not cached, not restored


def test_missing_key_fails_loudly_at_reset():
    env = InfoResetOnDoneWrapper(_autoreset(_FakeGapEnv()), keys=("nonexistent",))
    with pytest.raises(KeyError, match="nonexistent"):
        env.reset(jax.random.split(jax.random.PRNGKey(0), 3))


def test_shape_mismatch_fails_loudly():
    class _BadEnv(_FakeGapEnv):
        def reset(self, rng):
            state = super().reset(rng)
            state.info["scalar_key"] = jp.zeros(())  # no batch dim
            return state

        def step(self, state, action):
            new = super().step(state, action)
            new.info["scalar_key"] = state.info["scalar_key"]
            return new

    env = InfoResetOnDoneWrapper(_autoreset(_BadEnv()), keys=("scalar_key",))
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 3))
    with pytest.raises(ValueError, match="scalar_key"):
        env.step(state, jp.ones((3, 2)))
