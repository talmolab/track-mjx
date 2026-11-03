"""Maze-forage invariants of the kl-anchor entry's reset handling.

``RodentMazeForageVision`` draws its spawn pose, heading and treat cells into
``data.qpos`` at reset. ``BraxAutoResetWrapper(full_reset=False)`` restores
``data``/``obs`` from the FIRST reset and never clears ``state.info``, so with
it ``env.reset`` would run once per env index for a whole run: every worker
replays one frozen layout and the ``collected`` mask ratchets. The entry's
``wrappers:`` block and ``_validate_reset_requirements`` exist to prevent
that. Pinned here:

1. every shipped maze arm enables full resets, and every ``env_config`` key it
   sets reaches the env (the entry silently drops keys the env's
   ``default_config()`` lacks, unless the entry reads them itself);
2. the layout is re-drawn on the other side of a ``done`` (GPU: the env needs
   the warp backend), and stays frozen without ``full_reset``;
3. ``_validate_reset_requirements`` refuses a frozen-layout config.
"""

import pathlib

import pytest
from omegaconf import OmegaConf

maze_forage_vision = pytest.importorskip(
    "vnl_playground.tasks.rodent.maze_forage_vision",
    reason="needs the maze-forage env, which the pinned vnl-playground lacks",
)
INFO_RESET_KEYS = maze_forage_vision.INFO_RESET_KEYS
MazeForageVision = maze_forage_vision.MazeForageVision
default_config = maze_forage_vision.default_config

ENV_NAME = "RodentMazeForageVision"
MAZE_CONFIG_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "track_mjx"
    / "config"
    / "dmpo"
    / "maze"
)
MAZE_ARMS = sorted(MAZE_CONFIG_DIR.glob("*.yaml"))

# env_config keys the entry reads itself (vision wrapper, episode wrapper,
# eval HUD) or that are YAML-only metadata; every other key must be an env key.
ENTRY_ENV_CONFIG_KEYS = frozenset(
    {
        # dropped by the entry before the env is built
        "env_name",
        "flatten_obs",
        # metadata
        "task_name",
        "walker_name",
        "render_camera_name",
        # read by the entry
        "episode_length",
        "action_repeat",
        "ctrl_dt",
        "vision_width",
        "vision_height",
        "grayscale",
        "left_camera_name",
        "right_camera_name",
        "use_textures",
        "use_shadows",
        "eye_dropout_rate",
        "eval_eye_mode",
        "vision_noise_std",
        "vision_blur_tau_ms",
        "reward_terms",
    }
)


class _DeclaredMazeEnv:
    """Carries the reset contract ``MazeForageVision`` declares, without a model."""

    requires_per_episode_reset = MazeForageVision.requires_per_episode_reset
    info_reset_keys = MazeForageVision.info_reset_keys


class _Wrapper:
    def __init__(self, env):
        self.env = env


def test_maze_env_declares_per_episode_resets():
    assert MazeForageVision.requires_per_episode_reset is True
    assert tuple(MazeForageVision.info_reset_keys) == tuple(INFO_RESET_KEYS)


def test_entry_refuses_a_frozen_layout_config(kl_anchor_entry):
    validate = kl_anchor_entry["_validate_reset_requirements"]
    # The guard unwraps to the innermost env, as it does on the real stack.
    env = _Wrapper(_Wrapper(_DeclaredMazeEnv()))

    # No `wrappers:` block at all.
    with pytest.raises(ValueError, match="requires_per_episode_reset"):
        validate(OmegaConf.create({}), env, ENV_NAME)

    # Info-reset alone does NOT unfreeze the layout, so it is still refused.
    info_only = {"info_reset_on_done": True, "info_reset_keys": list(INFO_RESET_KEYS)}
    with pytest.raises(ValueError, match="requires_per_episode_reset"):
        validate(OmegaConf.create({"wrappers": info_only}), env, ENV_NAME)

    # Full resets pass.
    validate(OmegaConf.create({"wrappers": {"full_reset": True}}), env, ENV_NAME)

    # An explicit frozen-layout ablation is allowed, but only with a complete
    # info reset behind it...
    ablation = {"allow_frozen_layout": True, **info_only}
    validate(OmegaConf.create({"wrappers": ablation}), env, ENV_NAME)
    # ...and an incomplete key list is not a complete info reset.
    ablation["info_reset_keys"] = ["prev_action"]
    with pytest.raises(ValueError, match="requires_per_episode_reset"):
        validate(OmegaConf.create({"wrappers": ablation}), env, ENV_NAME)


def _compose_arm(arm: pathlib.Path):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(MAZE_CONFIG_DIR.parent), version_base=None
    ):
        return compose(config_name=f"maze/{arm.stem}")


@pytest.mark.parametrize(
    "arm", MAZE_ARMS or [None], ids=[a.stem for a in MAZE_ARMS] or ["no-maze-arms"]
)
def test_shipped_maze_arm_enables_per_episode_resets(kl_anchor_entry, arm):
    if arm is None:
        pytest.skip(f"no maze arm configs under {MAZE_CONFIG_DIR}")
    cfg = _compose_arm(arm)
    if cfg.get("env_name") != ENV_NAME:
        pytest.skip(f"{arm.name} is not a {ENV_NAME} arm")
    # Every maze arm under config/dmpo must be a DMPO arm. One that is not
    # (e.g. it lost its `base` default) fails here instead of being skipped.
    assert cfg.get("agent") == "dmpo", f"{arm.name} must compose to agent: dmpo"

    full_reset, _, _ = kl_anchor_entry["_wrapper_settings"](cfg)
    assert full_reset is True, f"{arm.name} must set wrappers.full_reset: true"
    kl_anchor_entry["_validate_reset_requirements"](cfg, _DeclaredMazeEnv(), ENV_NAME)

    # The entry forwards only default_config() keys to the env and drops the
    # rest without a word, so a key that is neither an env key nor one the
    # entry reads itself is a setting that never takes effect.
    env_keys = set(default_config().keys())
    ignored = sorted(set(cfg.env_config) - env_keys - ENTRY_ENV_CONFIG_KEYS)
    assert not ignored, f"{arm.name}: env_config keys that reach nothing: {ignored}"


# --- GPU: the real env through the entry's wrap helper --------------------------


@pytest.fixture(scope="module")
def maze_env():
    return MazeForageVision(config=default_config())


def _treat_xy(env, data):
    import numpy as np

    return np.asarray(data.xpos[..., np.asarray(env._treat_body_ids), :2])


def _end_every_episode(env, wrapped, batch=3):
    """Resets ``batch`` envs, then forces one step that ends every episode.

    Returns:
        ``(first, after_done, next_step)`` states, ``after_done`` being the
        state on the far side of the auto-reset.
    """
    import jax
    import jax.numpy as jp

    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), batch))
    step = jax.jit(wrapped.step)
    action = jp.zeros((batch, env.action_size))
    first = state
    # Collecting every treat ends the episode.
    state.info["collected"] = jp.ones((batch, env.n_treats), dtype=bool)
    after_done = step(state, action)
    return first, after_done, step(after_done, action)


@pytest.mark.gpu
def test_full_reset_redraws_the_layout_every_episode(kl_anchor_entry, maze_env):
    import numpy as np

    cfg = OmegaConf.create({"wrappers": {"full_reset": True}})
    wrapped = kl_anchor_entry["_wrap_for_brax_training"](
        maze_env, cfg, episode_length=4, action_repeat=1
    )
    first, after, nxt = _end_every_episode(maze_env, wrapped)
    assert np.all(np.asarray(after.done) > 0.5)

    root = maze_env._rodent_root_qpos
    first_root = np.asarray(first.data.qpos[:, root : root + 7])
    new_root = np.asarray(after.data.qpos[:, root : root + 7])
    assert not np.allclose(
        _treat_xy(maze_env, after.data), _treat_xy(maze_env, first.data), atol=1e-6
    ), "treat layout did not change across the auto-reset boundary"
    assert not np.allclose(new_root[:, :2], first_root[:, :2], atol=1e-6), (
        "spawn xy did not change across the auto-reset boundary"
    )
    assert not np.allclose(new_root[:, 3:7], first_root[:, 3:7], atol=1e-6), (
        "spawn heading did not change across the auto-reset boundary"
    )
    # The info half is reset with it...
    assert not np.any(np.asarray(after.info["collected"]))
    assert np.all(np.asarray(after.info["n_collected"]) == 0)
    # ...and the next episode runs instead of ending on its first step.
    assert np.all(np.asarray(nxt.done) < 0.5)


@pytest.mark.gpu
@pytest.mark.parametrize("info_reset", [False, True])
def test_layout_stays_frozen_without_full_reset(kl_anchor_entry, maze_env, info_reset):
    """The control: without full resets the layout never changes.

    An info reset fixes the ``collected`` ratchet only; it cannot unfreeze a
    layout that lives in ``data.qpos``.
    """
    import numpy as np

    wrappers = {"full_reset": False}
    if info_reset:
        wrappers |= {
            "info_reset_on_done": True,
            "info_reset_keys": list(INFO_RESET_KEYS),
        }
    wrapped = kl_anchor_entry["_wrap_for_brax_training"](
        maze_env, OmegaConf.create({"wrappers": wrappers}), episode_length=4
    )
    first, after, _ = _end_every_episode(maze_env, wrapped, batch=2)
    assert np.all(np.asarray(after.done) > 0.5)

    root = maze_env._rodent_root_qpos
    np.testing.assert_allclose(
        _treat_xy(maze_env, after.data), _treat_xy(maze_env, first.data), atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(after.data.qpos[:, root : root + 7]),
        np.asarray(first.data.qpos[:, root : root + 7]),
        atol=1e-7,
    )
