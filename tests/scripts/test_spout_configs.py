"""The spout-maze smoke arms differ from their bases only where they say.

``_smoke_t_ff_c3`` is ``mm_l0a_ff_c3`` plus the T-maze keys and smoke sizes, and
``_smoke_t_hold_ff_c3`` is ``_smoke_t_ff_c3`` plus the hold keys. The arms are
composed through hydra from ``track_mjx/config/dmpo/spout_maze`` with
interpolations left unresolved, so the assertions pin the environment-relative
paths (``DMPO_PRIOR_ROOT``, ``DMPO_CKPT_ROOT``) rather than a machine path.
"""

import pathlib

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

spout_maze = pytest.importorskip(
    "vnl_playground.tasks.rodent.spout_maze",
    reason="needs the spout-maze env, which the pinned vnl-playground lacks",
)

CONFIG_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "track_mjx" / "config" / "dmpo"
)
SPOUT_DIR = CONFIG_DIR / "spout_maze"
CKPT_ROOT = "${oc.env:DMPO_CKPT_ROOT,dmpo_checkpoints}"
PRIOR_ROOT = "${oc.env:DMPO_PRIOR_ROOT}"

# The T-maze spec defaults the smoke arm lists for the record.
_T_DEFAULTS = dict(
    t_corridor_width=0.25,
    t_segment_length=0.50,
    t_n_stem_segments=5,
    t_arm_length=0.6875,
    t_spout_offset=0.10,
    t_wall_thickness=0.05,
    t_entry_margin=0.03,
)


def _arm(name: str) -> dict:
    """The arm composed through hydra, interpolations unresolved."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=f"spout_maze/{name}")
    return OmegaConf.to_container(cfg, resolve=False)


def _first_line(name: str) -> str:
    return (SPOUT_DIR / f"{name}.yaml").read_text().splitlines()[0]


def _valid_env_keys() -> set[str]:
    """Keys the entry forwards to the env."""
    return set(spout_maze.default_config().keys()) | {"env_name", "flatten_obs"}


def test_smoke_t_config():
    """``_smoke_t_ff_c3`` = ``mm_l0a_ff_c3`` + the T keys + smoke sizes.

    Every env_config key survives the entry point's filter (unknown keys are
    DROPPED there).
    """
    assert _first_line("_smoke_t_ff_c3") == "# @package _global_"
    smoke = _arm("_smoke_t_ff_c3")
    base = _arm("mm_l0a_ff_c3")

    env_cfg = smoke["env_config"]
    valid = _valid_env_keys()
    # Keys the entry point consumes elsewhere (vision wrapper / brax wrapper /
    # logging).
    passthrough = set(base["env_config"]) - valid
    assert set(env_cfg) - valid <= passthrough, set(env_cfg) - valid - passthrough
    assert env_cfg["maze_shape"] == "T"
    assert env_cfg["appear_distance"] == pytest.approx(0.70)
    for key, value in _T_DEFAULTS.items():
        if key in env_cfg:
            assert env_cfg[key] == pytest.approx(value), key

    tc = smoke["train_config"]
    assert tc["num_envs"] == 256
    assert tc["num_timesteps"] == 3_000_000
    assert tc["eval_every_steps"] == 1_000_000
    assert tc["max_replay_size"] == 230_400
    assert smoke["logging_config"]["exp_name"] == "_smoke_t_ff_c3"
    # Run output and the graft source hang off the checkpoint root; the prior
    # hangs off the prior root.
    assert smoke["checkpoint_dir"] == f"{CKPT_ROOT}/_smoke_t_ff_c3"
    assert smoke["transfer"]["graft_ff_checkpoint"] == (
        f"{CKPT_ROOT}/maze_gen1_prior_mlp/DMPONetwork_482816000"
    )
    assert smoke["transfer"]["prior_checkpoint_path"] == (
        f"{PRIOR_ROOT}/260827_153154_576196"
    )
    top = [c for c in smoke["eval_render_config"]["cameras"] if c["label"] == "top"]
    assert len(top) == 1 and top[0]["distance"] >= 4.0

    # Everything else is mm_l0a_ff_c3 verbatim.
    changed_env = {"maze_shape", "appear_distance", "trial_timeout_steps", *_T_DEFAULTS}
    assert {k: v for k, v in env_cfg.items() if k not in changed_env} == {
        k: v for k, v in base["env_config"].items() if k not in changed_env
    }
    changed_train = {"num_envs", "num_timesteps", "eval_every_steps", "max_replay_size"}
    assert {k: v for k, v in tc.items() if k not in changed_train} == {
        k: v for k, v in base["train_config"].items() if k not in changed_train
    }
    for section in (
        "network_config",
        "transfer",
        "kl_anchor",
        "wrappers",
        "env_name",
        "walker",
    ):
        assert smoke[section] == base[section], section
    # T runs use a 15 s trial timeout and a full 30 s eval episode.
    assert env_cfg["trial_timeout_steps"] == 1500
    assert smoke["eval_render_config"]["episode_length"] == 3000
    erc, base_erc = dict(smoke["eval_render_config"]), dict(base["eval_render_config"])
    erc.pop("cameras"), base_erc.pop("cameras")
    erc.pop("episode_length"), base_erc.pop("episode_length")
    assert erc == base_erc


def test_smoke_t_hold_config():
    """``_smoke_t_hold_ff_c3`` = ``_smoke_t_ff_c3`` + the hold keys + its own run name."""
    assert _first_line("_smoke_t_hold_ff_c3") == "# @package _global_"
    hold = _arm("_smoke_t_hold_ff_c3")
    base = _arm("_smoke_t_ff_c3")

    env_cfg = hold["env_config"]
    valid = _valid_env_keys()
    passthrough = set(base["env_config"]) - valid
    assert set(env_cfg) - valid <= passthrough, set(env_cfg) - valid - passthrough
    expected = dict(
        hold_enabled=True,
        hold_min_s=1.0,
        hold_max_s=3.0,
        gate_boundaries=[1],
        trial_timeout_steps=1500,
        maze_shape="T",
        treat_visible_until_teleport=True,
    )
    for key, value in expected.items():
        assert env_cfg[key] == value, key
    assert hold["eval_render_config"]["episode_length"] == 3000
    assert hold["logging_config"]["exp_name"] == "_smoke_t_hold_ff_c3"
    assert hold["checkpoint_dir"] == f"{CKPT_ROOT}/_smoke_t_hold_ff_c3"
    # Everything else is _smoke_t_ff_c3 verbatim.
    new_env = {
        "hold_enabled",
        "hold_min_s",
        "hold_max_s",
        "gate_boundaries",
        "treat_visible_until_teleport",
    }
    assert {k: v for k, v in env_cfg.items() if k not in new_env} == base["env_config"]
    for section in (
        "train_config",
        "network_config",
        "transfer",
        "kl_anchor",
        "wrappers",
        "env_name",
        "walker",
        "eval_render_config",
        "checkpoint_max_to_keep",
    ):
        assert hold[section] == base[section], section
    lc, base_lc = dict(hold["logging_config"]), dict(base["logging_config"])
    for key in ("exp_name", "notes"):
        lc.pop(key), base_lc.pop(key)
    assert lc == base_lc
