"""The full-maze memory arms differ from their bases only where they say.

The arms in ``track_mjx/config/dmpo/full_maze_memory`` share one full-maze
setup and vary the policy head, the env count and, in the ``mm_fmt_*``
ablations, one setting each. Pinned here, from the configs composed through
hydra with interpolations left unresolved:

* every arm grafts the aligned feed-forward maze checkpoint;
* each arm equals its base arm except for the listed keys (and its names);
* the privileged critic width of every arm is the width the full-maze env
  writes (``env.critic_priv_dim`` with ``maze_critic_priv``, else 0), in the
  dtype the replay stores it in. The env is built on CPU, which compiles the
  model and runs nothing;
* the contact-termination arms build the floor-contact criterion with their
  margin.
"""

import os
import pathlib

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

maze_forage_vision = pytest.importorskip(
    "vnl_playground.tasks.rodent.maze_forage_vision",
    reason="needs the maze-forage env, which the pinned vnl-playground lacks",
)

CONFIG_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "track_mjx" / "config" / "dmpo"
)
GROUP = "full_maze_memory"
ARMS = sorted(p.stem for p in (CONFIG_DIR / GROUP).glob("*.yaml"))
CKPT_ROOT = "${oc.env:DMPO_CKPT_ROOT,dmpo_checkpoints}"
GRAFT = f"{CKPT_ROOT}/unified_maze_gen1_prior_mlp/DMPONetwork_482816000"

# Keys every arm sets to its own name.
NAME_KEYS = {"logging_config.exp_name", "logging_config.notes", "checkpoint_dir"}

_N1K = {
    "train_config.num_envs",
    "train_config.max_replay_size",
    "train_config.sgd_steps_per_rollout",
    "eval_render_config.num_envs",
}
_TIMESTEPS = "train_config.num_timesteps"
_ANCHOR = {"kl_anchor.alpha_anchor", "kl_anchor.w_anchor"}

# arm -> (base arm, keys that differ besides NAME_KEYS)
DELTAS = {
    "mm_fm_ff_c3_n1k": ("mm_fm_ff_c3", _N1K),
    "mm_fm_gru_l40_n1k": ("mm_fm_gru_l40", _N1K),
    "mm_fm_lstm_l40_n1k": ("mm_fm_lstm_l40", _N1K),
    "mm_fm_lstm_l100_n1k": ("mm_fm_lstm_l100", _N1K),
    "mm_fm_lstm_l200_n1k": ("mm_fm_lstm_l200", _N1K),
    "mm_fm_lstm_l40": (
        "mm_fm_gru_l40",
        {
            "network_config.policy_head_slow_core.cell",
            "network_config.policy_head_slow_core.forget_bias",
        },
    ),
    "mm_fmt_A_nopriv_ff": (
        "mm_fm_ff_c3_n1k",
        {
            "env_config.maze_critic_priv",
            "network_config.critic_privileged_dim",
            _TIMESTEPS,
        },
    ),
    "mm_fmt_B_g999_ff": ("mm_fm_ff_c3_n1k", {"train_config.discount"}),
    "mm_fmt_B_g999_lstm_l100": (
        "mm_fm_lstm_l100_n1k",
        {"train_config.discount", _TIMESTEPS},
    ),
    "mm_fmt_C_upright_ff": (
        "mm_fm_ff_c3_n1k",
        {"env_config.reward_terms.upright.weight", _TIMESTEPS},
    ),
    "mm_fmt_D_contact_ff": (
        "mm_fm_ff_c3_n1k",
        {"env_config.termination_criteria.contact_termination", _TIMESTEPS},
    ),
    "mm_fmt_Dsoft_contact_ff": (
        "mm_fm_ff_c3_n1k",
        {"env_config.termination_criteria.contact_termination.margin", _TIMESTEPS},
    ),
    "mm_fmt_E_klanchor_ff": ("mm_fmt_B_g999_ff", _ANCHOR | {_TIMESTEPS}),
    "mm_fmt_E_klanchor_lstm_l100": ("mm_fmt_B_g999_lstm_l100", _ANCHOR),
}


def _arm(name: str) -> dict:
    """The arm composed through hydra, interpolations unresolved."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=f"{GROUP}/{name}")
    return OmegaConf.to_container(cfg, resolve=False)


def _leaves(node, path=()):
    """``{dotted path: value}`` of a nested dict; an empty dict is a leaf."""
    if isinstance(node, dict) and node:
        out = {}
        for key, value in node.items():
            out |= _leaves(value, (*path, str(key)))
        return out
    return {".".join(path): node}


def test_the_group_holds_the_full_maze_arms():
    assert len(ARMS) == 19
    assert set(DELTAS) <= set(ARMS)


@pytest.mark.parametrize("name", ARMS)
def test_every_arm_grafts_the_aligned_maze_checkpoint(name):
    arm = _arm(name)
    assert arm["env_name"] == "RodentMazeForageVision"
    assert arm["transfer"]["graft_ff_checkpoint"] == GRAFT
    assert arm["checkpoint_dir"] == f"{CKPT_ROOT}/{name}"
    assert arm["logging_config"]["exp_name"] == name


@pytest.mark.parametrize("name", sorted(DELTAS))
def test_arm_differs_from_its_base_only_where_it_says(name):
    base_name, changed = DELTAS[name]
    arm, base = _leaves(_arm(name)), _leaves(_arm(base_name))
    differing = {
        key
        for key in arm.keys() | base.keys()
        if arm.get(key, ...) != base.get(key, ...)
    }
    assert differing - NAME_KEYS == changed


def _env_args(arm: dict) -> dict:
    """The env_config keys the entry forwards to the env."""
    valid = set(maze_forage_vision.default_config().keys())
    return {k: v for k, v in arm["env_config"].items() if k in valid}


def _innermost(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _build(arm: dict, work_dir: pathlib.Path):
    """The arm's innermost env, built in ``work_dir``.

    MuJoCo writes its compile warnings to ``MUJOCO_LOG.TXT`` in the working
    directory, so the build runs in a temp directory.
    """
    from vnl_playground import tasks

    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        env = tasks.load(
            arm["env_name"], flatten_obs=False, config_overrides=_env_args(arm)
        )
    finally:
        os.chdir(cwd)
    return _innermost(env)


@pytest.fixture(scope="module")
def critic_env(tmp_path_factory):
    """The full maze with the critic vector, as every flagged arm builds it."""
    return _build(_arm("mm_fm_ff_c3_n1k"), tmp_path_factory.mktemp("build"))


def _geometry(arm: dict) -> dict:
    """The env keys besides the critic switch, rewards and terminations."""
    skip = {"maze_critic_priv", "reward_terms", "termination_criteria"}
    return {k: v for k, v in _env_args(arm).items() if k not in skip}


@pytest.mark.parametrize("name", ARMS)
def test_critic_width_is_the_width_the_env_writes(name, critic_env):
    arm = _arm(name)
    # The env the width was read from is this arm's env up to the switch.
    assert _geometry(arm) == _geometry(_arm("mm_fm_ff_c3_n1k"))
    flag = arm["env_config"]["maze_critic_priv"]
    width = critic_env.critic_priv_dim if flag else 0
    assert arm["network_config"]["critic_privileged_dim"] == width
    if flag:
        # The replay field's dtype is read from the same key.
        assert (
            critic_env._critic_priv_dtype == arm["env_config"]["maze_critic_priv_dtype"]
        )
        # The vector's time entry is normalised by the wrapper's horizon.
        assert arm["env_config"]["episode_length"] == (
            critic_env._config.episode_length
        )


def test_the_critic_width_is_264_on_the_default_maze(critic_env):
    assert critic_env.critic_priv_dim == 264


@pytest.mark.parametrize(
    ("name", "margin"),
    [("mm_fmt_D_contact_ff", 0.0), ("mm_fmt_Dsoft_contact_ff", -0.002)],
)
def test_contact_arms_build_the_floor_contact_criterion(name, margin, tmp_path):
    raw = _build(_arm(name), tmp_path)
    assert raw._contact_plan is not None
    assert raw._contact_plan.margin == pytest.approx(margin)
    assert raw._contact_plan.n_candidates > 0


def test_the_other_arms_have_no_contact_criterion(critic_env):
    assert critic_env._contact_plan is None
