"""The DMPO configs under ``track_mjx/config/dmpo`` compose and keep their rules.

Arms live in ``maze/``, ``full_maze_memory/``, ``spout_maze/`` and ``run_gap/``;
``base.yaml`` and the
two ``rodent-imitation*.yaml`` files are top-level configs. Paths are
environment-relative: priors hang off ``DMPO_PRIOR_ROOT`` and checkpoint
directories and graft sources off ``DMPO_CKPT_ROOT``, so no committed file
names a machine.
"""

import inspect
import pathlib
import re

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "track_mjx" / "config" / "dmpo"
MANIFEST = CONFIG_DIR / "priors" / "MANIFEST.md"

ARM_GROUPS = ("maze", "full_maze_memory", "spout_maze", "run_gap")
ARMS = sorted(p for group in ARM_GROUPS for p in (CONFIG_DIR / group).glob("*.yaml"))
TOP_LEVEL = ("base", "rodent-imitation", "rodent-imitation-intention")

PRIOR_ROOT = "${oc.env:DMPO_PRIOR_ROOT}"
CKPT_ROOT = "${oc.env:DMPO_CKPT_ROOT,dmpo_checkpoints}"

# Assembled from pieces so this file does not contain the string itself.
FORBIDDEN = "/" + "home" + "/"

# Legacy PPO high-level-transfer template that ships beside the maze DMPO arms.
NOT_A_DMPO_ARM = {"maze/maze_forage"}


def _name(path: pathlib.Path) -> str:
    return f"{path.parent.name}/{path.stem}"


ARM_NAMES = [_name(p) for p in ARMS]


@pytest.fixture(autouse=True)
def dummy_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DMPO_PRIOR_ROOT", "/dummy/priors")
    monkeypatch.setenv("DMPO_CKPT_ROOT", "/dummy/checkpoints")


def _compose(name: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name)


def _walk(node, path=()):
    """Yields ``(key path, value)`` for every leaf of a nested container."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, (*path, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, (*path, str(index)))
    else:
        yield path, node


def _raw(arm: pathlib.Path) -> dict:
    """The composed config with interpolations left unresolved."""
    return OmegaConf.to_container(_compose(_name(arm)), resolve=False)


def _manifest_ids() -> set[str]:
    rows = re.findall(
        r"^\| `([^`]+)` \| `PriorNetwork_\d+` \|", MANIFEST.read_text(), re.M
    )
    assert rows, f"no checkpoint rows found in {MANIFEST}"
    return set(rows)


def test_the_expected_arm_groups_are_present():
    counts = {g: len(list((CONFIG_DIR / g).glob("*.yaml"))) for g in ARM_GROUPS}
    assert all(counts.values()), counts


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_every_arm_composes(arm):
    cfg = _compose(_name(arm))
    # Resolving every interpolation proves the env-var roots are well formed.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved, f"{_name(arm)} composed to an empty config"


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_top_level_config_composes(name):
    resolved = OmegaConf.to_container(_compose(name), resolve=True)
    assert resolved["agent"] == "dmpo"


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_first_line_package_global(arm):
    assert arm.read_text().splitlines()[0] == "# @package _global_"


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_dmpo_arms_compose_to_agent_dmpo(arm):
    if _name(arm) in NOT_A_DMPO_ARM:
        pytest.skip(f"{_name(arm)} is a legacy PPO template")
    assert _compose(_name(arm)).get("agent") == "dmpo"


def test_no_absolute_home_paths():
    offenders = [
        str(path.relative_to(REPO))
        for path in sorted(CONFIG_DIR.rglob("*"))
        if path.is_file() and FORBIDDEN in path.read_text()
    ]
    assert not offenders, f"machine paths in: {offenders}"


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_no_finite_max_to_keep(arm):
    """Checkpoints are never pruned: an arm may not set a finite retention."""
    for key_path, value in _walk(_raw(arm)):
        if key_path[-1] in ("checkpoint_max_to_keep", "max_to_keep"):
            assert value is None, f"{'.'.join(key_path)} = {value!r}"


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_top_level_no_finite_max_to_keep(name):
    for key_path, value in _walk(OmegaConf.to_container(_compose(name))):
        if key_path[-1] in ("checkpoint_max_to_keep", "max_to_keep"):
            assert value is None, f"{'.'.join(key_path)} = {value!r}"


CKPT_PATH_KEYS = (
    "checkpoint_dir",
    "mimic_checkpoint_dir",
    "graft_ff_checkpoint",
    "warm_start_dmpo_checkpoint",
)


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_paths_hang_off_the_environment_roots(arm):
    raw = _raw(arm)
    for key_path, value in _walk(raw):
        key = key_path[-1]
        if value is None:
            continue
        if key == "prior_checkpoint_path":
            assert value.startswith(f"{PRIOR_ROOT}/"), (key_path, value)
        elif key in CKPT_PATH_KEYS:
            assert value.startswith(f"{CKPT_ROOT}/"), (key_path, value)
    if _name(arm) not in NOT_A_DMPO_ARM:
        # Run output always goes under the checkpoint root, never beside the cwd.
        assert raw["checkpoint_dir"].startswith(f"{CKPT_ROOT}/")


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_top_level_checkpoint_dir_hangs_off_the_checkpoint_root(name):
    """Configs that do not set their own run directory inherit a rooted one."""
    raw = OmegaConf.to_container(_compose(name), resolve=False)
    assert raw["checkpoint_dir"].startswith(f"{CKPT_ROOT}/")


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_prior_ids_are_in_the_manifest(arm):
    ids = _manifest_ids()
    for key_path, value in _walk(_raw(arm)):
        if key_path[-1] == "prior_checkpoint_path" and value is not None:
            prior_id = value.removeprefix(f"{PRIOR_ROOT}/")
            assert prior_id in ids, f"{_name(arm)}: {prior_id} not in the manifest"


def _warm_start_arm() -> str:
    """A run_gap arm that grafts a warm-start checkpoint."""
    for arm in ARMS:
        if arm.parent.name == "run_gap" and "warm_start_dmpo_checkpoint: ${" in (
            arm.read_text()
        ):
            return _name(arm)
    raise AssertionError("no run_gap arm sets warm_start_dmpo_checkpoint")


def test_resolved_paths_use_the_dummy_roots():
    name = _warm_start_arm()
    cfg = OmegaConf.to_container(_compose(name), resolve=True)
    assert cfg["checkpoint_dir"] == f"/dummy/checkpoints/{name.split('/')[1]}"
    assert cfg["transfer"]["prior_checkpoint_path"].startswith("/dummy/priors/")
    warm_start = cfg["transfer"]["warm_start_dmpo_checkpoint"]
    assert warm_start.startswith("/dummy/checkpoints/")


def test_checkpoint_root_defaults_to_a_relative_directory(monkeypatch):
    monkeypatch.delenv("DMPO_CKPT_ROOT")
    name = _warm_start_arm()
    cfg = OmegaConf.to_container(_compose(name), resolve=True)
    assert cfg["checkpoint_dir"] == f"dmpo_checkpoints/{name.split('/')[1]}"


# --- imitation configs ----------------------------------------------------------


def _domain_randomization_params() -> set[str]:
    from track_mjx.agent.domain_randomization import domain_randomization_maker

    return set(inspect.signature(domain_randomization_maker).parameters)


@pytest.mark.parametrize("name", ["rodent-imitation", "rodent-imitation-intention"])
def test_imitation_domain_randomization_keys_match_the_maker(name):
    """The entry reads each maker argument off ``domain_randomization``.

    A key the maker no longer takes is dead and a key it takes but the config
    lacks fails the entry at startup.
    """
    cfg = _compose(name)
    dr = set(cfg.env_config.domain_randomization) - {"use_domain_randomization"}
    assert dr == _domain_randomization_params()


def test_intention_imitation_inherits_rodent_full_clips():
    cfg = _compose("rodent-imitation-intention")
    assert cfg.env_config.env_name == "RodentImitation"
    assert cfg.train_setup.train_config.num_envs == 1024
    assert cfg.network_config.stochastic_latent is False
    # rodent-full-clips is the only source of these keys.
    assert "reward_terms" in cfg.env_config
    assert "termination_criteria" in cfg.env_config


def test_intention_imitation_composes_from_any_working_directory(tmp_path, monkeypatch):
    """``hydra.searchpath`` is package-relative, not cwd-relative."""
    monkeypatch.chdir(tmp_path)
    assert _compose("rodent-imitation-intention").agent == "dmpo"
