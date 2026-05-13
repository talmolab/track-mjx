"""Smoke: load the stick DMPO config, build the env, reset + step."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import jax
import jax.numpy as jp
import yaml


CONFIG_PATH = Path(
    "/home/talmolab/Desktop/SalkResearch/track-mjx/track_mjx/config/stick/stick-dmpo-imitation.yaml"
)


def test_config_keys_match_env_default_config():
    """Every env_config key in the YAML must exist in StickImitation default config."""
    from vnl_playground.tasks.stick import imitation
    yaml_cfg = yaml.safe_load(CONFIG_PATH.read_text())
    env_keys = set(yaml_cfg["env_config"].keys())
    default_keys = set(imitation.default_config().keys())
    # env_name is the registry key, not an env config field; skip it.
    env_keys.discard("env_name")
    missing = env_keys - default_keys
    assert not missing, f"YAML has keys absent from default_config: {missing}"


def test_walker_registry_resolves_stickbug_files():
    from track_mjx.config.walker_registry import get_walker_defaults
    entry = get_walker_defaults("stickbug")
    for k in ("walker_xml_path", "arena_xml_path", "reference_data_path"):
        assert Path(entry[k]).is_file(), f"{k}={entry[k]}"


def test_env_loaded_from_config_resets_and_steps():
    """Wire up the env from the YAML's env_config block and do one step."""
    from ml_collections import config_dict
    from vnl_playground import registry
    from etils.epath.gpath import PosixGPath

    yaml_cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = registry.get_default_config("StickImitation")
    # Selectively override only the keys that exist on the default config.
    for k, v in yaml_cfg["env_config"].items():
        if k in cfg and k != "env_name":
            cfg[k] = v
    # reference_data_path, walker_xml_path, arena_xml_path are typed as
    # PosixGPath in the config dict, so we must cast explicitly.
    cfg.reference_data_path = PosixGPath(
        "/home/talmolab/Desktop/SalkResearch/track-mjx/data/stick/stick_reference_clips.h5"
    )
    cfg.rescale_factor = yaml_cfg["walker_config"]["rescale_factor"]
    cfg.torque_actuators = yaml_cfg["walker_config"]["torque_actuators"]

    env = registry.load("StickImitation", config=cfg, flatten_obs=False)
    state = env.reset(jax.random.PRNGKey(0))
    assert jp.all(jp.isfinite(state.data.qpos))
    next_state = env.step(state, jp.zeros(env.action_size))
    assert jp.all(jp.isfinite(next_state.data.qpos))
