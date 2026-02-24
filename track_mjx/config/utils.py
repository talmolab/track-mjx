"""Configuration utilities for track-mjx training pipelines.

This module provides helper functions for preparing and updating OmegaConf
configurations, including environment default layering and config merging.
"""

import json
import logging
import os
import subprocess
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf


def _get_package_commit(package_name: str) -> str:
    """Get the git commit hash for an installed package.

    Works for both editable installs (file://) and VCS installs (git+https://).

    Args:
        package_name: Name of the installed package.

    Returns:
        Git commit hash, or "unknown" if not available.
    """
    try:
        dist = distribution(package_name)
        direct_url = json.loads(dist.read_text("direct_url.json"))
        url = direct_url.get("url", "")

        # Editable install: file:// URL pointing to local repo
        if url.startswith("file://"):
            path = url[7:]
            result = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()

        # VCS install: git+https:// with vcs_info
        if "vcs_info" in direct_url:
            return direct_url["vcs_info"].get("commit_id", "unknown")

    except Exception:
        pass
    return "unknown"


from vnl_playground import registry as vnl_registry
from vnl_playground.tasks.fruitfly import consts as fruitfly_consts
from vnl_playground.tasks.rodent import consts as rodent_consts

# Project root directory (track-mjx/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_data_path(relative_path: str) -> str:
    """Resolve a relative data path to an absolute path from the project root.

    Args:
        relative_path: Path relative to the project root (e.g., "data/rodent/file.h5").

    Returns:
        Absolute path as a string.
    """
    return str(_PROJECT_ROOT / relative_path)


def _to_omegaconf_compatible(value: Any) -> Any:
    """Recursively coerce values to OmegaConf-compatible primitive containers."""
    if isinstance(value, dict):
        return {k: _to_omegaconf_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_omegaconf_compatible(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _get_track_env_overrides(env_name: str) -> dict[str, Any]:
    """Return track-mjx env defaults layered on top of vnl-playground defaults."""
    if env_name == "RodentImitation":
        return {
            "walker_name": "rodent",
            "walker_xml_path": str(rodent_consts.RODENT_XML_PATH),
            "arena_xml_path": str(rodent_consts.ARENA_XML_PATH),
            "reference_data_path": _resolve_data_path(
                "data/rodent/rodent_reference_clips.h5"
            ),
        }
    if env_name == "FruitflyImitation":
        return {
            "walker_name": "fruitfly",
            "walker_xml_path": str(fruitfly_consts.FRUITFLY_XML_PATH),
            "arena_xml_path": str(fruitfly_consts.ARENA_XML_PATH),
            "reference_data_path": _resolve_data_path(
                "data/fruitfly/fly_reference_clip.h5"
            ),
        }
    return {}


def _merge_env_config_with_defaults(cfg: DictConfig) -> DictConfig:
    """Merge cfg.env_config on top of vnl-playground task defaults.

    Ensures omitted environment parameters inherit per-task defaults while still
    allowing explicit YAML values to override them.

    Args:
        cfg: Root training config containing env_config.env_name.

    Returns:
        The input cfg with env_config replaced by the merged result.

    Raises:
        ValueError: If env_config.env_name is missing or unknown.
    """
    env_name = OmegaConf.select(cfg, "env_config.env_name", default=None)
    if env_name is None:
        raise ValueError("Missing required config key: env_config.env_name")

    try:
        default_env_cfg = vnl_registry.get_default_config(env_name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown env_name '{env_name}' in env_config. "
            "Expected a registered vnl_playground task name."
        ) from exc

    merged_env_cfg = OmegaConf.merge(
        OmegaConf.create(_to_omegaconf_compatible(default_env_cfg.to_dict())),
        OmegaConf.create(_get_track_env_overrides(env_name)),
        cfg.env_config,
    )

    OmegaConf.update(cfg, "env_config", merged_env_cfg, merge=False)
    return cfg


def prepare_config(
    cfg: DictConfig,
) -> tuple[DictConfig, dict[str, Any], config_dict.ConfigDict]:
    """Prepare config by merging env defaults and resolving track-mjx overrides.

    Takes a Hydra/OmegaConf configuration, validates that deprecated compatibility
    keys are not used, merges vnl-playground defaults with track-mjx defaults and
    user env_config overrides, then returns multiple config formats.

    Args:
        cfg: The root OmegaConf configuration containing env_config.

    Returns:
        A tuple containing:
            - cfg: The updated OmegaConf DictConfig with resolved paths.
            - cfg_dict: The full config as a plain Python dictionary.
            - env_cfg_ml: The env_config as an ml_collections ConfigDict.

    Raises:
        ValueError: If deprecated keys are present, if env_config.env_name is
            missing/unknown, or if no reference_data_path can be resolved.
    """
    if OmegaConf.select(cfg, "walker_config", default=None) is not None:
        raise ValueError(
            "`walker_config` has been removed. Move walker fields into `env_config`."
        )
    if OmegaConf.select(cfg, "data_path", default=None) is not None:
        raise ValueError(
            "`data_path` has been removed. Use `env_config.reference_data_path`."
        )

    cfg = _merge_env_config_with_defaults(cfg)

    if OmegaConf.select(cfg, "env_config.reference_data_path", default=None) is None:
        raise ValueError(
            "Missing env_config.reference_data_path after config preparation. "
            "Set env_config.reference_data_path explicitly."
        )

    # Create ml_collections ConfigDict for env_config
    env_cfg_dict = OmegaConf.to_container(cfg.env_config, resolve=True)
    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    # Convert full config to dict and log
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["env_config"]["vnl_playground_commit"] = _get_package_commit(
        "vnl-playground"
    )
    logging.info(f"Configs: {cfg_dict}")

    return cfg, cfg_dict, env_cfg_ml


def update_config(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    """Update an OmegaConf configuration with override values.

    Temporarily disables struct mode to allow adding/modifying keys,
    applies all overrides, then re-enables struct mode.

    Args:
        cfg: The OmegaConf DictConfig to update.
        overrides: Dictionary of key-value pairs to apply. Keys can use
            dot notation for nested updates (e.g., "env_config.num_envs").

    Returns:
        The updated DictConfig (modified in-place).
    """
    OmegaConf.set_struct(cfg, False)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    OmegaConf.set_struct(cfg, True)
    return cfg
