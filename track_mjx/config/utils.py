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

from vnl_playground import registry as vnl_registry
from vnl_playground.tasks.fruitfly import consts as fruitfly_consts
from vnl_playground.tasks.rodent import consts as rodent_consts
from vnl_playground.tasks.celegans import consts as celegans_consts

# Project root directory (track-mjx/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


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


def _resolve_data_path(relative_path: str) -> str:
    """Resolve a relative data path to an absolute path from the project root.

    Args:
        relative_path: Path relative to the project root (e.g., "data/rodent/file.h5").

    Returns:
        Absolute path as a string.
    """
    return str(_PROJECT_ROOT / relative_path)


_TRACK_ENV_OVERRIDES_BY_ENV_NAME: dict[str, dict[str, Any]] = {
    "RodentImitation": {
        "walker_name": "rodent",
        "walker_xml_path": str(rodent_consts.RODENT_XML_PATH),
        "arena_xml_path": str(rodent_consts.ARENA_XML_PATH),
        "reference_data_path": _resolve_data_path(
            "data/rodent/rodent_reference_clips.h5"
        ),
    },
    "FruitflyImitation": {
        "walker_name": "fruitfly",
        "walker_xml_path": str(fruitfly_consts.FRUITFLY_XML_PATH),
        "arena_xml_path": str(fruitfly_consts.ARENA_XML_PATH),
        "reference_data_path": _resolve_data_path(
            "data/fruitfly/fly_reference_clip.h5"
        ),
    },
}


def _to_omegaconf_compatible(value: Any) -> Any:
    """Recursively coerce values to OmegaConf-compatible primitive containers."""
    if isinstance(value, dict):
        return {k: _to_omegaconf_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_omegaconf_compatible(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if not isinstance(value, (str, int, float, bool, type(None))):
        logging.debug(
            "Passing through non-primitive value in _to_omegaconf_compatible: %s",
            type(value).__name__,
        )
    return value


def _get_track_env_overrides(env_name: str) -> dict[str, Any]:
    """Return track-mjx-specific env overrides for a given env_name."""
    overrides = _TRACK_ENV_OVERRIDES_BY_ENV_NAME.get(env_name)
    if overrides is not None:
        # Return a copy so callers cannot mutate module-level defaults.
        return dict(overrides)

    if env_name:
        logging.warning(
            "No track-mjx-specific env overrides found for env_name='%s'. "
            "Using vnl-playground defaults and user env_config only.",
            env_name,
        )
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

    track_overrides = _get_track_env_overrides(env_name)

    merged_env_cfg = OmegaConf.merge(
        OmegaConf.create(_to_omegaconf_compatible(default_env_cfg.to_dict())),
        OmegaConf.create(track_overrides),
        cfg.env_config,
    )

    # Force-apply installation-specific paths from track-mjx overrides.
    # These are determined by the local package installation and must not
    # be overridden by stale absolute paths saved in checkpoints.
    for key in ("walker_xml_path", "arena_xml_path", "reference_data_path"):
        if key in track_overrides:
            OmegaConf.update(merged_env_cfg, key, track_overrides[key], merge=False)

    OmegaConf.update(cfg, "env_config", merged_env_cfg, merge=False)
    return cfg


def prepare_config(
    cfg: DictConfig,
) -> tuple[DictConfig, dict[str, Any], config_dict.ConfigDict]:
    """Prepare config by merging env defaults and resolving track-mjx overrides.

    Takes a Hydra/OmegaConf configuration, merges vnl-playground defaults with
    track-mjx defaults and user env_config overrides, then returns multiple
    config formats.

    Args:
        cfg: The root OmegaConf configuration containing env_config.

    Returns:
        A tuple containing:
            - cfg: The updated OmegaConf DictConfig with merged env defaults.
            - cfg_dict: The full config as a plain Python dictionary.
            - env_cfg_ml: The env_config as an ml_collections ConfigDict.

    Raises:
        ValueError: If env_config.env_name is missing/unknown or if no
            reference_data_path can be resolved.
    """
    cfg = _merge_env_config_with_defaults(cfg)

    # If data_path is explicitly set (e.g., from a notebook pointing to
    # locally-downloaded data), use it as reference_data_path so that stale
    # absolute paths baked into checkpoints are overridden.
    data_path = OmegaConf.select(cfg, "data_path", default=None)
    if data_path is not None:
        OmegaConf.update(
            cfg,
            "env_config.reference_data_path",
            str(data_path),
            merge=False,
        )

    if OmegaConf.select(cfg, "env_config.reference_data_path", default=None) is None:
        raise ValueError(
            "Missing env_config.reference_data_path after config preparation. "
            "Set env_config.reference_data_path explicitly."
        )

    # Preserve this field in cfg/env_cfg_ml/cfg_dict for run metadata parity.
    # Use force_add for struct configs where this key is not predeclared.
    OmegaConf.update(
        cfg,
        "env_config.vnl_playground_commit",
        _get_package_commit("vnl-playground"),
        merge=False,
        force_add=True,
    )

    # Create ml_collections ConfigDict for env_config
    env_cfg_dict = OmegaConf.to_container(cfg.env_config, resolve=True)
    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    # Convert full config to dict and log
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
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
