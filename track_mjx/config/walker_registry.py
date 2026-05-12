# track_mjx/config/walker_registry.py
"""Registry of default walker/arena XML paths per walker type.

Centralizes the mapping from walker names to their default XML and reference
data paths sourced from vnl_playground. This replaces the hardcoded if/elif
chain in utils.prepare_config() and makes it easy to add new walkers.

Users can override any of these defaults via walker_config in their YAML.
"""

from pathlib import Path
from typing import Any

from vnl_playground.tasks.fruitfly import consts as fruitfly_consts
from vnl_playground.tasks.mouse import consts as mouse_consts
from vnl_playground.tasks.rodent import consts as rodent_consts

# Project root directory (track-mjx/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_data_path(relative_path: str) -> str:
    """Resolve a relative data path to an absolute path from the project root."""
    return str(_PROJECT_ROOT / relative_path)


# Registry: walker_name -> dict of default paths
# Each entry must have: walker_xml_path, arena_xml_path, reference_data_path
# Entries with status="not_implemented" will raise NotImplementedError unless
# the user provides all required path overrides in walker_config.
WALKER_DEFAULTS: dict[str, dict[str, Any]] = {
    "rodent": {
        "walker_xml_path": str(rodent_consts.RODENT_NO_TAIL_COLLISION_XML),
        "arena_xml_path": str(rodent_consts.ARENA_XML_PATH),
        "reference_data_path": _resolve_data_path(
            "data/rodent/rodent_reference_clips.h5"
        ),
    },
    "fruitfly": {
        "walker_xml_path": str(fruitfly_consts.FRUITFLY_XML_PATH),
        "arena_xml_path": str(fruitfly_consts.ARENA_XML_PATH),
        "reference_data_path": _resolve_data_path(
            "data/fruitfly/fly_reference_clip.h5"
        ),
    },
    "mouse": {
        "walker_xml_path": str(mouse_consts.MOUSE_XML_PATH),
        "arena_xml_path": None,
        "reference_data_path": None,
        "status": "not_implemented",
    },
    "celegans": {
        "walker_xml_path": None,
        "arena_xml_path": None,
        "reference_data_path": None,
        "status": "not_implemented",
    },
    "stickbug": {
        "walker_xml_path": None,
        "arena_xml_path": None,
        "reference_data_path": None,
        "status": "not_implemented",
    },
}


def get_walker_defaults(walker_name: str) -> dict[str, Any]:
    """Get default paths for a walker by name.

    Args:
        walker_name: Name of the walker (e.g., "rodent", "fruitfly").

    Returns:
        Dict with keys: walker_xml_path, arena_xml_path, reference_data_path.

    Raises:
        ValueError: If walker_name is not in the registry.
        NotImplementedError: If the walker is registered but not yet implemented.
    """
    if walker_name not in WALKER_DEFAULTS:
        raise ValueError(
            f"Unknown walker name: '{walker_name}'. "
            f"Available walkers: {list(WALKER_DEFAULTS.keys())}"
        )
    entry = WALKER_DEFAULTS[walker_name]
    if entry.get("status") == "not_implemented":
        raise NotImplementedError(
            f"Walker '{walker_name}' is not yet fully implemented. "
            f"Provide walker_xml_path, arena_xml_path, and reference_data_path "
            f"in walker_config to use it with custom paths."
        )
    return entry
