"""Confirm stickbug is registered and points at existing files."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path


def test_stickbug_entry_has_paths():
    from track_mjx.config.walker_registry import get_walker_defaults
    entry = get_walker_defaults("stickbug")
    for k in ("walker_xml_path", "arena_xml_path", "reference_data_path"):
        assert k in entry and entry[k] is not None, k
        assert Path(entry[k]).is_file(), f"{k}={entry[k]} does not exist"


def test_stickbug_xml_is_mesh_fast():
    from track_mjx.config.walker_registry import get_walker_defaults
    entry = get_walker_defaults("stickbug")
    assert Path(entry["walker_xml_path"]).name == "stick_mesh_fast.xml"


def test_stickbug_reference_data_is_under_track_mjx_data():
    from track_mjx.config.walker_registry import get_walker_defaults
    entry = get_walker_defaults("stickbug")
    parts = Path(entry["reference_data_path"]).parts
    assert "data" in parts and "stick" in parts
