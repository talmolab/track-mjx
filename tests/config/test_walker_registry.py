# tests/config/test_walker_registry.py
"""Tests for walker path registry."""

import pytest
from track_mjx.config.walker_registry import get_walker_defaults, WALKER_DEFAULTS


class TestWalkerDefaults:
    """Tests for the WALKER_DEFAULTS registry."""

    def test_rodent_defaults_exist(self):
        defaults = get_walker_defaults("rodent")
        assert "walker_xml_path" in defaults
        assert "arena_xml_path" in defaults
        assert "reference_data_path" in defaults

    def test_fruitfly_defaults_exist(self):
        defaults = get_walker_defaults("fruitfly")
        assert "walker_xml_path" in defaults
        assert "arena_xml_path" in defaults
        assert "reference_data_path" in defaults

    def test_unknown_walker_raises(self):
        with pytest.raises(ValueError, match="Unknown walker name"):
            get_walker_defaults("unknown_walker")

    def test_not_implemented_walker_raises(self):
        """Walkers with status='not_implemented' raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet fully implemented"):
            get_walker_defaults("celegans")
        with pytest.raises(NotImplementedError, match="not yet fully implemented"):
            get_walker_defaults("stickbug")

    def test_mouse_not_implemented(self):
        """Mouse walker raises NotImplementedError (arena/data not ready)."""
        with pytest.raises(NotImplementedError, match="not yet fully implemented"):
            get_walker_defaults("mouse")

    def test_rodent_xml_path_ends_with_xml(self):
        defaults = get_walker_defaults("rodent")
        assert str(defaults["walker_xml_path"]).endswith(".xml")
        assert str(defaults["arena_xml_path"]).endswith(".xml")

    def test_all_implemented_walkers(self):
        """All implemented walkers should return valid defaults."""
        for walker_name, entry in WALKER_DEFAULTS.items():
            if entry.get("status") == "not_implemented":
                continue
            defaults = get_walker_defaults(walker_name)
            assert "walker_xml_path" in defaults
