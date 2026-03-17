# tests/config/test_walker_registry.py
"""Tests for walker path registry."""

import pytest
from unittest.mock import patch
from omegaconf import OmegaConf
from track_mjx.config.walker_registry import get_walker_defaults, WALKER_DEFAULTS
from track_mjx.config.utils import prepare_config


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


class TestPrepareConfigOverrides:
    """Tests that walker_config can override default XML paths."""

    def _make_cfg(self, walker_name="rodent", **walker_overrides):
        """Helper to create a minimal OmegaConf config for testing."""
        cfg_dict = {
            "walker_config": {
                "walker_name": walker_name,
                "torque_actuators": True,
                "rescale_factor": 0.9,
                **walker_overrides,
            },
            "env_config": {
                "env_name": "RodentImitation",
            },
        }
        return OmegaConf.create(cfg_dict)

    @patch("track_mjx.config.utils._get_package_commit", return_value="test-commit")
    def test_default_paths_used_when_no_override(self, mock_commit):
        """Without overrides, registry defaults are used."""
        cfg = self._make_cfg()
        cfg, _, _ = prepare_config(cfg)
        assert "rodent_no_tail_collisions.xml" in cfg.env_config.walker_xml_path
        assert "arena.xml" in cfg.env_config.arena_xml_path

    @patch("track_mjx.config.utils._get_package_commit", return_value="test-commit")
    def test_walker_xml_path_override(self, mock_commit):
        """walker_config.walker_xml_path overrides the default."""
        custom_path = "/custom/path/to/my_rodent.xml"
        cfg = self._make_cfg(walker_xml_path=custom_path)
        cfg, _, _ = prepare_config(cfg)
        assert cfg.env_config.walker_xml_path == custom_path

    @patch("track_mjx.config.utils._get_package_commit", return_value="test-commit")
    def test_arena_xml_path_override(self, mock_commit):
        """walker_config.arena_xml_path overrides the default."""
        custom_path = "/custom/path/to/my_arena.xml"
        cfg = self._make_cfg(arena_xml_path=custom_path)
        cfg, _, _ = prepare_config(cfg)
        assert cfg.env_config.arena_xml_path == custom_path

    @patch("track_mjx.config.utils._get_package_commit", return_value="test-commit")
    def test_reference_data_path_override(self, mock_commit):
        """walker_config.reference_data_path overrides the default."""
        custom_path = "/custom/path/to/my_clips.h5"
        cfg = self._make_cfg(reference_data_path=custom_path)
        cfg, _, _ = prepare_config(cfg)
        assert cfg.env_config.reference_data_path == custom_path

    @patch("track_mjx.config.utils._get_package_commit", return_value="test-commit")
    def test_partial_override(self, mock_commit):
        """Can override just one path, others use defaults."""
        custom_walker = "/custom/rodent.xml"
        cfg = self._make_cfg(walker_xml_path=custom_walker)
        cfg, _, _ = prepare_config(cfg)
        assert cfg.env_config.walker_xml_path == custom_walker
        assert "arena.xml" in cfg.env_config.arena_xml_path

    @patch("track_mjx.config.utils._get_package_commit", return_value="test-commit")
    def test_not_implemented_walker_raises_without_overrides(self, mock_commit):
        """Not-implemented walkers raise NotImplementedError without overrides."""
        cfg = self._make_cfg(walker_name="celegans")
        with pytest.raises(NotImplementedError):
            prepare_config(cfg)

    @patch("track_mjx.config.utils._get_package_commit", return_value="test-commit")
    def test_not_implemented_walker_works_with_all_overrides(self, mock_commit):
        """Not-implemented walkers work when all paths are provided."""
        cfg = self._make_cfg(
            walker_name="mouse",
            walker_xml_path="/custom/mouse.xml",
            arena_xml_path="/custom/arena.xml",
            reference_data_path="/custom/data.h5",
        )
        cfg, _, _ = prepare_config(cfg)
        assert cfg.env_config.walker_xml_path == "/custom/mouse.xml"
        assert cfg.env_config.arena_xml_path == "/custom/arena.xml"
        assert cfg.env_config.reference_data_path == "/custom/data.h5"
