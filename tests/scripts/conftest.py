"""Fixtures for the entry-script tests."""

import pytest

from tests.scripts._entries import load_entry


@pytest.fixture
def kl_anchor_entry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Module globals of ``scripts/train_dmpo_highlvl_kl_anchor.py``."""
    return load_entry("kl_anchor", monkeypatch)


@pytest.fixture
def imitation_entry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Module globals of ``scripts/train_dmpo.py``."""
    return load_entry("imitation", monkeypatch)
