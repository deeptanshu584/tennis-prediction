"""Fixtures for the data-ingestion tests (tests/data only)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tennis.config import load_config
from tests.data.helpers import FIXTURE_DIR


@pytest.fixture(scope="session")
def _base_cfg() -> dict[str, Any]:
    return load_config()


@pytest.fixture
def cfg(_base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Fresh copy of the real config.yaml."""
    return copy.deepcopy(_base_cfg)


@pytest.fixture
def fixture_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config pointing at the CSV fixtures in tests/data/fixtures/sackmann (year 2023 only)."""
    cfg["paths"]["sackmann_dir"] = str(FIXTURE_DIR)
    cfg["data"]["first_year"] = 2023
    cfg["data"]["last_sackmann_year"] = 2023
    return cfg
