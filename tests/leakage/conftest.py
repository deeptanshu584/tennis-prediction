"""Fixtures for the leakage suite.

Real-data tests (`@pytest.mark.realdata`, `@pytest.mark.slow`) skip when data/interim/matches.parquet (or
data/processed/features.parquet) is absent.

Real-data input universe: to keep the suite runnable, the engine-level real tests (1, 2, 4) feed the builder
every match from REAL_START_YEAR onward: the environment variable TENNIS_LEAKAGE_REAL_START_YEAR if set (e.g.
1990 for the full history), else config `leakage_tests.real_start_year` if present, else 2018. Matches before that
year are NOT fed, so Elo and all other state start fresh at REAL_START_YEAR; this is fine for leakage checks (the
reference implementation sees exactly the same universe) but it is not the production feature table.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import pandas as pd
import pytest

from tennis.config import load_config, repo_path
from tests.fixtures.synthetic import make_synthetic_matches
from tests.leakage.helpers import build

DEFAULT_REAL_START_YEAR = 2018


def _real_start_year(cfg: dict[str, Any]) -> int:
    env = os.environ.get("TENNIS_LEAKAGE_REAL_START_YEAR")
    if env:
        return int(env)
    return int((cfg.get("leakage_tests") or {}).get("real_start_year", DEFAULT_REAL_START_YEAR))


@pytest.fixture(scope="session")
def _session_cfg() -> dict[str, Any]:
    return load_config()


@pytest.fixture
def cfg(_session_cfg: dict[str, Any]) -> dict[str, Any]:
    """A fresh deep copy of config.yaml for each test (tests may modify it)."""
    return copy.deepcopy(_session_cfg)


@pytest.fixture(scope="session")
def _synthetic_session(_session_cfg: dict[str, Any]) -> pd.DataFrame:
    return make_synthetic_matches(seed=0, cfg=_session_cfg)


@pytest.fixture
def synthetic_matches(_synthetic_session: pd.DataFrame) -> pd.DataFrame:
    """Synthetic unified match table (3 seasons, shuffled rows). A private copy per test."""
    return _synthetic_session.copy(deep=True)


@pytest.fixture(scope="session")
def synthetic_build(_synthetic_session: pd.DataFrame, _session_cfg: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
    """(features, k_log) for the synthetic fixture with the default config. Skips if the engine is missing."""
    return build(_synthetic_session, _session_cfg, record_k=True)


@pytest.fixture(scope="session")
def real_matches(_session_cfg: dict[str, Any]) -> pd.DataFrame:
    path = repo_path(_session_cfg["paths"]["matches"])
    if not path.exists():
        pytest.skip(f"real data absent: {path} not found (built by the data-engineer)")
    df = pd.read_parquet(path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    return df


@pytest.fixture(scope="session")
def real_universe(real_matches: pd.DataFrame, _session_cfg: dict[str, Any]) -> pd.DataFrame:
    """Real matches from REAL_START_YEAR on: the whole input universe of the real engine-level tests."""
    start = _real_start_year(_session_cfg)
    return real_matches[real_matches["tourney_date"].dt.year >= start].reset_index(drop=True)


@pytest.fixture(scope="session")
def real_build(real_universe: pd.DataFrame, _session_cfg: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
    """(features, k_log) of the real universe with the default config. Skips if the engine is missing."""
    return build(real_universe, _session_cfg, record_k=True)


@pytest.fixture(scope="session")
def features_table(_session_cfg: dict[str, Any]) -> pd.DataFrame:
    """The production training table data/processed/features.parquet; skips if absent."""
    path = repo_path(_session_cfg["paths"]["features"])
    if not path.exists():
        pytest.skip(f"{path} not found (produced by the feature engine in Phase 2)")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
