"""Smoke tests for the config file."""

from tennis.config import config_hash, load_config


def test_config_loads_with_required_sections() -> None:
    cfg = load_config()
    for key in ["seed", "paths", "data", "cleaning", "elo", "features", "splits", "xgb", "evaluation"]:
        assert key in cfg, key


def test_round_order_matches_plan() -> None:
    cfg = load_config()
    assert cfg["data"]["round_order"] == [
        "Q1", "Q2", "Q3", "Q4", "R128", "R64", "R32", "R16", "RR", "QF", "SF", "BR", "F",
    ]


def test_config_hash_is_stable() -> None:
    cfg = load_config()
    assert config_hash(cfg) == config_hash(load_config())
