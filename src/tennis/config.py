"""Config loading helpers. All tunable numbers live in config.yaml at the repo root."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = REPO_ROOT / "config.yaml"


@lru_cache(maxsize=4)
def _load_cached(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (or another YAML file) and return it as a fresh dict."""
    p = Path(path) if path is not None else CONFIG_PATH
    return yaml.safe_load(_load_cached(str(p.resolve())))


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a config dict (used to tag models and experiments)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def repo_path(rel: str | Path) -> Path:
    """Resolve a repo-relative path from config to an absolute path."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p
