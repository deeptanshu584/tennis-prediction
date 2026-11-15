# PROGRESS.md

## 2026-11-16 — Phase 0: Setup
**Done**
- `git init`; created the repository layout from PLAN.md §6 (data/, models/ and .venv/ are gitignored).
- `pyproject.toml` (Python ≥ 3.11; the plan's dependency list plus `xlrd`), `.venv` on Python 3.13.5, editable install.
- `CLAUDE.md` with PLAN §2 verbatim, the §1 conventions and how to run tests.
- `.claude/agents/`: data-engineer, feature-engineer, ml-engineer, leakage-auditor, scraper (copied verbatim from PLAN §3).
- `config.yaml` holding every tunable number (seed, Elo start/K/m/layoff, N, k list, years, levels, round order,
  split dates, retirement flags, harness grid, XGBoost search space, evaluation and leakage-test bands).
- `src/tennis/config.py` (load_config, config_hash, repo_path); `tests/test_config.py` smoke tests.
- `config/player_overrides.csv` (header only), README with the licence note, DECISIONS D-001…D-006.

**Tests:** `pytest -q` → 3 passed (config smoke tests). Installed pandas 3.1.1, xgboost 3.4.1, optuna 5.0.0, scikit-learn 1.9.1.
**Open issues:** custom agent types may need a session restart to register (D-003 fallback).
