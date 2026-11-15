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

## 2026-11-16 — Phase 1: Foundations ✅
**Data blocker → resolved by the human**
- `github.com/JeffSackmann/tennis_atp` returns 404. The TML GitHub copy is tour-level only and stops in mid-January
  2026. 🛑 Raised with the human, who supplied an archive copy at `data/raw/tennis_atp/` (D-007).
- Orchestrator verified the archive: `reports/data/source_check.md`. All files and columns are present; no read
  errors, missing IDs or duplicate match IDs. Tour-level counts for 2000–2025 match TML within 0–6 rows a year. The
  archive covers 2025 fully and 2026 to Roland Garros (early June). **No 2026 H2 data, which affects the Phase 7 test window.**
- Human decision: K-factor m = 1 + matches already played in the current event (D-008).

**Agents (run in parallel)**
- feature-engineer: `src/tennis/engine/elo.py` (expected score, update, dynamic K, layoff bonus, EloParams) and
  `tests/engine/test_elo.py`; 58 tests, including 2063 beats 2120 at K = 24 → +13.95.
- leakage-auditor: the full PLAN §9 suite in `tests/leakage/` (tests 00–09). It includes an independent reference
  implementation for test 2 and a synthetic fixture (`tests/fixtures/synthetic.py`, ~2,060 matches, every edge case).
  It was checked against a mock engine, and all planted leaks were caught. Its audit found 7 contradictory duplicate
  pairs in the real data → fixed (D-014).
- data-engineer: `src/tennis/ingest/{sackmann,player_map,merge}.py`, `tests/data/` (96 tests),
  `reports/data/ingest_sackmann.md`. `python -m tennis.ingest.merge` builds `data/interim/matches.parquet` in about 13 s:
  **337,721 matches** (1990-01-01 → 2026-06-01), all match_ids unique, 0 order ties, 9,934 players, 0 unmatched.
  Tour-level training rows run 2,641–3,313 a year for 2000–2025 (2020: 1,430, COVID; 2026: 1,405, partial).
  Flags: walkover 1,506 · retirement 10,229 · qualifying 59,999 (none of them training-level) · surface_imputed 53 ·
  stats_invalid 5 · Carpet→Hard 14,672.
- Orchestrator: `src/tennis/engine/CONTRACT.md` pins the shared schema and APIs. Rulings on the auditor's 7
  ambiguities (int order_key, a total match order with match_id as tiebreaker, layoff list per tourney_id, k_log incl.
  RET/DEF, time_split per split type, config-generated feature names). Decisions D-007…D-016.

**Tests:** `pytest -q` → 169 passed, 32 skipped. Every skip names a target not built yet: engine (Phase 2), splits
(Phase 3), features.parquet, `cfg.model.features`, or no experiment logs. Leakage test 5 (duplicates) and test 0
pass on the real data; test 7 passes.
**Open issues:** 2026 H2 data for Phase 7 → resolved by the human: shortened forward test, Apr–early Jun 2026 (D-017).
