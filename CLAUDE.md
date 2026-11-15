# CLAUDE.md — ATP tennis match predictor

## Project summary
A **leakage-free** model that predicts the winner of ATP men's singles matches. It replicates a published approach:
Elo (overall + surface, dynamic K-factor, layoff bonus) + rolling serve/return stats + XGBoost.
Data: Jeff Sackmann `tennis_atp` (1990–2024), TML-Database (2025–2026), betting odds for a bookmaker baseline.
The full brief is in `PLAN.md`. Progress is logged in `PROGRESS.md`; judgement calls in `DECISIONS.md`.

Targets: Elo-only ~64–70% accuracy on recent seasons; XGBoost beats Elo on log loss with ~64–68% accuracy.
**Any full-season validation accuracy > 75% = leakage until proven otherwise.**

## Data leakage rules (non-negotiable; verbatim from PLAN.md section 2)

Every feature for a match must use only information available **before that match starts**.

1. **One chronological pass builds all features.** For each match, in order:
   read both players' state → write the feature row → **then** update state with the result and stats.
   Never compute features with `groupby`/`rolling`/`merge`/`expanding` over the whole table.
2. **Match order** = `tourney_date`, `tourney_id`, round rank, `match_num`.
   Round rank: `Q1 < Q2 < Q3 < Q4 < R128 < R64 < R32 < R16 < RR < QF < SF < BR < F`.
   `tourney_date` is the tournament **start** date, shared by every match in it. It is not the match date.
3. **Same-match columns are never model inputs:** `score`, `minutes`, all `w_*`, `l_*`, and anything prefixed
   `winner_`/`loser_`. They are used only to update state after the row is written.
4. **No winner/loser layout in the model table.** Assign Player A/B by a seeded coin flip (`config.seed`).
   Label `y = 1` if A won.
5. **Time-based splits only.** No random shuffles, no plain K-fold.
6. **The test set is evaluated once,** after all settings are frozen. Tuning and selection use validation only.
7. **Walkovers are dropped everywhere. Retirements (RET/DEF) are never labels.** Whether retirements update Elo is a
   config flag, default `false`.
8. **No duplicate matches** after merging sources.
9. **Elo K-factor leakage trap:** `m` in K = D/(Ne+m) must not depend on how far the player goes in the
   current event. Use a constant, or the matches played so far in the event.
10. Any learned preprocessing (imputers, encoders, calibrators) is fitted on training rows only.
    Calibration may be fitted on a validation slice.

## Conventions (from PLAN.md section 1)
- After every phase / significant decision, append to `PROGRESS.md` (date, phase, what was done, tests, metrics, open issues).
  Record judgement calls in `DECISIONS.md` (decision, alternatives, why).
- Commit after every phase. **Never commit `data/`** (or `models/`, `.venv/`).
- Tests are the gate: a phase is done only when its acceptance checks pass. Run `pytest -q` before every commit.
- **Never weaken, skip, xfail or delete a leakage test to make it pass.** A failing leakage test means the code is wrong.
- Simple, readable code. Pure functions where possible. Type hints. Docstrings on public functions.
- **All tunable numbers live in `config.yaml`** (load via `tennis.config.load_config()`). Nothing magic is hard-coded.
- Routine choices: pick the sensible default, log it in `DECISIONS.md`, keep going. Ask the human only at 🛑 STOP points.
- Be polite to websites: cache raw downloads on disk under `data/raw/`; never re-download what exists; ≥ 3 s between scrape requests.
- Parallel subagents write only to disjoint files. Only the orchestrator edits `config.yaml`, `PROGRESS.md`,
  `DECISIONS.md`, `pyproject.toml`, `CLAUDE.md`.

## Layout
```
src/tennis/config.py      config loader (load_config, config_hash, repo_path)
src/tennis/ingest/        sackmann, tml, scrape_tennisabstract, odds, player_map, merge
src/tennis/engine/        ordering, state, elo, rolling, h2h, build_features
src/tennis/model/         dataset, splits, elo_harness, train, tune, evaluate, predict
tests/{data,engine,leakage,model,fixtures}/
data/{raw,interim,processed}/   (gitignored)
experiments/  reports/{data,audits,figures}/  models/ (gitignored)
```

## How to run
Environment (Windows, Git Bash): a venv lives at `.venv/`.
```bash
.venv/Scripts/python -m pip install -e .          # install
.venv/Scripts/python -m pytest -q                 # all tests
.venv/Scripts/python -m pytest tests/leakage -q   # leakage suite
.venv/Scripts/python -m pytest -q -m "not slow"   # skip real-data tests
```
Always invoke Python as `.venv/Scripts/python` (the system `python` lacks the dependencies).
