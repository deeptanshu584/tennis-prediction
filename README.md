# ATP tennis match predictor

This is a leakage-free model that predicts the winner of ATP men's singles matches. It combines Elo ratings (overall
and per surface, with a dynamic K-factor and a layoff bonus), rolling serve and return statistics, and XGBoost.

**Results:** see [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md).
- 2026 forward test (April–June, n = 473): 67.6% accuracy, log loss 0.592.
- Wimbledon 2025 (n = 122): 65.6%.

The build brief is in `PLAN.md`, the progress log in `PROGRESS.md`, and every judgement call in `DECISIONS.md`.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .     # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pytest -q -m "not slow"   # fast tests (about 2 min)
.venv/Scripts/python -m pytest -q                 # everything, including real-data and slow leakage tests (~8 min)
```
Every tunable number is in `config.yaml`. Paths there are relative to the repository root.

## Data
Put the raw files under `data/raw/`. This folder is never committed, and nothing is downloaded automatically.

| What | Where (config key) |
|---|---|
| Jeff Sackmann `tennis_atp` CSVs (`atp_matches_YYYY.csv`, `atp_matches_qual_chall_YYYY.csv`, `atp_players.csv`) | `paths.sackmann_dir` |
| TML-Database CSVs (2025–2026 cross-check) | `paths.tml_dir` |
| Optional: tennis-data.co.uk odds files, for a bookmaker baseline | `paths.odds_dir` |

## Run end to end
```bash
PY=.venv/Scripts/python
$PY -m tennis.ingest.merge                    # → data/interim/matches.parquet, player_map.csv
$PY -m tennis.engine.build_features           # → data/processed/features.parquet (one chronological pass, ~2 min)
$PY -m tennis.model.evaluate                  # baselines: higher rank, Elo, surface Elo, blend, rank logistic
$PY -m tennis.model.elo_harness               # Elo K-factor grid (Phase 4; results already in config.yaml)
$PY -m tennis.model.tune                      # XGBoost Optuna search on the dev split (Phase 5; slow)
$PY -m tennis.model.evaluate_final --split replication   # validation only; never touches the test
$PY -m tennis.model.evaluate_final --split forward --tune  # re-tune on 2026 Q1 (validation only)
```
Test scoring (`evaluate_final --split replication|forward --score-test`) is a one-shot step. It refuses to run a
second time and needs the frozen settings in `config.yaml`. Both tests have already been scored once
(`experiments/replication.jsonl`, `experiments/forward.jsonl`).

## Predict a match
```bash
$PY -m tennis.model.predict --p1 "Jannik Sinner" --p2 "Carlos Alcaraz" --surface Grass --level G --round F --best-of 5
```
- It prints P(p1 wins) from the production XGBoost model, plus the overall and surface Elo probabilities.
- The state is the one after the latest data (cut-off 2026-06-01).
- `--date YYYY-MM-DD` sets the tournament start date (default: the day after the cut-off).
- Players are matched by full name, ignoring case and accents, or by Sackmann id. An ambiguous name lists candidates.

Score a whole round from a CSV with columns `p1,p2,surface,level,round,best_of[,date]`:
```bash
$PY -m tennis.model.predict --draw draw.csv --out draw_predictions.csv
```

- The first call builds and caches the engine state (`models/state_<hash>.pkl`, about 1 min). It also fits the
  production model, using all rows to the cut-off with the frozen 2026 settings. That model is never evaluated.
- The XGBoost output is symmetrised: swapping p1 and p2 gives exactly 1 − p.

## Leakage safeguards
See the rules in `CLAUDE.md` and the test suite in `tests/leakage/`. In short:
- Every feature comes from a single chronological pass: read state, write the row, then update.
- No same-match columns ever reach the model.
- Players A and B are assigned by a seeded coin flip.
- Splits are by time only.
- Each test set was scored once, after its settings were frozen.

Audit reports are in `reports/audits/`.

## Data and licence
Match data comes from Jeff Sackmann's [`tennis_atp`](https://github.com/JeffSackmann/tennis_atp), licensed
**CC BY-NC-SA 4.0**: non-commercial use only, attribution required, share-alike. Results derived from it inherit
those terms, including this project's models and reports. 2025–2026 data comes from the
[TML-Database](https://github.com/Tennismylife/TML-Database). No betting-odds data is used (DECISIONS D-029), and raw
data is never committed to this repository.
