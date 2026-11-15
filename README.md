# ATP tennis match predictor

A leakage-free model that predicts the winner of ATP men's singles matches, using Elo (overall and surface, with a
dynamic K-factor), rolling serve/return statistics and XGBoost.

> Work in progress. The full build plan is in `PLAN.md`; progress is in `PROGRESS.md`.
> End-to-end run instructions will be added in Phase 8.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .     # Windows (use .venv/bin/python on Linux/macOS)
.venv/Scripts/python -m pytest -q
```

## Data and licence
Match data comes from Jeff Sackmann's [`tennis_atp`](https://github.com/JeffSackmann/tennis_atp), licensed
**CC BY-NC-SA 4.0** (non-commercial use only, attribution required, share-alike). Results derived from it, including
this project's models and reports, inherit those terms. 2025–2026 data comes from the
[TML-Database](https://github.com/Tennismylife/TML-Database). Betting odds (bookmaker baseline only) come from
tennis-data.co.uk. Raw data is never committed to this repository.
