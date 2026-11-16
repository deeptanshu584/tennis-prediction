# PLAN.md — Autonomous build plan: ATP tennis match predictor

> **For Claude Code.** This file is your full brief. Read all of it before doing anything.
> Then run Phase 0 and work through the phases in order. Run work in parallel where this plan says so.
> Stop only at the checkpoints marked **🛑 STOP**.

---

## 0. Mission

Build a **leakage-free** model that predicts the winner of ATP men's singles matches. It replicates a published approach:
Elo (overall + surface) + rolling serve/return stats + XGBoost, with a dynamic Elo K-factor.

Targets:
- Elo-only baseline: ~64–70% accuracy on recent seasons.
- XGBoost: beat the Elo baseline on **log loss**, with ~64–68% accuracy on a full held-out season.
- Reference points: the original author got 66.3% on Wimbledon 2025 (IBM 63.8%, bookmakers ~72%). His best validation accuracy was ~67%.
- **Any full-season validation accuracy > 75% = leakage until proven otherwise.** Stop and investigate.

Success = all phases complete, all leakage tests green, and a final report written to `reports/FINAL_REPORT.md`.

---

## 1. Ground rules for working autonomously

1. **Keep a log.** After every phase (and any significant decision), append to `PROGRESS.md`:
   date, phase, what was done, test results, metrics, open issues.
   Record judgement calls in `DECISIONS.md`: the decision, the alternatives, and why.
2. **Commit after every phase** with a clear message (`git init` in Phase 0). Never commit `data/`.
3. **Tests are the gate.** A phase is done only when its acceptance checks pass. Run `pytest -q` before every commit.
4. **Never weaken, skip, xfail or delete a leakage test to make it pass.** A failing leakage test means the code is wrong.
5. **Prefer simple, readable code** over clever code. Pure functions where possible. Type hints. Docstrings on public functions.
6. **All tunable numbers live in `config.yaml`.** Nothing magic is hard-coded.
7. **Don't ask the human about routine choices.** Pick the sensible default, log it in `DECISIONS.md`, keep going.
   Ask only at 🛑 STOP points.
8. **Be polite to websites.** Never hammer a site. Cache raw downloads on disk and never re-download what you already have.
9. **Context hygiene.** Delegate large, self-contained jobs to subagents (section 3) so the main session stays focused on orchestration.

### 🛑 STOP and ask the human only when:
- A data source needed for a phase can't be downloaded (network or permission blocked). Ask the human to fetch it manually.
- A leakage test still fails after 3 honest fix attempts.
- Validation accuracy > 75% and the cause hasn't been found after investigating.
- Scraping is needed (Phase 5, option B). Confirm with the human before you start scraping.
- All phases are done. Present the final report.

---

## 2. Data leakage rules (non-negotiable; copy into CLAUDE.md in Phase 0)

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

---

## 3. Agent team (create these in Phase 0)

Create these subagent files in `.claude/agents/`. Use them through the Task/Agent tool.
Run independent agents **in parallel** (several Task calls in a single message) wherever section 6 marks a phase ∥ parallel.

**File-ownership rule for parallel work:** agents running at the same time must write to **disjoint files/directories**,
listed in each phase. Only the main session (orchestrator) edits shared files: `config.yaml`, `PROGRESS.md`,
`DECISIONS.md`, `pyproject.toml`, `CLAUDE.md`. Subagents return any config or dependency changes they need as text,
and the orchestrator applies them.

### 3.1 `.claude/agents/data-engineer.md`
```markdown
---
name: data-engineer
description: Downloads, parses, cleans and merges tennis match data (Sackmann tennis_atp, TML-Database, scraped data, betting odds) into the unified Parquet schema. Use for any ingestion, schema, cleaning, de-duplication or player-ID mapping task.
tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch
---
You are a careful data engineer on a tennis prediction project. Read CLAUDE.md and PLAN.md sections 2, 4 and 5 first.
- Never modify anything under data/raw/. Write outputs to data/interim/ as Parquet.
- Every ingestion step must produce a short markdown report in reports/data/ (row counts per year and level, nulls per column, dropped rows and why).
- Map every player to a Sackmann player_id. Unmatched players are an error, not a warning.
- Never drop or alter the result of a match to "fix" data; flag it instead.
- Write pytest tests for your parsing and cleaning functions in tests/data/.
- Only touch the files and directories the orchestrator assigned to you. Return any needed config.yaml or dependency changes as text in your final message.
- End with a summary: files created, row counts, anomalies, open questions.
```

### 3.2 `.claude/agents/feature-engineer.md`
```markdown
---
name: feature-engineer
description: Implements the chronological feature engine — Elo (overall and surface, dynamic K, layoff bonus), rolling serve/return stats, head-to-head, form — and the training table. Use for any change under src/tennis/engine/ or src/tennis/model/dataset.py.
tools: Read, Write, Edit, Bash, Glob, Grep
---
You build the feature engine for a tennis predictor. Read CLAUDE.md and PLAN.md sections 2, 5 and 6 first.
The non-negotiable pattern: for each match in order -> snapshot both players' state -> emit row -> THEN update state.
- Never use groupby/rolling/merge over the full table to compute a feature.
- Rates are sums of counts divided by sums of denominators over the buffer, not means of per-match percentages.
- Write unit tests first (tests/engine/), including the Elo arithmetic check (2063 beats 2120 with K=24 -> +13.95).
- Keep state updates in small pure functions that are easy to test.
- Only touch the files assigned to you. Return any config.yaml changes as text.
- End with a summary: functions added, tests added and their status, and any assumptions (also logged for DECISIONS.md).
```

### 3.3 `.claude/agents/ml-engineer.md`
```markdown
---
name: ml-engineer
description: Trains, tunes, calibrates and evaluates models (Elo-only harness, XGBoost random search/Optuna, baselines, backtests, reports). Use for anything under src/tennis/model/ except dataset.py, and for experiment reports.
tools: Read, Write, Edit, Bash, Glob, Grep
---
You are the ML engineer on a tennis predictor. Read CLAUDE.md and PLAN.md sections 2, 7 and 8 first.
- Time-based splits only. Select on validation log loss (accuracy as a tiebreaker). Never compute test metrics during tuning.
- Log every experiment (params, metrics) to experiments/ as JSON lines, plus a readable markdown summary in reports/.
- Always report the Elo-only and rank-only baselines next to the model.
- If any validation accuracy exceeds 75%, stop the experiment and report it as suspected leakage.
- Only touch the files assigned to you. End with a summary: best settings, metrics table, plots saved, recommendations.
```

### 3.4 `.claude/agents/leakage-auditor.md`
```markdown
---
name: leakage-auditor
description: Independent reviewer that hunts for data leakage and look-ahead bias. Use after every phase that touches data, features, splits or models, and whenever results look too good. Must not have written the code it reviews.
tools: Read, Bash, Glob, Grep, Write
---
You are a skeptical auditor. Your only job is to find ways information from the future (or from the match itself) reaches the model.
Read CLAUDE.md and PLAN.md section 2 and section 9.
Checklist:
1. Read the feature engine and confirm features are snapshotted BEFORE state updates, for every feature.
2. Confirm match ordering (tourney_date, tourney_id, round rank, match_num) and that same-day ordering is correct.
3. grep the model feature list for w_, l_, winner, loser, score, minutes.
4. Check splits: max(train date) < min(val date) < min(test date); no match id in two splits; no duplicates.
5. Check the K-factor's m does not use future matches in the event.
6. Check that nothing (imputer, encoder, calibrator, feature selection) is fitted on validation/test, except a calibrator explicitly fitted on a validation slice.
7. Run the full leakage suite (pytest tests/leakage -q) and report the results.
8. Sanity-check the metrics: Elo-only accuracy should be ~64-70%; the model above 75% is a red flag.
You may write NEW tests in tests/leakage/ that reproduce a problem you find. Do not modify source code; report the fixes needed.
Output: PASS or FAIL, then a numbered list of findings with file:line and the suggested fix. Write the same to reports/audits/<phase>.md.
```

### 3.5 `.claude/agents/scraper.md` (only if Phase 5 option B is approved)
```markdown
---
name: scraper
description: Polite, cached web scraper for tennisabstract.com match data when no ready-made dataset covers the needed dates. Use only after the human has approved scraping.
tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch
---
Read robots.txt first and obey it. One request every 3+ seconds. Cache every raw page under data/raw/scraped/ and never re-fetch a cached page.
Parse into the same columns as Sackmann's atp_matches_YYYY.csv (see PLAN.md section 4). Write tests against saved HTML fixtures.
Report coverage per tournament and any parsing failures. Never invent missing values.
```

---

## 4. Data specification

### 4.1 Sources
| Source | Files | Use |
|---|---|---|
| Jeff Sackmann `tennis_atp` — `git clone --depth 1 https://github.com/JeffSackmann/tennis_atp data/raw/tennis_atp` | `atp_matches_YYYY.csv` (tour level), `atp_matches_qual_chall_YYYY.csv` (qualifying + Challengers), `atp_players.csv`, `atp_rankings_*.csv` | 1990–2024 history |
| TML-Database `https://github.com/Tennismylife/TML-Database` | Sackmann-like format, live-updated | First choice for 2025–2026. Verify coverage of Challengers, qualifying and stats before relying on it |
| tennisabstract.com | Match pages | Fallback for 2025–2026 (🛑 needs approval) |
| Betting odds (e.g. tennis-data.co.uk) | Yearly Excel/CSV files with bookmaker odds | Phase 7 bookmaker baseline only. If it can't be downloaded, 🛑 ask the human |

License: Sackmann data is CC BY-NC-SA 4.0 (non-commercial). Note it in README.md.

### 4.2 Sackmann columns (reference)
`tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date (YYYYMMDD), match_num,`
`winner_id, winner_seed, winner_entry, winner_name, winner_hand, winner_ht, winner_ioc, winner_age,`
`loser_* (same), score, best_of, round, minutes,`
`w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced, l_* (same),`
`winner_rank, winner_rank_points, loser_rank, loser_rank_points`

`tourney_level`: G = Grand Slam, M = Masters, A = other tour, F = Tour Finals, D = Davis Cup, O = Olympics, C = Challenger,
S/others = satellites/ITF. **Training-row levels = {A, G, M, F, D, O}**, as in the original author's code.
Check how qualifying rounds of tour events are labelled in the qual_chall files. Qualifying rounds (Q1–Q4) never produce
training rows, but they DO update state.

### 4.3 Unified schema (`data/interim/matches.parquet`)
One row per match, still winner/loser oriented at this stage, plus:
`match_id` (stable hash of tourney_id + match_num + both player ids), `source`, `round_rank` (int), `order_key` (sortable),
`is_walkover`, `is_retirement`, `is_training_level` (bool), `surface` normalised to {Hard, Clay, Grass} (Carpet → Hard; logged).

### 4.4 Cleaning rules
- Drop walkovers (`W/O`, `Walkover`, or no points played) everywhere.
- Flag RET/DEF. Never use them as labels. Elo updates for them follow config (default no). Their partial stats may update the rolling buffers (config, default yes).
- Missing stats stay NaN, and the match is skipped for stat buffers only.
- De-duplicate on `match_id` across sources. When merging, prefer Sackmann, then TML, then scraped.
- Player mapping: `data/interim/player_map.csv` (source, source_id or name → sackmann player_id), plus a hand-edited
  `config/player_overrides.csv`. Normalise accents, case, hyphens and name order. **Unmatched players fail the build.**

---

## 5. Feature engine specification

### 5.1 State per player
Overall Elo; Elo per surface (Hard/Clay/Grass); total match count; per-surface match count; list of tournament start
dates played; a deque of the last `N` matches' raw serve/return counts (own and opponent's); a deque of the last results;
head-to-head win counts vs each opponent (overall and per surface); the Elo history (for the trend feature).

### 5.2 Elo
```
E_A  = 1 / (1 + 10 ** ((R_B - R_A) / 400))
R_A' = R_A + K_A * (S_A - E_A)          # S_A = 1 win, 0 loss; each player uses their own K
start rating = 1500 (config)
```
Surface Elo: same update, using only matches on that surface and the surface match count for Ne.
Unit test: Alcaraz 2063 beats Djokovic 2120 with K = 24 → Alcaraz +13.95 (±0.01).

### 5.3 Dynamic K (from the author's code)
His signature: `updateStats(row, prev_stats, k_factor, base_k_factor, max_k_factor, div_number, bonus_after_layoff)`.
His "vanilla" test: `k_factor=24, base_k_factor=0, max_k_factor=0, div_number=0, bonus_after_layoff=False` → fixed K = 24.
```
if div_number == 0:
    K = k_factor
else:
    K = min(max_k, max(base_k, div_number / (Ne + m)))     # chess formula: div_number = 800
if bonus_after_layoff:
    K *= k_bonus_after_layoff(tourney_dates_played)
```
`m` was not shown in the videos. Default `m = 1`; it is tuned in Phase 4.

### 5.4 Layoff bonus (the author's code, verbatim logic)
```python
def k_bonus_after_layoff(played, threshold=100):
    gaps = gaps_between_tourneys(played)   # gaps in DAYS between consecutive tournaments, MOST RECENT FIRST
    if len(gaps) <= 15:
        return 1.0
    bonus = 1.5
    for g in gaps:
        if g < threshold:
            bonus -= 0.05
        else:
            break
    return max(1.0, bonus)
```
`played` = start dates of tournaments before this one, plus the current tournament's start date (known in advance, so not leakage).
The ordering and units are assumptions; log them in DECISIONS.md.

### 5.5 Features (per player, snapshotted pre-match)
Rates over the last `N` matches with stats (default N = 50). Each is a sum of numerators divided by a sum of denominators. "opp" = the opponent's serve counts in those matches.

| Feature | Definition |
|---|---|
| p_ace | ace / svpt |
| p_df | df / svpt |
| p_1stIn | 1stIn / svpt |
| p_1stWon | 1stWon / 1stIn |
| p_2ndWon | 2ndWon / (svpt − 1stIn) |
| p_bpSaved | bpSaved / bpFaced |
| p_rpw | (opp_svpt − opp_1stWon − opp_2ndWon) / opp_svpt |
| p_retAceAgainst | opp_ace / opp_svpt |
| p_ret1stWon | (opp_1stIn − opp_1stWon) / opp_1stIn |
| p_ret2ndWon | (opp_svpt − opp_1stIn − opp_2ndWon) / (opp_svpt − opp_1stIn) |
| p_bpConv | (opp_bpFaced − opp_bpSaved) / opp_bpFaced |
| p_totalPtsWon | (1stWon + 2ndWon + return points won) / (svpt + opp_svpt) |
| dominance_ratio | p_rpw / (1 − serve points won rate) |
| elo, elo_surface | pre-match ratings |
| elo_grad | slope of a linear fit over the player's last 20 Elo values (assumption; config) |
| matches_played, matches_surface_played | counts before this match |
| last_k_winrate | win rate over the last k matches, for k ∈ {5, 10, 50} |
| h2h_wins, h2h_surface_wins | the player's wins vs this opponent before this match |
| n_stat_matches | how many matches with stats are in the buffer (reliability) |
| extras | rank, rank_points, age (at tourney_date), height, hand, days_since_last_match, matches_last_14_days |

State is updated by ALL matches (tour, Challenger, qualifying). Rows are emitted only for training-level, non-qualifying,
non-walkover, non-retirement matches.

### 5.6 Training table (`data/processed/features.parquet`)
- One row per emitted match. A/B by seeded coin flip; `y = 1` if A won.
- For each player feature: `a_<f>`, `b_<f>`, `d_<f> = a − b`.
- Plus `elo_prob_a = 1/(1+10**((elo_b − elo_a)/400))` and the same for surface Elo.
- Context: surface (one-hot), tourney_level, round_rank, best_of.
- Keep `match_id`, `date`, `tourney_name` and the player ids as non-feature columns. The model's feature list is an explicit allow-list in config.
- Optional (config): add flipped copies of TRAINING rows only.

---

## 6. Phases (∥ = run agents in parallel)

Repository layout to create in Phase 0:
```
Tennis_Prediction/
  PLAN.md  CLAUDE.md  PROGRESS.md  DECISIONS.md  README.md  config.yaml  pyproject.toml  .gitignore
  .claude/agents/  (section 3)
  config/player_overrides.csv
  data/raw/  data/interim/  data/processed/        # gitignored
  src/tennis/ingest/   sackmann.py tml.py scrape_tennisabstract.py odds.py player_map.py merge.py
  src/tennis/engine/   ordering.py state.py elo.py rolling.py h2h.py build_features.py
  src/tennis/model/    dataset.py splits.py elo_harness.py train.py tune.py evaluate.py predict.py
  tests/data/  tests/engine/  tests/leakage/  tests/model/  tests/fixtures/
  experiments/  reports/data/  reports/audits/  reports/figures/
  notebooks/    # exploration only
```

### Phase 0 — Setup (orchestrator, sequential)
- `git init`; create the layout above; `pyproject.toml` (Python ≥ 3.11: pandas, pyarrow, numpy, xgboost, scikit-learn,
  optuna, matplotlib, pyyaml, tqdm, pytest, requests, beautifulsoup4, unidecode, openpyxl); create a venv and install.
- Write `CLAUDE.md`: project summary, section 2 verbatim, conventions from section 1, how to run tests.
- Create the 5 agent files from section 3.
- Write `config.yaml` with every tunable number (seed, Elo start, K settings, layoff threshold, N, k list, years, levels, split dates, retirement flags).
- ✅ Accept: `pytest -q` runs (zero tests is fine); commit "phase 0: scaffold".

### Phase 1 — Foundations ∥ (3 agents in parallel)
| Agent | Task | Owns |
|---|---|---|
| data-engineer | Clone Sackmann; load 1990–2024 tour + qual_chall; build the unified schema, ordering key and cleaning; write the data report | `src/tennis/ingest/sackmann.py`, `merge.py`, `player_map.py`, `tests/data/`, `reports/data/` |
| feature-engineer | Pure Elo module: expected score, update, dynamic K, layoff bonus, plus unit tests (incl. 13.95), all on synthetic data | `src/tennis/engine/elo.py`, `tests/engine/test_elo.py` |
| leakage-auditor | Write the leakage test suite FIRST against a small synthetic fixture (section 9), designed to run on any feature-builder function | `tests/leakage/`, `tests/fixtures/` |
- ✅ Accept: all three report done; `pytest -q` green (leakage tests may be skipped only because the engine doesn't exist yet — mark them with a clear reason); matches.parquet row counts look plausible (~2.5–4.5k tour-level matches per recent year).
- Orchestrator: merge, update PROGRESS.md, commit.

### Phase 2 — Feature engine (feature-engineer, then leakage-auditor)
- Build `ordering.py`, `state.py`, `rolling.py`, `h2h.py` and `build_features.py` as specified in section 5. Produce `features.parquet` (2000–2024 rows).
- Then run **leakage-auditor** (fresh agent) on the engine. Fix every finding (via feature-engineer), then re-audit until PASS.
- ✅ Accept: leakage tests 1–4 and 7 pass on REAL data; audit PASS; the Elo-only accuracy (higher Elo wins) on 2023–2024 training-level matches is in ~64–70%. Commit.

### Phase 3 — Data extension ∥ + baselines ∥ (parallel)
| Agent | Task | Owns |
|---|---|---|
| data-engineer | 2025–2026 data: try TML-Database first. Check coverage (tour, Challengers, qualifying, stats) and map player ids. If it's insufficient → 🛑 ask about scraping (option B, scraper agent) | `src/tennis/ingest/tml.py`, `reports/data/tml_coverage.md` |
| data-engineer (2nd instance) | Betting odds ingestion and matching to match_id (for Phase 7) | `src/tennis/ingest/odds.py`, `data/interim/odds.parquet` |
| ml-engineer | `splits.py` + baselines (higher rank, Elo, surface Elo, Elo/surface blend) on the 2000–2024 features | `src/tennis/model/splits.py`, `evaluate.py`, `tests/model/` |
- ✅ Accept: merged matches through the latest available 2026 date; zero unmatched players; test 5 (splits/duplicates) passes; baseline table in `reports/baselines.md`. Commit.

### Phase 4 — Elo tuning (ml-engineer)
- `elo_harness.py`, like the author's `run_ELO_evaluation`: walk every match, predict the higher-rated player, update after;
  warm-up 10,000 matches; score only training-level matches.
- Grid: `div_number ∈ {400, 800, 1200}`, `m ∈ {1, 5, 10}`, `base_k ∈ {0, 8, 16}`, `max_k ∈ {32, 64, 100}`,
  `bonus_after_layoff ∈ {False, True}`, plus fixed K = 24 (on/off). Separate runs for overall and surface Elo.
- Score on **log loss of the Elo probability** over 2015–2024 (accuracy reported too). **Never on the test period.**
- Parallelise grid runs with multiprocessing.
- Write the best settings to config (orchestrator), regenerate features.parquet, and re-run leakage tests 1–2.
- ✅ Accept: `reports/elo_tuning.md` with a results table and the chosen settings; leakage tests green. Commit.

### Phase 5 — XGBoost ∥ (parallel: ml-engineer + leakage-auditor reviewing the dataset code)
- ml-engineer: XGBoost `binary:logistic`, eval_metric logloss, early stopping on validation.
  Random search / Optuna, 500–2000 trials. Search space: learning_rate 0.01–0.1 (log), n_estimators ≤ 2000 with early stopping,
  max_depth 3–8, min_child_weight 1–10, subsample 0.5–1, colsample_bytree 0.5–1, gamma 0–1, reg_alpha 0–10, reg_lambda 1–50.
  Select on validation log loss. Save `experiments/xgb_trials.jsonl`, feature importance and a calibration plot.
- leakage-auditor: in parallel, audit `dataset.py`, `splits.py` and the tuning code (section 2, rules 5–6 and 10).
- ✅ Accept: the model beats Elo-only on validation log loss; validation accuracy ≤ 75%; audit PASS; tests 6 and 9 green. Commit.

### Phase 6 — Replication backtest (ml-engineer)
- Split A (the author's): train 2000–2024 · validate 2025-01-01 → day before Wimbledon 2025 · test = Wimbledon 2025.
- Retrain the best config on train+val, predict the test once, report accuracy, log loss and Brier, with a 95% bootstrap CI.
  Compare to 66.3% (author), 63.8% (IBM) and bookmakers (if odds are available).
- ✅ Accept: `reports/replication_wimbledon_2025.md`. Commit.

### Phase 7 — Forward test (ml-engineer, then leakage-auditor final audit)
- Split B: train 2000–2025 · validate 2026 H1 (to 30 Jun) · test 2026 H2 (or whatever 2026 data exists after 30 Jun).
- Re-tune lightly on the new validation set (≤ 300 trials), freeze, then evaluate the test once.
- Report the full test season plus each 2026 Grand Slam in the test window separately, the bookmaker baseline, and calibration.
- Final leakage-auditor audit of the whole repo.
- ✅ Accept: `reports/forward_test_2026.md`; audit PASS. Commit.

### Phase 8 — Prediction tool + final report (orchestrator)
- `python -m tennis.model.predict --p1 "Jannik Sinner" --p2 "Carlos Alcaraz" --surface Grass --level G --round F --best-of 5`
  → win probability, using the state as of the latest data. Also a `--draw draw.csv` mode for a whole round.
- `README.md`: how to run everything end-to-end.
- `reports/FINAL_REPORT.md`: approach, data, leakage safeguards, all metrics tables, what worked or didn't, next ideas.
- 🛑 STOP: present the summary to the human.

---

## 7. Model & evaluation details
- Metrics everywhere: accuracy, log loss, Brier, calibration curve (10 bins); broken down by surface, level and round.
- Baselines: higher rank wins; Elo prob; surface Elo prob; 50/50 blend; bookmakers (implied prob, margin removed).
- Small-sample warning: one Slam (~127 matches) has a ~±8 pt 95% CI. Always print CIs for slam-level results.
- Save the models with their config hash under `models/` (gitignored).

## 8. Experiment hygiene
- Each run writes one JSON line: timestamp, git commit, config hash, split, params, metrics.
- Figures go to `reports/figures/`, named `<phase>_<what>.png`.

## 9. Leakage test suite (tests/leakage/)
| # | Test | Fails if |
|---|---|---|
| 1 | **Future invariance:** scramble the results and stats of every match after a cutoff, rebuild the features → every row at or before the cutoff is identical | any feature reads the future |
| 2 | **Slow recompute:** for 200 random matches, recompute the features in a simple, obviously correct way from strictly earlier matches → equal to the engine | engine or ordering bug |
| 3 | **Banned columns:** the model feature list has no `w_`, `l_`, `winner`, `loser`, `score`, `minutes` | same-match stats leak |
| 4 | **Orientation:** label mean in [0.48, 0.52]; swapping A/B negates every `d_` column and flips the elo_prob | winner/loser layout leaks |
| 5 | **Splits:** max train date < min val date < min test date; no match_id in two splits; no duplicate match_ids | time or duplicate leakage |
| 6 | **Shuffled labels:** training on permuted labels → validation accuracy in [0.47, 0.53] | row structure reveals the label |
| 7 | **Elo arithmetic:** 2063 beats 2120 with K = 24 → +13.95 | wrong formula |
| 8 | **K-factor m:** K for a match doesn't change when later rounds of the same event are removed | m uses the future |
| 9 | **Too-good alarm:** validation accuracy ≤ 0.75 | something slipped through |

## 10. Known assumptions (log any changes in DECISIONS.md)
m = 1 default (tuned); layoff gaps in days, most recent first; elo_grad = slope over the last 20 Elo values; N = 50;
k ∈ {5, 10, 50}; retirements don't update Elo; Carpet → Hard; Davis Cup and Olympics produce training rows.

---

## 11. How to start (for Claude Code)
1. Read this whole file.
2. Run Phase 0 yourself.
3. From Phase 1 on, act as **orchestrator**: launch the subagents named in each phase (in parallel where marked ∥), give each one
   its task, the files it owns, and the acceptance criteria, then review their results, run `pytest -q`, update
   PROGRESS.md, and commit.
4. After any phase that touches data, features, splits or models, run the **leakage-auditor** as a fresh agent before moving on.
5. Continue phase by phase without waiting for the human, except at 🛑 STOP points.

---

## 12. Added task (human, 2026-11-16) — Project summary PDF
**When:** only after Phase 8 is committed, and before presenting the final summary to the human (the Phase 8 🛑 STOP).
Don't start it earlier, and don't change the current work for it.

**TASK:** write `reports/Project_Summary.pdf`. It is a complete, plain-language PDF of the whole project, which the human
will use to understand the project and to put it on a resume.

Rules:
- Use only real results from the repo (PROGRESS.md, DECISIONS.md, reports/, experiments/, config.yaml). Never invent
  or round up a number. If a number is missing, leave it out.
- Be honest about the limits:
  - the shortened 2026 forward test and its small sample;
  - the bookmaker figure of ~72% is a published number, not one we measured;
  - the best of 800 tuning trials was chosen on validation data;
  - the assumptions made (m in the K-factor, elo_grad, etc.).
- Read-only on the repo. Write only to reports/. Don't edit src/, tests/ or config.
- Build it with whatever works (markdown → PDF via pandoc, weasyprint or reportlab; install what is needed). Open the
  finished PDF, check that the pages render properly (no cut-off tables or broken figures), and fix any problems.
- Use charts already produced in reports/figures/ where they help.

Contents, in this order:
1. One-page overview: the problem, the approach, and the headline results in a small table.
2. Data: sources, years, what each file contains, and the cleaning rules, including the data errors found and how
   they were handled.
3. Features: Elo (the formula, the dynamic K-factor, the layoff bonus) and every statistic, explained simply with
   its formula.
4. How data leakage is prevented: the one-chronological-pass design, the ordering rules, and each leakage test with
   what it catches. Mention the original author's leakage mistake and how we avoid it.
5. Models: baselines, Elo tuning, XGBoost, the search space, the best settings, feature importance.
6. Evaluation: the split design, the Wimbledon 2025 replication, and the shortened 2026 forward test (overall and
   Roland Garros), with confidence intervals and the comparison to the published figures (66.3%, 63.8%, ~72%).
7. Engineering: repo structure, the agent team, parallel phases, and the test suite (counts).
8. Limitations and what to do next.
9. "Things I should know": about 12 likely interview questions about this project, each with a short, accurate
   answer based on the actual work.
9b. "What I did differently": a short section answering "How is your work different from the YouTube project you
   replicated?"
   - Base it only on real decisions in DECISIONS.md and the code. Include true differences only:
     - leakage-safe design by construction, and its test suite;
     - the independent auditor agent;
     - Elo tuned on validation log loss rather than on test years;
     - no tuning on any test period;
     - a fixed-K versus dynamic-K comparison;
     - confidence intervals on small test sets;
     - anything else we really did differently.
   - Give a 30-second spoken answer and a 2-minute detailed answer.
10. Resume material:
    - 3 to 5 bullet points, each starting with an action verb and holding one real number;
    - a 2-line project description;
    - a list of the technologies used.
    Do not exaggerate.

Keep it readable: about 8 to 12 pages. When it's done, give the human the file path and nothing else about it.
