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

## 2026-11-16 — Phase 2: Feature engine (audit pending)
**Done (feature-engineer)**
- `src/tennis/engine/{ordering,state,rolling,h2h,build_features}.py` and `src/tennis/model/dataset.py`: one
  chronological pass that snapshots, emits, then updates. `python -m tennis.engine.build_features` runs over all
  337,721 matches in about 12 s.
- `data/processed/features.parquet`: 76,479 rows × 115 columns (106 allow-listed), 2000 → 2026-05. Label mean 0.4989.
  NaN ≤ 3% per a_ feature (rates 2.9%, rank 1.4%).
- tests/engine: 116 unit tests, written first.

**Tests:** `pytest -q` → 254 passed, 5 skipped (splits for Phase 3, and no experiment logs yet). Leakage tests 1–4,
7 and 8 pass on synthetic AND real data. That includes test 2 against the auditor's independent reference at 1e-9
(also run on the full 1990+ history).

**Metrics (Elo-only, untuned K = 800/(Ne+m)):** 2023–24 accuracy **63.96%** (surface Elo 63.55%; rank-only 63.76%),
elo_prob log loss 0.6376. Seasons 2015–2024 range 61.2–68.0%; the 2000–2026 maximum is 68.6% (2012). Nothing near
75%. The 2023–24 figure is at the low edge of the expected ~64–70%; Phase 4 tunes K.
**Next:** a fresh leakage-auditor reviews the engine and re-checks the unified table (D-016).
**Audit 1 (fresh auditor): FAIL**, with 1 blocker and 2 should-fix findings (reports/audits/phase2.md):
RR rounds not chronological (858 rows), hindsight in missing height/hand/age, and no guard on augmentation. Fixed per
D-019…D-022 by the feature-engineer (engine), the suite author (reference.py updated independently, plus an
RR-trap fixture) and the data-engineer (Laver Cup excluded, level overrides; it was cut off by the usage limit after
its tests passed, and the orchestrator finished the regex fix). Auditor tests 10–13 added and passing.
Rebuilt: matches 337,663 · features **76,421** rows. Elo-only 2023–24 **63.99%**, log loss 0.6377.
**Tests:** 282 passed, 5 skipped. **Next:** re-audit (running) → commit Phase 2 → Phase 3.
Committed as `def6147`. **Re-audit: PASS** (reports/audits/phase2.md). Leakage suite 45 passed, 5 skipped, plus new test 14 (4 passing). Elo-only 2023–24 63.95% (independent recompute). Follow-ups D-024. Phase 2 ✅

## 2026-11-16 — Phase 3: Data extension + baselines (starting)
- TML coverage: `reports/data/tml_coverage.md`. TML is not ingested (D-023): it adds nothing the archive lacks.
  Phase 3 acceptance "matches through the latest available 2026 date, zero unmatched players" is already met.

### EXACT NEXT STEPS (written 2026-11-16 while the auto-mode permission check was failing transiently)
1. Wait for the Phase 2 re-audit (leakage-auditor resumed; it writes a "Re-audit" section in reports/audits/phase2.md).
   PASS → commit "phase 2: audit PASS". FAIL → send the findings to the feature-engineer, fix, re-audit.
2. Launch Phase 3 in parallel (the launches failed on the transient check; retry):
   a. data-engineer (fresh agent): tennis-data.co.uk odds 2015–2026 → data/raw/odds/ (polite, cached, ≥ 3 s, no
      June–Nov 2026 match data) → src/tennis/ingest/odds.py → data/interim/odds.parquet matched to match_id
      (≥ 97% tour main draw), plus reports/data/odds_ingest.md and tests/data/test_odds.py. If the download fails → 🛑 ask the human.
   b. ml-engineer (fresh agent): src/tennis/model/splits.py (CONTRACT §4) and evaluate.py (metrics, bootstrap CI,
      calibration, JSONL logger, leakage alarm); baselines on dev split val 2023–24 → reports/baselines.md,
      experiments/baselines.jsonl, reports/figures/phase3_calibration_elo.png; tests/model/. Leakage tests 05/06/09 must pass.
3. Phase 3 acceptance: zero unmatched players, test 5 passes, baseline table exists → pytest → commit "phase 3".
4. Then Phase 4 (ml-engineer: elo_harness.py, grid with multiprocessing, score on 2015–2024 log loss, write the best
   settings to config, rebuild features, re-run leakage tests 1–2).

## 2026-11-16 — Phase 3: Data extension + baselines ✅ (odds pending the human)
- **TML** not ingested (D-023): the archive already covers 2025 and 2026 to 2026-06-01 with 0 unmatched players.
- **Odds** 🛑 blocked (D-025): tennis-data.co.uk's robots.txt disallows Anthropic/Claude agents. No data file was
  fetched. The parser and matcher `src/tennis/ingest/odds.py` are built (30 tests, never downloads), and the `odds:`
  config block is added. Waiting on the human: manual download, drop the bookmaker baseline, or another source. Only
  the Phase 6–7 bookmaker baseline depends on it.
- **ml-engineer:** `splits.py` (dev/replication/forward per CONTRACT §4), `evaluate.py` (metrics, bootstrap CIs,
  calibration, breakdowns, leakage alarm, JSONL logger, baselines), 42 tests in tests/model/.
  Split sizes: dev 66,363 / 5,928 · replication 72,291 / 1,493 / test 122 (Wimbledon 2025) · forward 75,016 / 932 / test 473.
- **Baselines, dev val 2023–24 (n=5,928)** in reports/baselines.md:

  | Baseline | Acc | Log loss | Brier |
  |---|---|---|---|
  | Higher rank | 0.6385 | — | — |
  | Rank logistic (train-fit) | 0.6385 | **0.6297** | **0.2202** |
  | Elo | 0.6398 | 0.6377 | 0.2236 |
  | Surface Elo | 0.6354 | 0.6379 | 0.2237 |
  | 50/50 blend | **0.6426** | 0.6336 | 0.2217 |

  The untuned Elo is underconfident (ECE 0.028).
- **Tests:** 363 passed, 0 skipped. Leakage tests 05, 06 (shuffled labels) and 09 (too-good alarm) now run and pass.

## 2026-11-16 — Phase 4: Elo tuning ✅
- ml-engineer: `src/tennis/model/elo_harness.py` (all 336k non-walkover matches from 1990, the engine's exact rules,
  incremental layoff tracker, multiprocessing on 11 processes; bit-identical to the engine). 18 tests.
- Scored on 2015–2024 training-level matches after the 10k warm-up (n = 26,786), on log loss. No test period used.
- Plan grid (164 × 2): best div 1200 / m 1 / base 16 / max 64 / bonus, at the grid edge. Widened once (D-027, 650 × 2):
  - **overall:** div 3200, m 1, base_k 16, max_k 100, bonus → log loss 0.5987, acc 66.9% (default 0.6269, 64.3%)
  - **surface:** div 2400, m 10, base_k 24, max_k 64, bonus → log loss 0.6077, acc 66.4% (default 0.6257, 64.5%)
- Written to config.yaml; features.parquet rebuilt (76,421 rows; about 100 s with the bonus on).
- **Dev val 2023–24 baselines after tuning:** Elo acc 65.5%, log loss 0.6099 (was 64.0% / 0.6377). Surface Elo 64.5% /
  0.6214. Blend 65.7% / **0.6056**. Rank logistic 63.9% / 0.6297. Elo is now inside the expected 64–70% band.
- Reports: reports/elo_tuning.md, reports/baselines.md (regenerated), reports/figures/phase4_*.png.
- **Tests:** full suite 379 passed + 3 engine unit tests that had hard-coded the old default K. Those now pin their
  own Elo settings and pass (132/132 engine tests). Leakage tests 1–2 pass on the regenerated features.

## 2026-11-16 — Phase 5: XGBoost tuning ✅
- **Audit part 1** (reports/audits/phase5.md): `is_banned` fixed (D-028); leakage tests 15–18 added; committed 6919e78.
- **ml-engineer:** `train.py` (`fit_xgb`, allow-list only, augmentation guard, NaN native) and `tune.py` (Optuna TPE,
  8 processes × 1 thread, alarm check after every trial). 18 new tests.
- **800/800 trials** in 77 min on dev (train 2000–2022, val 2023–24). The surface is flat: log loss 0.5936–0.5971.
  Best is trial 720 → `xgb.best_params` (n_estimators 689), D-030.
- **Dev val 2023–24 (n = 5,928), 95% CIs:**

  | Model | Acc | Log loss | Brier |
  |---|---|---|---|
  | **XGBoost** | **0.6672** [0.6557, 0.6791] | **0.5936** [0.5831, 0.6048] | **0.2050** |
  | Elo | 0.6553 | 0.6099 | 0.2118 |
  | 50/50 blend | 0.6565 | 0.6056 | 0.2102 |
  | Rank logistic | 0.6385 | 0.6297 | 0.2202 |

  Paired Δ log loss vs blend: −0.0120 [−0.0163, −0.0075]. Shuffled labels give acc 0.489 (PASS). ECE 0.023.
  Height ranks 71–93 of 106, so it is kept (D-024 R1). Top features: d_elo, elo_prob_a, elo_surface_prob_a,
  d_elo_surface, d_dominance_ratio.
- **Odds:** human chose option (b), D-029. The ~72% bookmaker figure is quoted externally; the parser is kept.
- **Tests:** 422 passed (ml-engineer run, before the config edit). Final pytest at commit.
- **In flight:** the leakage auditor's final pass on train/tune, including a second look at the 0.012 gain over the
  blend. In parallel, the ml-engineer writes the Phase 6/7 code and runs the validation-only forward re-tune. Tests
  are not scored until the audit passes.
- **Audit final pass** (reports/audits/phase5.md): no leak found. F1 (rounded params) and F2 (val-window guard in
  `fit_xgb`) are fixed, as is F4 (nthread check); see D-031. New leakage test 19 (8 tests) passes.
  - The 0.012 gain over the blend is real: ~0.010 survives without selection on 2023–24.
  - 3,080 of 3,080 feature values recomputed from earlier matches agree.
- **Tests:** full suite **444 passed** (9.4 min).

## 2026-11-16 — Phases 6–7: frozen settings; test scoring in progress
- Frozen in config (D-032, hash 3c3f2599a311):
  - replication: 405 fixed trees on train+val with `best_params`;
  - forward: `forward_best_params` from trial 253/300 on 2026 Q1, 393 trees.
- Validation:
  - replication 2025-01..06 (n = 1,493): XGB 0.6932 / 0.5857 vs blend 0.6792 / 0.5986;
  - forward 2026 Q1 (n = 932): XGB 0.6867 / 0.5856 vs blend 0.6701 / 0.5998.
- The ml-engineer agent hit the usage limit before scoring. Verified that no test set had been scored: no
  replication.jsonl or forward.jsonl, the reports say "Not scored yet", and no --score-test command ran.
- **EXACT NEXT STEPS (if interrupted):**
  1. Check `experiments/replication.jsonl` and `experiments/forward.jsonl`. Any that exist were scored ONCE, so
     never re-run them. Score only a missing one, with the frozen config:
     `.venv/Scripts/python -m tennis.model.evaluate_final --split replication|forward --score-test`.
  2. Finalize reports/replication_wimbledon_2025.md and forward_test_2026.md: CIs, Roland Garros separately,
     small-sample note, D-028/D-029/D-031 notes.
  3. Run `pytest -q`, then commit Phase 6 and Phase 7 code, reports and experiments.
  4. Phase 7 final full-repo leakage audit (auditor agent) → fix any FIX items → commit.
  5. Phase 8: write `src/tennis/model/predict.py` (CLI), README.md and reports/FINAL_REPORT.md → pytest → commit →
     🛑 STOP and present the final report to the human.

## 2026-11-16 — Phase 6: Wimbledon 2025 replication ✅ (scored once, frozen config 3c3f2599a311)
- Code: `evaluate_final.py` (CLI `--split replication|forward [--tune] [--score-test]`; refuses a second scoring) and
  `train.refit_fixed` (fixed trees, no eval set, date/match_id guards). 14 new tests.
- The agent stopped once at the usage limit before scoring. It was resumed and scored each test exactly once.
- **Wimbledon 2025 (n = 122; accuracy CI ±8.2 pt):**

  | Model | Acc [95% CI] | Log loss | Brier |
  |---|---|---|---|
  | XGBoost | 0.6557 [0.574, 0.738] | 0.6162 | 0.2134 |
  | Elo | 0.6967 | 0.6139 | 0.2119 |
  | Blend | 0.6475 | 0.6041 | 0.2105 |
  | Rank logistic | 0.6639 | 0.6184 | 0.2137 |

  - Every paired difference's CI includes 0 (vs blend +0.012 [−0.023, +0.049]). XGBoost matches the author's 66.3%
    and beats IBM's 63.8%. Bookmakers ~72% is quoted only (D-029).
  - Report: reports/replication_wimbledon_2025.md.

## 2026-11-16 — Phase 7: forward test scored once (final audit pending)
- Re-tune: 300 trials on 2026 Q1, best trial 253 (393 trees).
- **Test 2026-04-01..06-07 (n = 473, all clay; accuracy CI ±4.2 pt):**
  - XGBoost 0.6765 [0.634, 0.719], log loss 0.5920.
  - Elo 0.6702 / 0.6050; blend 0.6681 / 0.6069; rank logistic 0.6342 / 0.6308.
  - Paired Δ log loss: vs blend −0.0149 [−0.0301, −0.0002] (just significant); vs Elo −0.0130 [−0.035, +0.009].
  - ECE: XGBoost 0.031, Elo 0.041.
- **Roland Garros 2026 (n = 118):** XGBoost 0.7288 [0.653, 0.805], log loss 0.5958, against Elo 0.6610 / 0.6179. Every
  paired CI includes 0.
- Report: reports/forward_test_2026.md. No full-window accuracy is above 0.75.
- Phase 8 prediction tool is in progress in parallel: final-state snapshot plus a production model on all data that is
  never evaluated.

## 2026-11-16 — Phase 8: prediction tool done; final report drafted (uncommitted, waiting on Phase 7 audit)
- Phase 6 and the Phase 7 scoring are committed as 13c7986. The full suite passed on a clean worktree: 444 passed.
- Phase 8 agent (D-033):
  - `engine/snapshot.py` (final state, `query_rows`), with a `build_features.run_pass` refactor that gives
    bit-identical output;
  - `model/predict.py` (single and `--draw` modes, symmetrised output, production model never evaluated);
  - 25 new tests; `tests/engine` + `tests/model` gives 250 passed.
- README.md rewritten end to end; reports/FINAL_REPORT.md drafted. AUDIT_VERDICT is a placeholder in section 3.
- **EXACT NEXT STEPS (if interrupted):**
  1. Wait for the Phase 7 auditor. It writes reports/audits/phase7.md and tests/leakage/test_20_*. Fix any FIX or
     BLOCKER items. NEVER re-score a test.
  2. Optionally add a `predict:` block to config (symmetrize, threads, production params) and wire predict.py to read
     it.
  3. Fill in AUDIT_VERDICT in FINAL_REPORT.md, run `pytest -q` (full), then commit Phase 7 (audit) and Phase 8.
  4. 🛑 STOP: present FINAL_REPORT.md to the human.
- **Added task (human):** PLAN.md section 12, reports/Project_Summary.pdf. Do it AFTER the Phase 8 commit and BEFORE
  the 🛑 STOP presentation. Then give the human only its file path, plus the final summary.

## 2026-11-16 — Phase 7: final leakage audit PASS ✅
- reports/audits/phase7.md: PASS, no BLOCKER or FIX items.
  - Each test was scored once after freezing (timeline from the logs).
  - Refit windows end before the tests. An independent pre-test refit reproduces the scored probabilities to 2.9e-8.
  - Every logged metric recomputes from the prediction CSVs.
  - 4,902 recomputed forward-test feature values match.
  - Roland Garros 72.9% is normal for a Slam: Elo alone averages 73.6% on Slams 2015–24.
- New leakage test 20 (12 tests). Note 1 fixed: re-scoring is refused if prediction CSVs exist (D-034).
- **Tests:** full suite in the main repo (with the Phase 8 code) **482 passed**.

## 2026-11-16 — Phase 8: prediction tool + final report ✅
- `python -m tennis.model.predict` (single and `--draw` modes), engine snapshot, production model (D-033).
- README.md end to end; reports/FINAL_REPORT.md.
- **Next:** PLAN section 12, reports/Project_Summary.pdf, then 🛑 STOP and present to the human.
