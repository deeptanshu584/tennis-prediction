# Phase 7 final audit: the whole repo, end to end

> **Verdict: PASS.** No leak was found and there is no BLOCKER or FIX item. Each test set was scored once, after
> the freeze, by a model trained only on rows dated on or before its `val_end`. I confirmed this from the logs, the
> file timestamps and the agent transcripts. I also refit both final models independently on train+val only: the
> refits reproduce the saved test probabilities to within 3e-8. Every logged test metric can be recomputed from
> the saved prediction CSVs. Recomputing 43 forward-test rows (4,902 values) from strictly earlier matches matched
> the engine exactly. There are 5 NOTE items below. New leakage test 20 (12 tests) pins the final-test integrity.

Auditor: fresh leakage-auditor. I wrote none of the code under review. Date: 2026-11-16.
Code under review: the clean worktree at HEAD `60741b3` plus the uncommitted Phase 6/7 files (`evaluate_final.py`,
`tests/model/test_evaluate_final.py`, config/DECISIONS/PROGRESS diffs, experiments, reports). `data/` and `models/`
are the real ones. I ran it with `tennis.__file__` pointing into the worktree. The Phase 8 edits to
`src/tennis/engine/build_features.py` in the main repo are **not** covered (see N4).

## Test results (worktree)

| Run | Result |
|---|---|
| `pytest tests/leakage -q` (slow + realdata included; tests 00–20) | **96 passed, 0 skipped, 0 failed** (403 s) |
| New test 20 alone | 12 passed (10 fast/realdata in 2.9 s; 2 slow refits) |
| Full `pytest -q` (with test 20) | **456 passed, 0 skipped, 0 failed** (430 s) |

### New test: `tests/leakage/test_20_final_test_integrity.py` (12 tests)
| Test | Pins down |
|---|---|
| `test_exactly_one_test_record_with_the_frozen_settings[replication/forward]` | `experiments/<split>.jsonl` holds exactly ONE record with a `test` key. Test 16 only required one *hash*, so a second scoring under the same hash would have passed it. The logged model params and tree count must equal the frozen config values: `best_params` + `replication_n_estimators` 405, and `forward_best_params` with 393. The refit has no early stopping and its window ends ≤ `val_end`. The validation window lies inside the split's validation window and the test window inside the test period. n_test is 122 / 473, and the validation step reproduced the frozen tree count. |
| `test_both_final_tests_share_one_frozen_config_hash` | Both final records carry the same config hash (D-032: frozen together). |
| `test_forward_retune_is_validation_only_and_precedes_scoring` | All trials (≤ `n_trials_forward`) are forward-split runs with a validation window inside 2026 Q1 and no other metric keys. Every trial was logged before the forward test record. `forward_best_params` is exactly the best logged trial, with `n_estimators` = best_iteration + 1. The scoring run's validation log loss equals that trial's to 1e-12. |
| `test_rescoring_is_refused_before_loading_anything[…]` | Against the REAL experiment logs, `run_final(score=True)` raises "already been scored". `partitions`, `validate`, `score_test`, `write_report` and `log_experiment` are monkeypatched to fail if they are reached, so the check is safe to run. The jsonl files stay byte-identical. |
| `test_prediction_csv_is_the_test_partition_and_reproduces_the_logged_metrics[…]` (realdata) | The CSV's match_ids are exactly the `time_split` test partition, in the same order: 122 Wimbledon 2025 rows, and 473 forward rows of which 118 are Roland Garros. None of them is in train/val. Labels match, and `p_elo` equals `elo_prob_a`. The logged accuracy, log loss and Brier are recomputed from the CSV (tolerance 1e-6) for the model, the Elo and blend baselines, and each reported tournament. |
| `test_saved_final_model_is_the_frozen_refit[…]` (realdata) | The saved booster has exactly the frozen number of trees. Its meta has no early stopping, no augmentation, the frozen params, the 106 allow-listed features and a refit window ending ≤ `val_end`. Its predictions on the test partition equal the CSV. |
| `test_final_model_is_reproduced_from_train_and_validation_rows_only[…]` (realdata, slow) | An independent `refit_fixed` on train+val rows only reproduces the saved test probabilities (atol 1e-6). So the scored model saw no test row and no row after `val_end`. |

## Scope 1: end-to-end pipeline: PASS

- **Ingest/merge.** These are unchanged since the Phase 2 re-audit PASS (`git diff f06470c HEAD -- src/tennis/engine
  src/tennis/ingest` touches only the new `odds.py`, which never feeds the model; D-029).
  - matches.parquet has 337,663 rows and features.parquet 76,421. Neither has a duplicate `match_id` (rule 8).
  - No Laver Cup row remains (D-022).
  - Level overrides hold: every "<City> Olympics" row is O (575), every Tour Finals / Masters Cup / ATP Finals row
    is F (540), and NextGen Finals is A (122).
  - Feature rows contain 0 walkovers, 0 RET/DEF, 0 qualifying rows and 0 non-training-level rows (rule 7).
  - The label mean is 0.4988 overall and 0.496 from 2025-06-30 on (rule 4). One residual duplicate pair is in N3
    (harmless).
- **Engine (rules 1, 2, 9; D-019, D-020).** The source is unchanged since the audited version.
  - Tests 01 (future invariance), 02 (slow recompute), 08 (K-factor m), 10 (RR blocks), 12 (static fill) and 13
    (features.parquet = a fresh build with the current config) all pass on real data. Test 13 matters most here: the
    production table was built with the tuned Elo settings.
  - My own forward-test spot check is under Scope 5.
- **Dataset/allow-list (rule 3).** `is_banned` now covers a `w_`/`l_` behind one player prefix (phase-5 F1 fix;
  test 18 passes). The 106 allow-listed features are the meta feature list of both final models.
- **Orientation (rule 4).** Test 04 passes. Flip augmentation is off, the table has no `is_flipped` column, and both
  final metas say `augmented: false`.
- **Splits (rule 5).** `time_split` is date-only and checks disjointness and strict time order.
  - Replication: train 72,291 rows (→ 2024-12-30), val 1,493 (2025-01-06 → 06-23), test 122 (all 2025-06-30, one
    tourney).
  - Forward: train 75,016 (→ 2025-12-17), val 932 (2026-01-04 → 03-30), test 473 (2026-04-05 → 05-25).
  - Tests 05 and 15 pass.
- **Train/tune/evaluate_final (rules 6, 10).**
  - `refit_fixed` (`train.py:158-224`) raises on rows after `val_end`, on a shared match_id with the test, on a test
    row not dated after every refit row, and on augmented frames. It trains with `evals=[]` and exactly `n_trees`
    rounds.
  - `fit_xgb` refuses an early-stopping set outside the split's validation window (phase-5 F2).
  - Without `--score-test`, `evaluate_final` cuts the table at `val_end` before splitting. That includes
    `tune_forward` (`evaluate_final.py:111`).
  - `--score-test` requires the frozen config values (`model_params(require_frozen=True)`). It refuses when
    `experiments/<split>.jsonl` already has a test record (`evaluate_final.py:315-317`, checked before any data
    load), and the validation step must reproduce the frozen tree count.
  - Nothing is learned outside the booster: XGBoost handles NaN natively and there is no calibrator. The rank
    logistic baseline is fitted on train (validation step) or train+val (test step) only.

## Scope 2: Phase 6/7 test hygiene (rule 6): PASS

- `experiments/replication.jsonl` and `experiments/forward.jsonl` each hold **exactly one** line, with a `test` key,
  config hash `3c3f2599a311`.
- The current config hashes to `3c3f2599a311` and holds the record's settings exactly:
  - `best_params` at full precision, and `replication_n_estimators: 405`;
  - `forward_best_params` = trial 253 at full precision (bit-equal to the trial log), with `n_estimators: 393`.
- **Timeline** (UTC, from the record timestamps, file mtimes, the orchestrator transcript and the ml-engineer
  transcript `agent-ae9f068f…`):

  | Time | Event |
  |---|---|
  | 08:11 | features.parquet built |
  | 10:02–10:23 | 300 forward trials (hash `f703ebbdc434`, pre-freeze) |
  | 10:07 / 10:15 / 10:23 | Validation-only runs (`--split replication` twice, `--split forward` once; no `--score-test`) |
  | 10:30:27 | Agent reports "Neither test set has been scored" |
  | 10:30:44 | Orchestrator freezes config.yaml (mtime 16:00:44 IST) and DECISIONS D-032 |
  | 13:31:46 | Report-wording-only edits to evaluate_final.py; the agent's last code edit |
  | 13:32:41 | The **only** `--split replication --score-test` |
  | 13:33:38 | replication.jsonl record and model/CSV written |
  | 13:33:42 | The **only** `--split forward --score-test` |
  | 13:34:39 | forward.jsonl record and model/CSV written |
  | Afterwards | Only report summary text was edited. No code, config or setting changed. |

- The transcripts show no other command that scored, or loaded test rows into, a metric. The one real-data
  `time_split` call at 09:59 printed partition sizes only. The Phase 8 agent's transcript shows no metric
  computation.
- **Other experiment files:** test 16 passes on all 7 jsonl files: no test metric, no window overlapping a test
  period, and no 2025/2026 per-season metric outside the final splits. The forward trial records carry only
  `val` / `best_iteration` / `seconds` / `val_window` (2026-01-04 → 03-30).
- **Reports:** `baselines.md`, `elo_tuning.md`, `xgb_tuning.md` and `reports/data/*` contain no test-period metric.
  The only earlier test-season numbers are the disclosed D-028 finding 2 (`phase2.md`, `PROGRESS.md:62`), and both
  final reports carry that disclosure.
- **Re-scoring refusal:** verified on the real logs by test 20 (and on synthetic data by
  `tests/model/test_evaluate_final.py::test_test_is_scored_once_and_logged_in_test16_format`).

## Scope 3: the refit: PASS

| | Replication | Forward |
|---|---|---|
| Refit rows (train+val) | 73,784, dated 2000-01-03 → 2025-06-23 (≤ val_end 2025-06-29) | 75,948, dated 2000-01-03 → 2026-03-30 (≤ val_end 2026-03-31) |
| Shared match_id with test | 0 | 0 |
| Booster trees / meta | 405 / fixed_trees 405, early_stopping false, augmented false | 393 / 393, false, false |
| CSV rows == test partition (ids, order, y) | yes (122) | yes (473; Roland Garros 118) |
| Saved booster → CSV p_xgb | max abs diff 2.9e-8 | 2.9e-8 |
| **Independent refit on train+val only → CSV p_xgb** | **2.9e-8** | **2.9e-8** |
| Metrics recomputed from CSV (with my own formulas, no project code) vs logged | acc 0.655738 = ; ll 0.616211 (Δ 2e-9); Brier 0.213416 (Δ 1e-9) | acc 0.676533 = ; ll 0.591989 (Δ 7e-10); Brier 0.203216 |
| Baselines recomputed | Elo 0.6967 / 0.61393; blend 0.6475 / 0.60406 = logged | Elo 0.6702 / 0.60497; blend 0.6681 / 0.60687 = logged; RG 0.7288 / 0.59581 = logged |

The tiny differences come from storing p_xgb as float32 in the CSV. Test rows were never augmented: the guard runs
in `score_test`, `refit_fixed` and `FitResult.predict`, and the table has no `is_flipped` column.

## Scope 4: forward re-tune: PASS

- 300 trials (≤ 300), all `split: forward`, one hash, `val_window` 2026-01-04 → 2026-03-30.
- The data path is `tune_forward` → `pre_test_frame` (cut at `val_end`) → `tune.prepare_data` → `time_split`.
- Trial 253 is the best by (log loss, −accuracy, number), with best_iteration 392. Two pieces of evidence show the
  trial used exactly the train/val partitions of the scoring run:
  - the scoring run's validation step reproduced it bit-for-bit (val log loss 0.5855702504327451 in both, 393 trees);
  - `run_final` raises if the tree count differs.
- No trial record carries a test metric. All trials finished (10:23) before the freeze (10:30) and well before
  scoring (13:34).

## Scope 5: plausibility: PASS

- Full-window test accuracies are 0.6557 (Wimbledon 2025) and 0.6765 (forward). Both are inside the 0.64–0.70
  target and far from 0.75. Validation accuracies are 0.6932 and 0.6867.
- **Roland Garros 2026 at 0.7288 [0.653, 0.805] is plausible and is not a leak signature:**
  - **Slams are easier.** Elo accuracy on all Grand Slams 2015–2024 is 0.736 (n = 4,760), against 0.654 at
    non-Slam events. The median Slam edition scores 0.732, and 85% of editions score ≥ 0.70. Roland Garros
    2015–2024 ranged 0.648 (2020) to 0.770, median 0.725. RG 2026 at 0.729 sits at the historical median. Elo's own 0.661 on RG 2026 is
    unusually low, not XGBoost's high.
  - **The gain is not in the late rounds.** XGB accuracy by round on RG 2026:

    | Round | R128 | R64 | R32 | R16 | QF | SF | F |
    |---|---|---|---|---|---|---|---|
    | n | 60 | 29 | 16 | 8 | 3 | 1 | 1 |
    | XGB accuracy | 0.750 | 0.690 | 0.688 | 0.750 | 0.667 | 1/1 | 1/1 |
    | Elo accuracy | 0.683 | 0.621 | 0.625 | 0.750 | 0.333 | 1/1 | 1/1 |

    Over half of the RG rows (60 of 118) are R128, and so is half of the edge over Elo (+4 of +8 correct picks).
    At R128 no same-event main-draw information exists apart from a qualifier's own qualifying matches, which
    really were played before the first round. The R32–F rows, where same-event history does exist, add +2.
  - **Other events in the window look ordinary:** Rome 0.570, Monte Carlo 0.660, Madrid 0.714, Barcelona 0.724,
    Geneva 0.630.
  - The paired CIs against Elo and the blend include 0, and the report says so.
- **Spot check by recomputation (`tests/leakage/reference.py`, independent of the engine):**
  - Rows: 43 forward-test rows. 10 are random non-RG rows, 4 are random RG R128/R64 rows, and 29 are **every** RG
    row from R32 to F, where same-event history exists (in-event m, days_since_last_match = 0,
    matches_last_14_days, form, Elo after earlier rounds).
  - Method: every CONTRACT column, all 106 allow-listed features included, was recomputed from strictly earlier
    matches, with the Elo replayed from 1990.
  - Result: **4,902 of 4,902 values match** to 1e-9. They match again when every match after the last target is
    deleted from the input.

## Scope 6: test runs

Leakage suite: **96 passed / 0 skipped / 0 failed**. Full `pytest -q`: **456 passed / 0 skipped / 0 failed**. Both
ran in the worktree, and the full run includes test 20 and `tests/model/test_evaluate_final.py`. Neither run wrote
to data/ (matches.parquet and features.parquet mtimes are unchanged).

## Findings

No BLOCKER. No FIX.

1. **NOTE: scoring writes the model and CSV before the log line.**
   - Where: `src/tennis/model/evaluate_final.py:295-300` (save_model and CSV inside `score_test`) vs `:340` (log).
   - If the process died between the two, a scored test would leave no record. `already_scored` would then allow
     a second scoring that silently overwrites the model and CSV.
   - It did not happen: the transcripts show one `--score-test` per split, and the model/CSV mtimes equal the
     record timestamps to the second.
   - **Fix (hardening, optional now that both are scored):** in `run_final`, also refuse when
     `models/xgb_<split>_final_*_test_predictions.csv` exists. Or append a `{"split":…, "status": "scoring_started"}`
     line to `experiments/<split>.jsonl` before `score_test`, and have `already_scored` treat it as scored.
2. **NOTE: test 16 allows several test records under one hash.** Its rule 3 checks the number of hashes, not the
   number of records. Pinned now by test 20 (`test_exactly_one_test_record_with_the_frozen_settings`). No change to
   test 16 is needed.
3. **NOTE: a walkover twin survives de-duplication.**
   - Where: `data/interim/matches.parquet`, Parma CH 2020-10-05 F, Tiafoe d. Caruso. It appears twice: match_num
     270 with a null score (`is_walkover`) and match_num 301 with "6-3 3-6 6-4".
   - D-009's key includes the score, so the pair isn't caught.
   - Harmless: walkovers are skipped by the engine (`build_features.py:184`) and the Elo harness, and it is a
     Challenger, so it is not a row.
   - 1,506 walkover rows stay flagged in matches.parquet. Rule 7 ("dropped everywhere") is enforced at every
     consumer, not at the file.
   - **Fix (optional):** in `sackmann.py`'s D-009 step, also drop a walkover row when a played row has the same
     (tourney_id, round, player pair). Log it in the ingestion report.
4. **NOTE (Phase 8): re-run the leakage suite in the main repo before committing.**
   - The main repo has an uncommitted refactor of `src/tennis/engine/build_features.py` (`run_pass` / `PassResult`
     for `snapshot.py`). It was **not** audited here.
   - Before the Phase 8 commit, run `pytest tests/leakage -q` (slow + realdata) in the main repo. Tests 01, 02 and
     13 must stay green. Test 13 proves features.parquet is unchanged; if the refactor changes a single value it
     fails, and the Phase 6/7 results would then no longer describe the shipped engine.
   - The Phase 8 "production model on all data" must never be scored on either test period. Save it under a name
     other than `xgb_<split>_final_*`. It must not append to `replication.jsonl` / `forward.jsonl` (tests 16/20
     would fail).
5. **NOTE: the trial log's `git_commit` is 6919e78.** `evaluate_final.py` was uncommitted when the trials ran, so
   the trial log cannot point at the code that produced it. The exact reproduction of trial 253 by the scoring run's
   validation step (Scope 4) makes up for this. Committing Phases 6/7 now records the code.

## Checklist (role definition)
1. Snapshot before update: unchanged engine, tests 01/02/13 green, plus 4,902/4,902 values on test rows. ✔
2. Ordering (tourney_date, tourney_id, round rank, match_num; RR blocks): unchanged, tests 02/10 green. ✔
3. Banned names: none in the allow-list (tests 03, 18), and the meta feature lists equal the allow-list. ✔
4. Splits: max(train) < min(val) < min(test), disjoint, no duplicates (tests 05/15 plus the numbers above). ✔
5. K-factor m: in-event matches so far only (test 08). ✔
6. Fitted preprocessing: none outside the booster. The rank logistic baseline is fitted on train / train+val only.
   No calibrator. ✔
7. Leakage suite: 96 passed. ✔
8. Metrics: Elo-only 0.6967 / 0.6702 on the tests. XGB 0.6557 / 0.6765. No full-window value above 0.75. ✔
