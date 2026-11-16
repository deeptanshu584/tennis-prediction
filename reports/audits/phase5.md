# Phase 5 audit, part 1: dataset, splits, evaluation, Elo tuning, test-set hygiene

> **Verdict (part 1): FAIL — formally, on one should-fix guard gap. No leak was found.**
> New leakage test 18 is red until a one-line hardening of `dataset.is_banned` lands (finding 1). No current
> feature is affected. Everything else in scope checks out: the allow-list, the augmentation guard, the three
> splits on the real table, train-only fitting in `evaluate.py`, a blind Elo-tuning window, and no test-window
> metric in `experiments/`. Once finding 1 is fixed and test 18 is green, part 1 is a **PASS**. That needs no
> re-audit, only a re-run of `pytest tests/leakage -q`.
>
> `train.py` / `tune.py` appeared during this audit. My review of them (end of file) is **preliminary**. A final
> pass is due once the ml-engineer has finished.

Auditor: fresh leakage-auditor. I wrote none of the code under review. Date: 2026-11-16.
Scope: PLAN §2 rules 5, 6 and 10; checklist items 4, 6 and 8. The feature engine was already audited
(phase2.md: PASS) and was not redone.

## Leakage suite results
| Run | Result |
|---|---|
| `pytest tests/leakage -q` before the new tests (slow + realdata included; tests 00–14) | **54 passed, 0 skipped, 0 failed** (390 s) |
| New tests 15–18 (22 tests) | **12 passed, 10 failed**. All 10 failures are test 18 and reproduce finding 1 |
| Full suite with the new tests | **66 passed, 10 failed** (test 18 only) |

### New tests (tests/leakage/)
| File | Tests | Status now | What it pins down |
|---|---|---|---|
| `test_15_split_test_windows.py` | 5 (3 realdata) | **pass** | Pins the test windows to D-005/D-017 (Wimbledon 2025; val_end 2025-06-29; forward 2026-04-01..06-07; `odds.max_date` = test_end). The dev split and `elo_harness.tune_end` end before every test period, and tune_end is before replication val_start. On the real table: replication test = every Wimbledon-2025 row and nothing else (122 rows, one tourney_id, one date 2025-06-30), and none of them sits in replication train/val. Forward test = exactly the rows dated in the pinned window (473). No test row of either split is in any dev partition. |
| `test_16_test_set_hygiene.py` | 2 | **pass** | Scans every `experiments/*.jsonl` line. Records outside the final splits may carry no `test` metric, no date window overlapping a test period, and no per-season metric for 2025/2026. Trial/tuning logs may carry no test metric at all. Per file and final split, test metrics come from ONE config hash (evaluated once, after freezing). Final-split validation windows don't overlap their test. The checker is self-tested on 10 synthetic violations. |
| `test_17_elo_tuning_window.py` | 3 (1 realdata+slow) | **pass** | Every `elo_tuning.jsonl` line has window 2015-01-01..2024-12-31 and only seasons 2015–2024. The config's Elo settings are the best logged 2015–24 log loss per flavour. On the real data, the harness's scored probabilities are **bit-identical** when every match after `tune_end` is deleted from the input. |
| `test_18_banned_player_prefixed.py` | 12 | **10 FAIL, 2 pass** | `is_banned` / `feature_columns` must reject same-match stats behind a player prefix (`a_w_ace`, `b_l_svpt`, `d_w_bpFaced`, …). Reproduces finding 1. |

The proposed fix for finding 1 was checked in a scratch session by monkeypatching, with no source change. It bans
all five probe names and none of the 106 allow-listed features.

## Findings

1. **SHOULD-FIX: the allow-list guard misses same-match stats behind a player prefix.**
   `src/tennis/model/dataset.py:28-32` (`is_banned`).
   - **The gap.** `low.startswith(("w_", "l_"))` only looks at the start of the name. `a_w_ace`, `b_l_svpt` and
     `d_w_bpFaced` pass the guard, and `feature_columns(cfg, df)` accepts them when the table carries them. Those
     are the natural names for same-match serve stats re-oriented to A/B. The banned substrings (`winner`,
     `loser`, `score`, `minutes`) are caught wherever they appear, but `w_`/`l_` are not.
   - **Impact today.** None. The allow-list in `config.yaml` has no such name (test 18
     `test_allow_list_has_no_player_prefixed_same_match_names` passes), and the engine emits none. This is
     defence-in-depth for rule 3.
   - **Fix (one line).** In `is_banned`, also check the name with one leading `a_`/`b_`/`d_` stripped:
     ```python
     core = low[2:] if low[:2] in ("a_", "b_", "d_") else low
     return (low.startswith(BANNED_PREFIXES) or core.startswith(BANNED_PREFIXES) or any(...) ...)
     ```
     A plain substring test for `l_` would be wrong: `level_A` contains `l_`.
   - **Test:** `test_18_banned_player_prefixed.py` (10 cases red now; green with the fix).

2. **SHOULD-FIX (process, rule 6): Elo-only metrics on the 2025 and 2026 full seasons were already reported.**
   - **Where.** `reports/audits/phase2.md:316` ("2025 and 2026: 63.71% and 64.91%"). Also `phase2.md:190`
     ("Highest season over 2000–2026") and `PROGRESS.md:62` ("the 2000–2026 maximum is 68.6%"). The 2025 season
     contains the Wimbledon 2025 test (122 rows). The 2026 season contains the whole forward test (473 rows).
   - **Severity: low.** These were sanity numbers for the *untuned default* Elo baseline, computed in the Phase 2
     audit. They were not used for any selection. Phase 4 selected on 2015–2024 only, and the choice follows
     exactly from the 2015–24 logs (test 17). The Elo settings they describe no longer exist. No model or
     tuned-Elo metric has been computed on a test window, and no `experiments/*.jsonl` line touches one
     (test 16).
   - **Fix.**
     - Disclose this in the Phase 6/7 reports: the test tournaments were inside whole-season sanity numbers for
       default Elo.
     - From now on, sanity checks before Phase 6/7 use seasons ≤ 2024 or the validation windows only (replication
       val 2025-01-01..06-29, forward val 2026-01-01..03-31), never a full 2025/2026 season.
     - Test 16 enforces this for the experiment log. Reports are not scanned automatically.

3. **NOTE: Elo tuning overlaps the dev validation years (D-004/D-027). The inflation is quantified and is tiny;
   the test numbers are not threatened.**
   - **Method.** From `experiments/elo_tuning.jsonl` (650 settings per flavour, per-season metrics), I
     re-selected each flavour on 2015–2022 only ("honest", never seeing 2023–24) and compared it with the
     chosen 2015–2024 setting on 2023–24:

     | Flavour | Chosen (2015–24) | Honest (2015–22) | 2023–24 log loss, chosen vs honest | 2023–24 acc, chosen vs honest | Oracle (best on 2023–24 alone) |
     |---|---|---|---|---|---|
     | overall | div 3200, m 1, base 16, max 100, bonus | **identical** | 0.60986 vs 0.60986 (**0**) | 65.53% vs 65.53% | 0.60972 (−0.00014) |
     | surface | div 2400, m 10, base 24, max 64, bonus | div 3200, m 10, base 16, max 64, bonus | 0.62135 vs 0.62253 (**−0.0012**) | 64.52% vs 64.20% (+0.3 pt) | = chosen |

   - **Phase 5 validation numbers.** The overall Elo feature (the dominant input) is not inflated at all.
     Surface Elo is optimistic by about 0.001 log loss and 0.3 pt accuracy. This affects the Elo baselines, the
     blend (the Phase 5 bar, 0.6056) and XGBoost alike, so the model-vs-baseline comparison stays roughly fair.
     XGBoost's own selection on 2023–24 (hundreds of trials plus early stopping) is a much larger optimism
     source. `tune.py` states both caveats.
   - **Phase 6/7 test numbers: not affected.** The tuning window ends 2024-12-31 (tourney_date), before
     replication val (first row 2025-01-06), Wimbledon 2025 (2025-06-30), forward val and forward test. Test 17
     shows the scored probabilities don't depend on any later match. They are bit-identical after deleting
     every post-2024 match. Elo settings were fixed using data that is training data in both final splits.

4. **NOTE: "2024" windows include events played in January 2025.**
   - Dev validation and the Elo tuning window end on tourney_date 2024-12-31, so they contain United Cup 2025
     (tourney_date 2024-12-27), Brisbane and Hong Kong 2025 (2024-12-30): 77 rows, played in January 2025.
   - This follows the tourney_date convention (rule 2). It is no test period, and the replication split puts
     the same rows in train. No action; worth one sentence in reports that say "2023–2024".

5. **NOTE (D-024 R2, still open): the `split_name=None` guard is weaker than it looks for the replication
   split.** `src/tennis/model/dataset.py:89-97,100`.
   - With no split name, the guard uses the latest train_end (forward, 2025-12-31). That is *after*
     Wimbledon 2025, so replication **test** rows would pass `maybe_augment_train(..., split_name=None)`.
   - `train.fit_xgb` and `tune.prepare_data` always pass the split (preliminary review), so there is no live
     issue. Make `split_name` required once the orchestrator updates test 11's call, or at least have
     `fit_xgb`/Phase 6–7 code never call it with None.

6. **NOTE: test-partition row counts are computed for reports.** `src/tennis/model/evaluate.py:528-534` and
   `reports/baselines.md:93-94` list sizes and date ranges of the replication/forward test partitions. These are
   counts only, with no labels or metrics, so this is no rule-6 issue. Keep it that way.

7. **NOTE (sanity, checklist 8): metrics are plausible.**
   - On dev validation 2023–24 (n = 5,928, tuned Elo): Elo 65.53% / log loss 0.6099; surface Elo 64.52% /
     0.6214; blend 65.65% / 0.6056; rank logistic 63.85% / 0.6297.
   - Per-season Elo 2015–2024 is 64.9–70.1%. The 70.1% is 2015, inside the Elo tuning window and the dev
     training years, so it is in-sample for the Elo hyper-parameters. Everything is under the 0.75 alarm and
     inside the expected 64–70% band.
   - The Phase 5 bar (D-026) is now the **blend, 0.6056**, not the rank logistic. `tune.py` reports both.

### Checked and OK
- **Rule 5, splits (checklist 4).** `splits.py` partitions on inclusive date windows only, with no shuffle.
  `_check_spec` rejects inverted or overlapping windows. `_check_partitions` (`splits.py:198-213`) enforces
  duplicate-free, disjoint and strictly time-ordered partitions on every call. Real partitions on
  features.parquet (76,421 rows, match_id unique):

  | Split | Train | Val | Test |
  |---|---|---|---|
  | dev | 66,363 (2000-01-03 → 2022-11-27) | 5,928 (2023-01-02 → 2024-12-30) | — |
  | replication | 72,291 (→ 2024-12-30) | 1,493 (2025-01-06 → 2025-06-23) | 122 (Wimbledon, all 2025-06-30, tourney_id 2025-540) |
  | forward | 75,016 (→ 2025-12-17) | 932 (2026-01-04 → 2026-03-30) | 473 (2026-04-05 → 2026-05-25; Roland Garros 118) |

  - Every Wimbledon-2025 row is in the replication test (122 = 127 matches minus W/O and RET).
  - Dev ∩ either test = 0, and dev ∩ either final split's validation = 0.
  - Wimbledon 2025 is in the *forward* train. That is legitimate: it is a different split.
- **Rule 3 / allow-list.** `dataset.feature_columns` is the only input path and validates against `is_banned`,
  duplicates and the table's columns. All 106 allow-listed names are clean (finding 1 is guard hardening only).
- **Flip augmentation.**
  - `maybe_augment_train` refuses rows after the split's train_end and refuses already-flipped frames.
  - `assert_not_augmented` is called in `evaluate_frame`, `breakdown`, `RankLogistic.fit` and
    `run_dev_baselines`.
  - `flip_orientation` is correct: a↔b swapped, d negated, probabilities and y complemented.
  - `features.flip_augment_train` is false.
- **Rule 10 (checklist 6).**
  - Nothing is learned on validation or test in `evaluate.py`. `RankLogistic.fit` (`evaluate.py:409-424`) fits β
    and `fill_rank` on the train frame only, via `baseline_predictions(train, frame)` (`evaluate.py:430-441`).
  - Bootstrap and paired bootstrap only resample rows. Calibration tables are descriptive.
  - There is no imputer, scaler, encoder or calibrator anywhere in `src/tennis/model/`.
- **Elo harness.**
  - It scores training-level, non-qualifying, non-RET matches with tourney_date in [2015-01-01, 2024-12-31]
    after a 10k warm-up (`elo_harness.py:261-263`). All 1,637 logged lines carry window 2015-01-01..2024-12-31
    and seasons 2015–2024 only.
  - The walk continues to 2026, but it predicts before it updates and RR blocks never span two events, so later
    matches cannot reach a scored prediction (test 17).
  - `m` uses only matches already played in the event (`k_arrays`, `prepare`).
  - The configured settings equal the logged best (test 17).
- **Test-set hygiene.**
  - `experiments/` holds `baselines.jsonl` (10 lines, split `dev`, windows ≤ 2024-12-30, seasons 2015–2024) and
    `elo_tuning.jsonl` (1,637 lines, split `elo_tune_2015_2024`).
  - No line carries a test metric, a 2025/2026 season or a window touching a test period.
  - Under `reports/`, the only test-window metrics are those in finding 2. `models/` is empty, and no other cache
    holds test predictions.

## Preliminary review of `train.py` / `tune.py` (in progress by the ml-engineer; NOT final)
At the time of reading (`train.py` 147 lines, `tune.py` 454 lines; no `xgb_*.jsonl` logged yet):
- **OK.**
  - Inputs come only from `dataset.feature_columns(cfg, train_df)` (`train.py:113`, `tune.py:76`).
  - `maybe_augment_train(train, cfg, split_name)` is always called with the split (`train.py:115`,
    `tune.py:78`). `assert_not_augmented(val)` is called on every validation frame (`train.py:114`,
    `tune.py:77`), and `FitResult.predict` calls it too.
  - Early stopping and Optuna selection use the validation rows only (`train.py:97-98`, `tune.py:105,129`).
  - `prepare_data` keeps only `train`/`val` from `time_split`, so a test partition is never loaded into the
    arrays (`tune.py:72-80`). The dev split has no test anyway.
  - Nothing is fitted on validation: there is no imputer or calibrator. The baselines in `run_report` refit
    `RankLogistic` on `data.train` (`tune.py:254`).
  - The leakage alarm runs on every trial and on the per-season breakdowns. Trial lines go to
    `experiments/xgb_trials.jsonl` with split `dev` and no test key, which test 16 will check.
  - The selected trial's `best_iteration + 1` is recorded for refits (`tune.py:180`).
- **Watch items for the final pass** (checklist below).
  - The shuffled-label check (`tune.py:293-297`) early-stops on the real validation labels. It is fine as a sanity
    check, but it can only make the shuffled model look slightly better than chance, never hide a leak.
  - `fit_xgb` *requires* an early-stopping set. Phase 6/7's "retrain on train+val" must therefore NOT reuse it
    with the test as `val_df`. See the checklist.

## Checklist for the final pass on train.py / tune.py (and the Phase 6/7 code that will reuse them)
1. `xgb_trials.jsonl` / `xgb_final.jsonl` exist and test 16 passes on them. Every trial is split `dev`, with no
   `test` key, `val_window` = 2023-01-02..2024-12-30, and accuracy ≤ 0.75 (test 09).
2. The selected trial is the argmin of logged validation log loss (tie-break as documented). The refit reproduces
   it exactly (`run_report` raises otherwise). The model is saved with its config hash.
3. No fitted preprocessing or calibrator. If a calibrator is added, it is fitted on a validation slice only and
   never scored on that same slice without saying so.
4. `maybe_augment_train` is called with `split_name` at every call site. `assert_not_augmented` is called on every
   validation and test frame. No early-stopping holdout is drawn from the augmented frame.
5. **Phase 6/7 refit on train+val** uses a fixed number of trees taken from validation (`best_iteration + 1`,
   possibly scaled). It has **no eval set at all**, or an eval set that is not the test, so test rows never drive
   early stopping. It asserts that the refit frame's max date ≤ the split's `val_end` and that it shares no
   match_id with `time_split(...)["test"]`. Note that `maybe_augment_train(split_name="replication")` will (rightly)
   refuse val rows. Don't work around that with `split_name=None` (finding 5).
6. The test partition is scored once, in a separate entry point that runs after the config is frozen. Its JSON
   line uses split `replication`/`forward`, test metrics under a `test` key, and the frozen config hash (test 16
   rule 3).
7. Height (D-024 R1): check the gain rank reported in `reports/xgb_tuning.md`. If any height column is in the top
   15, drop it from the allow-list (orchestrator) and re-run.
8. Shuffled-label and too-good tests (06, 09) are green on the final config. The validation accuracy of the chosen
   model is ≤ 0.75, ideally about 64–68%. A gain over the blend (0.6056) of more than about 0.01 log loss deserves
   a second look.
