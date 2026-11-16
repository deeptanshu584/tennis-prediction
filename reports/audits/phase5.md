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

---

# Final pass: train.py, tune.py, the 800-trial study and the frozen config

> **Verdict (final pass): FAIL — formally, on two FIX-level gaps. No leak was found.**
> The +0.012 log-loss gain over the blend was investigated. It is real, not leakage: 0.010 of it survives a protocol
> that never early-stops on 2023–24, and only about 0.0017 comes from the serve/return stats. A spot-check of
> 3,080 validation feature values against a strictly-earlier recomputation matched exactly.
> New test 19 is red on two items:
> - (a) `config.xgb.best_params` is a 6-significant-digit copy of trial 720, which is a *different* model;
> - (b) `fit_xgb` accepts a test frame as its early-stopping set.
>
> Both fixes are a few lines. Once test 19 is green, Phase 5 is a **PASS**. That needs no re-audit, only
> `pytest tests/leakage -q`.

Auditor: the same independent leakage-auditor. I wrote none of the code under review. Date: 2026-11-16.
Scope: `src/tennis/model/train.py` (including `refit_fixed`, which the Phase 6 agent added during this pass),
`tune.py`, `tests/model/{conftest,test_train,test_tune}.py`, `experiments/xgb_trials.jsonl` (800),
`experiments/xgb_final.jsonl` (2), `reports/xgb_tuning.md`, `models/xgb_dev_1f47f5c50fc9*`, `config.yaml` `xgb.*`
(D-030). `evaluate_final.py` and `xgb_trials_forward.jsonl` (Phase 6/7, in progress) were only checked for test
metrics.

## Leakage suite results
| Run | Result |
|---|---|
| `pytest tests/leakage -q`, tests 00–18 (slow + realdata included) | **76 passed, 0 failed** (500 s). Finding 1 of part 1 is fixed; test 18 is green |
| New `test_19_xgb_selection_integrity.py` (8 tests, 2 realdata+slow) | **6 passed, 2 failed** (24 s). The failures are findings F1 and F2 below |
| Tests 06, 09 and 16 re-run at the end, after `xgb_trials_forward.jsonl` appeared | **5 passed** |
| `pytest tests/model -q`, with the F2 guard monkeypatched in (scratch, no source change) | **93 passed**. With the patch, test 19's F2 case passes too |

### Test 19 (`tests/leakage/test_19_xgb_selection_integrity.py`)
| Test | Status | Checks |
|---|---|---|
| `test_trials_are_dev_only_and_never_see_a_test` | pass | 800 trials. All split `dev`, one config hash (`1f47f5c50fc9`), unique trial numbers 0–799, one `val_window` 2023-01-02..2024-12-30 inside `splits.dev`, n = 5,928, no `test` key at any depth |
| `test_config_best_params_are_exactly_the_logged_argmin` | **FAIL (F1)** | `xgb.best_params` must equal the argmin trial's parameters to rel 1e-12, and `n_estimators` = best_iteration + 1 |
| `test_final_refit_is_the_argmin_and_reproduces_it` | pass | The `best_refit` line is trial 720 = argmin, with the same best_iteration (688), log loss equal to 1e-12 and the same hash. The model path carries that hash. The shuffled-label line is in band |
| `test_saved_model_metadata_matches_the_selection` | pass | `models/xgb_dev_1f47f5c50fc9_meta.json`: hash, split, not augmented, features = allow-list, best_iteration, log loss |
| `test_fit_xgb_refuses_an_early_stopping_frame_outside_the_val_window` | **FAIL (F2)** | `fit_xgb(train, test, ..., split_name="replication")` must raise. It currently trains with the test as its early-stopping set |
| `test_refit_fixed_guards_and_has_no_eval_set` | pass | `refit_fixed` refuses `split_name=None`, a test row in the refit rows (even with an empty `test_df`) and a shared match_id. It builds exactly `n_trees` trees, and its predictions are bit-identical when the test labels are flipped |
| `test_saved_dev_model_reproduces_logged_validation` (slow) | pass | The saved booster scores dev validation at the logged log loss (to 1e-6); accuracy ≤ 0.75 |
| `test_shuffled_labels_tuned_params_fixed_trees` (slow) | pass | Tuned params, 689 fixed trees, no eval set, permuted training labels: **accuracy 0.5084**, log loss 0.6959 |

## Findings

**F1 — FIX: the frozen `xgb.best_params` are not trial 720. They are a rounded copy, which is another model.**
- **Where.** `src/tennis/model/tune.py:184` (`_yaml_params` writes `float(f'{v:.6g}')`) → `config.yaml`
  `xgb.best_params`.
- **Evidence.**
  - Exact trial-720 params through `fit_xgb` on dev (nthread 1): best_iteration **688**, log loss 0.5935729945.
    This is bit-identical to the log, so the selection and refit are correct.
  - The config's 6-s.f. params through the same call: best_iteration **518**, log loss 0.5938098, accuracy 0.6675.
  - Config params with 689 *fixed* trees (the Phase 6 recipe): log loss 0.5937788, accuracy 0.6660.
  - So the frozen config (hash `f703ebbdc434`) describes a model that was never selected or validated, and
    `n_estimators: 689` belongs to different parameters. The performance impact is tiny (+0.0002 log loss). It is
    not leakage. But "the chosen trial is frozen and reproduces exactly" is false, and every Phase 6/7 early-stopping
    run on these params will pick a tree count unrelated to dev.
- **Fix.**
  - In `_yaml_params`, write `repr(float(v))` (full double precision).
  - The orchestrator replaces the seven float values in `config.yaml` with the logged ones:
    - learning_rate 0.017403824245571284
    - min_child_weight 3.7070046985199285
    - subsample 0.7246419405377439
    - colsample_bytree 0.5151019491450788
    - gamma 0.1045539002555385
    - reg_alpha 0.06810140421533839
    - reg_lambda 3.0385256550684665
  - Do the same for `forward_best_params` in Phase 7.
- **Test:** test 19 `test_config_best_params_are_exactly_the_logged_argmin` (red now).

**F2 — FIX (rule 6, defence in depth): `fit_xgb` accepts any frame as its early-stopping set, including the test.**
- **Where.** `src/tennis/model/train.py:103-123` (`fit_xgb`) and `:85-100` (`train_booster`, unguarded by design).
- **The gap.** `split_name` is used only for the *training* rows (`maybe_augment_train`, `:115`). Nothing checks that
  `val_df` lies in the split's validation window. So `fit_xgb(train, parts["test"], ...)` early-stops on the test
  and returns test metrics as `val_metrics`.
- **Impact today.** No current caller does this:
  - `tune.prepare_data` and `run_report` take `val` from `time_split`;
  - `evaluate_final.py:232-235` checks val ∩ test = ∅ before calling it;
  - the final refit uses `refit_fixed`, which has no eval set.
- **Fix.** At the top of `fit_xgb`:
  ```python
  spec = cfg["splits"][split_name]
  vd = pd.to_datetime(val_df["date"])
  out = (vd < pd.Timestamp(spec["val_start"])) | (vd > pd.Timestamp(spec["val_end"]))
  if bool(out.any()):
      raise ValueError(f"fit_xgb: {int(out.sum())} early-stopping rows outside split {split_name!r} val window")
  ```
  I verified this by monkeypatch: test 19 goes green and `tests/model` stays 93/93.
- **Test:** test 19 `test_fit_xgb_refuses_an_early_stopping_frame_outside_the_val_window` (red now).

**F3 — NOTE: the shuffled-label check in the Phase 5 report proves little.**
- **Where.** `tune.py:297-302`. That run early-stops on the real validation labels and stopped after **1 tree**
  (`xgb_final.jsonl`: best_iteration 0), so its accuracy of 0.4889 is close to a constant prediction.
- **Replacement.** Test 19 trains the tuned params with the frozen 689 trees, no eval set and permuted labels:
  0.5084, inside [0.47, 0.53].
- **Fix.** Use fixed trees in `run_report` too, or cite test 19 in the report.

**F4 — NOTE: `xgb.nthread` in config is not read.**
- **Where.** `train.py:31` hard-codes `NTHREAD = 1`, and every booster uses it. `config.yaml` `xgb.nthread: 1` is
  documentation only: changing it changes nothing, which breaks the "all tunable numbers in config" convention.
- **Fix.** Read `cfg["xgb"]["nthread"]` in `native_params`/`refit_fixed`/`predict`, or assert
  `cfg["xgb"]["nthread"] == NTHREAD`.

**F5 — NOTE: the saved model's config hash is the tuning-time hash, not the frozen one.**
- The model and all trial lines carry `1f47f5c50fc9`. The frozen config (after writing best_params) hashes to
  `f703ebbdc434`. This is unavoidable, and the link is recorded (xgb_final `model_path`, checked by test 19).
- Phase 6/7 models must be saved under the frozen hash, after F1 is fixed.

**F6 — NOTE (cosmetic):** the first 7 logged trials carry `git_commit` 9d0c170 and the rest 6919e78. The part-1
commit landed while the study ran, and the label reads HEAD at log time. The study used the code already loaded,
and the config hash is the same throughout.

## Checklist (from part 1)
| # | Item | Result |
|---|---|---|
| 1 | Tests 16 and 09 pass on the trial/final logs. Every trial is dev with val window 2023-01-02..2024-12-30 | **PASS.** 800/800, no test key. Max trial accuracy 0.6692 |
| 2 | Chosen trial = lowest logged log loss; refit reproduces it; model saved with its config hash | **FAIL (F1, FIX).** Trial 720 (0.5935730; next best is trial 558 at 0.5936237) is the argmin. The refit and the saved model reproduce it exactly (run_report guard, test 19, and my rerun: 688 / 0.5935729945). But the frozen config params are a rounded copy that does not reproduce it |
| 3 | No preprocessing outside training; any calibrator on a validation slice | **PASS.** NaN goes to native missing handling. There is no imputer, scaler, encoder or calibrator in `train.py`/`tune.py`. ECE is descriptive only. The baselines' `RankLogistic` is fitted on `data.train` |
| 4 | `maybe_augment_train` with split name everywhere; `assert_not_augmented` on every val/test frame; no early-stopping holdout from augmented rows | **PASS.** `fit_xgb:115`, `prepare_data:79` and `refit_fixed` (rows ≤ train_end) all pass the split. `assert_not_augmented` runs on val (`fit_xgb:114`, `prepare_data:78`), in `FitResult.predict`, and on both frames in `refit_fixed`. The early-stopping set is always the `time_split` val, never drawn from the training frame. `tests/model/test_train.py` spies on the call |
| 5 | Phase 6/7 retrain readiness | **PASS for `refit_fixed`, but the F2 gap remains in `fit_xgb`.** `refit_fixed` (new) uses a fixed `n_trees` and `evals=[]`. It refuses `split_name=None`, refit rows after `val_end`, shared match_ids and non-later test rows. Because test windows start after val_end in both splits, the date guard alone also keeps the test out when the caller passes an empty `test_df` (tested). Required guard still missing: F2 |
| 6 | No test metrics anywhere yet | **PASS.** No `test` key in any `experiments/*.jsonl`, including the new `xgb_trials_forward.jsonl` (133 lines, split `forward`, window 2026-01-04..03-30). Test 16 is green. `models/` holds only the dev model |
| 7 | Height not in the gain top 15 | **PASS.** Ranks are b 71, d 79, a 93 (of 106). Keep |
| 8 | Tests 06/09 green; val acc 64–68% and ≤ 0.75; investigate the gain over the blend (> 0.01) | **PASS.** 06, 09 and test 19's shuffled test are green. Val accuracy is 0.6672. The gain investigation is below: no leak |

## Gain over the blend (Δ log loss −0.0120): investigation
All runs are on the dev split. 2023–24 is scored only. No test-period row was loaded. Δ = XGB − blend log loss,
paired bootstrap CI.

**1. Does the gain survive a protocol that never touches 2023–24 before scoring?** Yes: about 0.010 of it does.
| Protocol (tuned params) | Trees | Val log loss | Acc | Δ vs blend (0.6056) |
|---|---|---|---|---|
| As reported: early stopping and selection on 2023–24 (saved model, nthread 1) | 689 | 0.5936 | 0.6672 | −0.0120 [−0.0163, −0.0075] |
| Train 2000–2021, early stopping on **2022**, score 2023–24 | 505 | 0.5954 | 0.6678 | **−0.0102 [−0.0145, −0.0059]** |
| Refit on all of 2000–2022 with those 505 trees fixed, **no eval set** (the Phase 6 recipe) | 505 | 0.5959 | 0.6678 | **−0.0097 [−0.0139, −0.0054]** |

About 0.002 of the reported gain is early-stopping/selection optimism, as the D-030 caveat says. The hyper-parameters
themselves were still chosen on 2023–24, but the search surface is flat (0.5936–0.5971 over 800 trials).

**2. Feature-group ablations** (best params, early stopping on val as in tuning, nthread 4; the full model at
nthread 4 gives Δ −0.0103, which is the reference here):
| Inputs | k | Val log loss | Δ vs blend |
|---|---|---|---|
| All 106 | 106 | 0.5953 | −0.0103 [−0.0142, −0.0062] |
| Without the serve/return stats (all `p_*`, `dominance_ratio`, `n_stat_matches`) | 64 | 0.5970 | **−0.0086** [−0.0123, −0.0050] |
| Without rank / rank_points | 100 | 0.5953 | −0.0103 |
| Without stats and rank | 58 | 0.5965 | −0.0091 |
| Elo (11) + context (surface, level, round, best_of) | 22 | 0.6027 | −0.0029 [−0.0049, −0.0009] |
| Elo + context + matches_played / matches_surface_played | 28 | 0.5992 | −0.0064 [−0.0100, −0.0029] |
| Elo + context + rank | 28 | 0.6017 | −0.0039 |

- The rolling serve/return stats add only about 0.0017. Even if one of them were same-match by construction, it
  could not explain the gain, and the spot-check (4) shows they are not.
- Most of the gain is the rest:
  - Elo re-weighted by context, about 0.003 (best_of and level);
  - experience (matches_played), about 0.0035, which shows how unreliable Elo is for new or sparse players;
  - form, rest and age, about 0.003.
- A train-fitted logistic on logit(Elo), logit(surface Elo) gives 0.6055, the same as the blend. Adding best-of-5
  interactions gives 0.6039, and XGBoost is still −0.0103 better.

**3. Where the gain sits** (saved model vs blend; share = fraction of the total log-loss gain):
- **best_of:**
  - 5 sets (n 1,011): −0.0248, 35 % of the gain; accuracy 0.735 vs 0.707.
  - 3 sets: −0.0094.
- **Level:**
  - G −0.0255, D −0.0174, M −0.0095, A −0.0079.
  - Slam accuracy 0.745 vs blend 0.715: plausible, because best-of-5 favourites win more often than an Elo
    probability says.
- **Surface:** Clay −0.0186, Grass −0.0202, Hard −0.0071.
- **Season:** 2023 −0.0084, 2024 −0.0155.
- **Rows with a missing rank** (n 92, 72 of them Davis Cup): −0.0861, 11 % of the gain; accuracy 0.804 vs 0.674.
  - **Checked as a possible outcome-dependent missingness:** in 2023–24, rank is constant within every
    (tourney, player): 0 groups mixing missing and present, 0 with two distinct ranks. So missingness cannot depend
    on the match result.
  - In the training years, the missing-rank player wins 19 % against an Elo-implied 39 %. Unranked newcomers start
    at Elo 1500 and are over-rated, so the signal is genuine and pre-match.
- **Rows where either player has < 10 stat matches** (n 184): −0.041, 11 %. Same mechanism.
- **Blend-favourite bands:** gains in every band except > 0.9 (+0.0023).
- **Olympics** 0.762 (n 63): the blend gets 0.778 there. That is small-n noise, not a signal.

**4. Spot-check of validation features against matches.parquet.**
- **Sample.** 55 dev-validation matches: 40 random (seed 20261116), plus 5 RR, 5 Davis Cup and 5 Slam.
- **Method.** Every non-Elo player feature for A and B (28 × 2 per match) was recomputed from `matches.parquet`
  using only strictly earlier matches (before the block start for RR), with `tests/leakage/reference.py`'s
  filter-based `player_features`.
- **3,080 / 3,080 values match** (tol 1e-6).
- **The check is discriminative.** Including the current match would have changed at least one serve/return rate
  in 102 of 110 player-rows.
- **Hand-worked example.** Ruud, Roland Garros 2024 R16. Last 50 stat matches 2023-10-23..2024-05-27 (including
  RG R128–R32). `a_p_1stWon` = 1786/2445 = 0.730470, equal to the table.

**Conclusion.** The gain is plausible and explained:
- it uses information Elo does not see (match format, experience, unranked status, form) in a coin-flip-oriented
  table;
- it survives an early-stopping-free protocol at about 0.010;
- it does not come from the serve/return stats;
- the stats are strictly pre-match.

Expect the Phase 6/7 test gain over the blend nearer 0.010 than 0.012.

## Checked and OK (final pass)
- **Inputs.** Only `dataset.feature_columns` (`train.py:113`, `tune.py:77`, `refit_fixed`). `test_train.py`
  shows that an extra `zz_leak = y` column is ignored and a banned name raises.
- **Search isolation.**
  - `prepare_data` keeps only train/val and raises if the dev split ever grows a test partition.
  - Workers get arrays only.
  - The leakage alarm runs per trial, on the final refit and per season.
  - The model tests run entirely in `tmp_path` (`small_cfg` redirects experiments, reports, figures and models),
    so no synthetic line can pollute `experiments/`.
- **Selection.** `best_record` tie-breaks on log loss, then accuracy, then trial number, matching the logs.
  `run_report` raises unless the refit reproduces the trial to 1e-12.
- **Determinism.** The fixed seed and nthread 1 make each trial bit-reproducible. I re-ran trial 720 and got an
  identical log loss. With 8 async workers the sequence is not reproducible (D-030); this affects which trials were
  proposed, not the integrity of any result.
- **Report** (`reports/xgb_tuning.md`): no 2025/2026 metric. Its numbers match `xgb_final.jsonl`, and it states the
  early-stopping/selection and Elo-window caveats.
