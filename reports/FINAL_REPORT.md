# Final report: leakage-free ATP match predictor

**Bottom line.** The model is Elo plus rolling serve/return stats plus XGBoost, and on validation and tests it is
in the expected 64–73% accuracy range.
- **Validation:** it beats every baseline on log loss on all three validation windows (2023–24, 2025 H1, 2026 Q1).
- **2026 forward test** (473 matches, April–June, all clay): 67.6% accuracy, log loss 0.592, against 0.607 for the
  best simple baseline. That gain is significant, but only just.
- **Wimbledon 2025** (122 matches): 65.6%, level with the published replication target (66.3%). On a sample this
  small it can't be told apart from plain Elo (69.7%) or the Elo blend (64.8%).
- **Roland Garros 2026:** 72.9% [65.3, 80.5].
- **Overall:** most of the signal is in a well-tuned Elo. XGBoost adds about 0.010–0.015 log loss on top of it.

Written for: the project owner and anyone reading the repository. Details live in the linked reports.

## 1. Approach
- **Elo ratings,** overall and per surface (Hard / Clay / Grass, Carpet → Hard):
  - dynamic K = min(max_k, max(base_k, div / (Ne + m))) times a layoff bonus;
  - m = base m + the matches the player has *already* played in the current event (D-008);
  - tuned on 2015–2024 (reports/elo_tuning.md).
- **Per-player features,** written as a_ / b_ / d_ (A minus B) columns: Elo, surface Elo, Elo slope, experience,
  win rate over the last 5/10/50 matches, H2H, 13 rolling serve/return rates over the last 50 matches with stats,
  rank, rank points, age, height, handedness, rest days and matches in the last 14 days.
- **Context columns:** surface, level, round, best-of, and the Elo win probabilities.
- **Model:** XGBoost (`binary:logistic`) on an explicit 106-column allow-list, tuned with Optuna. That was 800
  trials on the dev split and 300 trials to re-tune for 2026.

## 2. Data
| Source | Coverage | Use |
|---|---|---|
| Jeff Sackmann `tennis_atp` (archive copy, D-007) | 1990 → Roland Garros 2026 | All matches: tour level, Challengers and qualifying |
| TML-Database | 2000 → mid-January 2026 (tour level) | Cross-check only, never ingested (D-023): tour-level counts per year agree with the archive within 0–6 rows |
| Betting odds | — | Not used. tennis-data.co.uk opts out of AI agents (D-025); human decision D-029 |

- **Unified table:** 337,663 matches (walkovers dropped; Laver Cup excluded; conflicting duplicate pairs dropped,
  D-014).
- **Feature table:** 76,421 training-level rows from 2000 onward (levels A, G, M, F, D, O). Challengers and
  qualifying update player state but are never rows.
- No data after 2026-06-07 exists or was downloaded (D-017).

## 3. Leakage safeguards
- **One chronological pass** builds every feature: read state → write row → update state.
  - Order is tourney_date, tourney_id, round rank, match_num.
  - Round-robin blocks are snapshotted from the state before the block (D-019).
  - Missing static attributes get constants, not hindsight values (D-020).
- **Rules 3–4:** same-match columns are banned by name (`is_banned`, also behind a_/b_/d_), the model sees only the
  allow-list, and A/B is a seeded coin flip.
- **Time-only splits.** The Elo tuning window ends 2024-12-31. Each test was scored exactly once, after its settings
  were frozen (D-032), and `evaluate_final` refuses a second scoring. The final refits use fixed tree counts with no
  early-stopping set.
- **Leakage test suite (tests/leakage, tests 00–20):**
  - the 9 tests in PLAN section 9: future invariance, slow recompute against an independent reference, banned
    columns, orientation, splits, shuffled labels, Elo arithmetic, K-factor m, the too-good alarm;
  - plus test-window pinning, experiment-log hygiene, the Elo tuning window and the XGBoost selection integrity.
- **Independent audits** in reports/audits:
  - Phase 1: PASS.
  - Phase 2: initial FAIL on three leaks (the round-robin block, static-attribute hindsight, the augmentation guard),
    fixed, then PASS.
  - Phase 5: no leak found. It fixed the rounded best params and added the validation-window guard.
  - Phase 7 final audit of the whole repo: **PASS**, no leak.
    - Each test was scored once after freezing.
    - An independent refit on pre-test rows reproduces the scored probabilities to 3e-8.
    - 4,902 recomputed feature values of forward-test rows all match.
- **Disclosure (D-028):** Elo-only accuracy for the full 2025 and 2026 seasons appeared in an early audit report.
  Nothing was selected on it.

## 4. Results
### 4.1 Validation (model selection)
| Window | n | XGBoost acc / log loss | Elo | 50/50 blend | Rank logistic |
|---|---|---|---|---|---|
| Dev 2023–24 | 5,928 | **0.667 / 0.594** | 0.655 / 0.610 | 0.657 / 0.606 | 0.639 / 0.630 |
| 2025-01-01 → 06-29 | 1,493 | **0.693 / 0.586** | 0.669 / 0.603 | 0.679 / 0.599 | 0.658 / 0.624 |
| 2026 Q1 | 932 | **0.687 / 0.586** | 0.672 / 0.603 | 0.670 / 0.600 | 0.649 / 0.612 |

These windows were used for early stopping and selection, so they are slightly optimistic: about 0.002 log loss on
dev (D-031).

### 4.2 Test sets (each scored once)
| Test | n | XGBoost acc [95% CI] | XGB log loss | Elo acc / LL | Blend acc / LL | Δ LL vs blend [95% CI] |
|---|---|---|---|---|---|---|
| Wimbledon 2025 | 122 | 0.656 [0.574, 0.738] | 0.616 | 0.697 / 0.614 | 0.648 / 0.604 | +0.012 [−0.023, +0.049] |
| 2026-04-01 → 06-07 | 473 | **0.676** [0.634, 0.719] | **0.592** | 0.670 / 0.605 | 0.668 / 0.607 | **−0.015** [−0.030, −0.000] |
| ↳ Roland Garros 2026 | 118 | 0.729 [0.653, 0.805] | 0.596 | 0.661 / 0.618 | 0.678 / 0.615 | −0.020 [−0.049, +0.012] |

- **Reference points for Wimbledon 2025:** the original author 66.3%, IBM 63.8%, bookmakers ~72%. The bookmaker
  figure is quoted from the source and was not computed on our rows (D-029).
- **Small samples:** one Slam is about 120 matches, with an accuracy CI of about ±8 points. Differences between
  models on a single Slam are not meaningful.
- **Calibration (ECE):** XGBoost 0.023 on dev val and 0.031 on the forward test, against 0.041 for Elo. No
  calibrator was needed.
- **Details:** reports/replication_wimbledon_2025.md, reports/forward_test_2026.md, reports/xgb_tuning.md,
  reports/baselines.md.

## 5. What worked, what didn't
- **Worked:**
  - Tuning the Elo K-factor was the biggest single gain: dev log loss went from 0.638 to 0.610, accuracy from 64.0%
    to 65.5%.
  - XGBoost then adds a small, consistent gain, mostly from best-of-5 Slam matches, from interactions of Elo with
    level and format, and from unranked newcomers whose starting Elo is too high (D-031).
  - The leakage tests and the audits caught three real leaks in the feature engine before any model was trained.
- **Didn't, or little gain:**
  - The serve/return stats add only about 0.002 log loss beyond Elo and context.
  - Hyper-parameter search barely matters: all 800 trials fell within 0.004 log loss of each other.
  - On Wimbledon 2025 the model did not beat plain Elo. The CIs say this is noise, but there is no evidence of a
    grass-specific advantage.
- **Limitations:**
  - No bookmaker baseline.
  - The 2026 test is clay-only and short (D-017).
  - The Elo hyper-parameters were tuned on 2015–24, a window that includes the dev validation years. This makes the
    dev numbers slightly optimistic.

## 6. Next ideas
1. A computed bookmaker baseline. The odds parser is ready; it needs the human to supply odds files (D-029).
2. Extend the forward test to June–November 2026 once that data is available. That needs the human's go-ahead, since
   it requires a download.
3. Recency-weighted serve/return stats, or stats per surface, in place of the plain last-50.
4. A logistic or stacked model over the Elo probabilities and format terms, as a simpler and more robust alternative.
   Per the audit, it already captures part of the gain.
5. Point-level or set-level models (serve-point win probability → match probability).
6. Make XGBoost orientation-symmetric, either by flip-augmenting the training rows or by averaging both orientations
   at scoring. Choose on validation, then score a fresh test period.

## 7. Prediction tool
`python -m tennis.model.predict` scores any matchup from the state after the latest data (cut-off 2026-06-01), or a
whole draw from a CSV (D-033). Its feature rows equal the engine's to 1e-9 on real 2026 first-round matches. Example:

```
$ python -m tennis.model.predict --p1 "Jannik Sinner" --p2 "Carlos Alcaraz" --surface Grass --level G --round F --best-of 5
  P(Jannik Sinner wins)   XGBoost (production): 0.618   Elo (overall): 0.617   Elo (surface): 0.378
```

- **Production model:** the frozen 2026 settings, fitted on all data and never evaluated.
- **Symmetrised output:** the tool averages both A/B orientations. Without flip augmentation, XGBoost isn't
  orientation-symmetric: for the example above, the two orientations give 0.52 and 0.70. The evaluated numbers used
  one coin-flip orientation per match. Symmetrising on validation is an obvious next experiment (next idea 6).

## 8. Reproduce
See README.md. In short: `pip install -e .` → `python -m tennis.ingest.merge` → `python -m tennis.engine.build_features`
→ `python -m tennis.model.tune` → `python -m tennis.model.evaluate_final --split replication|forward` →
`python -m tennis.model.predict ...`.
