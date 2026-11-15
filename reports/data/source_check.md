# Source check — Sackmann `tennis_atp` archive copy

Checked 2026-11-16 by the orchestrator, before any ingestion code was built on it.
Scripts were run read-only against the files; nothing under `data/raw/` was modified.

## Provenance
- Location: `data/raw/tennis_atp/tennis-sackmann-archive-main/atp/`. The archive's root also holds `wta/` and
  `slam_pointbypoint/`, which this project doesn't use.
- The archive's README describes it as an archival mirror of Jeff Sackmann's public datasets, with a twin copy at
  huggingface.co/datasets/Aneeshers/tennis-sackmann-archive. The ATP snapshot is taken from upstream commits made in
  June 2026. Licence: CC BY-NC-SA 4.0 (LICENSE file included).
- `github.com/JeffSackmann/tennis_atp` itself returns 404 (checked 2026-11-16), so this copy can't be diffed against
  the original. Integrity was checked against the independent TML-Database instead (section 4).

## 1. Files and years covered
| File family | Years | Gaps | Used |
|---|---|---|---|
| `atp_matches_YYYY.csv` (tour level) | 1968–2026 (59 files) | none | 1990–2026 |
| `atp_matches_qual_chall_YYYY.csv` (Challengers + tour qualifying) | 1978–2026 (49 files) | none | 1990–2026 |
| `atp_matches_futures_YYYY.csv` | 1991–2026 (36 files) | none | no (`include_futures: false`) |
| `atp_players.csv` | 66,912 players, no duplicate `player_id` | — | yes |
| `atp_rankings_{70s,80s,90s,00s,10s,20s,current}.csv` | 7 files, all readable | — | available |
| `atp_matches_doubles_*`, `atp_matches_amateur.csv`, `matches_data_dictionary.txt`, `UPSTREAM_README.md` | — | — | no |

## 2. Columns
Every tour and qual_chall file from 1990 to 2026 has exactly the 49 columns in PLAN.md §4.2: no missing columns,
no extra ones. That includes all 18 serve-stat columns (`w_ace … w_bpFaced`, `l_ace … l_bpFaced`). Every non-empty
stat cell is numeric.

## 3. Tour-level rows per year (since 2000)
| Year | Rows | Year | Rows | Year | Rows |
|---|---|---|---|---|---|
| 2000 | 3,378 | 2009 | 3,085 | 2018 | 2,897 |
| 2001 | 3,307 | 2010 | 3,030 | 2019 | 2,806 |
| 2002 | 3,213 | 2011 | 3,015 | **2020** | **1,462** |
| 2003 | 3,218 | 2012 | 3,009 | 2021 | 2,733 |
| 2004 | 3,288 | 2013 | 2,944 | 2022 | 2,917 |
| 2005 | 3,264 | 2014 | 2,901 | 2023 | 2,986 |
| 2006 | 3,267 | 2015 | 2,943 | 2024 | 3,076 |
| 2007 | 3,192 | 2016 | 2,941 | 2025 | 2,944 |
| 2008 | 3,123 | 2017 | 2,911 | **2026** | **1,449** |

Every year is inside the 2.5–4.5k band except **2020**, when the COVID suspension cut the season to 33 events, and
**2026**, which is a partial year (section 5). Stats coverage on tour-level rows is 87–89% for 2000–2015, 94–99% for
2016–2024, 91% for 2025 and 82% for 2026.

## 4. Integrity
- **Readability:** every file decodes as strict UTF-8 and parses with no bad lines. Raw line counts equal parsed row
  counts, and every `tourney_date` parses as a valid date.
- **Player IDs:** no null `winner_id`/`loser_id`, and every ID used in 1990–2026 matches exists in `atp_players.csv`.
- **Match IDs:** 0 duplicates of the contract `match_id` (tourney_id + match_num + both IDs), and 0 duplicates of
  (tourney_id, match_num), across all 1990–2026 tour + qual_chall rows (337,753 rows).
- **Cross-check against TML-Database** (independent re-scrape, GitHub copy, 2000–2025): excluding Davis Cup, tour-level
  row counts per year agree exactly in 20 of 26 years and differ by at most 6 rows in the others (2015 −1, 2021 −2,
  2022 −1, 2023 −6, 2025 −2). Event counts agree too. The 2024 and 2025 per-event differences are only naming
  ("Us Open" vs "US Open"). Davis Cup counts differ more (up to 29 rows in 2016–2018); the sources cover Davis Cup
  zone ties differently.
- **Spot checks:** all 2024 and 2025 Grand Slam finals match the known results (e.g. Wimbledon 2025: Sinner d. Alcaraz
  4-6 6-4 6-4 6-4; Wimbledon 2025 has 127 rows from R128 to F, `tourney_date` 2025-06-30).

### Data quirks found (small; none is corruption)
| Issue | Rows | Detail | Handling (DECISIONS.md) |
|---|---|---|---|
| Same match entered twice under different `match_num` | 14 pairs | Viña del Mar 2007 qualifying (9 matches, `match_num` 1–9 and 261–269); the finals of Seoul CH 2024, Antofagasta CH 2025 and Kigali 1 CH 2026 (384/385); Davis Cup 2024 AUT–TUR and 2025 ESA–ROU (rubbers 2 and 3 identical) | D-009: de-duplicate on (tourney_id, round, winner_id, loser_id, score), keeping the first; logged |
| Same pair, same score, different rounds | 2 pairs | Tour Finals 2024 Sinner d. Fritz 6-4 6-4 in both RR and F (real); Santiago CH 2022 Torres d. Dutra Da Silva in Q2 and R32 (plausible lucky loser) | kept |
| Player listed as their own opponent | 4 | Challengers: Ho Chi Minh 2004, Happy Valley 2016, Chengdu 2016, Astana 2016 | D-010: flagged and excluded |
| Null surface | 53 | 15 Davis Cup 2023 World Group I/II play-off ties | D-011: imputed Hard, flagged `surface_imputed` |
| Impossible serve stats | 1 | Roland Garros 2024 R64 Machac d. Navone: `w_2ndWon` 21 > 13 second serves | D-012: stats treated as missing; result kept |
| Round code `ER` (not in the plan's round order) | 32 | 2007 round-robin experiment events (Adelaide, Buenos Aires, Delray Beach, Las Vegas): elimination round before the RR groups | D-013: `ER` ranked between R16 and RR |
| Carpet | 14,672 | — | → Hard (plan) |
| Walkovers / RET / DEF / null score | 1,434 / 10,117 / 97 / 50 | — | cleaning rules (§4.4) |
| `tourney_date` year ≠ file year | 1,652 | events starting in late December are filed under the next season | expected; ordering uses `tourney_date` |

## 5. Most recent data
- **2025: complete season.** 2,944 tour-level rows plus 11,629 qual_chall rows; the last event is the Next Gen Finals
  (`tourney_date` 2025-12-17). All four Slams are present.
- **2026: partial, January to early June.** 1,449 tour-level rows; the last event is Roland Garros 2026
  (`tourney_date` 2026-05-25, complete through the final). There are also 5,745 qual_chall rows, the latest with
  `tourney_date` 2026-06-01.
- **No data after early June 2026.** Grass season, Wimbledon 2026, the US Open 2026 and everything after are absent.
  I couldn't check the 2026 results against an independent source, because TML's GitHub copy stops in mid-January 2026.

## Verdict
The archive is complete, readable and consistent for 1990–2025 and suitable for Phase 1. It also covers 2025 and
2026 H1, which removes most of Phase 3's reliance on TML.
**Open issue for Phase 7:** PLAN split B tests on 2026 H2 (after 30 June 2026). This archive has no data in that
window, so Phase 7 will need another source or a different split (to be raised at that point).
