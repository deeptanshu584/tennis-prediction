# TML-Database coverage check (Phase 3)

Checked 2026-11-16 by the orchestrator. Sources: the TML GitHub copy (`github.com/Tennismylife/TML-Database`; yearly
files 2000–2025 plus 2024 cached under `data/raw/tml/`, 2026 inspected remotely) and the per-year comparison in
`reports/data/source_check.md`.

| Need (PLAN §4.1) | TML GitHub copy | Sackmann archive (D-007) |
|---|---|---|
| Tour level 2025 | yes (2,944 rows, same as the archive except naming and 2 rows) | **yes, complete** |
| Tour level 2026 | only to mid-January 2026 (Brisbane/Auckland/Adelaide, about 2 weeks) | **yes, to Roland Garros 2026** |
| Challengers + qualifying | **no** (tour level only) | yes (qual_chall files, through 2026-06-01) |
| Match stats | yes | yes |
| Player IDs | ATP codes (e.g. `D875`), would need mapping | Sackmann ids natively |
| Data after early June 2026 | no (the repo moved to stats.tennismylife.org in January 2026) | no |

## Decision
TML is **not ingested** (DECISIONS D-023). It adds no matches the archive lacks for any date up to 2026-06-01, has
no Challengers or qualifying, and would need an ATP-code → Sackmann-id mapping for no gain. Data after early June 2026
is out of scope by the human's decision (D-017: no downloads or scraping for June–November 2026).
Phase 3 acceptance ("merged matches through the latest available 2026 date; zero unmatched players") is met by the
archive alone: `data/interim/matches.parquet` runs to 2026-06-01 and has 0 unmatched players.
`src/tennis/ingest/tml.py` is not created; the generic `merge.py` and `player_map.match_names` remain ready if a
TML or scraped source is approved later.
