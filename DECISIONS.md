# DECISIONS.md

Judgement calls made while executing PLAN.md. Format: decision · alternatives · why.

---

### D-001 (2026-11-16, Phase 0) — Python 3.13 in a local `.venv`
- **Decision:** use the system Python 3.13.5 to create `.venv/`, installing the project editable (`pip install -e .`).
- **Alternatives:** conda; uv; pin 3.11.
- **Why:** 3.13 satisfies the plan's ≥ 3.11, it's already installed, and every dependency ships 3.13 wheels.

### D-002 (2026-11-16, Phase 0) — Extra dependency `xlrd`
- **Decision:** add `xlrd` alongside `openpyxl`.
- **Alternatives:** convert the old `.xls` files by hand.
- **Why:** tennis-data.co.uk ships pre-2013 odds as `.xls`, which `openpyxl` can't read.

### D-003 (2026-11-16, Phase 0) — How subagents are invoked
- **Decision:** the role files live in `.claude/agents/` as specified. If the harness doesn't register them as agent types
  (custom agents are usually loaded only at session start), the orchestrator launches `general-purpose` agents whose
  prompt starts with the full text of the role file. The role, tool discipline and file ownership stay the same.
- **Alternatives:** restart the session; do the work inline.
- **Why:** keeps the plan's division of labour and parallelism without stopping for a routine choice.

### D-004 (2026-11-16, Phase 0) — Development split for Phases 3–5
- **Decision:** dev split = train 2000-01-01 → 2022-12-31, validate 2023-01-01 → 2024-12-31, with no test period.
- **Alternatives:** train ≤ 2021 / val 2022–23 / test 2024.
- **Why:** the plan's only real test sets are Wimbledon 2025 (split A) and 2026 H2 (split B), and Phase 4 already
  tunes Elo on 2015–2024. Calling 2024 a "test" would be false, because Elo selection has seen it. Phase 2 also
  checks Elo-only accuracy on 2023–2024, which matches this validation window.
  Caveat: Elo hyper-parameters are chosen on 2015–2024, which overlaps the XGBoost validation window. That is
  selection overlap on validation data only; no test period is involved.

### D-005 (2026-11-16, Phase 0) — Wimbledon 2025 boundary
- **Decision:** split A validation ends 2025-06-29. The Wimbledon 2025 main draw started Monday 2025-06-30, which
  should be its Sackmann/TML `tourney_date`. Phase 6 will check this against the data.
- **Why:** PLAN.md says "validate 2025-01-01 → day before Wimbledon 2025".

### D-006 (2026-11-16, Phase 0) — Retirements and H2H/form
- **Decision:** RET/DEF matches don't update H2H win counts or the last-k win-rate deque (config flags
  `retirement_updates_h2h: false`, `retirement_updates_form: false`). They don't move Elo (plan default) and their
  partial stats do enter the rolling buffers (plan default). Overall/surface match counts still increase for them.
- **Alternatives:** count a retirement as a win for the player who finished.
- **Why:** consistent with "retirements are never labels"; a retirement says little about relative strength.
  Revisit if needed (config flags).
