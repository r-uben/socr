# Backlog fixes — tickets

One ticket per confirmed-still-valid defect. Dispatch one `socr-implementer` per READY ticket.

---

## GH-249 — the verifier grades chart pages against a phantom table

**Status:** READY
**Branch:** `fix/249-verifier-grid-gate-v2`
**Write ownership:** `src/socr/tables/native_verifier.py`, `tests/test_native_table_verifier.py`

### Context
Confirmed still real on `main@8bf34d5` by 2026-09-15 triage (high confidence).
`native_verifier._verify_from_words` goes from lane detection straight into `_value_guard`.
It neither imports nor calls `rows_establish_grid`, while `orchestrator.py:5443` already uses
that predicate for `table_not_scorable`. So a chart page's axis tick labels are treated as a
native table, every ladder rung is graded against that phantom, all rungs are rejected, and
the tick labels ship.

PR #444 attempted this and was CLOSED 2026-09-15: 593 commits behind, and `native_verifier.py`
moved 1219 -> 1540 lines beneath it. Do NOT revive that branch. Its three review findings are
the acceptance criteria below.

### Plan
Gate value verification on the native rows actually establishing a grid, using the existing
`socr.core.table_grid.rows_establish_grid` predicate. Reuse it — do not write a second one.

### Acceptance Criteria
1. A page whose native "rows" are axis tick labels does NOT reach `_value_guard`; it abstains
   with a reason that names the non-grid cause.
2. **Page-wide ticks must not mask a real table on the same page** (PR #444 review finding 1).
3. **A single-numeric-column table must still be verified** — it must not lose verification as
   collateral (finding 2).
4. Tests must exercise the value guard, NOT pass by early abstention (finding 3). For each new
   test, show it fails without the change.
5. No new threshold. No magic number.

### Verification
- `~/venvs/socr/bin/pytest tests/test_native_table_verifier.py tests/test_agentic.py -q`
- `uvx ruff@0.16.0 format --check .`
- Pin a DIFFERENCE, not a locally-measured tuple: run the same page with the gate on and off in
  one process and assert the outcomes differ exactly as intended. CI has no provider, so an
  absolute outcome measured here will not reproduce there.
- Do not `Closes #249` unless criteria 1-4 all hold; say which remain otherwise.
