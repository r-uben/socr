# #703: A2's row-shortfall term must not floor a text table

Branch `fix/703-text-table-shortfall` off `main@cf28858`, worktree
`~/repos/.worktrees/socr-703`.

## Problem

Third-institution census (BoE, `docs/log/2026-09-10_third-institution-census.md`):
`boe-meetings-2018-scan-p28-30` p1 is Inflation Report Table 3.B, a two-column
comparison box whose cells are sentences carrying zero or one number each. The
cached qwen candidate holds 23/23 of the page's numbers with 0 extras and the
judge ladder accepted it. Shipped: the fail-closed marker, 0/23, because
`candidate_truncated` fired on qwen and gemini alike.

Reproduced on the real PDF + cached attempt:

- `numeric_body_rows` over the candidate yields exactly two rows, `('4%',)` and
  `('32.',)` — width 1.
- so `row_shape_min = 1`, at which `table_shaped_native_row_count` counts 19
  native bands on the page (every prose line that mentions a figure).
- 2 vs 19 is a huge shortfall → term (b) calls a complete candidate truncated.

At `row_shape_min = 2` the same page has 4 native bands; at 3, two. The count is
an artefact of the floor, not of a dropped row.

## Rule

Term (b) now applies only when the candidate's own body rows are
**numeric-dominant**: a strict majority carry at least two genuine numeric
tokens (`structure_check._numeric_dominant`, `_ROW_SHAPE_DISCRIMINATING_MIN`).
Otherwise term (b) abstains and the page is left to term (a) (the style break)
and the ladder verdict.

Two tokens is the floor of the existing per-candidate derivation, not a new
tuned threshold: at width 1 the predicate "this band is a table row" is true of
essentially every prose line carrying a figure, so the reconciliation is
vacuous; there is no value between 1 and 2.

Strict majority rather than `min(len(row)) >= 2` deliberately — one stray
single-numeric row (a total line, a footnote-marker row that survived
`numeric_body_rows`) inside an otherwise numeric table would take the minimum to
1 and switch the guard off for the whole candidate.

The issue's alternative ("use B1's prose corroboration witness") is not
available: PR #695 deleted model-prose corroboration by ruling.

## Evidence

- Real BoE p1, ungated vs gated: `(True, False)` — term (a) never fired on this
  page, so the gate is the only thing that changed.
  (`tests/tables/test_gh703_text_table_dominance.py`, corpus-skipped.)
- Hermetic text-table fixture: same difference, same direction.
- Numeric-dominant candidates: gated and ungated verdicts identical
  (complete `False`, truncated `True`) — the gate is inert there.
- The two real ECB truncation fixtures still fire:
  `tests/tables/test_structure_check_truncated.py::test_real_fixture_bulletin_p2_picks_complete_candidate`
  and `::test_real_fixture_bulletin_p3_picks_complete_candidate`. Every one of
  their numeric body rows carries 3+ tokens.

## Residuals

- **#643 is untouched.** Its narrow tables are 2-3 columns, i.e. `row_shape_min`
  is already 2 and the candidate IS numeric-dominant — measured: a 3-row
  `[('2018','1.0'), …]` candidate returns `_numeric_dominant() == True`. The gate
  neither helps nor hurts it. #643 also lives at a different call site
  (`manifest._row_shape_reconciliation_ok`, A1b), which this change does not
  modify.
- That A1b call site derives `row_shape_min` the same way and has the same text-
  table blind spot. It fails closed rather than shipping wrong content, and no
  measured page hits it, so it is left alone here rather than changed unmeasured.
- The BoE page now ships the accepted candidate with NO table defect at all, so
  it is flagged only by whatever the ladder itself says — this change does not
  add a flag of its own.
