# #714: A1b's row-shape reconciliation must abstain where the page has no lanes

Branch `fix/714-a1b-text-table-blind-spot` off `main@f7e83ee` (which carries
PR #715 for #703 and PR #725 for #713), worktree `~/repos/.worktrees/socr-714`.

## Problem

`manifest._row_shape_reconciliation_ok` (TICKET-A1b, #640) is the twin of A2's
term (b) at a different call site, and carries the same defect #703 fixed
there. It derives `row_shape_min` from the candidate's own minimum numeric
tokens per body row, so on a text table the minimum collapses to 1 and every
native prose band that mentions a figure counts as a native table row.

Measured on the real BoE 2018 Inflation Report box page
(`~/Data/socr/census-boe-2026-09-10/`, cached qwen candidate for p1):

| quantity | value |
| --- | --- |
| candidate numeric body rows | `[('4%',), ('32.',)]` |
| `row_shape_min` | 1 |
| `table_shaped_native_row_count(words, 1)` | 19 |
| `_row_shape_reconciliation_ok` | **False** |

2 against `ceil(19 * 36/39) = 18`, so a candidate holding 23/23 of the page's
numbers with 0 extras was vetoed out of the corroboration pool.

## Rule

The rule #703 settled, reused rather than copied: the reconciliation is
meaningful only where the NATIVE page shows recurring numeric column lanes,
which a model cannot truncate. `_row_shape_reconciliation_ok` now consults
`structure_check._native_page_has_column_lanes` (the seeded-lane builder,
rounds 3-6) and abstains where it is closed.

No second gate, no new constant, no candidate-side dominance test — all three
were falsified on #703 and none is reintroduced here. `row_shape_min` stays the
candidate's own minimum, for #703's reason: once the page is known to have
column structure, counting bands with one or more numerals is no longer
"every prose line on a prose page", and the minimum is what catches the
sparse-prefix truncation.

## Abstain maps to True, and why that is not a shortcut

A1b's contract is a **veto, not a vote**. Its sole caller
(`_row_corroborated_grid_winner`) does:

```python
if not _row_shape_reconciliation_ok(words, text):
    continue
```

The candidate is dropped from the scored pool; a True is not a positive
finding, it is the absence of this particular objection. Nothing surfaces the
outcome separately either — `_apply_row_corroboration_disclosure` carries the
`RowCorroboration`, `region_kind` and `coverage_share`, none of which this
predicate touches. So there is no caller and no report that could act on a
third state, and a tri-state would be a distinction with no consumer.

This is also how the predicate's two PRE-EXISTING abstentions already behave:
`if not candidate_rows: return True` and `if native_table_rows <= 0: return
True`. The new gate is placed with them and returns the same value for the same
reason. (It sits FIRST, before the imports of the row-corroboration helpers, so
the vacuous case does no work.) The empty-`words` case is unchanged in
behaviour: `_native_page_has_column_lanes([])` is False, and previously
`table_shaped_native_row_count([], 1)` was 0 — both abstain.

Abstaining removes one veto; it admits nothing. The candidate still has to
clear A1a's `corroborate_rows` gate immediately above (`rc.clears is True`),
still has to be a grid-reading attempt, still has to survive A2's truncation
drop and A1c's binding checks downstream.

## Evidence — every pin is a difference, gate real vs gate forced open

Forcing `_native_page_has_column_lanes` to True restores the pre-#714
predicate exactly, since the gate is the only change.
`tests/tables/test_gh714_a1b_text_table_gate.py`, `(gate_open, gate_real)`:

| page / candidate | lanes | ungated | gated |
| --- | --- | --- | --- |
| real BoE 2018 p1, cached qwen (corpus-skipped) | False | False | **True** |
| #703's hermetic text table | False | False | **True** |
| sparse-prefix numeric, truncated to 2 sparse rows | True | False | False |
| sparse-prefix numeric, complete | True | True | True |
| ECB bulletin p2, truncated / complete | True | False / True | False / True |
| ECB bulletin p3, truncated / complete | True | False / True | False / True |
| 20-row grid, last 2 rows deleted | True | False | False |
| 20-row grid, first 2 rows deleted | True | False | False |

Only the two text tables move, and both move from refusal to abstention. Every
numeric refusal A1b made before it still makes, including the two real ECB
truncation fixtures and both edges of the selection suite's deleted-row
reproducers.

## What the BoE page ships end to end

Executed, not inferred, against this worktree's source: `finalized_page_record`
on a `DocumentState` over the real PDF, with the run's own sidecar geometry and
page-level flags (`detected_table_bboxes`, `native_table_structure_failed`,
`native_table_content_defect='table_content_empty'`, `native_table_region_*`,
`table_ladder_disposition=None`) and the run's own cached qwen `PageOutput`.

| cache state | gate | status | ending | body |
| --- | --- | --- | --- | --- |
| as cached (`audit_passed=False`, `judge raised: timed out`) | real | `warning` | `model_output` | 3539 chars, carries `fall to 4%` |
| as cached | forced open | `error` | `fail_closed_marker` | 47 chars, 0/23 |
| counterfactual `audit_passed=True` | real | `success` | `model_output` | 3539 chars |
| counterfactual `audit_passed=True` | forced open | `success` | `model_output` | 3539 chars |

So this change IS the last blocker on the census page as cached. #703's log
recorded the page still fail-closed with term (b) fixed; the remaining veto was
A1b, and removing it routes the page through the corroboration fallback with
the table intact. It ships flagged (`warning`, primary reason
`structure_class`), not clean — the judge-timeout question is a separate gate
and this change makes no claim about it.

The counterfactual row is the one #703 measured, and it is unchanged here: on a
page whose candidate was accepted outright, A1b's branch is never reached, so
the gate is inert in both directions.

## Residuals

- **#713 (judge timeout) is a separate gate and is untouched.** The cached
  candidate still carries `audit_passed=False` with `judge raised: timed out`.
  The page ships its content because A1b's corroboration fallback admits it,
  not because the judge verdict changed. What ships is `warning`, not
  `success`.
- **Every #703 residual applies verbatim at this call site**, because it is the
  same predicate: the gate is page-wide (numbered source citations can arm the
  reconciliation for a candidate whose table is elsewhere,
  `test_citation_rows_on_a_text_page_are_a_known_scope_limitation`); a
  space-grouped number can register as two token lanes; a column whose anchor
  drifts more than one rounding bin between rows is still two positions; and a
  genuinely tabular page whose columns are too sparse to recur over three bands
  now relies on A1a's corroboration alone at this call site. No such page was
  found in the 45-page census corpus.
- **The gate is now consulted from two modules.** A future change to
  `_native_page_has_column_lanes` moves both A2's term (b) and A1b's
  reconciliation at once. That is the intended coupling — one rule, one
  implementation — but it widens the blast radius of that function.
- **#643 is still not fixed.** #703's log noted that on the synthetic GH-643
  shape the lane gate closes, so if that shape reached A1b the reconciliation
  now abstains. That is a consequence of this change, not a fix for #643, and
  #643's own defect is untouched.
- **The reconciliation's abstention is not surfaced anywhere.** A page whose
  A1b reconciliation abstained is indistinguishable in the sidecar from one
  that reconciled. That was already true of the two pre-existing abstentions;
  this change adds a third with the same silence. If disclosure is wanted, it
  is a separate ticket at `_apply_row_corroboration_disclosure`.
- **No CI-visible claim about the real page.** The BoE pins are corpus-skipped;
  in CI only the hermetic text table, the sparse-prefix reproducers, the two
  ECB fixtures and the deleted-row reproducers run.
