# TICKETS — GH-151 structure lost at full recall

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.

Context: `2024__bauer_pflueger_sunderam` p26 ships at **100% word recall with 0
tokens missing** and an unusable table — spanning headers in body cells, `R2` and
its values on different rows, coefficients unbound from their standard errors.
This falsifies word recall as a sufficient routing gate (proposed on GH-49):
recall measures TOKEN loss and is blind to STRUCTURAL loss.

## Stream A — structural signals (all deterministic, no model)

### TICKET-A1 — grid-shape checks · TODO · depends-on: none · wave 1
**Problem:** Nothing detects a grid whose rows disagree on width or that is mostly empty.
**Do:** Add pure functions over a parsed grid: row-width consistency, empty-cell
density, orphan rows (empty label cell with populated neighbours). Return a report
object, not a boolean — the caller decides policy.
**Files:** `src/socr/tables/structure_check.py` (new)
**Done when:** `~/venvs/socr/bin/pytest tests/test_structure_check_gh151.py -q` exits 0; the p26 grid (checked in as a fixture string) is reported defective and a clean 3x4 grid is not.

### TICKET-A2 — x-position binding check · TODO · depends-on: none · wave 1
**Problem:** The strongest available signal is unused: for born-digital pages the
native geometry knows each value's x-position, so a value assigned to a column
whose lane it does not sit under is misbound and detectable for free.
**Do:** Given a page and an emitted grid, verify each body value's x-position falls
within its assigned column's lane. Reuse the lane clustering in `native_verifier`
rather than reimplementing it.
**Files:** `src/socr/tables/native_verifier.py`
**Done when:** a synthetic page whose grid has one value shifted a column reports a binding failure; the same page with the correct grid reports none.

## Stream B — consequence

### TICKET-B1 — surface structural failure at page level · TODO · depends-on: A1, A2 · wave 2
**Problem:** A defect nothing consumes is not a gate. Today p26 ships SUCCESS.
**Do:** Wire the A1/A2 reports into a `PageState` flag and an `AuditEvent`; a page
failing the structural check must not ship as trusted native. Do NOT hard-fail the
run — mirror the existing fail-closed pattern used for `native_table_unverifiable`.
**Files:** `src/socr/core/born_digital.py`, `src/socr/core/state.py`, `src/socr/pipeline/orchestrator.py`
**Done when:** processing the p26 fixture yields an audit event of kind `table_structure_failed` and the page is not `audit_passed=True`.

### TICKET-B2 — record the gate correction on GH-49 · TODO · depends-on: B1 · wave 3
**Problem:** GH-49 currently carries my claim that word recall is the routing signal.
p26 disproves sufficiency; the design note must not stay wrong.
**Do:** Comment on GH-49 with the p26 evidence and the revised rule: recall (token
loss) AND structure (binding/shape) together gate escalation.
**Files:** none (issue comment)
**Done when:** the comment exists and names p26 with its 100%-recall figure.
