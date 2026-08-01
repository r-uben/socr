# TICKETS — metric blind spots (socr #123)

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Parallelizable = no shared files / no dep. Each ticket = one implementer agent,
then one reviewer pass before commit.

## Context

`benchmark/table_exactness` grades OCR'd tables against the PDF's native text layer.
It is not only a report: `tables/escalation_decision` uses it as the **accept rule**
for GH-96, deciding whether a second engine's table replaces the incumbent's. So a
blind spot in the metric is a blind spot in a production gate.

Seven defects were found in this metric on 2026-07-30/08-01. Six made the OCR look
**worse** than it was. The seventh is the dangerous direction: it makes a wrong table
look **right**.

Design input: an adversarial review recorded on #123. Its central points, which this
graph implements:

- The proximate bug is the **empty-cell filter**, not missing column identity. A
  column shift into an empty cell preserves value order and multiset; the only signal
  it destroys is the **gap pattern**, and the code discards that on both sides.
- Column identity is buildable and survives the earlier failed attempt, but only its
  cheap core is worth it: anonymous lanes plus one global monotone alignment.
  **Header-semantic identity is explicitly out of scope.**
- The check that would have caught all seven is a **metamorphic corruption battery
  over the scorer**. Nothing currently grades the metric — the doer grading its own
  work, one level up.

## Stream A — grade the metric itself

### TICKET-A1 — corruption battery over the scorer · TODO · depends-on: none · wave 1
**Problem:** Nothing tests the metric. Seven defects were found by accident, each
after it had already produced published numbers. All seven share one shape: a known
perturbation of the input moved the score in the wrong direction, or failed to move
it at all.
**Do:** Add a standing metamorphic test over `score_page`, built on fitz-generated
fixture pairs (native PDF + emitted markdown). Assert two properties:
- **benign transforms leave the score unchanged** — bold a cell, respell a footnote
  marker in each of the five known spellings, wrap a label across two lines, reflow
  whitespace, add a `Note:` row
- **corrupting transforms make the score strictly worse** — shift a value into an
  adjacent empty cell, swap two values between rows, drop a value, perturb a digit

The shift-into-empty-cell case **must fail on today's code**. Land it xfail-marked
with a reference to TICKET-B2, which makes it pass.
**Files:** `tests/test_metric_corruption_battery.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_metric_corruption_battery.py -q`
exits 0 with at least one `xfail` reported, and the xfail's reason names TICKET-B2.

## Stream B — scoring correctness

### TICKET-B1 — refuse to score a page with no table · DONE · depends-on: none · wave 1
**Problem:** Page 54 of the OBR reference document is prose and two fan charts. socr
correctly emitted no table; `native_rows_from_page` invented five rows from chart
legends and axis labels, and the page scored **0.0%**. Reported aggregates therefore
count pages-with-no-table as engine failures, understating every engine by an unknown
amount.
**Do:** `score_page` returns a not-scorable report rather than 0% when the ground
truth does not establish a grid. Reuse the GH-113 predicate already in
`orchestrator._table_page_needs_escalation`: at least two value columns, and at least
two rows sharing that width. Set `ceiling_note`; leave `pct` as `None`. Any aggregate
helper must exclude not-scorable pages rather than counting them as zeros.
**Files:** `src/socr/benchmark/table_exactness.py`, `tests/test_gh96_table_exactness.py`
**Done when:** a test builds a page whose only numerics are chart-legend text and
asserts `score_page(...).scorable is False` and `.pct is None`; and
`~/venvs/socr/bin/pytest tests/ -q` exits 0.

### TICKET-B2 — stop discarding sparsity on both sides · TODO · depends-on: B1 · wave 2
**Problem:** `markdown_rows` filters empty cells out of a row before comparing
(`values = [c for c in raw_values if _is_value(c)]`), and ground-truth rows store a
flat ordered list with no gaps. So `['-1.0','1.0','28.1','']` and
`['2.5','0.5','','13.8']` both reduce to three values and both match a three-value
ground truth. **A value in the wrong column scores as correct** — confirmed on OBR
page 53, where Margin figures sit under Forecast in one block and correctly in
another.
**Do:** Preserve cell positions on the markdown side rather than compacting them.
Ground truth keeps its native x-position per value (the field is added in B3; until
then compare positionally against the preserved markdown shape). Removing the filter
alone must make the shift-into-empty-cell case in TICKET-A1 fail the corruption
battery's "strictly worse" assertion — i.e. flip that xfail to a pass.
**Files:** `src/socr/benchmark/table_exactness.py`,
`tests/test_metric_corruption_battery.py`
**Done when:** the shift-into-empty-cell xfail from A1 is removed and passes; and
`~/venvs/socr/bin/pytest tests/ -q` exits 0.

### TICKET-B3 — column identity in the ground truth · TODO · depends-on: B2 · wave 3
**Problem:** `native_rows_from_page` reads values in x-order and never records which
column each landed in, so the ground truth cannot express a column shift even once
the markdown side preserves gaps.
**Do:** Attach an anonymous lane index to each ground-truth value.
- Cluster the **right edges** (`x1`) of value tokens, not left edges — financial
  columns are right-aligned and left edges drift with digit count. Where a lane's
  right-edge variance exceeds its centre-of-mass variance, anchor on the centre
  instead; choose per lane by lower variance, no constant.
- Cluster **only tokens right of the existing label boundary**, seeded by the current
  last-non-numeric-word rule. A previous attempt clustered numerics *to find* the
  boundary and a numeral inside a label ("Growth Plan after **17** October reversals")
  formed its own cluster and dragged the boundary left. Decoupling the two is what
  makes this survivable.
- Require each lane to have support in **at least two rows** — same justification as
  the GH-113 grid rule; a one-off numeral inside one label cannot form a lane.
- Lanes are **anonymous indices**. Do not read headers for identity. Spanning and
  multi-row headers are where native text layers are worst, and header-semantic
  identity is out of scope for this plan.
- A row whose values snap to no lane is flagged lane-ambiguous and scored
  conservatively, never guessed.
**Files:** `src/socr/tables/native_rows.py`, `tests/test_gh96_table_exactness.py`
**Done when:** a test builds a page with a right-aligned numeric column, a numeral
inside a label, and a sparse row, and asserts the numeral does not form a lane and the
sparse row's value snaps to the correct lane index; and
`~/venvs/socr/bin/pytest tests/ -q` exits 0.

### TICKET-B4 — global monotone column alignment · TODO · depends-on: B3 · wave 4
**Problem:** Native lane count and emitted column count legitimately differ — 14 vs 6
on OBR page 53, because paired-year headers and stub columns collapse. Index equality
cannot be used.
**Do:** Compute one global, monotone, injective mapping from emitted markdown columns
to native lanes per table (per panel), chosen to maximise total cell agreement across
all matched rows. Needleman-Wunsch over columns; the pairing score for (md column j,
lane k) is the count of rows where the values agree. No thresholds.

Because the mapping is **global**, a table where one block is shifted and another is
not cannot satisfy both — the correct block pins the mapping and the shifted block's
cells score as errors. That is exactly the OBR page 53 case.

**Document the limit in the module docstring:** a *uniformly* shifted table, where
every row consistently places Margin under Forecast, is absorbed by the mapping and
still scores clean. Fixing that needs header semantics and is deliberately out of
scope.
**Files:** `src/socr/benchmark/table_exactness.py`, `tests/test_gh96_table_exactness.py`
**Done when:** a test builds a two-block table with one block shifted one column and
asserts it scores strictly below the same table unshifted; the module docstring states
the uniform-shift limitation; and `~/venvs/socr/bin/pytest tests/ -q` exits 0.

## Stream C — pipeline response

### TICKET-C1 — surface unexplained lanes as content loss · TODO · depends-on: B4 · wave 5
**Problem:** The pipeline already computes `native_lanes=14, output_cols=6, gap=8` and
only writes it to the log. The raw gap must not be thresholded — 14-vs-6 is frequently
benign — but once B4's alignment exists there is a threshold-free derived fact:
**unexplained lanes**, meaning lanes that received no markdown column under the best
mapping *and* contain values in rows that matched. Unexplained lanes > 0 means values
exist in the native layer with no home in the output. That is content loss, and the
repo's no-silent-content-loss rule requires it to surface at every level.
**Do:** Expose an unexplained-lane count from the alignment. Emit a named audit kind
for pages where it is non-zero, add that kind to `tables_trust.TABLE_DISTRUST_KINDS`,
and roll it into the document-level surface. In `escalation_decision`, never accept a
candidate that **increases** unexplained lanes; prefer the candidate that decreases
them, with exactness as the tiebreak. "Zero versus non-zero" is a fact about the data,
not a tuned cutoff.
**Files:** `src/socr/benchmark/table_exactness.py`,
`src/socr/tables/escalation_decision.py`, `src/socr/core/tables_trust.py`,
`src/socr/pipeline/orchestrator.py`, `tests/test_gh96_escalation_decision.py`
**Done when:** a test asserts a candidate that increases unexplained lanes is rejected
even when its exactness is higher; the new kind is in `TABLE_DISTRUST_KINDS`; and
`~/venvs/socr/bin/pytest tests/ -q` exits 0.

## Explicitly out of scope

- **Header-semantic column identity.** Reading spanning/multi-row headers from the
  native layer to give lanes meaning. High parser-bug surface, and the uniform-shift
  case it would catch was not the observed failure.
- **Repairing** the OCR's column placement. This plan makes the error *visible*; what
  the pipeline does about it beyond C1 belongs with #49.
