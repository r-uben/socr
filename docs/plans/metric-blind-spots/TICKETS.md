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

### TICKET-A1 — corruption battery over the scorer · DONE · depends-on: none · wave 1
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

**Constraint from B1 (landed 2026-08-01):** every fixture must clear the grid predicate
in `native_rows.rows_establish_grid` — at least two value columns, and at least two rows
sharing that width. A fixture that does not is silently *not-scorable*: `pct` is `None`
and every "strictly worse" assertion then compares against `None`.
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

### TICKET-B2 — positional comparison on BOTH sides · DONE (`docs/log/2026-08-01_TICKET-B2.md`; reopened for a lane-splitter regression on regular grids and re-closed, `docs/log/2026-08-01_TICKET-B2-reopen-fix.md`) · depends-on: B1 · wave 2
**Merged with the former TICKET-B3 on 2026-08-01, after a first attempt was reverted
(`ad649b5`, reverted by `717914d`). They are one change — see "Why merged".**

**Problem:** `markdown_rows` filters empty cells out of a row before comparing
(`values = [c for c in raw_values if _is_value(c)]`), and ground-truth rows store a
flat ordered list with no gaps. So `['-1.0','1.0','28.1','']` and
`['2.5','0.5','','13.8']` both reduce to three values and both match a three-value
ground truth. **A value in the wrong column scores as correct** — confirmed on OBR
page 53, where Margin figures sit under Forecast in one block and correctly in another.
Separately, `native_rows_from_page` reads values in x-order and never records which
column each landed in, so the ground truth cannot express a column shift at all.

**Why merged — do not split these again.** The original plan fixed the markdown side in
B2 and the ground-truth side in B3, telling B2 to "compare positionally against the
preserved markdown shape" meanwhile. That interim state is incoherent: a positional list
cannot be compared against a compacted one. The first attempt implemented B2 exactly as
written, and on a **leading-gap** sparse table produced:

| transcription | score |
|---------------|-------|
| faithful — reproduces the empty cell | **80%** |
| sloppy — drops the column entirely | **100%** |

The metric rewarded the engine that destroyed the table's structure, inside a production
accept rule. That is not an implementation error; the seam was in the wrong place.

**Do:**
- Preserve cell positions on the **markdown** side rather than compacting them.
- In the same change, attach an anonymous **lane index** to each **ground-truth** value:
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
- Label-*boundary* reconstruction stays out of scope: wrapped labels are **TICKET-B5**.

**The alignment rule — settled by panel, 2026-08-01 (`/consilium`, codex:gpt-5.5 +
gemini:antigravity, 2 rounds, converged).** The map is fitted to the prediction being
scored, so a wider output has strictly more admissible maps — `C(M,L)`: at L=6, M=14 gives
**3003**. Three corrections were considered; ship the first:

- **(a) SHIP — score each engine under its own best map**, equal ground-truth denominator,
  and **document the freedom** in the module docstring beside the uniform-shift limitation.
- **(b) REJECTED — do not penalise unmapped candidate columns.** Both panelists rejected
  this independently and for the same reason: it makes strictness a function of the
  *prediction's shape*, which is precisely the rule the reverted first attempt violated. It
  also punishes a candidate for extracting a column the ground truth itself missed.
- **(c) REJECTED — do not score both sides under the incumbent's map.** Tempting (it removes
  the candidate's extra search freedom) and it was the panel's opening recommendation, but
  it is **structurally anti-improvement**: if the incumbent missed a left-hand stub column,
  a candidate that *correctly detects it* has every column shifted by one and scores near
  zero **for being right**. Column indices are ordinal, not semantic; one structural
  correction cascades misalignment across everything to its right.

**The freedom is real — do not dismiss it in the docstring.** The exploit mechanism is
low-entropy repeated cell values (`0.0`, `-`, `n/a`, blanks): a wider output gives the
alignment more paths to route a lane through them and harvest coincidental exact matches.

**Expose it as diagnostics, never as penalties** — `M`, `L`, the chosen map, and unmapped
non-empty columns. Diagnostics keep the accept rule uncontaminated while making a suspicious
win auditable. This is largely **TICKET-C1**'s unexplained-lane count arriving from an
independent direction; wire them to the same surface rather than building a second one.

**Files:** `src/socr/benchmark/table_exactness.py`, `src/socr/tables/native_rows.py`,
`tests/test_metric_corruption_battery.py`, `tests/test_gh96_table_exactness.py`
**Done when:**
- the `shift_into_adjacent_empty_cell` strict xfail from A1 is **removed** and passes
  (it is `strict=True` — fixing the bug without removing the marker turns the suite red);
- `test_a_perfect_transcription_scores_100` and
  `test_dropping_a_column_never_beats_keeping_the_gap` still pass. These guard the exact
  regression that got the first attempt reverted and **must not be weakened** to suit the
  implementation;
- **a padding test:** a candidate padded with extra columns of repeated low-entropy values
  (`0.0`, `-`, blanks) must not outscore a faithful narrower transcription. This is the
  panel's identified exploit for the map-search freedom and nothing currently covers it;
- **an escalation-asymmetry test:** a candidate that is a same-shift-plus-spurious-column
  variant of the incumbent must not be accepted. Without it the rejected option (a)-variant
  from the design note would land green;
- **`cells == number of ground-truth values`**, pinned as a test — it is what keeps
  `exact > exact` a single-denominator comparison. Blank lanes must **not** count as cells;
- a test builds a page with a right-aligned numeric column, a numeral inside a label, and
  a sparse row, and asserts the numeral does not form a lane and the sparse row's value
  snaps to the correct lane index (name the observable `lanes` field so the implementer
  does not invent a private API);
- the module docstring documents both the uniform-shift limitation and the map-search
  freedom, the latter without dismissing it;
- `test_wrapped_label_is_scored_the_same_as_unwrapped` is **still xfailed** (that is B5);
- `~/venvs/socr/bin/pytest tests/ -q` exits 0.

**Validation, separate from the unit suite:** re-score the two preserved runs in
`~/data/fiscal-ballast/_experiments/` before and after and record the flip count. STATUS.md
calls this the real acceptance test. Accepts will flip **both ways** — down where a column
shift was previously invisible, up where a leading gap previously misaligned an otherwise
correct row. Do **not** re-run OCR; local-model variance exceeds the effect.

**Also measure:** `orchestrator.py:1469` triggers escalation on `pct < 100.0`, so a stricter
metric puts more pages under 100 and fires **more second-engine calls**. Measure the
direction in the same re-score pass rather than discovering it as a bill. And
`escalation_decision`'s module-docstring calibration table (45.0 / 81.7 / 85.0) was measured
under the old meaning of `exact`; re-measure or annotate it — do not leave it standing
unqualified.

### TICKET-B3 — merged into B2 · CLOSED
Column identity in the ground truth. Absorbed into TICKET-B2 on 2026-08-01: the two
halves cannot land separately without the metric transiently rewarding
structure-destroying OCR. Number retained so existing references resolve.

### TICKET-B4 — merged into B2 · CLOSED
Global monotone column alignment. Absorbed into TICKET-B2 on 2026-08-01, after the design
pass (`docs/log/2026-08-01_TICKET-B2-design.md`) showed there is no coherent state in which
lanes exist but are not yet aligned. **The comparison contract between an L-lane space and
an M-column space *is* the alignment.** Every B2-alone rule either craters legitimate L≠M
pages or makes strictness a function of the prediction's shape — the latter measured to
create a *new* wrong-direction accept (a shifted candidate carrying one spurious empty
column scores 5/5 against the shifted incumbent's 4/5, where today both tie at 5 and are
rejected). Number retained so existing references resolve.

**Two corrections to the original B4 text, carried into B2:**
- *"per table (per panel)"* → **per page**. Per-panel is not implementable:
  `BenchmarkScorer._markdown_table_cells` (`benchmark/scorer.py:443`) flattens every pipe
  table on the page into one grid with no table boundaries, and there is no table
  segmentation on either side.
- The uniform-shift limitation still must be documented in the module docstring, and now
  has a sharper sibling: the map is fitted to the prediction being scored, so an output
  with more columns has strictly more admissible maps.

### TICKET-B5 — reconstruct a row label wrapped across two lines · TODO · depends-on: B2 · wave 3
**Found by TICKET-A1's corruption battery, 2026-08-01** — the eighth defect, and the
first one this plan caught *before* it produced published numbers rather than after.
That is the battery paying for itself; note it when judging whether Stream A was worth
the effort.

**Problem:** `native_rows_from_page` never reconstructs a label genuinely split across
two visual bands. Whichever band carries the row's values keeps only the text on that
band; the other line is silently dropped. Reproduced directly against the parser with a
two-line label (`"Central government net"` / `"debt"`) plus a plain row:

| line gap | parsed label | failure |
|----------|--------------|---------|
| 9pt, 12pt | `'debt'` | `"Central government net"` dropped entirely |
| 6pt | `'Central debt government net'` | bands merge, words x-sorted, label scrambled |

So a **perfect** transcription of a wrapped-label row matches no ground-truth label and
scores as if the row were missing — the same wrong-direction shape as the original
seven, understating the engine.

Distinct from B2: that is the markdown-side empty-cell filter, this is native-side row
identity. Distinct from `270cdab` ("don't merge a wrapped label with the line above
it"), which only stops the parser interleaving a wrapped label with an *unrelated*
neighbouring line — it was never a fix for a label spanning two lines of its own.

**Sequenced after B4, deliberately.** This is label-*boundary* work, and B3 explicitly
takes the boundary as given ("cluster only tokens right of the existing label boundary,
seeded by the current last-non-numeric-word rule"). The landmine list records that an
earlier attempt died by merging boundary-finding with lane clustering, and that
decoupling them "is what makes this survivable". Do **not** fold this into B3, and do
not touch lane clustering here.

**Do:** Reconstruct multi-line row labels before the grid predicate runs. A continuation
line is a text band with no values of its own, vertically adjacent to a value-bearing
band, sharing its left edge. Join in reading order — never x-sort across bands, which is
what produces the scrambled 6pt case. Derive adjacency from the page's own line metrics
(font size, band spacing), not a constant.
**Files:** `src/socr/tables/native_rows.py`, `tests/test_metric_corruption_battery.py`
**Done when:** the `strict=True` xfail
`test_wrapped_label_is_scored_the_same_as_unwrapped` is removed and passes at 9pt, 12pt
**and** 6pt gaps; the existing regression
`test_a_wrapped_label_is_not_merged_with_the_line_above` still passes (this must not
reintroduce what `270cdab` fixed); and `~/venvs/socr/bin/pytest tests/ -q` exits 0.

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

**Also — folded in from the B1 review (finding 1), 2026-08-01.** `ceiling_note` reaches
no surface: `grep -rn "ceiling_note" src/` outside `table_exactness.py` returns zero
hits. B1 made a chart page *not-scorable* but also **invisible** — OBR page 54 used to
report a loud, wrong `0.0%` and now reports `pct=None` with no trace anywhere. Wrong
number → no number and no trace is exactly what the no-silent-content-loss rule exists
to prevent. Route not-scorable pages through the **same** audit-kind machinery this
ticket builds (a second input to one mechanism, not a second mechanism). Details in
`docs/log/2026-08-01_TICKET-B1-review.md`.
**Files:** `src/socr/benchmark/table_exactness.py`,
`src/socr/tables/escalation_decision.py`, `src/socr/core/tables_trust.py`,
`src/socr/pipeline/orchestrator.py`, `tests/test_gh96_escalation_decision.py`
**Done when:** a test asserts a candidate that increases unexplained lanes is rejected
even when its exactness is higher; the new kind is in `TABLE_DISTRUST_KINDS`; a
not-scorable page (B1's grid gate) is visible on the document-level surface rather than
silently absent; and `~/venvs/socr/bin/pytest tests/ -q` exits 0.

## Explicitly out of scope

- **Header-semantic column identity.** Reading spanning/multi-row headers from the
  native layer to give lanes meaning. High parser-bug surface, and the uniform-shift
  case it would catch was not the observed failure.
- **Repairing** the OCR's column placement. This plan makes the error *visible*; what
  the pipeline does about it beyond C1 belongs with #49.
