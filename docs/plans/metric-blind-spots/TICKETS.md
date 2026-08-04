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

### TICKET-B2 — positional comparison on BOTH sides · DONE (`docs/log/2026-08-01_TICKET-B2.md`; reopened for a lane-splitter regression on regular grids and re-closed, `docs/log/2026-08-01_TICKET-B2-reopen-fix.md`; reopened again for float-sensitivity in the dedup fix and closed with an Otsu-style cut, `docs/log/2026-08-01_TICKET-B2-otsu-cut.md`; reopened again for paired-column collapse and closed with a left anchor + zero-floor cut, `docs/log/2026-08-01_TICKET-B2-paired-columns-fix.md`; reopened again for real-page over-splitting and closed with a best-first split bounded by the widest row's value count, `docs/log/2026-08-01_TICKET-B2-widest-row-cap.md`) · depends-on: B1 · wave 2
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

### TICKET-B5 — reconstruct a row label wrapped across two lines · DONE · depends-on: B2 · wave 3
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

**Done, see `docs/log/2026-08-01_TICKET-B5.md`.** Two new pieces of geometry, both derived
from the page's own data, no constants:

- `_merge_continuation_bands` folds a text-only band with no values into the value-bearing
  band immediately below it, single-hop, when the two share a left edge, the gap is no
  wider than the text-only band's own line height, it is not a hierarchy marker, **and**
  both bands are set in the same font. The font check was not in the original design and
  had to be added: the reference corpus carries bold section headers ("Key fiscal
  determinants") flush against the data row beneath them, geometrically identical to a
  genuine wrapped continuation (same indent, gap under one line height) — without the font
  check, 7 real pages regressed by folding the header into the row's label. A continuation
  is literally the same label carrying on, so it is set in the row's own font; a header is
  conventionally set apart.
- `_reading_order` replaces plain x0-sort inside a band with a same-page-derived bisection:
  split the band by its largest y0 gap, keep the split only if both halves independently
  share one left edge, and read top-half-then-bottom-half in that case, x0-order otherwise.
  Needed because the 6pt-gap case is already one merged band by the time step (2) runs, so
  the label/value split itself must know reading order, not just band construction. A
  cruder `(y1, x0)` sort tried first broke real pages (word bbox top edges are not level
  across glyphs of different heights on one physical line, e.g. "Nominal" vs "GDP1"'s
  digit) and, separately, exposed a *pre-existing* bug in step (1)'s band clustering (a
  transitively-chained y-overlap test can absorb x-unrelated content — e.g. a rotated axis
  title plus a chart-legend fragment — into one band on a generously-scoped table bbox);
  that bug is out of B5's scope and was not touched.

Corpus gate: all 19 preserved pages score byte-identical to `obr_efo_2022_11_baseline.json`
(no page regresses, none improves) — the baseline needed no re-recording. Full suite:
1401 passed, 1 xfailed (base 1398/2 + 3 new cases from the gap-parametrized fixture − the
removed xfail).

## Stream C — pipeline response

### TICKET-C1 — surface unexplained lanes as content loss · DONE (`docs/log/2026-08-01_TICKET-C1.md`) · depends-on: B2 (formerly listed as B4, merged into B2) · wave 5
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

### TICKET-C2 — surface content loss on local-only runs · DONE (`docs/log/2026-08-02_TICKET-C2.md`) · depends-on: C1 · follow-up
**Found reviewing C1, 2026-08-02.** C1 closes B1 review finding 1 **only for cloud-enabled
runs.** The surfacing lives in `_table_page_needs_escalation`, and the call chain is:

```
_escalation_profile = _pick_escalation_provider(available)   # None when no non-local provider
    -> if _escalation_profile is not None:  _escalate_table_page(...)
        -> if not _table_page_needs_escalation(...): return    # <- the only surfacing point
```

`_pick_escalation_provider` returns `None` when `[p for p in available if p.tier != TIER_LOCAL
and p.supports_per_page]` is empty. So on a **local-only run** — no cloud provider configured,
`--strict-local`, or a cost cap suppressing the lane — `_table_page_needs_escalation` never
runs and **not-scorable pages and unexplained lanes are as invisible as they were before C1.**

That is not an edge case: socr is local-first by design (CLAUDE.md — "local qwen VLM first,
cloud only on a judge signal"), so local-only is a normal, arguably default configuration. The
corpus re-score found **17 of 68 pages** not-scorable; on a local-only run every one of them
still ships with no trace, which is exactly what the no-silent-content-loss rule forbids.

**Do:** Emit the `table_not_scorable` and `table_unexplained_lanes` audit events from a point
in the page loop that does not depend on `_escalation_profile`.

**The real decision this ticket must settle — do not skip it.** Surfacing on every run means
scoring every table page against its native layer on every run, including local-only runs that
currently do none of that work. Measure that cost on the reference document before choosing:
always-on, or on-by-default-with-an-opt-out. This is a cost/visibility trade, not a bug fix,
and it is why C1 did not simply do it.
**Files:** `src/socr/pipeline/orchestrator.py`, `tests/test_gh95_tables_trust.py`
**Done when:** a test asserts a not-scorable page reaches `tables_trust` on a run with **no
non-local provider available**; the added per-page cost is measured and recorded; and
`~/venvs/socr/bin/pytest tests/ -q` exits 0.

**Done, see `docs/log/2026-08-02_TICKET-C2.md`.** One new method,
`_surface_table_scoring`, calls the existing `_table_page_needs_escalation` emitter
(no new predicate, no new event kind) from an `elif` arm of the per-page loop's
escalation block that fires whenever the `if` (provider present, lane not degraded)
does not — the `if` branch itself, and every existing escalation test, is unchanged.
Measured cost on the OBR reference document (68 pages, 18 clearing the grid
predicate): `native_rows_from_page` + `score_page` combined ≈ 9.3s total, ≈137ms
mean/page, ≈692ms worst case — negligible against the per-page VLM inference this
loop already pays for, so the decision is **always-on, no opt-out**. Corpus gate
unchanged (3 passed, score-neutral); full suite 1406 passed, 1 xfailed; lint clean.

### TICKET-B6 — stop absorbing a non-numeric data column into the row label · TODO · depends-on: B2 · follow-up
**Found by the engine-vs-metric separation on the corpus, 2026-08-02. Highest-value item
left in this plan: it zeroes an entire page class for *every* engine.**

**Problem:** OBR page 53 (printed p.49, Table 7) has a literal **Met / Not Met** column. socr
emits it correctly as its own column:

```
| March 2022 forecast | Met | -1.0 | 1.0 | 28.1 |
```

`native_rows_from_page`'s label boundary is the **last non-numeric word**, so it swallows the
`Met` into the label:

```
label='March 2022 forecast Met'   values=('-1.0', '1.0', '28.1')
```

`'March 2022 forecast Met'` never matches the emitted `'March 2022 forecast'`. **Normalized
label overlap on that page: 0 of 6.** All 17 rows fail to match, and the page scores **0.0%**.

**It is not engine-specific — that is the tell.** The preserved comparison run records
`agy 0.0%` and `gemini-ocr 0.0%` on the same page. Three independent engines scoring zero on
a page they transcribed well is the metric, not the OCR.

This behaviour is already in STATUS.md's landmine list ("tables with non-numeric data columns
are currently absorbed into the label by the last-non-numeric-word rule") but was never
connected to a score. It is worth a page of the reference document, and the whole class of
target/status/flag columns that financial tables routinely carry.

**Do:** Let the label boundary end before a **column** of non-numeric values, rather than
before the last non-numeric word in the row. Signal available without a new heuristic: B2's
lane machinery already establishes columns from token geometry with ≥2-row support — a
non-numeric column has consistent x-position across rows, whereas a numeral or word inside a
prose label does not.

**Landmine — read before starting.** A previous attempt died by clustering numerics *to find*
the boundary; a numeral inside a label ("Growth Plan after **17** October reversals") formed
its own cluster and dragged the boundary left. Keep boundary-finding and lane clustering
decoupled — that decoupling is what made B2 survivable. There is a regression fixture for the
in-label numeral; it must keep passing.
**Files:** `src/socr/tables/native_rows.py`, `tests/test_metric_corruption_battery.py`
**Done when:** a test builds a table with a literal `Met`/`Not Met` column and asserts the
label excludes it and the row matches an emitted table that keeps it as a column; the in-label
numeral fixture still passes; the corpus gate improves on p53 and regresses nowhere; and
`~/venvs/socr/bin/pytest tests/ -q` exits 0.

### TICKET-B7 — a non-numeric status column costs the row its last value · TODO · depends-on: none · follow-up
**Diagnosed from the preserved run's raw model cache, 2026-08-03. This is a real OCR defect,
not a metric defect. It was first written up as "qwen drops repeated values" — that framing
was wrong and is corrected below; do not go hunting for a dedup bug.**

**Mechanism.** On OBR page 53 the model emitted this header:

```
| | Per cent of GDP |        | £ billion |        |
| | Forecast        | Margin | Forecast  | Margin |
```

Four data columns. The printed table needs **five**: the literal `Met` / `Not Met` status
column plus those four numbers. No column was allocated for the status field, so `Met` takes
the first numeric slot and every fully-populated row pushes its final value off the right
edge:

```
printed:   March 2022 forecast   Met   1.3   1.3   36.2   36.2
emitted:   | March 2022 forecast | Met | 1.3 | 1.3 | 36.2 |
```

`36.2` was not deduplicated — it fell off the end. The same cell is missing from every row
with a full complement of values, including `| March 2022 forecast | Met | -1.0 | 1.0 | 28.1 |`
where the repeated-value theory does not apply at all.

**Confirmed as the model, not post-processing.** The raw cached blob
(`cache/1d/1db22f…json`, `engine: qwen`, `page_num: 53`) already has five columns before any
socr processing touches it.

**Content loss, but NOT silent — corrected 2026-08-05.** This was first written up as silent
loss violating the no-silent-content-loss rule. It is not: the preserved run flags page 53 with
`value_guard_row_count_warning` ("output has 19 numeric row(s) but native has 21 effective"),
`native_table_verifier_warn` (`output_cols=5` against `native_lanes=14`) and `dualpass_flagged`,
and the page is `untrusted` with `patch_eligible: false`. The value is still missing, so this is
a real defect worth fixing — but it announces itself, which lowers its severity and means it is
**not** the rule violation the first write-up claimed. A presence check cannot see the loss;
the trust surface can, and does.

**Do:**
- Establish first whether this is general or specific to a status column: build fixtures with
  a non-numeric column in first, middle and last position and see which lose cells. The
  mechanism above predicts the loss is about **column-count allocation**, not about the word
  `Met` — confirm or refute that before changing anything.
- **Do not tune the prompt against this page.** If the fix is prompt-side it must be justified
  by the general shape (a table whose data columns are not all numeric), not by one document.
- Check whether TICKET-B6 interacts: B6 fixes the *ground truth* swallowing the same column.
  The two are independent bugs about the same table feature, on opposite sides of the
  comparison. Fixing B6 alone would make this page's score go *up* while the value is still
  missing — worth knowing before either lands.
**Files:** to be determined by the diagnosis; likely the table-extraction prompt/handling path
**Done when:** the mechanism is established as general or status-column-specific with fixture
evidence; a fix lands with a test that does not depend on this one page; and
`~/venvs/socr/bin/pytest tests/ -q` exits 0. **Do not re-run OCR to validate** — local-model
variance exceeds the effect; use the preserved run.

### TICKET-B8 — CLOSED, INVALID (filed on a misreading, 2026-08-03; withdrawn 2026-08-05)
The ticket claimed the damage on page 53 "was detected and shipped anyway" because the cached
blob records `audit_passed: True` alongside `confidence: 0.0` and 56 dual-pass disagreements.

**That was wrong.** `audit_passed` is the per-page hallucination/accept check — a different
mechanism from the table trust surface. Checking the preserved run's own `tables_trust.json`:

```
untrusted_pages: [22, 25, 35, 37, 39, 41, 43, 45, 46, 53, 56, 59, 60, 61, 62, 64, 65, 66, 67]
page 53 reasons: dualpass_flagged, native_table_verifier_warn, value_guard_row_count_warning
  - value_guard_row_count_warning: output has 19 numeric row(s) but native has 21 effective
  - native_table_verifier_warn: ambiguous_lane_count_mismatch: native_lanes=14, output_cols=5, gap=9
  - dualpass_flagged: table 0: 'Per cent of GDP' -> '' (+55 more)
```

Three independent kinds fired, one of them naming the exact column shortfall that causes B7,
and the page is `flagged: true` with `patch_eligible: false`. **The surfacing works.** The page
was accepted as text and flagged as untrusted for tables, which is the intended behaviour.

Kept as a closed entry rather than deleted, so nobody re-derives the same false premise from
the same cache field. The lesson: `audit_passed` in a cached page blob says nothing about the
table trust surface — read `tables_trust.json`, not the blob.

## Explicitly out of scope

- **Header-semantic column identity.** Reading spanning/multi-row headers from the
  native layer to give lanes meaning. High parser-bug surface, and the uniform-shift
  case it would catch was not the observed failure.
- **Repairing** the OCR's column placement. This plan makes the error *visible*; what
  the pipeline does about it beyond C1 belongs with #49.
