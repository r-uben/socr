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

## Round 1 (c6767aa) and why it was wrong

Round 1 gated term (b) on the CANDIDATE's own row widths: apply the
reconciliation only when a strict majority of the candidate's numeric body rows
carry two or more numeric tokens. Astra's review falsified it. Dominance
computed from the surviving rows disappears together with the missing rows: a
numeric table with two legitimate one-number rows followed by eighteen dense
ones, truncated to those two sparse rows, fails the dominance test, term (b)
abstains, term (a) sees no style break, and through the real A2 cross-pool
selection the truncated qwen reading WINS over the complete gemini one -- the
exact loss A2 exists to prevent. Astra also rejected the constant: two is the
smallest integer above one, which is arithmetic, not a prose/table
discriminator, and on the BoE page the native count is table-specific at no
width (19/4/2 bands at widths 1/2/3).

## Rule

Eligibility is established on the NATIVE side, which a model cannot truncate.
Term (b) applies only when the native page shows **recurring numeric column
lanes** (`structure_check._native_page_has_column_lanes`): at least two
column-like x-lanes, each recurring over at least `_MIN_TABLE_ROWS` bands, with
at least `_MIN_TABLE_ROWS` bands populating two of them at once. Otherwise the
reconciliation is vacuous and term (b) abstains, leaving the page to term (a)
and the ladder verdict.

The discriminator is ALIGNMENT, not numeral count: a table's numerals recur in
shared x-lanes down the page, prose figures scatter. Measured:

| page | x0-lanes | column-like lanes | bands populating >= 2 | gate |
|---|---:|---:|---:|---|
| BoE p1 (real PDF) | 16 | 2 | 0 | closed |
| synthetic text table | 4 | 2 | 0 | closed |
| Astra's sparse-prefix numeric table | 2 | 2 | 18 | open |
| ECB bulletin p2 fixture | 11 | 11 | 3 | open |
| ECB bulletin p3 fixture | 11 | 11 | 3 | open |

The lane machinery is reused, not rewritten: `reconstruct.has_numeric_columns`
was factored into `has_recurring_numeric_columns(words, min_lanes_per_row)` (the
same factoring `native_verifier._lane_count_from_words` already did for its own
page-level twin), keeping GH-248's lane-reuse rule and GH-349's two-anchor
(x0 and x1) test. `has_numeric_columns` still calls it with the detector's
`_MIN_LANES_PER_ROW` and is behaviourally unchanged.

`_MIN_RECONCILABLE_LANES = 2` is an arity floor, not a fitted threshold: the
reconciliation compares row WIDTHS, and a single recurring lane gives every
native band width 1, which is the vacuous case #703 was filed for. It is
deliberately weaker than the detector's 3 because this is an ABSTENTION gate on
a content-loss guard: a wrong "no lanes" silently disarms A2, a wrong "lanes"
only leaves A2 armed as before.

`row_shape_min` stays the candidate's own minimum. The lane gate has already
established that the page has column structure, so counting bands with one or
more numerals is no longer "every prose line on a prose page" -- and keeping the
minimum is what catches the sparse-prefix truncation (2 surviving width-1 rows
against 20 native bands). A lane-derived width would change the native counts A2
measured on the ECB fixtures and buys nothing on any page measured here.

The issue's alternative ("use B1's prose corroboration witness") is not
available: PR #695 deleted model-prose corroboration by ruling.

## Evidence

- Real BoE p1, ungated vs gated: `(True, False)` — term (a) never fired on this
  page, so the gate is the only thing that changed.
  (`tests/tables/test_gh703_text_table_dominance.py`, corpus-skipped.)
- Hermetic text-table fixture: same difference, same direction. Its native words
  are laid out as running text (token widths, per-line indents) rather than on a
  fixed pitch, so the numerals genuinely scatter.
- Pages WITH lanes: gated and ungated verdicts identical (complete `False`,
  truncated `True`) — the gate is inert there.
- Astra's three reproducers, transcribed into the repo file with only the patch
  target renamed: `test_numeric_table_sparse_prefix_is_still_truncated`,
  `test_sparse_prefix_truncation_not_kept_in_the_strict_pool`,
  `test_sparse_prefix_truncation_does_not_win_over_complete_reading`. The
  complete gemini reading wins selection; round 1 flipped that to qwen.
- The two real ECB truncation fixtures still fire:
  `tests/tables/test_structure_check_truncated.py::test_real_fixture_bulletin_p2_picks_complete_candidate`
  and `::test_real_fixture_bulletin_p3_picks_complete_candidate`.

## What the real finalization path actually does on BoE p1

Executed, not inferred: `manifest.finalized_page_record` on a `DocumentState`
built over the real PDF with the run's own sidecar geometry and the run's own
cached qwen `PageOutput`, against this worktree's source.

- `table_output_defect` on that candidate is now `''` (was `table_truncated`).
- The page nevertheless still ships the fail-closed marker: `status=error`,
  `failure_mode=structure_class_ladder_exhausted`, ending `fail_closed_marker`,
  47-character body. The strict grid pool is EMPTY and the grid winner is
  `None`.
- Cause, and it is not term (b): the cached qwen `PageOutput` carries
  `audit_passed=false` with `judge_reason='judge raised: timed out'`, so
  `_grid_authored_attempt` never admits it, and a text table has no numeric rows
  to corroborate with.
- Counterfactual with `audit_passed=True` (representing the
  `table_ladder_accepted` event the run's audit log records for qwen, rung
  `ollama:glm-5.3-flash:cloud`, witness scope page): the page ships the full
  3539-character qwen body carrying the table, `fall to 4%` and `around 32`,
  ending `model_output`, status `success`. It does so with the gate ON **and**
  OFF — that path never reaches the structure-class branch.

So this change is proven at the predicate and at S1 selection; it is NOT proven
to be the last blocker on the census page itself.

## Round 3: recurrence-seeded lanes (Astra's bridge counterexample)

Round 2 asked `reconstruct.has_recurring_numeric_columns` its lane question with
that detector's own clustering: x positions are sorted and each one joins the
running lane when it is within `_LANE_X_TOL_PT` of the **previous** one. That is
single-linkage chaining, and a chain can span arbitrarily more than the
tolerance.

Astra's counterexample: in the sparse-prefix fixture the recurring columns sit
at x0 = 12 and x0 = 24. Print one unrelated footnote value `999` at x0 = 18,
x1 = 26, on a band of its own. Its left edge is 6pt from both column left edges
and its right edge is 6pt from both column right edges, so on **both** anchors
the two columns chain into a single lane. The gate returns False, term (b)
abstains, and selection flips from the complete gemini reading to the truncated
qwen one. The bridging word never recurs; it only has to exist once.

Chaining is safe where the detector's answer is used positively -- merging lanes
can only lower the lane count, so `has_numeric_columns` stays conservative about
claiming a grid. #703 introduced the first NEGATIVE use, where an under-count
switches a loss guard off. The fix is scoped to that use.

`has_recurring_numeric_columns(..., seeded_lanes=True)` clusters by recurrence
instead of adjacency (`reconstruct._seeded_lane_of`), in three steps and with no
constant of its own:

1. **Seed.** An x position is a seed when the bands carrying a numeral within
   `_LANE_X_TOL_PT` of it number at least `_MIN_TABLE_ROWS` -- the same
   recurrence already required of a column-like lane downstream.
2. **Found.** Seeds are taken in order of decreasing recurrence (ties by x) and
   each founds a lane unless it is within the tolerance of one already founded.
   Lane identity is a distance to a fixed centre, never a chain, so two centres
   more than the tolerance apart can never be merged.
3. **Assign.** Every other x joins the nearest centre within the tolerance; an x
   within reach of no centre is dropped and contributes to no row's lane set.

`has_numeric_columns` and every other existing caller keep adjacency clustering
(`seeded_lanes` defaults to False), pinned by
`test_detector_entry_point_keeps_adjacency_clustering`.

### Why direction (B) was not taken

The alternative offered was to treat an ambiguous lane count (a lane whose span
exceeds the tolerance, i.e. chaining occurred) as evidence that the page HAS
lanes, so the loss guard fails closed. Measured, that rule fires on the ticket's
own page: BoE 2018 p1 clusters its 16 x0 positions into chained lanes, so (B)
would report lanes there and leave #703 unfixed. (A) discriminates; (B) does not.

### The BoE 2003 "coverage loss" premise, measured

Astra's coverage probe is correct about the verdicts and wrong about the pages.
Seeded clustering changes no verdict anywhere on the census corpus (45 pages,
BoE + ECB + Banxico): the gate result is identical to round 2 on every one. The
three `boe-meetings-2003-table-p15-17` pages still close it -- because they are
not tables. They are the Bank's narrative annex ("ANNEX: SUMMARY OF DATA
PRESENTED BY BANK STAFF"), continuous prose whose lines quote two figures each:

| page | native bands at width 2 | recurring lanes (x0) | bands populating 2 recurring lanes | gate |
| --- | --- | --- | --- | --- |
| boe-meetings-2003 p1 | 9 | 1 | 0 | False |
| boe-meetings-2003 p2 | 17 | 3 | 0 | False |
| boe-meetings-2003 p3 | 10 | 2 | 1 | False |
| banxico-2018 p3 | 11 | 4 | 0 | False |
| boe-minutes-1997 p1-p3 | 0 | 0 | 0 | False (no text layer at all) |
| boe-meetings-2018 p1 (the ticket) | 4 | 2 | 0 | False |
| boe-meetings-2018 p2 / p3 | 12 / 14 | 6 / 6 | 2 / 4 | True |
| ecb-reports-2003 p2 | 41 | 23 | 41 | True |
| ecb-surveys-2018 p1-p3 | 13 / 12 / 13 | 11 / 12 / 10 | 12 / 11 / 12 | True |

Those width-2 bands are prose lines, not rows: no band puts numerals in two
different recurring lanes, which is the shape term (b) needs before it can
reconcile anything. This is exactly the class #703 was filed for, so closing the
gate there is the intended behaviour rather than a price paid for it. The 1997
scans have no text layer, so term (b) already abstained at `if not words`.

Pinned by `test_real_boe_2003_pages_are_prose_not_tables`, which asserts the
annex heading and the absence of any table before asserting the verdicts.

## Round 4: seed by own occupancy, split by co-occurrence

Astra reproduced two further defects in round 3's seeding, both ending in the
truncated candidate winning selection over the complete one.

**Seeding counted the neighbourhood, not the position.** A seed qualified on the
union of bands anywhere within `_LANE_X_TOL_PT` of it, so a position occurring
ONCE could borrow both neighbouring columns' support and outrank each of them.
Add a legitimate final row holding only the second column's value, so column 12's
bands no longer subsume column 24's, then print one numeral at x=18:

| position | bands it actually occupies | bands in its neighbourhood |
| --- | ---: | ---: |
| 12 | 20 | 21 |
| 18 (the bridge) | 1 | 22 |
| 24 | 19 | 20 |

The bridge is founded first and suppresses both real columns at distance 6.
Fixed centres stopped later merging; they did not stop founding the wrong centre.
Round 3's bridge test passed only because its column-12 bands happened to
subsume column 24's, so the tie broke toward a genuine centre.

**Tolerance merged two columns that share rows.** Two recurring anchors 5pt
apart were collapsed into one centre even with eighteen rows carrying a distinct
numeral in each. A tolerance calibrated for conservative positive detection is
not evidence of absence.

Round 4 replaces the seeding with four steps, still with no constant of its own:

1. **Quantise** x with `round`, the same rounding the band key already applies
   to y, so sub-point extraction jitter inside one printed column collapses to
   one position.
2. **Qualify** a position as recurring on its OWN occupancy: tokens at that
   position on at least `_MIN_TABLE_ROWS` bands. A one-off position can no
   longer found a lane at any ranking.
3. **Found** lanes from recurring positions in decreasing occupancy order, each
   founding a centre unless it is within the tolerance of an existing one AND
   does not co-occur with it. Two recurring positions carrying distinct numerals
   on the same band, on at least `_MIN_TABLE_ROWS` bands, are separate columns
   by direct evidence: one column cannot hold two cells of one row. Co-occurrence
   overrides the tolerance rather than shrinking it.
4. **Assign** every other x to the nearest centre within the tolerance, or drop
   it.

### Corpus re-measure: three pages flip, all of them non-tables

The 45-page sweep was re-run. Round 4 changes three verdicts against round 3,
all True to False, and nothing else:

| page | bands at width 2 | lanes | wide bands | round 3 | round 4 |
| --- | ---: | ---: | ---: | --- | --- |
| boe-meetings-2018 p2 | 12 | 2 | 0 | True | False |
| boe-meetings-2018 p3 | 14 | 3 | 1 | True | False |
| ecb-reports-2000 p3 | 5 | 2 | 2 | True | False |

None is a table. The two BoE pages are Inflation Report prose sections carrying
vector fan charts (58 and 28 drawing objects, no images); their recurring numeric
positions are y-axis tick labels, `180/160/140` down one axis and `90/80/70/60/50`
down another, each on a band of its own, and almost no band holds a cell in two
of them. Round 3 read those two axes as a two-column grid through neighbourhood
support. The ECB page is the Bulletin's imprint: a left-aligned block of address,
telephone, fax and telex numbers. Pinned by
`test_real_boe_2018_chart_pages_close_the_gate`.

Every other verdict is unchanged, including the ticket page (False), ECB
p2/p3 and the annex pages (False, prose -- Astra rendered all three and
retracted the coverage claim).

## Residuals

- **The gate's per-page verdicts over the census corpus** are tabulated in the
  round-3 section. Every page that closes it was measured to be prose or to
  have no text layer.
- **Superseded in round 3, refined in round 4.** The residual below described
  the round-2 gate.
  Lane chaining is now eliminated for this use (see the round-3 section), and
  the pages the round-2 residual named as lost coverage were measured to be
  prose, not numeric tables. What remains true is narrower: term (b) is armed
  only on pages showing recurring numeric columns, so a genuinely tabular page
  whose columns are too sparse to recur over three bands relies on term (a)
  alone. No such page was found in the 45-page census corpus.
- **#643 is untouched.** It lives at a different call site
  (`manifest._row_shape_reconciliation_ok`, A1b), which this change does not
  modify. On the synthetic GH-643 shape the lane gate closes (above), so if that
  shape ever reached term (b) the guard would abstain; that is neither a fix nor
  a regression for #643 itself.
- **The A1b twin also rejects the BoE page.** Measured:
  `_row_shape_reconciliation_ok(words, markdown)` returns `False` on the real
  BoE p1 text and words. The round-1 log claimed no measured page hits the twin;
  that was wrong. The twin fails closed rather than shipping wrong content, and
  it is deliberately left alone here, but it is a live second path to the same
  loss on this page shape.
- **The end-to-end BoE recovery is not demonstrated** (see the finalization
  section above). The census state cannot be reconstructed exactly: the run's
  audit log shows a gemini attempt for p1 (`native_table_verifier_warn`,
  `candidate_truncated`) but no gemini candidate was cached, and inventing its
  text would be fabrication.
- The same run records `native_table_verifier_hard_fail` on the qwen candidate
  (`value_guard_label_binding`, 4 interleaved pairs) and
  `ambiguous_lane_count_mismatch: native_lanes=16, output_cols=2`. Those guards
  are untouched here and may reject this candidate independently.
- This change adds no flag of its own; whether the page ships flagged is the
  ladder's and the verifier's business.

### Round-4 residuals

- **Exact-x recurrence alone is not usable**, which is why step 1 quantises.
  Measured: requiring float-equal recurrence closes the gate on four real
  tables in the corpus that round 3 opened, because their column anchors carry
  sub-point jitter. Quantising to whole points (and to tenths -- both were
  measured and agree on all 45 pages) keeps them.
- **The tolerance still merges two columns that never share a row.** Two
  recurring anchors within `_LANE_X_TOL_PT` are one lane unless co-occurrence
  proves otherwise, so a table whose two columns are never both populated on
  the same band is still read as one lane. That shape has no rows of width two,
  so term (b) has nothing to reconcile there in any case.
- Co-occurrence is counted between recurring positions only. A column that
  recurs and one that does not can still be merged; the non-recurring one
  contributes no lane of its own by design.
