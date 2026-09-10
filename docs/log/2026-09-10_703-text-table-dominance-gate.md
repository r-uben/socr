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

## Residuals

- **The gate disarms term (b) on some real numeric table pages.** Measured over
  the BoE and ECB census inputs (27 + 18 pages, gate verdict per page):
  every ECB annex page and the BoE speech table pages open the gate, but
  `boe-meetings-2003-table-p15-17` closes on all three pages despite carrying
  9/17/10 table-shaped native bands at width 2, and so do the Banxico and 1997
  scan pages. On those pages A2 now relies on term (a) alone. No content loss
  was observed there (that excerpt shipped 97% of its numbers in the census),
  but this is a genuine reduction in term (b)'s coverage, in the fail-open
  direction for A2 and the fail-closed direction for content.
- The likely mechanism is lane chaining: `_LANE_X_TOL_PT` clustering is greedy
  and adjacency-based, so a page whose numeric x-positions are dense collapses
  into one lane. Reproduced on a synthetic GH-643 shape (3 aligned data rows
  plus 5 footnote bands at a different pitch): all lanes chain into one and the
  gate closes. Fixing the clustering is out of scope here and belongs with
  #642/#643.
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
