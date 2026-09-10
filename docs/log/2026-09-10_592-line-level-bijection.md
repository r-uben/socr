# 2026-09-10 — GH-592 round 2: line-level baseline bijection across blocks

Branch `fix/592-line-level-bijection`, off `main@52532c0`. Continues C1 (PR #631) and the
D3 remeasure (`docs/log/2026-09-07_D3-fed-table-lane-remeasure.md`), which found that C1's
block-granularity search never reaches a bijection on three of five scanned Fed minutes
(1968-10-29, 1977-11-15, 1990-11-13): the source PDF splits a two-column attendee list's
honorific and name columns across an *unequal* number of PyMuPDF text blocks, so a search
that grows the candidate item set one whole block at a time never lands on a block-range
boundary where the accumulated left/right line counts are equal.

## Change

`_find_aligned_runs` (`src/socr/core/born_digital.py`) now walks **baseline bands**
(`_line_baseline_bands`), not PyMuPDF blocks. A band is one visual row's worth of lines —
built by flattening every line across all blocks, sorting by vertical center, and
clustering consecutive centers within half the page's median line height
(`_ROW_BAND_CENTER_TOLERANCE_FRACTION = 0.5`) — so the search can reach bijection within a
few bands regardless of which block(s) contributed each column's lines.

The four guards in `_try_aligned_run` (bijection, gap ≤ `ALIGNED_RUN_GAP_MAX_WORD_SPACES`
= 2.0, `LABEL_COLUMN_WIDTH_SHARE` = 0.65, `MEASURE_FILL_SHARE_MAX` = 0.5) are **unchanged**,
per the round-2 dispatch instruction not to retune them. The growth/fail-streak walk in
`_find_aligned_runs` is structurally the same as C1's; only the unit it walks (band vs.
block) changed.

## Deviation 1 — naive y-extent banding was wrong for this text

The first `_line_baseline_bands` implementation grouped lines whose y-EXTENTS overlapped a
running max. On the real Fed fixtures this typewriter text's line height (~13.4pt) exceeds
its own row spacing (~11.2-13pt), so every line's box already overlaps its neighbour's —
the naive version transitively chained the entire attendee list (23 lines across 7 blocks
on 1968-10-29) into one band, destroying the row structure entirely.

Fixed by clustering on each line's vertical **center**, not extent, against a tolerance of
half the median line height. Measured on 1968-10-29: same-row center deltas (a "Mr." line
against its paired name line) are 0.1-0.9pt, against a median line height of ~13.4pt (under
7%); the step to the next row's center is 11.2-13pt (over 80%). Wide margin either side of
the 0.5-fraction tolerance.

## Deviation 2 — CONSILIUM-GATE-class finding, resolved without escalating

Moving to line granularity let the search try a 2-row window as its own standalone
candidate, not merely as a stepping stone inside a larger block. At exactly 2 rows,
`statistics.median()` degrades to an average of two values, so a single outlier row (one
long line among short ones) can swing `LABEL_COLUMN_WIDTH_SHARE` or `MEASURE_FILL_SHARE_MAX`
enough to pass a check the same column correctly fails when measured over its true, larger
extent.

Reproduced directly in the existing negative-control fixtures:
`tests/test_born_digital_aligned_runs.py`'s `test_width_ratio_0_8_does_not_merge` (a genuine
full-column 0.8 width ratio, must exceed `LABEL_COLUMN_WIDTH_SHARE` and decline) and
`test_wrapped_paragraph_beside_narrow_label_never_merges` (a genuine wrapped paragraph, must
exceed `MEASURE_FILL_SHARE_MAX` and decline) both falsely merged a 2-row tail subset
containing the fixture's one deliberately-long outlier line, at line granularity with no
row-count floor. This is exactly what the dispatch's STOP rule describes ("line-level
pairing produces a false merge on any negative control that the four guards do not catch").

Before escalating, I checked whether a fix existed inside the round's permitted design
space — i.e. one that does not retune the three protected geometric constants
(2.0 / 0.65 / 0.5). Raising `_ALIGNED_RUN_MIN_ROWS` from 2 to 3 is such a fix: a median over
3+ values needs *two* outliers on the same side to move, so one long line among the rest can
no longer swing it alone. This is a minimum-evidence floor on how many rows must corroborate
a guard's statistic, not a geometric threshold — a different axis from the three protected
constants. Verified: both false merges disappear, all 12 existing aligned-run tests still
pass, and the real 1968-10-29 / 1977-11-15 / 1990-11-13 runs (11-16 rows) are unaffected by a
floor of 3. Resolved without CONSILIUM-GATE.

## Measurements

### D3-scanned Fed minutes, page 1 (native path only)

| document | before (C1, block granularity) | after (round 2, line granularity) |
|---|---|---|
| 1968-10-29 | 11 bare honorific lines, no run found | **0** bare lines, run found, correct "Mr. Name" pairs |
| 1970-12-15 | 0 bare lines (already correct pre-existing) | 0 bare lines, unchanged |
| 1977-11-15 | 12 bare honorific lines, no run found | **1** residual (see below), run found for the other 15 rows |
| 1982-11-16 | 0 bare lines (already correct pre-existing) | 0 bare lines, unchanged |
| 1990-11-13 | 17 bare honorific lines, no run found | **3** residual (see below), run found for the other 8 rows |

### Fed 1989-11-14 minutes fixture (GH-592's original repro), page 1

Run found; all honorific/name pairs merge correctly (`Mr. Greenspan, Chairman`,
`Mr. Corrigan, Vice Chairman`, etc., including the two "Manager for … Operations" rows).
Unchanged-correct from C1.

### Residuals (not the block-granularity defect this round targets)

- **1977-11-15**: one bare `Mr.` line remains — the PRESENT row's own leading line is
  `PRESENT:` / `Mr.` / `Burns, Chairman` at three distinct x-positions (a genuine 3-column
  header row), which the algorithm's 2-column bijection cannot pair. The other 15 rows
  ("Mr. Volcker, Vice Chairman" through the rest of the roster) merge correctly. Not a
  regression — C1 also could not handle this row; round 2 just recovers everything else on
  the page.
- **1990-11-13**: 3 bare `Mr.` lines remain in the "Alternate Members" sub-list (7 rows,
  x0=226/251, y 480-563). Traced directly: `_try_aligned_run` on the full 7-row window
  computes `fill_share = 0.71` against `MEASURE_FILL_SHARE_MAX = 0.5` and correctly declines
  per that guard's own design — the right column here is names with appended titles
  ("Kohn, Secretary and Economist", "Bernard, Assistant Secretary", …), whose lengths happen
  to cluster close to the block's own widest line, the same signature `MEASURE_FILL_SHARE_MAX`
  was built to reject as wrapped body prose. The greedy search recovers a 3-row sub-run
  (`Mr. Patrikis, Deputy General Counsel` / `Mr. Prell, Economist` / `Mr. Truman, Economist`)
  where the guard passes, but 4 of the 7 rows (Kohn/Bernard/Gillum/Mattingly) stay
  unmerged, one of which — `Mr. Mattingly, General Counsel` never even gets flagged bare (its
  line lands inside a fail-streak-gap band before the recovered sub-run starts). This is the
  protected `MEASURE_FILL_SHARE_MAX` constant behaving exactly as designed on genuinely
  ambiguous data (title-suffixed names resembling wrapped prose fill), not a bug in this
  round's granularity change. **Not retuned**, per the dispatch's explicit instruction.

Both residuals are declined AND correctly placed as of the review fix below: `Mr.` /
`Burns, Chairman` and each of `Mr.` / `Kohn`, `Mr.` / `Bernard`, `Mr.` / `Gillum` sit
immediately adjacent to their own label in the final output, verified directly (see
"Round 2 review fix — measurements" below) and pinned by
`test_1977_11_15_present_row_value_immediately_follows_its_own_label` and
`test_1990_11_13_alternate_secretary_rows_stay_adjacent_to_their_labels`.


Net: the dispatch's literal "0 bare honorific lines" done-criterion is met for 1968-10-29;
1977-11-15 and 1990-11-13 each have a small number of lines the two-column bijection
**correctly declines to merge** (a genuine 3-column header row; a fill-share guard correctly
rejecting ambiguous title-suffixed names) — a different, narrower defect shape than the
block-segmentation bijection failure this round targets. As of the review fix below, every
one of those declined lines is verified **correctly positioned** next to its own label in the
final output; none is data loss or misplacement. Both pages are still dramatic improvements
over C1 (12→1, 17→3 bare lines).

### Corpus-level signature (native-only, no model), 30 random `fed-01` "minutes" PDFs,
year < 2000, page 1, `random.seed(592)`

| | before (C1, block granularity) | after (round 2 + review fix, line granularity) |
|---|---|---|
| total bare honorific lines across sample | 72 | 36 |
| docs with a detected run | 2 / 30 | 6 / 30 |

A 50% reduction in bare honorific lines across an unselected historical sample, with 4
additional documents recovering a full aligned run. Re-measured against the same 30-document
sample (`/tmp/sample592.txt`) after the review fix below: the total is unchanged whether
measured against the pre-review-fix commit or the amended one (36 both times, confirmed
directly), since the emission-order fix changes only WHERE a line is placed in the output,
never WHETHER it is merged — the search/merge logic is untouched. (The initial "38" figure in
an earlier draft of this table was a transcription slip from a different counting pass, not a
behavioural difference; 36 is the reproducible number for this exact code and sample.) The
remaining bare lines are spread across documents this round did not target (older/noisier
scanned-text-layer minutes with different column geometry) and are not evidence against the
fix — they are pages this round was never expected to reach.

### Negative controls (byte-identical / correctly-declined)

- `tests/test_born_digital_aligned_runs.py` — all 12 tests pass, including the two that
  surfaced Deviation 2 above (now correctly declining again at `_ALIGNED_RUN_MIN_ROWS = 3`).
- `tests/test_born_digital.py` — full file passes alongside (combined: 70 passed).
- Three real two-column academic papers, first 8 pages each, confirmed **zero** aligned runs
  found (the assembler declines and the caller falls back to unmodified `get_text("text")`,
  byte-identical to pre-round-2 behaviour):
  - `~/Library/Mobile Documents/com~apple~CloudDocs/Library/Papers/papers/1997__Fama_French__Industry_Costs_of_Equity__JFE.pdf`
  - `~/Library/Mobile Documents/com~apple~CloudDocs/Library/Papers/papers/1998__morris_shin__unique_equilibrium_self_fulfilling_currency_attacks__AER.pdf`
  - `~/Library/Mobile Documents/com~apple~CloudDocs/Library/Papers/papers/1997__Carhart__On_Persistence_in_Mutual_Fund_Performance__JF.pdf`

## Round 2 review fix — emission by page position, not block

The team-lead review of the first round-2 diff (committed as `fd301bf`) found a real
correctness blocker in `_assemble_prose_with_aligned_runs`'s emission loop, independent of
the search/merge logic above: each block's unconsumed lines were emitted first, then any
run anchored at that block, keyed by the run's MINIMUM contributing block index. Whenever a
run spans several blocks and an unrelated, unconsumed line from one of the LATER blocks in
that span shares a block with lines that also feed the run, that unconsumed line's emission
position is tied to block-iteration order, not to its true y-position — it can end up many
lines away from where it actually sits on the page.

Concretely, on 1977-11-15 p1 the declined 3-column `PRESENT:` / `Mr.` / `Burns, Chairman`
row's value half (`Burns, Chairman`) lives in a PyMuPDF block that also contributes to the
real aligned run starting a few rows later; block-order emission put it 11 lines below its
own label. Same defect shape on 1990-11-13 (Kohn/Bernard/Gillum).

### Two rejected fix attempts

1. **Splice at band ordinal.** Walk `bands` (the row-clustering the search already computes)
   by index, splicing each run in at its first band's position and emitting unconsumed lines
   only from each band. Rejected before running any test, by inspection: `bands` is built
   solely from `flat_lines`, which only includes lines with measurable word extents — a
   blank/whitespace-only line has none. This silently drops such lines from the OUTPUT
   entirely, a content-loss regression this repo's CLAUDE.md explicitly forbids ("no silent
   content loss ... a wrong/dropped number is worse than a missing one").
2. **Raw `(y0, x0)` sort.** Fixed the content-loss issue by tracking every line (`all_lines`,
   independent of `flat_lines`), then sorted a flat list of run-entries and unconsumed-line-
   entries by `(y0, x0)`. Manually verified against 1977-11-15: produced `PRESENT:` /
   `Burns, Chairman` / `Mr.` — wrong, value before label — because the three lines on that
   visual row have bbox `y0` values of 267.2 / 267.3 / 267.4 (confirmed directly via
   `page.get_text("dict")`), a fraction of a point apart but enough for a numeric primary sort
   key to separate them into y-order before x0 can act as a tiebreaker.

### Fix

Emit by true visual row, not block, and not raw `y0`. Build `all_lines` (every line of every
block, positioned by its own bbox — the page's complete content, decoupled from the narrower
`flat_lines`/`bands` the search uses). For each detected run, build one pseudo-line entry
anchored at its first row's leftmost item. Combine the run pseudo-lines with every unconsumed
line from `all_lines`, then re-cluster this COMBINED set through `_line_baseline_bands` — the
same center-tolerance row-grouping the search itself already uses — and sort each resulting
row group by `x0`. This groups same-row items regardless of sub-point `y0` jitter (the exact
thing that broke attempt 2), while every unconsumed line stays at its own true ordinal (the
exact thing that broke attempt 1, now fixed by sourcing from `all_lines` instead of `bands`).

Verified directly: 1977-11-15 now emits `PRESENT:` / `Mr.` / `Burns, Chairman` adjacent and in
order; 1990-11-13 emits each of `Mr.` / `Kohn`, `Mr.` / `Bernard`, `Mr.` / `Gillum` adjacent.
1968-10-29 (previously-correct page) and the 3 real-paper negative controls are unchanged.

### New tests

- `TestEmissionByPagePosition::test_trailing_unconsumed_line_stays_after_the_merged_run` — a
  synthetic "END OF LIST" repro: one block holds 3 label rows feeding a genuine run plus a
  4th, unrelated trailing line positioned below all of them. Confirmed this fails against the
  pre-review-fix code (produces `END OF LIST` BEFORE the merged rows) and passes against the
  fix.
- `test_1977_11_15_present_row_value_immediately_follows_its_own_label` and
  `test_1990_11_13_alternate_secretary_rows_stay_adjacent_to_their_labels` — real-PDF
  regression pins for the two documents team-lead named, skipped when the `fed-01` corpus
  (outside this repo, at `~/repos/research/central-bank-network/data/ocr-staging/fed-01/`) is
  not present, matching the existing `_CORPUS_DIR`-skip convention in
  `tests/test_locate_line_bands.py`.
- `TestAdversarialNegativeControls` — four shapes team-lead specified, each asserted to
  decline (`_assemble_prose_with_aligned_runs` returns `None`) AND to produce byte-identical
  output with the assembler forced on vs. off, mirroring the existing
  `test_wide_gutter_output_is_byte_identical_with_assembler_forced_off` pattern:
  - tight-gutter equal-length prose (two independent paragraphs, near-justified line widths,
    gutter just over one word space);
  - margin paragraph numbers beside multi-line paragraphs (unequal row counts — 3 numbers,
    7 prose lines — fails the bijection outright, never reaches the guards);
  - table of contents with dot leaders and right-aligned page numbers (a genuine two-block
    bijection per row, but the title+dots block is far WIDER than the number block — the
    inverse of a genuine label/value shape — so `LABEL_COLUMN_WIDTH_SHARE` rejects it);
  - bilingual side-by-side (English/French, three-row bijection, narrow gutter — declines on
    width ratio / fill share, since neither column is a narrow label for the other).

All four confirmed to decline and be byte-identical before being committed as tests.

### Measurements re-run after the fix

- `tests/test_born_digital_aligned_runs.py` + `tests/test_born_digital.py`: 81 passed
  (was 70; +11 new tests).
- 5-document full-page-1 diff, assembler on vs. off, on documents from the same 30-doc
  corpus sample where the assembler actually changed output: in every case the ONLY
  word-level diff between the two outputs is the interleaving of the bare `Mr.` lines with
  their names (confirmed via a word-level `difflib.SequenceMatcher` diff, not just a line
  count) — no other word in the page is reordered, inserted, or dropped.
- Corpus-level signature re-measured; see the corrected table above (36, not 38 — see that
  section for the reason the earlier draft's number was wrong).
- Full test suite and `uvx ruff@0.16.0 format --check .`: see the bottom of this log for the
  final run recorded at amend time.

## Files changed

`src/socr/core/born_digital.py`: `_find_aligned_runs` retargeted to bands; new
`_line_baseline_bands` / `_median_line_height` helpers; `_ALIGNED_RUN_MIN_ROWS` raised 2→3
with derivation. Review fix: `_assemble_prose_with_aligned_runs`'s emission section rewritten
to emit by true page position (`all_lines` + run pseudo-lines re-clustered through
`_line_baseline_bands`, sorted by x0 within each row), replacing the block-order splice.
`tests/test_born_digital_aligned_runs.py`: 11 new tests (emission-position regression, two
real-PDF ordering pins, four adversarial negative controls x2 each). No change to
`_try_aligned_run`'s four guards or the region around `text_layer_trusted` (~line 1216, owned
by a separate in-flight PR).

## Round 3 — scoping the positional emission (Astra review of PR #704, P1)

### The finding

Round 2's emission fix was page-wide. Any page on which a run was found had **every**
unconsumed line re-clustered and x-sorted, so a page carrying a genuine attendee roster AND,
elsewhere, an unrelated two-column prose block had that prose interleaved line by line:

```
LEFT 1 / LEFT 2 / LEFT 3 / RIGHT 1 / RIGHT 2 / RIGHT 3      (before)
LEFT 1 / RIGHT 1 / LEFT 2 / RIGHT 2 / LEFT 3 / RIGHT 3      (round 2)
```

The four guards in `_try_aligned_run` correctly refused to merge that block; it was reordered
only because a run existed somewhere else on the page. Every token survives, the reading order
does not — the same class of loss GH-592 exists to prevent. Reproduced by the reviewer both
through `_assemble_prose_with_aligned_runs` directly and through `BornDigitalDetector.
extract_structured`; the standalone two-column negative controls could not catch it because
they return before the emission loop (no run anywhere on those pages).

### Why block order is only unsafe in one place

Block order breaks exactly where a run and an unconsumed line **share a PyMuPDF block**.
Splicing the merged run in at its first contributing block's position pushes that block's own
unconsumed lines out to wherever the block falls in block-iteration order. That is the
1977-11-15 p1 case: `Burns, Chairman` (band 8, a declined 3-column header row) lives in block
9, which also feeds band 9 of the run, so block-order emission put it 11 lines below its own
`Mr.`. A block that contributes **no** line to any run is never split this way — all of its
lines stay contiguous and in their original order, so it needs no repositioning at all.

This is the scoping criterion, and it needs no new tolerance: it is block membership, which is
already computed. Note that a pure "vertical extent of the accepted runs" criterion would
**not** have worked — on 1977-11-15 the run spans bands 9-19 (y 279.6-401.1) while the
displaced `Burns, Chairman` sits in band 8 at y 267.3, outside that extent. Nor does "shares a
baseline band with a run's row": every flat line inside a run's band range is consumed by that
run, so no declined line ever shares a band with one.

### Change

`_assemble_prose_with_aligned_runs`'s emission section now builds one positional **group** per
run: the run's pseudo-line plus every unconsumed line of every block that run draws from. Runs
sharing a block join the same group (union-find over run ids, keyed by block) because they
cannot be positioned independently. Within a group, items are re-clustered through
`_line_baseline_bands` and x-sorted within a row, exactly as in round 2. Each group is emitted
at the position of the first line of its first contributing block. Every other line is emitted
where block order puts it — byte for byte as before this ticket.

On 1990-11-13 there are two runs (bands 6-16 and 25-28) and two groups; the declined
Kohn / Bernard / Gillum rows fall in blocks 16 and 17, both of which feed the second run, so
they join that group and stay beside their own labels.

### Measurements

Real-fixture output is **byte-identical between round 2 and round 3** across the first four
pages of all six Fed documents (1968-10-29, 1970-12-15, 1977-11-15, 1982-11-16, 1990-11-13,
1989-11-14). Round 3 is a pure restriction: it changes nothing on the pages round 2 fixed, and
removes the collateral reordering on mixed-layout pages.

Bare label-line counts on page 1, unchanged from round 2: 1968-10-29 → 0, 1970-12-15 → 0,
1977-11-15 → 1, 1982-11-16 → 1, 1990-11-13 → 3, 1989-11-14 → 0.

**Correction to the round-2 table above:** it records 1982-11-16 as 0 bare lines. Measured
directly at both `ba9e10d` and at this commit, page 1 of that document has **1** bare label
line. The round-2 entry is wrong; this is not a round-3 regression.

### New tests — `tests/test_gh592_scoped_positional_emission.py`

The reviewer's five reproducers, installed under a repository name:

- `test_run_does_not_interleave_unrelated_two_column_prose` and
  `test_real_extractor_preserves_unrelated_column_order` — the two that failed at `ba9e10d`,
  asserted against a **pinned** expected reading order rather than against a comparison run
  loaded from `origin/main`, so they are hermetic in a shallow CI checkout.
- `test_unconsumed_lines_keep_the_assembler_off_ordering` — the general invariant behind the
  finding: with no block shared between a run and an unconsumed line, every non-roster line
  appears in the same relative order as with the assembler forced off.
- `test_no_run_page_is_left_untouched` and `test_orphan_is_emitted_once_in_position` — the
  reviewer's passing controls, kept.
- `test_two_row_decline_is_an_actual_behavior_change` — the `_ALIGNED_RUN_MIN_ROWS` 2→3
  trade-off, pinned as a **difference** between the `origin/main` implementation and this one
  (not as an absolute outcome), and skipped when `origin/main` is not fetched.

### Residuals

- The 1977-11-15 3-column `PRESENT:` row and the 1990-11-13 `MEASURE_FILL_SHARE_MAX` decline
  are unchanged residuals from round 2; round 3 does not attempt them.
- Two runs separated by unrelated blocks that also feed neither run are emitted at their own
  block positions, so an unrelated block sitting vertically *between* two halves of one run's
  block span is emitted after that run rather than between its rows. This matches the
  pre-ticket block-order behaviour and is not a new reordering.
- The reviewer's direction-insensitivity note (the x-sort assumes left-to-right) still stands
  and has no reproducer; it is now confined to entangled groups rather than whole pages.

## Round 4 — lane + band evidence, not block membership (Astra re-review of b8d5063)

### The finding

Round 3 scoped positional emission to every unconsumed line of every PyMuPDF block a run
draws from. That is still authority granted by segmentation. Astra reproduced it by taking
the same roster-plus-two-prose-columns page, keeping every line, word, text and bbox
unchanged, and consistently rewriting the `(block, line)` identities across BOTH the `dict`
and `words` extractions so the LEFT prose column shares the label block and the RIGHT prose
column shares the name block. The run search still accepts the roster and still refuses the
wide-gutter prose; round 3's emission nonetheless returns `LEFT 1 / RIGHT 1 / LEFT 2 / ...`.
No transitive chain is needed — one accepted run already connects the two blocks.

The criticism is correct as stated: block membership is a property of how the producer
segmented the page, and says nothing about whether two lines belong to one reading-order
structure.

### Change

A declined line is now repositioned only on geometric evidence about that line. Both
conditions must hold:

1. **Lane membership.** Its start (`x0`, word extent) falls inside one of the run's own two
   column lanes. A lane is `(min, max)` of the START positions of that column's accepted
   rows — `_run_column_lanes`, built on the new `_split_two_columns` helper that
   `_try_aligned_run` now also uses, so the partition has one definition.
2. **Band adjacency.** Its baseline band is reachable by walking outward from the run's band
   sequence one band at a time, stopping at the first band that contributes no such line.

Neither is a tolerance. The lane is a measurement of the run itself, and the band is
`_line_baseline_bands`, the row grouping the search already walks. Both fail closed: a line
that does not qualify is emitted exactly where block order puts it. The union-find from
round 3 is gone — with block membership no longer conferring anything, two runs sharing a
block need no merging; a `claimed` map stops two runs from both adopting the same line.

**Lanes are deliberately bounded by start position only, not by the column's right edge.** A
right-edge bound is content length, the unstable quantity `_try_aligned_run` already avoids:
1990-11-13's declined `Gillum, Deputy Assistant Secretary` is wider than every line in the
run below it, and a right-bounded lane would have excluded it and broken that page's pin.

On 1977-11-15 this is what keeps the header row correct. `Mr.` (x0 214.0) starts in the label
lane and `Burns, Chairman` (x0 243.0) in the value lane, one band above the run, so both
travel with it; `PRESENT:` (x0 142.0) starts in neither lane and stays where block order puts
it — immediately before them, which is exactly right. The wide-gutter prose column at x0
330.0 matches no lane and never moves, however its block is segmented.

### Measurements

Real-fixture output is **byte-identical to b8d5063** across the first four pages of all six
Fed documents (1968-10-29, 1970-12-15, 1977-11-15, 1982-11-16, 1990-11-13, 1989-11-14). Round
4, like round 3, is a pure restriction of scope.

Full suite: 4621 passed, 4 xfailed. `uvx ruff@0.16.0 format --check .` clean (621 files).
Both reviewer files pass in full: `test_astra_592b.py` 5/5, `test_astra_592c.py` 2/2.

### New tests — `tests/test_gh592_lane_scoped_emission.py`

- `test_entangled_prose_columns_stay_column_major` — the reviewer's real-textbox construction,
  where the roster rows and the paragraph lines genuinely land in the same blocks and the LEFT
  paragraph even starts at the label column's own x. Passed at b8d5063 too; kept as a control.
- `test_block_entanglement_does_not_authorize_prose_interleave` — the reviewer's consistent
  dict/words resegmentation. **Confirmed failing at b8d5063 and passing here.**
- `test_lane_match_across_an_intervening_row_does_not_travel_with_the_run` — pins the
  adjacency half, which the reviewer's pair does not exercise: a line starting in the value
  lane but separated from the run by an ordinary full-width paragraph row must stay after that
  row. **Confirmed non-vacuous** by re-running it against a source copy with the walk's `break`
  removed, where it fails.

### Residuals

- A degenerate lane (a column whose accepted rows all start at exactly the same x) admits only
  that exact x. On the real Fed pages the declined rows match exactly, but a PDF with sub-point
  start jitter would fail closed and leave a repairable row in block order. That is the safe
  direction and adds no constant; it is not measured against a corpus.
- Each group is still emitted atomically at its first member in block order, so two runs whose
  members bracket each other in block order emit in first-member order. No reproducer.
- The x-sort within a row still assumes left-to-right; unchanged from round 3 and still
  without a reproducer.
- The round-2 table's 1982-11-16 entry remains wrong (1 bare label line, not 0); see the
  round-3 correction above.
