# #1155 — routing Fed SEP figure pages to the chart lane (Phase 2A)

Date: 2026-10-16. Branch `feat/issue-1155`, base `b354a6bd`.
Baseline: cluster job 23352871 (`fed-repilot-1016`), same sample, settings and hardware as
job 23208354 (`fed-repilot-1014`, 05bc8678), so a difference between them is a code
difference. Full evidence: `scratch/routing-baseline/leads/1155/evidence.md`.

## The defect

Fed SEP figure pages shipped a bare 103-character failure marker — no prose, no numbers,
no chart reading. 15 such pages in three documents on `b354a6bd`. Number recall against
the PDF's own text layer: 33.8% / 26.8% / 54.6% (2008-10-29 / 2010-04-28 / 2015-09-17),
against 77.1% / 98.3% / 92.7% for the pre-rewrite `fed-01` run. #1144/#1145/#1178 moved
those by +0.2 / +2.7 / +0.6 points: the prose-recovery work did not restore the numbers.

Three gates in series produced it:

1. `_phase_agentic`'s chart-vs-table arbitration classed every figure page as
   `chart_mixed`, because it asked `_page_has_tables` — the lane-cooccupancy heuristic,
   which a chart's own axis tick labels satisfy unaided.
2. Mixed pages keep the table route into the structure-class ladder.
3. `manifest.table_floor_text_for_source` withholds the WHOLE page when
   `detected_table_count <= 0`: with no detected box there is nothing for #1144's prose
   recovery to splice around, so prose, numbers and chart reading go together.

## What changed

The arbitration now asks two questions instead of one.

- The **narrow** table question (`_page_has_scorable_table_evidence`:
  `detected_table_count > 0` OR `native_table_region_count > 0`) replaces the loose one.
  A page with real table evidence stays mixed, and the source-order invariant that class
  protects is untouched — it concerns a chart sinking below a table that followed it, and
  a page with no detected table region has no such table.
- A page with the loose signal only must additionally produce **positive evidence that the
  chart is the page**: `_chart_reader_resolves_page` requires `read_chart_page` to resolve
  at least one panel of the page's own `chart_region_bboxes`. Model-free, ~0.01 s/page,
  and run only for the pages where it can change the answer. It fails CLOSED: any error
  keeps today's route and emits `chart_reader_arbitration_probe_failed`.

## Why the second question is load-bearing (the harm census)

Over `fed-repilot-1016` plus four shards of the routing-baseline corpus, using existing
sidecars and the PDFs only — no OCR run:

- The narrow signals **alone** flip **138 pages** (10 Fed, 128 corpus).
- **12 of those 138 ship a markdown table today**, and most are REAL data tables the
  region detector missed: a confusion matrix (Dougherty p161), a diagnosis table with real
  counts (p187), a word-embedding matrix (Chen-Kelly-Xiu p5), a regression table with
  standard errors (Acosta p59), Beige Book summary statistics (Armesto et al p8), a
  p-value table (Li et al p18), model metrics (Chowdhury p15). Flipping those into a lane
  that ships page text plus an image would drop a real table. A blocker.
- Gated on the reader resolving a panel, **exactly 10 of the 138 flip**, and they are
  exactly the Fed SEP figure pages this ticket targets: 2015-09-17 p17 (5/5 panels), p18
  (5/5), p19 (1/5), p20 (4/4), p22 (1/5); 2008-10-29 p18 (4/4); 2010-04-28 p15/p16/p17
  (3/4), p18 (2/3). **11 of the 12 table-bearing pages resolve no panel** and keep their
  route.
- The twelfth is 2015-09-17 p19, whose table is **fabricated**: the model published the
  figure's y-axis TICK labels (18, 16, 14, 12, 10, 8, 6, 4, 2) as participant counts for a
  17-member committee. The page judge had already rejected it and #1145 marks the page
  unaccepted. Replacing it with the reader's own reading is a gain.

Absence of a detection is not evidence of absence. The flip requires a reading, not a
missing signal.

## Tests and mutants

`tests/test_chart_lane.py`, 43 passing. Three external mutants, each with the source
copied outside the repo, `socr.__file__` asserted inside the copy, and an UNCAPPED
`src.count(anchor) == 1` asserted before editing:

| mutation | test killed |
|---|---|
| revert the narrow question to `_page_has_tables` | `test_loose_signal_with_a_readable_chart_takes_the_chart_lane` |
| delete the arbitration (every chart page a winner) | `test_narrow_table_signal_still_keeps_its_normal_route` |
| delete the reader-resolved condition | `test_loose_signal_with_an_unreadable_chart_keeps_its_normal_route` |

Each mutant killed exactly one test and no other. The two routing tests assert their own
premise (`_chart_reader_resolves_page` returns the expected answer for their fixture)
rather than assuming it, so neither passes if the probe is deleted. The readable-chart
fixture is `test_gh635_chart_reader.build_chart`, a real two-series histogram, so the
reader runs for real.

The pre-existing GH-150 TICKET-B1 test keeps its fixture byte for byte; its original
assertions were correct for that fixture all along, and it is now named for what it pins
(an unreadable chart keeps its route). The genuinely-mixed case it used to cover moved to
a new test whose page carries `detected_table_count=1` — the state production reports for
2010-04-28 p12 (3 regions), 2015-09-17 p14 (4) and p15 (1).

## Not measured

- One corpus PDF aborts pymupdf at C level inside `find_tables` and was never scanned:
  `1995__strongin__identification_of_monetary_policy_disturbances_explaining_liquidity_puzzle__JME`.
  It is an academic paper from the routing-baseline corpus, **not a Fed document**, so it
  does not affect the Fed figures above. Not chased.
- End-to-end page outcomes under the change: no OCR run is permitted on this machine, so
  the 138→10 figure is from the arbitration's own inputs, not from a re-run.

## Phase 2B — the witness gate, and the per-panel lines

### The gate is built and NOT wired to any output

Stated plainly because it matters for reading this PR: **no chart count ships from this
change.** Today a flipped page ships prose plus crop images and no numbers at all — the
reader's counts reach a document only through `_reconcile_chart_table_grids`, which
returns 0 immediately when the candidate carries no markdown grid, and the chart lane's
body has none. So this PR lands the routing and the gate; a follow-up turns the counts on
behind it. (Verified by tracing the code path; no OCR run is permitted on this machine.)

### `figures/chart_witness.py`

The document's own projection table as an independent witness. `parse_table1_witness`
returns published ranges keyed by `(variable, series_row, year)`, for rows that pass their
own self-check — median ⊆ central tendency ⊆ range, and a value count equal to the
header's own column count. `gate_series` then requires the LOWEST filled bin to contain
the published minimum and the HIGHEST to contain the maximum.

Scope is only what is proven: the **upright single-block** layout. The 2008-10-29 release
splits the table into two stacked blocks and the 2015-09-17 release draws it rotated; both
REFUSE with a named reason (see #1184), so the gate withholds there rather than guessing.
Measured: 2010-04-28 p11 yields 24 self-checked rows; the other two refuse correctly.

**The gate's limit is stated in the module docstring AND in the shipped mark**, per the
#1181 precedent that a mark never claims a value was verified: it checks only the two
extreme bars, so any interior error, and any shift that leaves both extremes inside their
own bins, survives it. It narrows what can ship; it does not prove a reading correct.

### Never silent, now

Every page flipped by #1155 carries one canonical #1145 line per chart region — resolved
or not — as its LEADING block, because `core/marks` defines a mark as the first rendered
block and the acceptance meter reads only that. `PageState.chart_flipped_by_reader` scopes
this to pages whose routing actually changed; an ordinary chart-only page gains nothing.

### Three bugs found by building it, each of which had produced a wrong result first

1. A bin label's separator is a HYPHEN. `-?\d+\.\d+` parses `1.8-1.9` as `(1.8, -1.9)`;
   that made an early run report 0 gate passes where the true answer was 10.
2. Every month-prefixed series was classed as the PRIOR meeting's, so current-meeting
   series were checked against the previous meeting's range. A series is prior only when
   its own month is one the table's sub-rows name.
3. The first rotation test ("a word box taller than it is wide") fails on exactly this
   table: a rotated ONE-CHARACTER token is wider than tall, and a numeric table is mostly
   such tokens. Replaced by `born_digital.upright_rotation_for`.

### Mutants (all external: src+tests+pyproject copied out, `socr.__file__` asserted inside
the copy, uncapped `src.count(anchor) == 1` asserted before editing)

| mutation | test killed |
|---|---|
| drop the witness row self-check | `test_a_row_whose_nesting_breaks_is_not_a_witness` |
| delete the gate's LOW-end check | `test_a_reading_truncated_at_the_BOTTOM_is_refused` |
| delete the gate's HIGH-end check | `test_a_reading_truncated_at_the_TOP_is_refused` |
| an unrecognised layout no longer refuses | `test_the_rotated_layout_refuses_by_name` |
| never emit the panel lines | `test_a_flipped_page_accounts_for_every_chart_region_in_its_marks` |

**The high-end mutant initially survived**, killing nothing: every test written to that
point exercised only the lower bound, because the shift fixture moved both bins upward and
the low check alone decided it. The two truncated-reading tests were added for that, and
each endpoint's mutant now kills its own test. A surviving mutant found an untested branch
that review would have had to find instead.

## REDESIGN: the decision moved from the ROUTER to the FLOOR

The router change above was reverted. Its own full-suite run is why: 6 tests failed, and
the first one opened showed a page that used to ship `| September | 0 | 4 | 2 | 0 | 1 |`
shipping only an image. **Flipping a page at the router removes the MODEL from its path,
and with it the grid the chart reader fills** — so counts that ship today were being lost.
An earlier answer in this ticket said a flipped page could not lose reader numbers; that
was wrong, and the suite is what caught it. `_reconcile_chart_table_grids` does return 0
without a grid, as traced — but on these pages the model supplies the grid, and the router
change removed the model.

The decision therefore moved to where the page is already being given up on:

- **The router is byte-identical to origin.** Every page that ships counts today keeps its
  route and its bytes; the 6 tests pass with expectations unchanged (133 across the three
  affected files).
- **At the floor** — `manifest.table_floor_text_for_source`, the no-detected-box branch
  that returns the bare whole-page marker — a page the chart reader can speak for now
  ships its prose, its region pictures and one canonical #1145 line per region instead.

### Plumbing decisions (standing delegation, recorded per instruction)

1. **`DocumentState.figures_dir`**, set once in `_phase_agentic` from the value
   `_doc_and_figures_dir` already computed. The floor runs in `core.manifest` with only
   the DocumentState in hand; deriving the path a second way is how two code paths come to
   disagree about where a file went.
2. **`render_chart_region_crops` moved** from `orchestrator._render_chart_region_crops` to
   `figures/chart_regions.py` as a PURE move — the body never referenced `self`. Core may
   not import the pipeline layer, and this keeps one renderer, one naming scheme
   (`chart_region_p{page}_{index}.png`), one set of bytes. The orchestrator method remains
   as a thin delegate so existing callers and tests are untouched.
3. **Crops render AT the floor**, not earlier: a page that succeeds never reaches that
   code, so no figure file is written for it and no shipping page's output changes.
4. **The replacement is offered at ONE call site**, `structure_class_floor_text`. The other
   floor call leaves `prose_recovery` False, and `if not prose_recovery: return whole_page`
   fires before the no-box branch — passing a replacement there would be an argument that
   silently does nothing. Measured: all 9 bare Fed SEP pages reach the floor through the
   site that is wired.
5. **The rotated-lane refusal keeps its bare marker.** Its recovered "prose" is confetti
   (GH-147), so a chart body built from it would ship nonsense. Pinned by a test.

### Which target pages this reaches

Of the 10 pages the chart reader resolves, **9 ship the bare marker today and all 9 reach
the wired site** (`structure_class_ladder_exhausted`, disposition `fail_closed_marker`):
2008-10-29 p18; 2010-04-28 p15–p18; 2015-09-17 p17, p18, p20, p22.

**2015-09-17 p19 never floors.** It is a WARNING (`model_output_flagged`, disposition
`model_output`) shipping 2,641 characters — the fabricated table whose y-axis tick labels
(18, 16, 14, 12, …) were published as participant counts for a 17-member committee. The
page judge already rejected it and #1145 marks it unaccepted. This change does not touch
it. **Open item**, not extended into: replacing a model reading the judge refused is a
different decision from replacing a page that shipped nothing.

### Mutants (external copies, canary asserted, uncapped anchor count asserted)

| mutation | test killed |
|---|---|
| floor returns `whole_page` instead of the replacement | `test_the_floor_returns_the_chart_body_instead_of_the_bare_marker` |
| drop the "nothing resolved → leave it alone" guard | `test_a_page_with_regions_but_nothing_resolved_is_left_alone` |
| drop the page's prose from the replacement | 2 tests |
| drop the witness row self-check | `test_a_row_whose_nesting_breaks_is_not_a_witness` |
| delete the gate's LOW-end check | `test_a_reading_truncated_at_the_BOTTOM_is_refused` |
| delete the gate's HIGH-end check | `test_a_reading_truncated_at_the_TOP_is_refused` |
| an unrecognised layout no longer refuses | `test_the_rotated_layout_refuses_by_name` |

**Two mutants survived their first run** and each exposed a genuinely untested branch: the
gate's high-end check (every test to that point exercised only the low end) and the floor
wiring itself (the body was tested in isolation, never through the floor). Both were
closed with tests that fail under the mutation. A third, the no-panels guard, was being
satisfied by an earlier test that tripped a different guard first.

## The harm sweep at the floor, and the per-document cap

Criterion: a page is affected only if it ships the bare whole-page marker today AND the
chart reader resolves a panel on it. The router is untouched, so no page that ships
anything today can change. Existing sidecars and the PDFs only; no OCR run.

- **Fed (`fed-repilot-1016`, 21 documents): 12 pages affected**, listed in
  `leads/1155/evidence.md` section P. None of them carries a table in what it ships today
  — each ships 103 characters of marker and nothing else.
- **Routing-baseline corpus (`runs_23319732`, 395 documents): 0 pages affected.** The 12
  pages the earlier ROUTER design would have harmed are untouched here, because a page
  that ships a real table is by definition not shipping a bare marker and so never
  reaches this branch.

### The cap, and why it is this number

An uncapped first attempt had one document consuming 8.5 minutes of CPU, so the sweep was
re-run with a per-document wall-clock cap and every document's elapsed time recorded, so
the constant could be checked against the sweep's own distribution rather than asserted.

`CAP = 120` seconds. Measured over the 394 documents that completed: **median 1 s, p90
1 s, p99 6 s, slowest completed 30 s**. The cap is therefore **120x the median and 4x the
slowest document that finished**, and **no document sits within 2x of it** (none took
60 s or more). A sweep whose normal document takes one second is not being truncated by a
two-minute limit; the gap between 30 s and 120 s is the evidence for that.

**1 document UNMEASURED** — killed at the cap, and reported as unmeasured rather than
clean: `2025__white__new_keynesian_puzzle_reinterpreting_inflation_dynamics__WP`. A second
corpus document is unmeasured for a different reason, recorded earlier: it aborts pymupdf
at C level inside `find_tables`
(`1995__strongin__identification_of_monetary_policy_disturbances_explaining_liquidity_puzzle__JME`).
Neither is a Fed document.

## The one expectation edited in this change

`tests/test_gh304_derotate_before_ocr.py` keeps a registry of every `prerotate` call site.
The pure move of `render_chart_region_crops` into the figures layer changed one entry's
location, which is exactly the drift that registry exists to catch, so the RECORD was
updated and the code was not. Evidence the move itself changed nothing: the 86 tests in
`test_gh189_mixed_chart_preservation.py` and `test_chart_lane.py` pass before and after
it, and the registry's own 14 tests pass after the update. No other test expectation in
this PR was changed.

## Round 2 — the panel's findings, and what they changed

Astra: MERGE-AFTER-FIX. Argon: BLOCK. Both landed on the same P0 and were right.

### P0: the replacement shipped the page's raw text, so a chart's own numbers shipped

`floor_replacement_body` took `native_text` verbatim. On a chart page that text IS the
chart: y-axis ticks and vertically shredded bin labels. They shipped as loose prose
**beside a mark saying values were not shipped** — Argon on all 9 recovered pages
(2015-09-17 p22 dumping every endpoint from -0.37 to 4.37), Astra 149 numeric lines on
2010-04-28 p15.

Fixed by routing through #1144's own lane instead of building a second prose path:
`native_prose_floor_text` now takes `extra_withheld_boxes`, and the chart floor passes the
page's CHART regions, so a band touching one is withheld by the same geometry rule that
withholds a table. **Measured on the 12 real Fed pages with production `native_text`:
in-chart numerals shipped 489 → 0.**

### The chart-box rule is kept, and is honestly unexercised on the corpus

Measured separately: the prose lane ALONE takes in-chart numerals to 0 on all 12 pages.
The chart boxes change nothing there — the numeral-count rule already withholds every
band. An external mutant removing them killed no test, correctly.

Kept anyway, on the owner's standing preference for positive evidence: withholding by the
chart's own geometry is positive, while the numeral count is a deny-list over numeric
shapes, and this repo's record is that deny-list lanes leak in later rounds
(`denylist-safety-lanes-leak-every-round`). It is now SEEN to guard: a built page whose
in-region text is WORDS (a legend name, a panel heading) ships them as loose prose without
the rule and withholds them with it, and the mutant now dies on that test.
**It fires on none of the 12 real Fed pages today.**

### The other fixes, each with a mutant that kills its own test

| fix | mutant | test killed |
|---|---|---|
| chart regions join the withheld boxes | remove them | `..._withheld_by_geometry` |
| the lane runs AFTER the /Rotate guard, and the builder is lazy | delete the guard | `..._builder_is_not_called_...` |
| a picture already in the prose is not appended again | append always | `..._not_appended_again` |
| the body is flagged so chart preservation skips it | drop the flag | `..._marks_the_page_...` |
| crops render once across re-finalize and resume | drop the memo | `..._runs_once_...` |
| negative figures and the Unicode MINUS | disable the range match | 4 tests |

Ordering: the replacement is offered only after the /Rotate refusal, and the order gate is
applied to the recovered PROSE inside the builder — not to the assembled body, which leads
with socr's own canonical mark lines that the gate does not recognise. Gating the body
refused every page and the lane never fired.

### Two measurement errors of mine, recorded because both reached a claim

1. **"49 affected corpus pages"** was a miscount: 524 of the sweep's 536 output lines were
   pymupdf chatter on stdout ("Consider using the pymupdf_layout package", "MuPDF error:
   … structure tree"). The real count is **12 JSON records, all Fed, 0 corpus**, matching
   the earlier sweep.
2. **A reported "in-chart word leak"** was an ambiguous detector, not a defect. "Percent
   range" and "Number of participants" appear five times each on a page, and my check
   matched the string anywhere. Measured per band: **every surviving instance has 0 of its
   words inside any chart region**; the copies that ARE inside are withheld. Owner ruling
   2026-10-17: leave them — printed outside the regions, words not numbers, and widening
   "inside a chart" by guesswork is the drift being avoided. Known cosmetic repetition.

### The shipped-bytes sweep

Rebuilds each candidate page's actual replacement body rather than selecting on markers.
416 documents (395 corpus + 21 Fed), **415 completed, 1 capped**. Per document: median 5 s,
p90 10 s, p99 21 s, slowest completed 55 s; cap 120 s = **24x the median and 2.2x the
slowest completed**, and no document sits within 2x of it. UNMEASURED, not clean:
`2025__white__new_keynesian_puzzle_reinterpreting_inflation_dynamics__WP` (capped) and
`1995__strongin__…__JME` (aborts pymupdf at C level). Neither is a Fed document.
Result: **12 Fed pages affected, 0 corpus pages.**

## Round 3 — the panel's two remaining items, and one gap closed

Fresh lead, 2026-10-17. Both seats had said MERGE-AFTER-FIX with the same two items open.

### 1. An order refusal no longer loses the page's own image, and no longer says nothing

`_prose_follows_native_order` refusing the recovered prose set `recovered = ""`, and the
full-page image was EMBEDDED in that string. The page then shipped chart crops, no text,
and nothing to read the text from — beside no statement that any text had been withheld
(Astra round 2). `collapse_withheld_runs` could not speak for the case: it was handed the
emptied string, so it counted no runs and returned no image.

Fixed where the refusal is made: the refusal is remembered, the existing `png_ref` is
carried as the page image independently of whether the prose was accepted, and the page
gets a page-scoped `native_coverage=FAIL` line (`chart_witness.ORDER_REFUSED_REASON`)
saying its text was withheld whole and why. This lane may not withhold a page's whole
text silently.

**Measured, and the honest scope: the refusal fires on 0 of the 12 real Fed pages.** All
12 recover prose that IS in the page's own reading order, and all 12 carry a non-empty
`d3_floor_png_ref`, so all 12 ship the image either way
(`leads/1155/probes/order_gate_r3.py` → `round3/order_gate.jsonl`). Astra's case was
constructed and so is the fixture: a drawn chart page with prose in two columns, where
`get_text()` keeps the columns apart and the y-clustered bands weld them. The test
asserts that premise before asserting the fix, and pairs it with the one-column control.

### 2. Per-region verdict clauses, derived from the region results

All 12 finalized verdicts carried ZERO `chart_region` clauses while the Markdown showed a
mark per region (both seats). A rendered mark string is not machine-readable evidence:
nothing downstream parses a page back.

`chart_witness.region_clause_records` now builds one clause dict per region from the
region results, and the page's own lines are rendered FROM those same records
(`region_clause_mark`). **One source, so the line a reader sees and the clause a consumer
reads cannot drift.** The floor records them on `PageState.chart_floor_region_clauses`
only once a body exists, and `verdict._chart_floor_region_clauses` reads them back at the
seam, so they reach the sidecar and survive a resume (a published verdict wins).

Reachability is POSITIVE, not a flag over a class: the attribute exists only on a page
whose bare marker was actually replaced. `_merge_clause` keys on `(check, scope)` and
these are region-scoped, so the existing page-scoped `_chart_region_clause` is untouched.

**Measured on the 12 pages** (`probes/meter_class_r3.py` → `round3/meter_class.jsonl`):
`verdict.clauses` goes from 1 (`native_table_verifier=FAIL`, page-scoped) to 1 + one
`chart_region` clause per region — 50 region clauses over the 12 pages, each UNKNOWN,
each rendering the exact line already on its page. The page's bytes do NOT move: every
clause is suppressed by its own canonical line (`marks_to_write`'s pair coverage), which
the test asserts as `marked.text == body`.

### The shared surface: the acceptance meter does not move

`core/verdict.py` and `core/marks.py` are read by every lane, so the meter was measured
before and after over the whole corpus run, not just the 12 pages:
`socr-acceptance runs_23319732 --inputs corpus/manifest.csv`, base `c00651fd` vs this
tree. **METER = 7538 of 17,155 in both arms, and the two JSON reports are byte-identical**
(`leads/1155/round3/meter_{base,new}.{txt,json}`). (That 7538 is 7537 `document_not_assembly`
+ 3 `table_only_acceptance` + 311 `document_mark_not_first`, overlapping; the run predates
#1177, so it is a stale-run artefact and identical in both arms.)

**Stated honestly: that identity is weak evidence by construction.** The meter reads a
run's OUTPUTS (sidecar JSON + page Markdown) and imports one constant from
`core/verdict.py`, so it cannot see a derivation change at all. It rules out accidental
coupling, not a changed future run. The forward question was asked per page instead, on
the only pages whose bytes change: for all 12, the meter's own `judge_page` returns
**MARKED before and MARKED after** — status `error`, a leading mark either way. No class
moves.

### The recorded test gap is closed

The selector's override branch (`_select_page_output_tagged(..., text_will_be_overridden=)`
choosing `None` over a builder) was covered only by a mutant on the production line. It is
now driven through `_select_and_finalize_page` on the GH-317 floor fixture's own
`DocumentState`, so both halves are pinned: `saved_text` reaching the selector as
`text_will_be_overridden`, and the selector declining to build.

### Mutants (external copy of `src`+`tests`+`pyproject.toml`, an IN-PROCESS canary test
asserting `socr.__file__` is inside the copy, uncapped `src.count(anchor) == 1` before
editing)

| mutation | test killed |
|---|---|
| the order refusal keeps neither the image nor a line | `test_an_order_refusal_keeps_the_page_image_and_says_the_text_was_withheld` |
| the verdict ignores the recorded region clauses | `test_every_region_the_page_marks_is_also_a_clause_in_its_verdict` |
| the floor records no region clauses at all | (same test) |
| a region's UNKNOWN/FAIL result is inverted | `test_a_regions_result_is_UNKNOWN_unless_its_own_outcome_was_established` |
| the selector's override condition is deleted | `test_a_saved_body_stops_the_selector_offering_the_chart_floor_at_all` |
| the saved-body signal never reaches the selector | (same test) |

Each killed exactly its own test and nothing else. The canary is a test INSIDE the mutant
run, not a check outside it: `pyproject.toml`'s `pythonpath = ["src"]` is rootdir-relative,
so an external canary cannot see which tree pytest actually imported.

**The result mutant SURVIVED its first run** and that is a real finding, not a formality.
Routing both the line and the clause through one record meant the floor no longer calls
`panel_mark_line`, and the existing UNKNOWN-vs-FAIL test covers `panel_mark_line` — so the
refactor silently un-covered the distinction on the path that ships, and inverting it
passed the whole file. The new test drives the reader's own reading of a drawn page for
the UNKNOWN cases and a real frozen `PanelReading` with its label replaced for the FAIL
case, which is the only one of the three not reachable from a drawable fixture.

### Two suite findings from this round

1. **One existing test's SETUP was edited**, and it is the only such edit in this PR
   beside the GH-304 registry record. `test_public_wrapper_returns_the_output_not_the_tuple`
   monkeypatches the selector with a stand-in carrying the pre-#1155 signature, so round
   2's `text_will_be_overridden` keyword (`manifest.py:5809`) made it fail on the
   signature rather than on its own contract. The stand-in now accepts that keyword and
   ASSERTS it is `False` on both calls that save no body, so the forced change buys
   coverage of the wiring instead of permission. A mutant returning the TUPLE from the
   public wrapper still kills it, so the test keeps its teeth. No expectation changed.

2. **Editing a source file while the suite runs breaks a real test, and it is not a
   defect.** `test_the_source_digest_is_what_makes_dropping_the_key_free` recomputes a
   SHA-256 over every shipped `socr/**/*.py` and compares it to `_SOURCE_DIGEST_CACHE`,
   which is filled once per process. A docstring edit to `figures/chart_witness.py` while
   a full run was in flight made the two disagree. It passes alone and beside every file
   it was paired with, and the frozen re-run is green: **7994 passed, 2 skipped, 4
   xfailed, 0 failed** (5:57). The lesson is procedural — freeze the tree for the run
   whose number you intend to quote.

## Round 4 — the panel's three remaining items

Both seats MERGE-AFTER-FIX, everything from round 2 and 3 verified fixed. All three items
were real and all three are in this round.

### 1. Round 2's restored-recovery guard ERASED the page it was added to protect (Argon)

Round 2 made `chart_floor_replacement` return `None` for a page whose winner is an
already-finalized prose recovery, so the body would not be recomputed from the
`native_words` a restored page does not carry. But the caller reads a `None` builder as
"this page has no chart body" and returns the bare whole-page marker — so on 2015-09-17
p22 the note, the banner and all four crops were deleted and `scanned_prose_recovered`
flipped True → False. A guard that erases more than the bug it closes.

Fixed by the ruling that already governs this case: #649 round 2's **the frozen result
stands**, which `_select_page_output_tagged`'s D3 branch implements by keeping
`p.best_output.text`. This lane now returns `lambda: frozen` with the same credential —
`_is_restored_prose_recovery` requires both the typed field and this module's own banner,
neither forgeable by a model attempt. An empty winner still falls through to the marker.

Round 2's test asserted `builder is None` on the HELPER, so it pinned the defect as if it
were the contract, and it kept passing after the fix because its fixture had no
`best_output` at all. Replaced by a test that drives `_select_and_finalize_page`.

### 2. A saved body shipped without its own recovery facts (Argon)

`_phase_assemble` re-finalizes every page with the fragment it already wrote as
`saved_text`. Selection is told its text will be overridden, the chart floor declines, and
the structure-class ending derives `scanned_prose_recovered` and `audit_notes` from the
bare marker it builds instead — so the LAST write of the run rewrote `pages/00022.json`
with `scanned_prose_recovered=False, audit_notes=[]` over a body that visibly carries the
banner. The record contradicting the bytes it describes, on the write a resume reads.

`_recovery_facts_of_saved_body` now re-derives them from the saved body, by the same
evidence the ending and the restore path use (`_is_prose_recovery_body`).

**ADDITIVE only, and a surviving mutant is why.** The first version also cleared a stale
True. A mutant deleting the clearing killed nothing: a saved body forces
`text_will_be_overridden`, which makes the ending derive its flag from the bare marker, so
the output arriving at that seam already says False and a stale True cannot reach it by any
path this ticket touches. Clearing was therefore an unexercised branch that REMOVES a
disclosure. It was dropped rather than shipped, and the test arm written for it was deleted
rather than left asserting vacuously. If a future ending sets the flag on a path the
override does not suppress, that is when to add the other direction — with the arm that
reaches it.

### 3. The page-scoped line existed only as a string (Astra P2)

A page could say `native_coverage=FAIL; scope=page` in its Markdown while its terminal JSON
carried no such clause, so a machine-readable audit could not tell that the page's text had
been withheld. Same defect class as round 3's region clauses, one level up.

Every mark this lane writes is now rendered by `clause_mark` from a record that is also
handed to the verdict: `collapse_withheld_runs` returns the RECORD rather than the line,
`order_refused_record` replaces `order_refused_mark`, and `PageState.chart_floor_clauses`
(renamed from `chart_floor_region_clauses`) carries all of them. `verdict._chart_floor_clauses`
publishes any record whose check is in `_CHART_FLOOR_CHECKS` (`chart_region`,
`native_coverage`) and still refuses a `chart_region` record that names no region.

**Wider than Astra asked, deliberately.** The withheld-runs line had exactly the same gap
and fires on all 12 real Fed pages, where the order refusal fires on none; recording only
the one a reviewer named would have left the commoner case unrecorded — the shape
`denylist-safety-lanes-leak-every-round` describes.

### Measured

- **The 12 Fed pages: MARKED → MARKED, all 12.** Clauses now 1 + one `chart_region` per
  region + one page-scoped `native_coverage`: 50 region clauses and 12 coverage clauses
  over the 12 pages (`leads/1155/round3/meter_class_r4.jsonl`).
- **No shipped byte moves**: the round-4 bodies are identical to round 3's on all 12
  pages, because every line is rendered from the record it is now also recorded as
  (`order_gate_r4.jsonl` diffs clean against `order_gate.jsonl`).
- **The acceptance meter is still byte-identical to base** `c00651fd` over the corpus run,
  7538 of 17,155 (`meter_new_r4.json` vs `meter_base.json`).

### Mutants (external copy, in-process canary, uncapped anchor count)

| mutation | test killed |
|---|---|
| a restored recovery falls back to `None` again | `test_a_restored_prose_recovery_keeps_its_frozen_bytes_through_the_floor` |
| the saved body carries no recovery facts | `test_a_saved_body_carries_its_own_recovery_facts` |
| the helper records nothing at all | (same test) |
| the page-scoped record is never recorded | `test_what_the_page_says_about_withheld_text_is_in_its_terminal_verdict` |
| the clearing branch is removed | **nothing — see item 2; the branch was dropped** |

### Test expectations changed, with the justification

Three, all mine from earlier rounds, none a fixture edit that hides production behaviour:
`test_a_restored_prose_recovery_is_not_re_finalized_through_the_chart_floor` is replaced
(it pinned the defect); `test_repeated_withheld_run_markers_collapse_to_one_page_scoped_line`
now renders the returned record and keeps every assertion it made about the line; and the
region-clause test follows the renamed attribute.

### The round-4 fix that hung the suite for 55 minutes, and the latent bug under it

The first frozen run of round 4 never finished: 55 minutes at 100% CPU, stalled at
`test_1100_blank_pages.py::test_halted_blank_page_is_not_reported_as_shipped`.

`_is_prose_recovery_body` walks a body dropping leading blank lines and this page's own
canonical lines, under `while True`. On an EMPTY or whitespace-only body,
`"".partition("\n")` returns `("", "", "")` — blank head, empty separator — so the loop
reassigned `body = ""` and span forever. **The bug is older than this ticket and was
unreachable**: every previous caller passed a body that always carried at least a marker
line. Item 2 above added a caller that passes a SAVED FRAGMENT, and a halted blank page's
fragment is empty.

`while body:` is the whole fix; an empty body carries no banner, so falling out and
returning False is also the right answer. The regression test asserts with a THREAD
TIMEOUT, because a hang is the one failure that reports nothing: a mutant restoring
`while True` now makes that test FAIL in 25 s instead of stalling the run. The caller also
short-circuits an empty saved body, and that guard is annotated as belt-and-braces in the
source — a mutant removing it kills no test, now that the walk terminates by itself.

Two procedural notes, both earned this round: a hang is invisible in `-q` output, so the
diagnosis came from `-v` plus a stall watcher (the test name, immediately) rather than from
waiting on `faulthandler_timeout`; and every full run from here carries
`-o faulthandler_timeout=300 -p faulthandler`, so a repeat names itself.

Suite after the fix, frozen tree: **7997 passed, 2 skipped, 4 xfailed, 0 failed** (6:05).

## Round 5 — the repo reviewer's blocker on PR #1190

One blocker, two parts, both about the preservation-pass skip
(`orchestrator.py:5487`). CI was green and the lane itself held up; 13 of the reviewer's
16 mutants were already killed.

**1. The skip was UNPINNED.** Deleting
`if getattr(ps, "chart_floor_body", False): continue` kept every test green, because no
test outside this ticket's file ever set the flag and
`test_selecting_the_replacement_marks_the_page_for_the_preservation_pass` only checked
that the FLOOR sets it, never that the PASS honours it. Now driven for real:
`_preserve_chart_regions` over the body `chart_floor_replacement` built, asserting
identical bytes, no `chart_region_*` flag and no event — with the flag-off control, which
does rewrite the body. A mutant deleting the skip kills that test and leaves the GH-189
suite that owns the pass green.

**2. The flag lived only in memory.** `chart_floor_body` was never written to the sidecar
and never restored, so a resumed run brought the body back without the fact that describes
it, and the pass would re-enter a page whose in-chart words this lane withheld. Persisted
beside the GH-682 `chart_region_*` siblings and OR-restored the same way.
`chart_floor_pictures` and `chart_floor_clauses` are deliberately NOT persisted: the first
is a within-process render cache, and the second reaches a resumed page through #1145's
`published_verdict`, restored verbatim. Two mutants — dropping the writer line, and
ignoring the restored value — each kill the resume test.

**The sidecar surface is policed, and it caught this.** The new key failed seven P6 tests.
That is the guard working: a sidecar key may not appear unless it is enumerated per page
with its expected value. Registered from the oracle's OWN measured report — twelve leaf
paths, each `None -> False`, and nothing else — in `ADDITIVE_SIDECAR_KEYS` plus
`CHART_FLOOR_BODY_SIDECAR_1155_DIFFERENCES`, and in the canonical key set of
`test_p6_disposition_persistence.py`. No guard was relaxed: the writer mutant now fails
the P6 surface test as well as the resume test. False on every page of that corpus,
page 10 included — it is a structure-class floor page whose chart reader resolves no panel,
which is why its markdown is unchanged too.

**Honest limit of the resume test**, stated in its docstring: it drives
`_flush_page_sidecar` and `_restore_terminal_page_state`, the writer and the restorer the
review named. It does NOT assert that `_load_terminal_page`'s ledger gate admits such a
page — a floored page is ERROR, and that gate decides whether a page is skipped at all,
which is a different question.

One more hermeticity trap, caught by the repo's own guard rather than by CI:
`_flush_page_sidecar` reaches `_run_fingerprint` -> `_resolve_judge_model`, which POSTs to
a live Ollama daemon whatever the judge backend. Patched to `""` in the test, per CLAUDE.md.

Suite, frozen tree, with `-o faulthandler_timeout=300 -p faulthandler`: **7999 passed,
2 skipped, 4 xfailed, 0 failed** (6:00). Format gate: 968 files already formatted. The
softs in that review are untouched — the orchestrator is filing them as a follow-up.
