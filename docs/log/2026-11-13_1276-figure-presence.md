# #1276: a chart the page draws ships as a picture or the page says it did not

Branch `fix/issue-1276-a3`. Plan A3 (`scratch/army/phaseA/BRIEF-A3.md`) with the binding changes
from the three advisors and `DECISIONS.md`.

## What was wrong

* **Native lane.** A native page whose PDF draws a chart, with no table signal, reached no lane
  that ships a picture and nothing recorded the outcome. `_preserve_chart_regions` only visits
  pages with a table signal; `_is_chart_asset_page` routes on `has_chart_marks`; the box finder
  `chart_region_bboxes` disagrees with both.
* **Model lane.** `<!-- Image (x, y, w, h) -->` ships inert. Nothing in `src/` read it.

## The rule (`src/socr/figures/figure_presence.py`)

Native winner (any engine starting `native`, except `chart_asset`; **status is not part of the
gate**, so WARNING and `native+math` pages are covered):

1. Skip pages that are GH-189's (`_page_has_tables`: `_preserve_chart_regions` crops every box
   of exactly that set at assembly), pages with a floor body, and pages that ship their own
   page image.
2. `chart_region_bboxes(page)`. No box: outside this check, no record.
3. Per box, evidence is **tied to that box**. The words whose top-left corner lies in the box,
   minus the words inside any detected table, are put back into the text layer's own
   `(block, line)` lines; `tick_run` (existing, `MIN_TICK_RUN` = 3) runs on those lines. A box
   whose words have recurring numeric columns (`has_recurring_numeric_columns`, existing) is a
   **table region**: no evidence. This is the delimiter-free year-header exclusion.
4. A region is covered only by **its own crop** (`chart_region_pN_k` or `figure_presence_pN_k`
   with the same ordinal) or the page's own image. A logo, a table crop or another panel's
   image excuses nothing. The guard that appends the crop does not use the "any image on the
   page" abstention for these crops (`manifest._apply_unshipped_figure_guard`).
5. Firing regions are cropped. Records:
   * crop shipped: event `figure_presence_shipped` plus the existing
     `visual_values_not_transcribed` event (detail "figure shipped as image, values not
     transcribed"), which is #1174's carrier: its clause puts the visible #1174 mark line on the
     page and demotes SUCCESS to WARNING. The words on the page are the #1174 sentence.
   * crop cannot be saved: `figure_presence_render_failed`, a figure-specific clause per lost
     region, failure mode `FIGURE_RENDER_FAILED` (transient: retried on resume).
   * box but no firing region: `figure_presence_pass`, reason "no chart evidence under this
     check ... not a statement that the page carries no figure". Never marks (DECISIONS).
   * PDF unreadable: `chart_region_inventory_failed` (existing kind), never a pass.

Model reading: each `<!-- Image` placeholder (fenced code ignored) must be matched one-to-one by
a shipped **figure** image (stems `chart_region_`, `chart_page_`, `unshipped_figure_`,
`figure_presence_`, `figure_`; `failed_table_` and `uncovered_text_` are not figures), plus the
crops the #1187 lane just recorded, plus, on a GH-189 page, its chart boxes. Unmatched
placeholders get a figure-specific clause (`image placeholder N`, `FIGURE_NOT_SHOWN`). A page the
#1187 lane already marked for a figure keeps that mark.

`audit_passed` is untouched everywhere. Resume: the four new kinds are in the replay set
(`tests/test_resume_restore_kinds.py`, 69 -> 73) and the crop refs and outcome kind ride the
existing `unshipped_figure_*` sidecar fields.

## Measured (cached `fed_rerun_51270eb5_words`, 156 docs, no OCR)

`scratch/army/phaseA/a3_measure.py`; firings in `a3-firings.jsonl`, crops in `a3-crops/`.
The scratch files are local working files, not committed; the numbers are reproducible from the
scripts and the cached run named above.

| native pages | 10,216 (8,017 success, 902 warning, 1,297 error) |
| --- | --- |
| with a box | 421 |
| (before the box test, not in the funnel below: floor / failed marker 1,138; own page image 52) | |
| skipped: table signal (GH-189's) | 93 (421 - 93 = 328) |
| **checked (box, no table signal)** | **328** |
| **ship a crop** | **80 pages, 128 regions** (43 SUCCESS, 37 WARNING; 8 pages `native+math`) |
| **pass record, no mark** | **248** (all "no chart evidence"; 0 had every box already shipped) |
| **marked (render failure)** | 0 observed (cannot be measured without rendering failing) |

* The 43 SUCCESS firings equal the brief's 43; WARNING pages add 37 the SUCCESS-only gate would
  have missed.
* The 17 auditor pages: 17 of 17 fire. Two (mpr 2011-07 p35, FSR 2020-05 p46) did not under the
  first grouping of words into lines (by vertical overlap, which merges `400 ABS 80`); grouping
  by the layer's own `(block, line)` fixed both. Measured, then changed.
* The year-header table exclusion fired on **0** real pages of this run: every table-like page
  with a box is GH-189's (93). It is tested synthetically only; the mutant shows the test is
  load-bearing, the corpus does not exercise it.
* One checked page has a tick run in its whole body and a box but no tick run in the box
  (2020-07-29 transcript materials p23): the region tie declined it.
* Model side: 34 pages carry a placeholder (33 WARNING, 1 SUCCESS). 22 have more placeholders
  than figure images (57 unmatched placeholders): each gets the new clause, all 22 are already
  WARNING. 12 match (incl. FSR p57, the SUCCESS residue). New demotions on this side: 0.

Hand check (what I looked at myself): 5 of the 128 crops (tealbook-parta 2014-03 p23 r3, mpr
2024-07 p18, mpr 2022-06 p40, projections 2008-06 p9 r2, mpr 2011-03 p16): 5 real charts. 8 of
the 248 pass pages (seed 20261113, `a3-pass-sample/`): 0 charts. **This does not certify the
other 123 crops or the other 240 passes**; a visual pass over `a3-crops/` is still owed.

Known looseness of the crops: the box is expanded to the page's left edge and by one row
height, so a crop can include adjacent prose and clip a chart title (mpr 2011-03 p16 includes a
paragraph). The crop only adds an image beside unchanged text; no words are lost.

## Not covered

* Charts with no numeric ticks, or ticks drawn as shapes; pages with no box (hairline
  monochrome). They are outside this check, with no record.
* Pages that ship a floor or their own page image are skipped, not recorded.
* The measurement approximates `ps.has_tables` with `BornDigitalDetector._detect_tables` and
  has no `detected_table_bboxes`; the live loop uses the real values.
* #1268 (rules read as chart marks) is untouched: `chart_region_bboxes` was not widened.

## Mutants (src+tests copied outside the repo, `socr.__file__` asserted inside the copy,
uncapped anchor count asserted == 1 before editing)

| mutant | killed by |
| --- | --- |
| evidence from the whole page, not the box (region tie) | `test_evidence_must_lie_in_the_same_region_as_the_box` |
| table-region test removed | `test_a_delimiter_free_year_header_table_is_not_a_chart` |
| detected-table words not removed | `test_words_inside_a_detected_table_never_count_as_evidence` |
| any image excuses a placeholder | `test_a_two_figure_page_with_one_image_is_marked_for_the_second` |
| any own-crop link excuses every region | `test_another_regions_image_never_excuses_a_region` |
| guard abstains on any image again | `test_the_crop_is_appended_even_when_another_image_is_already_on_the_page` |
| tick condition dropped | `test_the_fixture_differs_only_in_its_tick_numbers` |
| pass record not written | `test_a_chart_with_ticks_ships_a_crop_whatever_the_status_or_native_engine[native-success]` |
| table-signal handoff to GH-189 removed | `test_a_page_with_a_table_signal_is_left_to_the_chart_region_pass` |
| fenced code not ignored | `test_placeholders_inside_fenced_code_are_ignored` |
| presence kinds dropped from the replay set | `test_resume_restore_kinds_is_exactly_the_pre_refactor_union` |

The box condition has no mutant of its own: with no box there is no region to assess.

## Tests

`tests/test_gh1276_figure_presence.py` (26): shipping-boundary pairs (ticks vs no ticks, SUCCESS
and WARNING, `native` and `native+math`), the year-header table, the region tie, per-region
excuse, render failure, unreadable PDF, the two-figure model page with one image, residue
placeholder, table crop, fenced code, and through `process()` in both provider states plus resume
restore of the crop, the record and the pass record. `tests/test_orchestrator.py` patches the new
step off in the one test whose PDF path is a stand-in (same reason as its `_is_chart_asset_page`
patch). Full suite: 10,082 passed.

## Review round on PR #1309 (Fable, Carbon, Hydrogen's visual check of all 128 crops, cubic)

### The visual check, and the crop extent (the main finding)

Gemini Hydrogen looked at all 128 crops: 128 real charts (precision of the evidence rule is
not in question), but **121 partial** (title above the top rule cut, `NOTE:`/`SOURCE:` cut,
x-axis labels or right-hand ticks clipped, multi-panel figures split) and **37 with the
neighbouring column's prose**. Cause: the crop was `chart_region_bboxes`' box, which exists to
keep labels out of a table rowizer: it runs to the page's left edge, ends one row below the
drawing and starts one row above it. 8 of 8 sampled pass pages had no chart.

Fix: `src/socr/figures/figure_crop.py`, used only by this lane (GH-189's crops and
`chart_region_bboxes` are unchanged; the latter now delegates to `chart_region_clusters`, which
also returns the bare drawing cluster, same boxes in the same order).

* Start from the drawing cluster, not the box (never the page edge).
* Sideways: numeric-only blocks beside the drawing within one line pitch (tick labels on both
  sides), reached without crossing a non-numeric word.
* Upward: the title block directly above in the drawing's column; downward: x-axis labels, then
  `NOTE:`/`SOURCE:` blocks. A walk stops at the first block farther than the page's body line
  pitch (median step between consecutive lines of one block, from the page). A block is a label
  if it is at most `MAX_LABEL_LINES` (2) lines long or opens with its side's lead word (title
  lead above: `Figure`, `Chart`, `Exhibit`, `Panel`, `N.`; note lead below: `Note`, `Source`).
  A title lead below the drawing is the NEXT figure's title and is not taken (found by looking
  at the first re-render: mpr 2014-02 p20 carried "25. U.S. net financial inflows").
* A block is claimed once: the region above owns the blocks it took below it.
* Regions that took a common title block, or whose rectangles overlap, are one crop named by
  the first region's ordinal; a page's covered set expands over the group.
* Padding: half the leading (line pitch minus median word height), from the page.

Re-measured on the same cached run (`a3_measure_v2.py`; crops in `a3-crops-v2/`, firings in
`a3-firings-v2.jsonl`, same naming, plus the crop rectangle and the old box so the visual check
can be repeated):

| | v1 | v2 |
| --- | --- | --- |
| pages shipping a crop | 80 | 80 |
| crops | 128 | 126 (2 panel merges) |
| crops containing a multi-line, non-label text block by centre (my own crude proxy, not Hydrogen's count) | 48 | 19 |

I opened 7 v2 crops: 2008-07 p16 (complete: title, axes, note, source), 2014-02 p20 (complete),
2011-07 p35, 2022-06 p14, 2005-07 p13, 2008-07 p20, 2019-10 p76. **Not certified**: the same
visual check on `a3-crops-v2/` is still owed.

Remaining defects, seen in those crops:

* The detector's own cluster sometimes spans two charts or two columns (a rule or the footer bar
  bridges them): 2011-07 p35 (figures 43 and 44), 2005-07 p13, 2008-07 p20 give two complete
  charts in one crop; 2011-07 p35 and 2022-06 p14 also take a few lines of body prose below
  the longer note. The crop cannot be narrower than the cluster it is cut from.
* Stacked panels with their own sub-titles (A., B.) are not merged (only a shared title or
  overlap merges); each ships complete.
* A tealbook chart whose left tick labels are farther than one line pitch from the axis keeps
  them clipped (2019-10 p76).

### Fable and cubic findings

1. **Link text is not a picture.** A placeholder counts as matched only by a figure image that
   resolves to an existing file under the document directory (`manifest.page_asset_refs`, the
   manifest's own scanner: literal context, URLs, anchors, missing files and escaping paths are
   skipped). Test: same body, link target saved or not.
2. **Table-signal model pages** count only the chart boxes that pass the native lane's regional
   tick test (`RegionAssessment.has_evidence`, one definition for both lanes). Test: same page,
   same placeholder, same table signal; the box's tick numbers are the only difference.
3. **One row of consecutive years is not a scale.** The tick run must be carried by two or more
   text-layer lines (`tick_layer_lines`: a column's length, plus each row that is itself a
   run). `2019 2020 2021 2022` beside prose gives 1; a y-axis column gives its length; a year row
   plus a tick column gives both. Measured on the cached run: **0 of the 128 firing regions
   change** (all carry a column), so on this corpus the rule costs nothing and removes the
   false alarm the three advisors named.
4. **Lane chosen from the best output, not the finalised record. Not changed; the limit:** the
   lane runs on `bo` before `finalized_page_record` selects what ships. If the selector then
   replaces a refused model reading by the native text (`NATIVE_FALLBACK`) or a floor, the
   placeholder rule judged text that does not ship (its mark and `unshipped_figure_unlocated`
   stay attached; the page is demoted for a figure the shipped text never named) and the
   native chart rule never ran on the text that does ship (a lost chart, unmarked). Deciding
   from the finalised record means calling the selector, which has side effects (the chart
   floor renders crops) and whose result depends on events this lane writes. Not confirmed in
   a live loop. Owed: a follow-up that moves the decision after selection and re-finalises.
5. **Every checked page has a record.** A table-signal page records `figure_presence_pass`
   ("left to the chart-region pass") and a page with no chart box records `figure_presence_pass`
   ("no chart box ... not a statement that the page carries no figure"). Neither marks. Cost: one
   small event per native page that reaches the lane (8,605 + 93 on the cached run).
6. Fable nit, cubic and Carbon items not changed in this round: placeholders inside a table cell
   or inline code are counted; the `chart_region_` boxes counted for a table-signal page are not
   proven cropped before assembly (cubic P2); the early return on a #1187 outcome (cubic P2);
   clause scope uses list position, not the region ordinal (cubic P2); resume does not retry a
   failed native crop (cubic P1: the guard returns for the restored kind); the document note
   and CLI sentence for the presence kinds still name "a figure socr could not locate" (Carbon 4,
   cubic); only the first ref in the #1174 record. Each is real until shown otherwise.

### Mutants for this round (src+tests copied outside the repo; `socr.__file__` asserted inside the
copy; uncapped anchor count asserted == 1 before editing; the guard removed; the test must fail)

| mutant | killed by |
| --- | --- |
| crop starts at the page edge | `test_the_crop_stays_in_the_charts_own_column` |
| no upward title walk | `test_the_crop_takes_its_title_and_notes_and_no_body_prose` |
| no downward note walk | same |
| upward gap limit removed | same |
| downward label test removed (paragraph taken) | same |
| next figure's title taken below the notes | `test_the_title_just_below_a_figures_notes_is_the_next_figures` |
| panel merge removed | `test_panels_of_one_figure_under_one_title_are_one_crop` |
| bare box counts on a table-signal model page | `test_a_bare_box_does_not_excuse_a_placeholder_on_a_table_page` |
| placeholder matched by link text only | `test_an_image_link_to_a_file_nobody_saved_ships_nothing` |
| tick-line requirement dropped | `test_one_row_of_years_beside_prose_is_not_chart_evidence` |
| no-box record not written | `test_a_page_with_no_chart_box_is_recorded_not_silent` |

No mutant for item 4: nothing was changed there.
