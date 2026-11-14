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

| native pages | 10,216 (8,017 success, 902 warning, 1,297 error) |
| --- | --- |
| with a box | 421 |
| skipped: floor / failed marker | 1,138 (counted before the box test) |
| skipped: own page image | 52 |
| skipped: table signal (GH-189's) | 93 |
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
| pass record not written | `test_a_chart_with_ticks_ships_a_crop_whatever_...[native-success]` |
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
