# #1407: a chart's own labels no longer ship as crops on top of the chart's picture

Date: 2026-11-25. Branch `fix/1407-chart-label-crops`.

## The defect

A chart page in the chart-asset lane ships its chart as a whole-page picture
(`![Chart page N](figures/chart_page_N.png)`) under `figure_text=UNKNOWN`. The
uncovered-text step (#1050) then ran on the same page and emitted one crop and one
`> Text drawn as shapes (OCR, not verified): …` line for every label drawn as shapes:
axis years, "Percentage points", panel titles.

## What the code knows at the point where crops are emitted

`_apply_text_coverage` (`pipeline/orchestrator.py`) has both sides:

- **The uncovered regions' boxes**: `CoverageRegion.pdf` (and `figure_pdf` for a merged
  figure region), in PDF points, from `coverage_for_document`.
- **The shipped chart picture**: its ref is in the page body. It has **no box of its
  own**. `_render_chart_page_png` renders the full page, so the picture's area is the
  page rect and every region is inside it by construction.

Containment in the picture therefore decides nothing. The rule needs a second fact: is
the region part of a chart. `chart_region_bboxes(page)` (`tables/reconstruct.py`)
already answers that for the chart lane: drawing clusters with chart marks, each grown
by the page's own median row gap so the axis labels fall inside.

## Measured before writing code

December 2017 Tealbook part A, page 25 (`analyze_page` on the PDF, no model):

| Region lies inside | Regions (of 73) |
| --- | --- |
| the shipped picture's area (the whole page) | 73 |
| a detected chart region (grown box) | 52 |
| a chart's drawing cluster alone (no growth) | 7 |
| no chart region | 21 |

The 21 outside are the page title, the note and source lines under each panel, the
page's closing note, and small pieces of those. All 21 carry a reading in the shipped
page.

Whole run (156 documents, shipped `pages/*.md`, counted by the picture the page
carries):

| Page ships | Pages | Fragments |
| --- | --- | --- |
| a whole-page chart picture | 215 | 5,095 |
| a chart-region crop | 1 | 1 |
| no chart picture | 84 | 2,767 |
| total | 300 | 7,863 |

## Decision

A region is left out when both hold:

1. The page ships a whole-page chart picture and its mark. Checked three ways: the
   chart lane's `visual_values_not_transcribed` event (the event `figure_text=UNKNOWN`
   is keyed on) names the ref with `png_saved`, the ref is in the body, and the PNG is
   on disk.
2. The region's crop box lies wholly inside ONE box of `chart_region_bboxes(page)`.

Left-out regions are not rendered, not read (no model call) and not listed. The page
gets one `uncovered_text_in_chart_picture` audit event with the count and the region
numbers, and `text_ink_unread.data.in_chart_picture` repeats the count. Status,
failure mode, lane, the `native_coverage=FAIL` mark and the page mark are unchanged.
Surviving regions keep their original numbers.

### Alternatives not taken

- **Leave out every region on a whole-page-picture page** (73 → 0 on page 25). The
  picture does show all of them, but the mark says the chart's "labels and values"
  were not transcribed; it says nothing about a three-line note under a panel. Dropping
  those readings would remove words that ship today. This is an owner ruling if wanted,
  and it is a one-line change to the second condition.
- **A containment tolerance.** Not added. Containment is exact. A region box is the
  ink's pixel box mapped back to points, so a label on a box edge can miss by a fraction
  of a point, and then it keeps its crop: the behaviour before this change. There is no
  measured number to set a tolerance from.
- **The crop route (#1053) and chart-region crops of other lanes.** 1 fragment of 7,863.
  Not handled.

## Result on page 25

Single-page copy of the PDF, crop reader stubbed, two runs in one process differing
only in whether the rule runs: 73 crops → 21 crops; 52 regions recorded in the event.
Same status (`warning` / `text_ink_unread`), same engine, same two verdict clauses.

## Known limits

- The grown chart box is a heuristic made for another purpose (keeping tick labels out
  of the table rowizer). On page 25 it reaches the "Source: Federal Reserve Board." line
  directly under the top-right panel, so that line is left out with the labels. It also
  extends to the page's left edge.
- 21 fragments remain on page 25, several of them slivers of a note that is already one
  region. That is a grouping defect in the uncovered-text step, not addressed here.
- The 84 pages with fragments and no chart picture (2,767 fragments) are untouched.
- The share of the run's 5,095 whole-page-picture fragments that this removes was
  measured on one page only (71%). It was not measured across the run.
- The new event is not in the resume replay allowlist, the same as
  `uncovered_text_recovered`. Whether a resumed document's `audit_log.json` still shows
  it was not tested.
- Real readings were not produced: no model was called. Counts of crops do not depend
  on the reader; which crops carry a reading does.

## Guard seen to fail

`src` and `tests` copied outside the repo, the `if i in in_chart_picture` skip
disabled (anchor count asserted to be 1 first), the test module run against the copy
(`test_loaded_source_is_this_checkout` passes there, so the copy was loaded):
`test_a_charts_own_labels_are_not_cropped_on_top_of_its_picture` fails with
`assert [1, 2] == [2]`. The other tests in the file pass, as they should.
