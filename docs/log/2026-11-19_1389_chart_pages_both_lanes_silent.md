# #1389: chart pages that ship SUCCESS with no picture and no mark

## Cause, per page (measured on the real PDFs and saved readings)

- **p074** (three line panels). The page has chart marks and a table signal, so the routing step
  emits `chart_table_arbitration` and the #1385 lane abstains: `chart_region_owns` in
  `_agentic_unshipped_figure_page` (and the same expression in
  `refinalize_unshipped_figures.refinalize_page`) is true on that event. The chart-region lane
  that "owns" the page crops only what `tables.reconstruct.chart_region_bboxes` returns, and that
  inventory is empty on p074, so `_preserve_chart_regions` takes its silent
  `if not bboxes: continue`. Both lanes abstain. `locate_figure_regions` finds 3 regions.
- **p002** (six chart panels). Table signal, no arbitration event, `locate_figure_regions` = 0
  regions, `chart_region_bboxes` = 0. The #1385 lane needs a located region; the #1225 mark needs
  `tick_run >= 3`, and the reading wrote its axes as `- **Left Axis (Percent):** 1.0, 1.5, ...`,
  a form `tick_run` does not read (it needs a line made only of numbers).

## Census (100 hard Fed pages, saved readings of the Nov 18 run, decisions replayed through the
re-finalize path on this branch's code and on main's)

Chart present, no picture, no figure mark on main: **3 pages**, p002, p074 and p037. p037 is a
scanned page (status WARNING, carries only a prior-OCR-layer mark) whose reading is three charts
written as inline tick lists. **p037 is NOT fixed**: the live lane abstains on it at
`page_ships_its_own_page_image` (a prior-OCR-layer page, whose mark is in the body), although no
picture ships; the re-finalize tool does not restore that state, so the two callers already differ
there (pre-existing, separate cause, left for its own issue). Pages with rule lines and many drawings but no chart (p009, p063,
p097: tables) and the blank/cover pages p026, p045 were checked at full size and are not charts.

## Change

1. `chart_region_lane` (one function, both callers) returns `(owns, crops)`. An ARBITRATED page
   whose chart-region inventory is empty is not owned by that lane, so the #1385 picture branch
   applies to it. A table page with no arbitration event keeps `owns` (nothing says it has a
   chart) but `crops` is false when the inventory is empty (p002). `crops` true excuses the inline
   tick mark: that lane's pictures ship at assembly, after the page loop (live replay found p024,
   already pictured by it, gaining a false mark in the first draft; pinned by a test).
2. `tick_run(..., inline=True)` also reads a labelled inline tick list (`label: n, n, n`), asked
   only AFTER the picture branch, so a page that can be pictured keeps its picture. Evidence only,
   never a transcribed value; the result is the existing `CHART_SKELETON_KIND` mark. No new
   constant: `MIN_TICK_RUN` is unchanged. No new event kind, so no new resume entry.

## Measured effect (decisions on the 100 saved readings)

p074: crop (picture + mark). p002: skeleton mark (live replay: SUCCESS to WARNING, `figure_not_shown`). No other page changes: the 13 pages
#1388 already handles keep their answers (the first draft downgraded p017, p053, p093 from picture
to mark because the inline rule ran first; the order above is the fix, pinned by a test). No text
or table page gains a picture or a mark.

## Not done

p002 gets a mark, not a picture: the locator finds no region on it and the page picture needs a
located region (#1385 owner ruling 2026-11-19). A picture on tick evidence alone would extend that
ruling; not taken here.
