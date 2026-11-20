# #1393 item 1: a partly pictured chart page no longer ships SUCCESS

## Cause

Fed p000 has six chart panels. `chart_region_clusters` keeps a drawing cluster only if
`_has_filled_rects_or_thick_strokes` holds (`tables/reconstruct.py`), which keeps ruled tables out
of the chart inventory and also drops thin-stroke line charts. The inventory holds the one bar
chart; `_preserve_chart_regions` crops that box (a box over the middle row, axis labels clipped)
and `chart_region_lane` returned `crops=True`, which excused the #1389 tick mark. Four charts in
no picture, no mark, SUCCESS.

## Change

* `chart_region_lane` returns `(owns, crops, uncovered)`. `uncovered_charts` finds, among every
  chart-sized drawing cluster, those lying outside every crop box that carry both a scale
  (`figure_presence.assess_regions` tick evidence) and a plotted series (`plots_a_series`: a path
  of `MIN_SERIES_RUN` = `MIN_TICK_RUN` consecutive slanted segments or curves, slant measured
  against the stroke's own width). Such a page is not the lane's (`owns`/`crops` False), so the
  #1385/#1388 whole-page picture and `figure_not_shown` mark apply, and `decide` takes `uncovered`
  as evidence when the locator finds nothing. No new audit kinds (UNCAPTIONED_CROP / _LOST reused),
  so `test_resume_restore_kinds.py` needs no entry.
* Rotated pages abstain (drawing clusters and table boxes are in different coordinate spaces).
* Tool: `chart_region_preserved` is now treated as a table signal and the same rule runs, replacing
  the `(True, True)` shortcut.
* `tables/reconstruct.py`: `chart_region_clusters` is now `_expanded_clusters(page, require_marks=True)`;
  new `sized_drawing_clusters` is the same function with `require_marks=False`. Callers of
  `chart_region_clusters` and `chart_region_bboxes` see identical output to before (mutant m12,
  flipping the flag, fails 15 tests). The mark gate was the cause, and the uncovered-chart check
  needs the clusters it discards.

## Census (rep100 outputs, full-size renders, crop boxes overlaid)

44 pages ship a chart-region picture. Partly pictured: p000 (SUCCESS, 4 of 6 charts), p099
(warning, 2 line charts; the crop is the redacted boxes), p025 (warning, raster chart, no
drawing evidence: NOT fixed), p024 (warning, rotated page, crop geometry wrong, 12 charts: NOT
fixed). The rest are covered, or table false-positive boxes. p075, the only other SUCCESS with a
picture, is fully covered. Not decidable by the decision-only census (finished bodies hold the
picture); decided from renders.

## Live replay (65 pages, local qwen, paid reader replayed)

Fix arm vs current-main arm on the 8 pages that differed from rep100: only p000 and p099 differ
(2 fires, both true positives). p000 SUCCESS -> warning, chart-region crop replaced by the
full-page picture plus mark. p099 warning -> warning, chart-region crop replaced by the full-page
picture. The other six differences from rep100 (p002, p017, p053, p074, p093) are #1388/#1391,
already on main, and p018 is local-model nondeterminism (main arm also SUCCESS). Fix arm vs
rep100 on all 65: 8 differ, all explained above. False positives 0, lost pictures 0, no page got
worse, SUCCESS pages changed: p000 (intended).

## Mutants (copies outside the repo, loaded source asserted)

14 mutants (rotation, tick half, series half, containment, lane, decide, tool signal, live and
tool pass-through, MIN_SERIES_RUN, slant `and`->`or`, mark gate, Bezier, save-figures reset);
the last was equivalent (line removed). All others fail at least one test.

## Not fixed

p025 (raster chart), p024 and other rotated pages, `--save-figures` (#1393 item 2), refinalize of
finished documents whose body already holds the partial picture. p099 ends with two pictures
(failed-table floor image plus the full-page picture), as the baseline had floor plus crop.
