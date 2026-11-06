# #1269 — chart predicates disagree on 6 Fed pages; `chart_region_bboxes` > 60 s on Tealbook pages

Date: 2026-11-06. Branch `fix/issue-1269`. Evidence files live in the army scratch area
(`scratch/army/1269/`, gitignored); Fed PDFs and outputs never enter git.

## Part B — the 60-second pages (fixed by algorithm, no time budget)

**Cause.** Three 2018 Tealbook part-B pages (06-13 p30, 01-31 p34, 05-02 p34) carry about 52,300 drawings each: stacked
area charts painted as hairline fill slivers 0.02 pt wide. `get_drawings` takes 0.7 s. Both drawing clusterers,
`_union_find_clusters` (production chart boxes) and `_cluster_drawings` (the census `has_chart_marks`), compared every
pair in Python (about 1.4e9 checks). Both ran for more than 60 s on all three pages.

**Change.** One shared leaf helper, `socr/core/box_clusters.py::cluster_box_indices`, used by both callers. It puts the
boxes on a grid with cell = gap/2 and unions boxes that share a cell without a check, since such boxes are within gap.
It checks cross-cell pairs exactly within ±3 cells: 2 cells cover the gap and 1 is a margin for floor() rounding.
Output order is unchanged, so both callers' results are byte-identical to before. It falls back to the exact pairwise
loop in five cases: a box inverted, a non-finite gap or coordinate, a degenerate cell, coordinates beyond cell·2^50,
or registration plus exact checks exceeding the pairwise count (follow-up: cell memberships are counted before any is
allocated and charged to the same budget). A grid attempt therefore never stores more memberships, or does more
registration-plus-check work, than one pairwise pass; registration plus exact checks total at most two passes
(union-find and dictionary bookkeeping are not counted in that bound).

**Measured** (Rorqual, 674 pages: the 550 checks345 diagnostic pages plus the other 124 pages of the three Tealbook
documents, 129 Tealbook pages in all, 5 of them already among the 550;
main 4e170962 against the grid commit, 60 s alarm per call):

| | main | grid |
|---|---|---|
| pages timing out | 6 | 0 |
| slowest page | over 60 s | 6.2 s |
| total time on the other 668 pages | 57.7 s | 23.3 s |

The six main timed out on are the three named in the issue plus 01-31 p33, 05-02 p33 and 06-13 p29. Chart-box counts
and `has_chart_marks` are identical on all 668 pages main finished. The 52k-drawing pages need 7,469 exact checks
against a fallback budget of 1.37e9.

**Why no time budget.** A budget has to skip something and then surface the skip. The algorithmic fix skips nothing
and changes no output.

**Review.**
- My review caught a floor() rounding miss at reach 2: boxes 14.999999999999998 and 45.0 are 3 cells apart. Fixed
  with reach 3.
- Astra built a two-band layout that forced Θ(n³) checks (7.1M at n=160). Fixed with the check budget.
- Astra also found that inverted boxes were dropped, that coordinates of 1e16 and above produced false same-cell
  unions, and that a 1e21 span overflowed. All three now fall back to pairwise.
- The one surviving mutant, cell = gap, gave identical output on every tested input; it shrinks the rounding margin
  and is not proven equivalent in general.

## Part A — census says chart, production draws no box (not reconciled; reason recorded)

**Why per page.** On all six pages (2005-08-09 gb2 p28/p63/p64, 2007-08-07 transcript p19, 2007-09-18 gb2 p13/p14)
the census fires only through `_has_framed_data_cluster`: thin black framed clusters, no fills, no colour. Production
`_has_filled_rects_or_thick_strokes` deliberately has no framed branch. Five pages are monochrome line-chart grids.
On p19 the framed clusters are the two text-box brackets, not the charts.

**Why not reconcile.** `read_chart_page` returns 0 panels on all six for every box choice tested: the census boxes, a
whole-page box, and 131 other choices (Kimi). That is not a proof over arbitrary rectangles. The refusals are "fewer than two ticks carry a numeric label", "no
vector plot frame", and two tick ladders of the same plot read as two frames. Widening the production predicate would
therefore not make the #1155 floor fire. On p19 it would delete 209 of 293 words through the rowizer's page-edge
expansion.

**Change.** The chart floor's probe result is kept on the page (`chart_floor_probe`, transient). The persisted sparse
sidecar key `chart_floor_abstention` is set only where the floor actually declines a box-less, table-free page. Its
outcomes are `no_chart_bboxes` (with `census_chart_marks`), `no_panels` (with the reader's refusals), `empty_body`,
`page_rotated`, `probe_not_run`, `probe_failed` and `probe_unavailable`. Every finalize from source resets it. On a saved-body re-finalize it is kept only
if two things hold. First, the saved bytes are exactly the bare floor body: once socr's own mark lines and the page-image
line are removed, what remains is the floor's whole-page marker line alone. Second, the latest selection from source
produced no chart body. That is a transient per-selection flag, because the #1155 `chart_floor_body` flag is never
cleared and a stale True erased valid reasons (Astra re-review). On the six pages it reads `no_chart_bboxes`,
`census_chart_marks: true`.

**Accepted limits.**
- A rotated page whose chart builder exists and is then declined by the /Rotate guard records nothing. Recording it
  needs a line inside the function #1266 is editing; follow-up after #1266 merges.
- The restorer brings the key back only for ledger-skipped pages. The ledger loader has non-SUCCESS exceptions, but it
  excludes structure-class floor modes and refuses whole-page failure-marker bodies, so a bare chart-floor page is
  reprocessed from source on resume and its key is recomputed there, not restored. A direct restore followed by a
  re-flush of such a state drops the key; no production path does that.

**Out of scope.** The 143 pages where chart regions are found but `read_chart_page` returns no panels for line/area
charts are a reader feature. Where such a page reaches the floor's decline seam, it now records `no_panels` with the
reader's refusals. This was not re-measured across all 143 pages.
