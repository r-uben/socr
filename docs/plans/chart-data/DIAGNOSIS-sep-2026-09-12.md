# SEP dot-plot pages: why the panels collapse and the frame finder refuses

Read-only measurement at `main@3cbf8a9`, CPU-only `fitz`, using the repo's own
`chart_region_bboxes`, `page_marks`, `find_frame`, `read_chart_page`. Corpus:
23 pages in `~/Data/socr/sep-dotplots/in/` plus the working reference
`~/Data/socr/fixtures/dotplot/dotplot-p20.pdf`.

## Verdict first

The issue's hypothesis is wrong on the drawing style. The SEP pages do **not** draw
their axes as thin filled rectangles. They draw them exactly as the reference does:
stroked `l` paths. Bars are filled `re` rects in both. The page is not the problem
in the way the issue guessed.

Three independent defects in socr's own code explain everything, and they are
separable:

1. **Region collapse** — every SEP panel is wrapped in an *invisible* rectangle
   (white fill, white stroke) that spans the whole panel slab. Consecutive
   wrappers are separated by 0.4–0.6 pt, so union-find merges all five panels
   into one cluster. The reference has no such rectangle.
2. **The tick-ladder agreement test compares floats for exact equality.** On the
   13 pages from Dec 2020 to Dec 2023 the left and right tick ladders disagree by
   0.036–0.053 pt — far below the ticks' own 0.336 pt stroke width — and every
   axis candidate is discarded. Frame is `None`, reader refuses.
3. **On the 10 pages where a frame IS found, the reader publishes garbage, not a
   refusal.** This is worse than the issue reports and is the most urgent finding.

## The urgent one: the later pages already ship a fabricated geometric table

`sep-20240612` through `sep-20260617` do not refuse. `find_frame` picks the LOWEST
axis in the merged region, which is the bottom panel's. `read_bins` then takes the
row below that axis with the most alphanumeric tokens — and the page's footnote has
more tokens than the bin-label row. The reader emits this, under the
"counts read from the source" banner that exists to signal geometric authority:

```
| Series | Deﬁnitions | of | variables | and | other | explanations | are | in | the | notes | to | table | 1. |
| June projections | 0 | UNRESOLVED | UNRESOLVED | ... | 0 | UNRESOLVED | UNRESOLVED |
```

The calibration residual on that panel is **211 pt** (the reference's is 0.006 pt),
and it does not gate anything. Some cells still emit a hard `0`, so the zero path
bypasses the residual entirely. Four of the five real panels are silently dropped.

A reader whose whole purpose is to stop fabricated counts is currently fabricating
its own on 10 of 23 pages. Fix 3 (a residual gate, and a bin-row rule that cannot
select a prose sentence) should land regardless of whether 1 and 2 do.

## Per-page measurement

`regions now` / `frame now` / `reader now` are `main@3cbf8a9`. The patched columns
are the two-line prototype described below.

| page | drawings | regions now | region bbox now | frame now | reader now | regions patched | panels patched | refusals patched |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: | :--- |
| `sep-20201216-p09` | 382 | 1 | 0.0, 100.8, 544.3, 712.7 | **none** | 0 (refused) | 5 | 0 | r1, r2, r3, r4, r5 |
| `sep-20210317-p09` | 310 | 1 | 0.0, 99.8, 545.3, 701.2 | **none** | 0 (refused) | 4 | 0 | r1, r2, r3, r4 |
| `sep-20210616-p09` | 311 | 1 | 0.0, 99.8, 545.3, 701.2 | **none** | 0 (refused) | 4 | 3 | r4 |
| `sep-20210922-p09` | 389 | 1 | 0.0, 100.8, 544.3, 712.7 | **none** | 0 (refused) | 5 | 0 | r1, r2, r3, r4, r5 |
| `sep-20211215-p09` | 398 | 1 | 0.0, 100.8, 544.3, 712.7 | **none** | 0 (refused) | 5 | 5 | none |
| `sep-20220316-p09` | 337 | 1 | 0.0, 98.8, 546.3, 702.2 | **none** | 0 (refused) | 4 | 4 | none |
| `sep-20220615-p09` | 325 | 1 | 0.0, 98.8, 546.3, 702.2 | **none** | 0 (refused) | 4 | 0 | r1, r2, r3, r4 |
| `sep-20220921-p09` | 425 | 1 | 0.0, 100.8, 544.3, 712.7 | **none** | 0 (refused) | 5 | 1 | r2, r3, r4, r5 |
| `sep-20221214-p09` | 457 | 1 | 0.0, 100.8, 544.3, 712.7 | **none** | 0 (refused) | 5 | 0 | r1, r2, r3, r4, r5 |
| `sep-20230322-p09` | 382 | 1 | 0.0, 98.8, 546.3, 702.2 | **none** | 0 (refused) | 4 | 4 | none |
| `sep-20230614-p09` | 373 | 1 | 0.0, 98.8, 546.3, 702.2 | **none** | 0 (refused) | 4 | 0 | r1, r2, r3, r4 |
| `sep-20230920-p09` | 472 | 1 | 0.0, 100.8, 544.3, 712.7 | **none** | 0 (refused) | 5 | 5 | none |
| `sep-20231213-p09` | 467 | 1 | 0.0, 100.8, 544.3, 712.7 | **none** | 0 (refused) | 5 | 5 | none |
| `sep-20240320-p09` | 138 | 1 | 0.0, 98.8, 546.3, 702.2 | y=654.7, 40 ticks | 0 (refused) | 4 | 3 | r3 |
| `sep-20240612-p09` | 136 | 1 | 0.0, 98.8, 546.3, 702.2 | y=654.7, 40 ticks | 1 panel(s) | 4 | 4 | none |
| `sep-20240918-p09` | 167 | 1 | 0.0, 100.8, 544.3, 712.7 | y=667.2, 50 ticks | 1 panel(s) | 5 | 5 | none |
| `sep-20241218-p09` | 168 | 1 | 0.0, 100.8, 544.3, 712.7 | y=667.2, 50 ticks | 1 panel(s) | 5 | 4 | r5 |
| `sep-20250319-p09` | 135 | 1 | 0.0, 99.8, 545.3, 701.2 | y=648.0, 40 ticks | 1 panel(s) | 4 | 3 | r4 |
| `sep-20250618-p09` | 136 | 1 | 0.0, 99.8, 545.3, 701.2 | y=648.0, 40 ticks | 1 panel(s) | 4 | 3 | r4 |
| `sep-20250917-p09` | 184 | 1 | 0.0, 72.9, 545.3, 687.6 | y=640.1, 50 ticks | 1 panel(s) | 5 | 4 | r5 |
| `sep-20251210-p09` | 189 | 1 | 0.0, 72.9, 545.3, 687.6 | y=640.1, 50 ticks | 1 panel(s) | 5 | 4 | r5 |
| `sep-20260318-p09` | 144 | 1 | 0.0, 70.9, 547.3, 677.0 | y=627.5, 40 ticks | 1 panel(s) | 4 | 3 | r4 |
| `sep-20260617-p09` | 148 | 1 | 0.0, 70.9, 547.3, 677.0 | y=627.5, 40 ticks | 1 panel(s) | 4 | 3 | r4 |
| `dotplot-p20` | 197 | 5 | 0.0, 91.2, 539.8, 200.9 | y=193.9, 9 ticks | 5 panel(s) | 5 | 5 | none |
## (c) How the pages draw their axes, ticks and bars

Measured on panel 1 of each file via `page.get_drawings()`.

| | SEP (all 23) | reference `dotplot-p20` |
| :--- | :--- | :--- |
| axis / plot rules | stroked `l`, width 0.946 (0.954 from Sep 2025) | stroked `l`, width 0.999 |
| tick marks | stroked `l`, width 0.336–0.469 (0.383–0.472 from Sep 2025), 8.27 pt long, one ladder at each end of the axis | stroked `l`, width 0.38, 9.17–9.32 pt long, one ladder at each end |
| bars | filled `re`, `fill=(0.72, 0.85, 0.97)`, ~24–27 pt wide | filled `re`, `fill=(0.59, 0.72, 0.83)`, 24.54 pt wide |
| prior-survey series | dashed stroked `l`, width 0.469 (0.946 in 2020–21), dash `[1.88 1.88]` | dashed stroked `l`, width 1.5, dash `[4.5 7.5]` |
| legend swatches | filled `re` 5.74 × 5.74 plus an 11.2 pt dashed rule | filled `re` 4.28 × 3.56 |
| **panel wrapper** | **`fs` `re`, `fill=(1,1,1)`, `color=(1,1,1)`, 445.5 × 114.6 pt, one per panel** | **absent** |

No filled rectangle is used as an axis or a tick anywhere in the corpus. The
drawing vocabulary is the same as the reference's; the wrapper rect is the only
structural difference.

## (d) Tick and bin label layout

Both corpora print y tick values as numeric words outside the plot's horizontal
span, which is what `calibrate_y` requires, and both print two-part bin labels
below the axis.

* SEP: two lines, `0.13−` above `0.37`, the first line ending in U+2212 MINUS.
  `read_bins` joins them with `_ATOM_JOIN` to `0.13−-0.37`, a label carrying two
  dashes. Cosmetic, but it will not match a model-authored `0.13-0.37`.
* Reference: three lines, `1.88 ` / `   -` / `2.12 `, joined to `1.88-2.12`.

The reference's bin row therefore carries 13 alphanumeric tokens. Its footnote
carries 13 as well, and `read_bins` keeps the first row on a tie — so the reference
survives defect 3 by coincidence, not by rule. SEP bin rows carry 10 or 12 tokens
and lose outright.

## Why the reference reads and the SEP pages do not

`chart_region_bboxes` clusters raw drawing bboxes with `_union_find_clusters` at a
30 pt gap. On the reference, the vertical gap between one panel's ink and the next's
is 33.3 pt, so five clusters survive. On the SEP pages the same gap is 37.4 pt —
wider — and the panels would separate cleanly **were it not for the wrapper rects**,
whose slabs are 0.4–0.6 pt apart. Measured y-bands on `sep-20201216`:

```
121.7–236.3   236.7–351.3   351.8–466.3   466.8–581.4   581.8–696.4
```

The assumption violated is in `_drawing_bboxes`: it treats every drawing operator as
ink that can join a cluster. A rectangle painted white on white paints nothing. The
SEP generator emits one per panel as a container; the reference's does not.

Once the region is one page-tall box, `find_frame` sees every panel's marks at once,
`region_word_rows` sees every panel's labels at once, and nothing downstream can
recover: the frame is the bottom panel's, the calibration is fitted across five
panels' tick labels, and the bins come from the footnote.

`find_frame`'s second assumption is that two tick ladders which tick the same
heights will report identical floats. Its guard is
`len({tuple(ys) for ys in ladders}) != 1`, over values rounded to three decimals.
Measured left-vs-right disagreement per page:

| pages | max ladder disagreement | tick stroke width | outcome |
| :--- | ---: | ---: | :--- |
| Dec 2020 – Dec 2023 (13) | 0.036 – 0.053 pt | 0.336 pt | every candidate discarded, frame `None` |
| Mar 2024 – Jun 2026 (10) | 0.000 pt | 0.336–0.383 pt | frame found (bottom panel only) |
| reference | 0.000 pt | 0.38 pt | frame found per region |

`Mark.tolerance` — half the stroke width — is already this module's declared
coordinate resolution and is already used for the horizontal/vertical tests. The
ladder test simply does not use it.

## Minimal, threshold-free change

Two edits, both reusing geometry the modules already have. Prototyped by
monkeypatching; the reference's five panels and every one of its cell readings
are **byte-identical before and after** (verified by dict comparison).

**Fix A — `_drawing_bboxes` in `src/socr/tables/reconstruct.py`.** Skip a drawing
whose only item is a `re`, whose fill is white, and whose stroke is absent or white.
It paints nothing, so it is not ink and must not join two clusters. No threshold:
the test is on the page's own declared colours.

**Fix B — `find_frame` in `src/socr/figures/chart_reader.py`.** Two ladders agree
when they have the same number of ticks and no pair differs by more than half the
ladder marks' own stroke width, instead of exact tuple equality. The tolerance is
the page's own `Mark.tolerance`, already defined and documented as "the page's own
coordinate resolution here". No new constant.

Effect across the corpus:

| | regions | pages reading ≥1 panel | panels read |
| :--- | ---: | ---: | ---: |
| `main@3cbf8a9` | 1 per page | 10 (all garbage, see above) | 10 |
| Fix A only | 4 or 5 per page | 10 | 37 real |
| Fix A + Fix B | 4 or 5 per page | 16 | 60 real |
| reference, either | 5 | 1 | 5, unchanged |

The readings that come out are credible. Series totals land on the known FOMC
participant counts without any total being supplied to the reader:

```
sep-20231213   2023=19  2024=19  2025=19  2026=19  Longer run=18
sep-20230322   2023=18  2024=18  2025=18  Longer run=17
sep-20211215   2021=18  2022=18  2023=18  2024=18  Longer run=17
```

### Risk to the reference

None measured. Fix A removes a drawing class the reference does not contain. Fix B
widens an equality test the reference passes at 0.000 pt disagreement. The
before/after readings for all five reference panels compare equal.

### What A and B do not fix

Seven SEP pages still refuse after A+B, and the bottom panel refuses on almost every
page. Both trace to defect 3 and to legend binding, which are separate tickets:

* **Bin row selection.** `read_bins` picks the row below the axis with the most
  alphanumeric tokens. The footnote sentence wins on every SEP page. The bottom
  panel then lands in its own figure group (the group key is the bin labels plus the
  tick values), no legend binds it, and it refuses. This is the same root cause as
  the garbage table above.
* **Legend swatch rejection, Dec 2020 – Dec 2023.** `read_legend` discards a swatch
  that covers any bin's printed label centre. On these pages the legend sits
  directly above a bin label, so both swatches are discarded and the whole figure
  loses its legend. That is why `sep-20201216` still reads nothing.
* **Legend naming window.** A swatch is named by a word-row whose centre lies within
  half the swatch's stroke width of it. For a zero-height dashed rule that window is
  0.23 pt, and the text baseline sits 0.29 pt off. `sep-20210616` binds its dashed
  entry (width 0.946, window 0.473); `sep-20250618` does not (width 0.469, window
  0.234). The dashed series is lost at random on the strength of a stroke width.
* **Dashed staircase reading.** Where the dashed series does bind, it sometimes
  reads all zeros (`sep-20210616`, `sep-20220316`). Not investigated here.

## Reproduction

Scripts used, all read-only and CPU-only: `/tmp/sepdiag.py` (census),
`/tmp/frame_why.py` (per-region refusal cause), `/tmp/ladder.py` (ladder
disagreement), `/tmp/panel1.py` (drawing-operator classification), `/tmp/leg2.py`
(legend candidate trace), `/tmp/proto.py` and `/tmp/final.py` (prototype and
before/after comparison, results in `/tmp/final.json`).
