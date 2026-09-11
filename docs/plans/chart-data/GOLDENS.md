warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead.
# #635 — chart-count goldens for `dotplot-p20.pdf`

**Status: machine-read, awaiting human annotation.**

Read by `socr.figures.chart_reader` (version `635-stage1/1`) from the page's own PDF
vector operators — not from a raster, not from a model. The issue's worked example is NOT
the oracle: its June column sums to 19, and a visual check of the 2018 panel found it
misreads the dashed heights. These numbers replace it.

Every count below is written out with the evidence a human needs to check it without
running any code: the bar's measured top in PDF points, the baseline it stands on, and
where that top falls relative to the two printed ticks either side of it. The y axis
prints a tick every **2** participants, so an odd count sits exactly halfway between two
printed ticks — that is expected, not an error.

To annotate: open the page at high magnification, read each bar against the printed
ticks, and mark each row below `ok` or give the count you read. A disagreement is a bug
in the reader, and the reader's interval is printed so you can see how much room it had.

600-DPI crops of the five panels, rendered for exactly that purpose, are at
`~/Data/socr/fixtures/dotplot/goldens-crops-2026-09-11/panel_{1..5}.png` (outside the
repo; they are regenerable from the fixture).

**Checks already made, and their standing.** Two, neither of them a human annotation:

1. The page's raw drawing operators were decoded BY HAND, before this reader was
   written, and the counts that decode produced are the counts below — panel by panel,
   bin by bin.
2. The rendered crops above were then read visually against the printed ticks, for
   panels 1 (2018), 3 (2020) and 5 (Longer run) — the two most crowded staircases and
   the one whose totals are not 16 — and agreed on every cell.

Both checks were made by the same agent that wrote the reader, so neither is
independent of it in the sense this file needs. The status stays **awaiting human
annotation**.


## Panel 1 — “2018”  (chart region 1, crop `chart_region_p20_1.png`)

- Baseline (0 participants) at y = **193.90**; scale **3.5606 pt per participant** (fitted to 9 labelled ticks, worst residual 0.0064 pt; half a count is 1.780 pt).
- Printed y ticks: 18, 16, 14, 12, 10, 8, 6, 4, 2 at y = 129.80, 136.92, 144.05, 151.17, 158.29, 165.41, 172.53, 179.65, 186.77.
- Printed x bins: 1.88-2.12 … 4.88-5.12 (13 bins).

### September projections — total **16**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.13-2.37 | 4 | 179.66 | 14.24 | [3.944, 4.054] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 179.65), offset +0.01 pt — half a count is 1.78 pt |
| 2.38-2.62 | 12 | 151.17 | 42.73 | [11.946, 12.056] | top sits at 12.00 on the fitted axis; nearest printed tick is **12** (y 151.17), offset -0.00 pt — half a count is 1.78 pt |
| 2.63-2.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.88-3.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.13-3.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.38-3.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.63-3.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.88-4.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.13-4.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

### June projections — total **15**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 2 | 186.77 | 7.13 | [1.790, 2.215] | top sits at 2.00 on the fitted axis; nearest printed tick is **2** (y 186.77), offset -0.00 pt — half a count is 1.78 pt |
| 2.13-2.37 | 5 | 176.09 | 17.81 | [4.790, 5.214] | top sits at 5.00 on the fitted axis; nearest printed tick is **6** (y 172.53), offset +3.56 pt — half a count is 1.78 pt |
| 2.38-2.62 | 7 | 168.97 | 24.93 | [6.789, 7.214] | top sits at 7.00 on the fitted axis; nearest printed tick is **6** (y 172.53), offset -3.56 pt — half a count is 1.78 pt |
| 2.63-2.87 | 1 | 190.34 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 186.77), offset +3.57 pt — half a count is 1.78 pt |
| 2.88-3.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.13-3.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.38-3.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.63-3.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.88-4.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.13-4.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

## Panel 2 — “2019”  (chart region 2, crop `chart_region_p20_2.png`)

- Baseline (0 participants) at y = **305.50**; scale **3.5606 pt per participant** (fitted to 9 labelled ticks, worst residual 0.0064 pt; half a count is 1.780 pt).
- Printed y ticks: 18, 16, 14, 12, 10, 8, 6, 4, 2 at y = 241.40, 248.52, 255.65, 262.77, 269.89, 277.01, 284.13, 291.25, 298.37.
- Printed x bins: 1.88-2.12 … 4.88-5.12 (13 bins).

### September projections — total **16**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.13-2.37 | 1 | 301.94 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 2.38-2.62 | 1 | 301.94 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 2.63-2.87 | 1 | 301.94 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 2.88-3.12 | 4 | 291.26 | 14.24 | [3.944, 4.054] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 291.25), offset +0.01 pt — half a count is 1.78 pt |
| 3.13-3.37 | 4 | 291.26 | 14.24 | [3.944, 4.054] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 291.25), offset +0.01 pt — half a count is 1.78 pt |
| 3.38-3.62 | 4 | 291.26 | 14.24 | [3.944, 4.054] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 291.25), offset +0.01 pt — half a count is 1.78 pt |
| 3.63-3.87 | 1 | 301.94 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 3.88-4.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.13-4.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

### June projections — total **15**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 1 | 301.94 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 2.13-2.37 | 1 | 301.94 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 2.38-2.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.63-2.87 | 1 | 301.94 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 2.88-3.12 | 4 | 291.25 | 14.25 | [3.790, 4.215] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 291.25), offset +0.00 pt — half a count is 1.78 pt |
| 3.13-3.37 | 4 | 291.25 | 14.25 | [3.790, 4.215] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 291.25), offset +0.00 pt — half a count is 1.78 pt |
| 3.38-3.62 | 3 | 294.81 | 10.69 | [2.790, 3.215] | top sits at 3.00 on the fitted axis; nearest printed tick is **4** (y 291.25), offset +3.56 pt — half a count is 1.78 pt |
| 3.63-3.87 | 1 | 301.94 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 298.37), offset +3.57 pt — half a count is 1.78 pt |
| 3.88-4.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.13-4.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

## Panel 3 — “2020”  (chart region 3, crop `chart_region_p20_3.png`)

- Baseline (0 participants) at y = **417.10**; scale **3.5606 pt per participant** (fitted to 9 labelled ticks, worst residual 0.0064 pt; half a count is 1.780 pt).
- Printed y ticks: 18, 16, 14, 12, 10, 8, 6, 4, 2 at y = 353.00, 360.12, 367.25, 374.37, 381.49, 388.61, 395.73, 402.85, 409.97.
- Printed x bins: 1.88-2.12 … 4.88-5.12 (13 bins).

### September projections — total **16**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.13-2.37 | 1 | 413.54 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +3.57 pt — half a count is 1.78 pt |
| 2.38-2.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.63-2.87 | 1 | 413.54 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +3.57 pt — half a count is 1.78 pt |
| 2.88-3.12 | 1 | 413.54 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +3.57 pt — half a count is 1.78 pt |
| 3.13-3.37 | 4 | 402.86 | 14.24 | [3.944, 4.054] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 402.85), offset +0.01 pt — half a count is 1.78 pt |
| 3.38-3.62 | 2 | 409.98 | 7.12 | [1.944, 2.055] | top sits at 2.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +0.01 pt — half a count is 1.78 pt |
| 3.63-3.87 | 6 | 395.74 | 21.36 | [5.944, 6.054] | top sits at 6.00 on the fitted axis; nearest printed tick is **6** (y 395.73), offset +0.01 pt — half a count is 1.78 pt |
| 3.88-4.12 | 1 | 413.54 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +3.57 pt — half a count is 1.78 pt |
| 4.13-4.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

### June projections — total **15**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 1 | 413.54 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +3.57 pt — half a count is 1.78 pt |
| 2.13-2.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.38-2.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.63-2.87 | 2 | 409.97 | 7.13 | [1.790, 2.215] | top sits at 2.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +0.00 pt — half a count is 1.78 pt |
| 2.88-3.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.13-3.37 | 2 | 409.97 | 7.13 | [1.790, 2.215] | top sits at 2.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +0.00 pt — half a count is 1.78 pt |
| 3.38-3.62 | 5 | 399.29 | 17.81 | [4.790, 5.214] | top sits at 5.00 on the fitted axis; nearest printed tick is **6** (y 395.73), offset +3.56 pt — half a count is 1.78 pt |
| 3.63-3.87 | 3 | 406.41 | 10.69 | [2.790, 3.215] | top sits at 3.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset -3.56 pt — half a count is 1.78 pt |
| 3.88-4.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.13-4.37 | 2 | 409.97 | 7.13 | [1.790, 2.215] | top sits at 2.00 on the fitted axis; nearest printed tick is **2** (y 409.97), offset +0.00 pt — half a count is 1.78 pt |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

## Panel 4 — “2021”  (chart region 4, crop `chart_region_p20_4.png`)

- Baseline (0 participants) at y = **528.70**; scale **3.5606 pt per participant** (fitted to 9 labelled ticks, worst residual 0.0064 pt; half a count is 1.780 pt).
- Printed y ticks: 18, 16, 14, 12, 10, 8, 6, 4, 2 at y = 464.60, 471.72, 478.85, 485.97, 493.09, 500.21, 507.33, 514.45, 521.57.
- Printed x bins: 1.88-2.12 … 4.88-5.12 (13 bins).

### September projections — total **16**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.13-2.37 | 1 | 525.14 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 521.57), offset +3.57 pt — half a count is 1.78 pt |
| 2.38-2.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.63-2.87 | 1 | 525.14 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 521.57), offset +3.57 pt — half a count is 1.78 pt |
| 2.88-3.12 | 4 | 514.46 | 14.24 | [3.944, 4.054] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 514.45), offset +0.01 pt — half a count is 1.78 pt |
| 3.13-3.37 | 1 | 525.14 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 521.57), offset +3.57 pt — half a count is 1.78 pt |
| 3.38-3.62 | 5 | 510.90 | 17.80 | [4.944, 5.054] | top sits at 5.00 on the fitted axis; nearest printed tick is **4** (y 514.45), offset -3.55 pt — half a count is 1.78 pt |
| 3.63-3.87 | 2 | 521.58 | 7.12 | [1.944, 2.055] | top sits at 2.00 on the fitted axis; nearest printed tick is **2** (y 521.57), offset +0.01 pt — half a count is 1.78 pt |
| 3.88-4.12 | 1 | 525.14 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 521.57), offset +3.57 pt — half a count is 1.78 pt |
| 4.13-4.37 | 1 | 525.14 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 521.57), offset +3.57 pt — half a count is 1.78 pt |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

### June projections — **unresolved**

no dashed outline of this series is drawn anywhere in the panel, so the series being absent and every bin being zero are not distinguishable from this panel's geometry


## Panel 5 — “Longer run”  (chart region 5, crop `chart_region_p20_5.png`)

- Baseline (0 participants) at y = **640.30**; scale **3.5606 pt per participant** (fitted to 9 labelled ticks, worst residual 0.0064 pt; half a count is 1.780 pt).
- Printed y ticks: 18, 16, 14, 12, 10, 8, 6, 4, 2 at y = 576.20, 583.32, 590.45, 597.57, 604.69, 611.81, 618.93, 626.05, 633.17.
- Printed x bins: 1.88-2.12 … 4.88-5.12 (13 bins).

### September projections — total **15**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.13-2.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.38-2.62 | 3 | 629.62 | 10.68 | [2.944, 3.055] | top sits at 3.00 on the fitted axis; nearest printed tick is **2** (y 633.17), offset -3.55 pt — half a count is 1.78 pt |
| 2.63-2.87 | 4 | 626.06 | 14.24 | [3.944, 4.054] | top sits at 4.00 on the fitted axis; nearest printed tick is **4** (y 626.05), offset +0.01 pt — half a count is 1.78 pt |
| 2.88-3.12 | 6 | 618.94 | 21.36 | [5.944, 6.054] | top sits at 6.00 on the fitted axis; nearest printed tick is **6** (y 618.93), offset +0.01 pt — half a count is 1.78 pt |
| 3.13-3.37 | 1 | 636.74 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 633.17), offset +3.57 pt — half a count is 1.78 pt |
| 3.38-3.62 | 1 | 636.74 | 3.56 | [0.945, 1.055] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 633.17), offset +3.57 pt — half a count is 1.78 pt |
| 3.63-3.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.88-4.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.13-4.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

### June projections — total **14**

| bin | count | top y | height (pt) | interval | evidence |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1.88-2.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 2.13-2.37 | 1 | 636.74 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 633.17), offset +3.57 pt — half a count is 1.78 pt |
| 2.38-2.62 | 1 | 636.74 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 633.17), offset +3.57 pt — half a count is 1.78 pt |
| 2.63-2.87 | 5 | 622.49 | 17.81 | [4.790, 5.214] | top sits at 5.00 on the fitted axis; nearest printed tick is **6** (y 618.93), offset +3.56 pt — half a count is 1.78 pt |
| 2.88-3.12 | 5 | 622.49 | 17.81 | [4.790, 5.214] | top sits at 5.00 on the fitted axis; nearest printed tick is **6** (y 618.93), offset +3.56 pt — half a count is 1.78 pt |
| 3.13-3.37 | 1 | 636.74 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 633.17), offset +3.57 pt — half a count is 1.78 pt |
| 3.38-3.62 | 1 | 636.74 | 3.56 | [0.787, 1.212] | top sits at 1.00 on the fitted axis; nearest printed tick is **2** (y 633.17), offset +3.57 pt — half a count is 1.78 pt |
| 3.63-3.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 3.88-4.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.13-4.37 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.38-4.62 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.63-4.87 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |
| 4.88-5.12 | 0 | — | 0.00 | — | no mark over this bin; the axis is stroked across it |

## What is NOT claimed here

- **Panel 4 (2021) has no June row.** The panel draws no dashed outline at all, so
  “the series is absent from this panel” and “every June bin is zero” cannot be told
  apart from this panel's geometry. The reading records the series as `unresolved` and
  publishes nothing for it. It is never filled with zeros.
- **No total was used to produce any number above.** The reader holds no expected total
  for any survey, series or horizon; the September columns summing to 16 on four panels
  is an observation about this page, not a constraint that shaped the reading.
- **Longer-run totals differ from the dated horizons** (15 September, 14 June) because
  not every participant submits a longer-run value. That is normal and is exactly why a
  universal equal-total rule was rejected.

