# #1184: a reusable Table 1 witness parser (decision log)

## Diagnosis

`parse_table1_witness` counted a row's values against the header's column total.
A count cannot tell a cell the table never printed from one the parser lost, so
it rejected both: 2010-04-28's core PCE rows (longer-run core PCE is not
collected — that release's own footnote 3), and every row of 2008-10-29 (its
groups are stacked, so a row legitimately carries half the columns). 2015-09-17
is drawn at 90°, which nothing read.

## What changed and why

`src/socr/figures/chart_witness.py`: the parser is now geometric. Columns come
from the table's own header; each value binds to one column or the row is
refused by name. Groups are placed by where their headings sit — beside the
columns when the year sequence repeats, stacked blocks when it does not. Cells
are three consecutive tokens, so a range cannot be paired across an empty
column. Labels are taken beside the first value, not at the start of the band.
Tolerance is a quarter of a cell's own lo-to-hi span: derived, and measured to
sit 5× inside a real cell's deviation and 2× outside a dropped figure's shift.

Per the orchestrator's ruling the count self-check is replaced by per-column
binding, with missing and extra both refused by name. A blank COLUMN stays a
hole (flagged and unanswered); a lost figure is refused.

## Measured

| release | layout | before | after |
|---|---|---|---|
| 2010-04-28 p11 | side-by-side | 24 ranges, 2 rows rejected | **30**, 0 rejected |
| 2008-10-29 p12 | stacked blocks | 0 (refused) | **28**, 1 prose band refused |
| 2015-09-17 p13 | rotated 90° | 0 (refused) | **43**, 0 rejected |

2015 scored against the Fed's own published table: **43/43 exact, 0 wrong, 0
missing, 0 spurious**. (`fomcprojtabl20150917.htm`; #1155's URL 404s — the live
one omits `/files/`.) 2008 and 2010 have no standalone published table, so they
were cross-checked against an independent line reading: 2010 agrees 30/30, 2008
25/25, zero disagreements — and the 5 cells the LINE reader lost are the
prose-contaminated bands, which is the defect the geometry removes.

Pilot census (`fed-repilot-1016`): 11 chart-floor pages in 9 documents. The 3
modern SEP releases now all yield a witness, 101 published ranges against 24
before. The 6 pre-1970 documents carry no projection table and refuse by name.
Exactly one page per document was accepted — no false positives.

## Tests and mutants

`tests/test_gh1184_table1_layouts.py`, 18 tests on grid-drawn fixtures. Each
changes one thing and pins the difference: stacked vs side-by-side and upright
vs rotated both yield identical witnesses; a collapsed range reads the same as a
spelled one; a blank column costs only its own key; a lost figure, an off-grid
extra, a displaced cell and a second cell in one column are each refused by
name; prose arriving in a row's band moves nothing.

Seven mutants, external copy, `socr.__file__` asserted inside it, uncapped
anchor-count assertion before each edit: **all seven killed by their named
test**. Two needed repair first — the tolerance mutant SURVIVED the first run
(the containment half of the binding rule was doing all the work), which is why
`test_a_whole_cell_printed_off_its_column_is_REFUSED_by_name` exists; and the
separator mutant was killed by the layout tests, not the one I had named, so the
naming was corrected rather than the result.

`tests/test_gh1155_chart_witness.py`: its 7 parser tests used fixtures that draw
each row as one run of text, which is not a grid and which no real Table 1 is.
Their assertions were moved to the #1184 file on grid fixtures; one test remains
in place pinning that an ungridded line now refuses. Nothing was dropped.

**Suite: 8010 passed, 0 failed, 2 skipped, 4 xfailed** (494s). **ruff 0.16.0
format --check: 969 files already formatted.** Nothing committed.

## Argon

Consulted before building (`argon-design.md`). It falsified the first design's
core mechanism: the grid was clustered out of the data, so every row seeded it
and the tolerance accepted whatever formed it. Rebuilt unsupervised that grid
gives a 38.01pt within-column spread against a 36.49pt gap on 2008 — no grid at
all. Columns now come from the header. It also killed the nearest-group-header
rule (measured wrong on 2015: 89.65 vs 95.55), found the row-label theft on
2008's shared bands, and caught two measurement errors of mine (token clusters
quoted as cell clusters; an edge gap conflated with a centre distance). I
rejected one finding: the orphan `inflation4` band cannot rebind a variable,
because only a band with bound cells sets one.

## Not measured

- Panel ship counts. Wiring the gate to an output path is PR 2; the census above
  measures witness availability, not shipped numbers.
- A dash printed as an empty cell's placeholder would be paired as a range by
  the consecutive-token rule. Not present on any of the three releases and no
  corpus rate exists, so no machinery was added; the fix if it appears is to
  bound a cell's span by its own column's pitch.
- Pre-2007 fractional projections, horizon headers that are not years, and
  180°/270° content: refused, not handled.
- The suite baseline on the base commit was not re-measured; only the
  post-change total above.

DONE
