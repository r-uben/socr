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

## Round 2: the PR #1195 review

Two holes on the wrong-number path, both real.

**A lost cell was indistinguishable from a column the release never prints.**
The self-check only compared neighbouring groups, so a missing central cell
left the range with nothing above it: on head `c9e38d8`, GDP 2010 shipped
`(4.1, 4.6)` on a row whose own median is 3.5, with `rejected` empty, and a row
carrying only the range group shipped every year unchecked. Two rules now: the
range is checked against the median DIRECTLY, and a year that prints a range
while a group the table declares is absent is refused by name
(`REJECTED_MISSING_GROUP`). The distinction holds because a column these
releases do not print is blank in EVERY group — longer-run core PCE prints no
median, no central tendency and no range — so a gap in one group alone is a
cell this parser lost. The blank-column case still costs only its own key: the
three releases still yield 30, 28 and 43 published ranges, and 2015-09-17
still scores 43/43 against the Fed's published table.

**A rejected row lent its name to the sub-row below it.** The reset existed and
was commented, but no test failed when it was deleted: the reviewer's four-row
fixture then returned GDP's `(2.3, 4.0)` for
`range_for("Unemployment rate", "prior", "2010")` — one variable's figures
under another's key. A sub-row names a meeting, not a variable, so an orphan
one is now refused by name (`REJECTED_ORPHAN_PRIOR_ROW`) rather than falling
back on the last variable that survived or on its own printed label, which
would invent a variable called "january projection".

Also closed from that review: the median-outside-central check, the
repeated-sequence column check, the one-group-word-per-block rule, the trailing
column bound (the "Federal Open Market Committee" printed down the edge of
2015-09-17 p13), the pitch coming from the LAST repetition (2015's 26.4pt
median columns against its 41.4pt range columns), and the first-data-band
on-grid rule were each in the code and in no test; each now has one. Holes are
bound like figures instead of being filtered out before binding, so a
misplaced `n/a` is refused rather than dropped — the branch that read them was
dead code. A figure can no longer name a column. The module docstring
described the parser it replaced.

**The real geometry is now in the repository.** `tests/fixtures/gh1184/` holds
the frozen word boxes of the three pages, so the suite drives the real layouts
with no PDF: 30, 28 and 43 rows, the 2015 figures checked against the Fed's own
published table carried as literal data, and a difference test that deletes one
printed figure from a real cell and requires that row to be refused.

19 mutants, all killed by their named test. One of them survived its first run
and the mutation was at fault, not the test: it added a self-assignment instead
of deleting the reset, and the anchor was not unique between the two reset
sites. Suite and ruff results are in the PR.
## PR 2 census: 1 panel, then 43

The first PR 2 census reported 1 resolved panel in 15 regions and was wrong. It
took its page set from the `chart_region` marks in the `fed-repilot-1016`
output, but that run is dated 2026-10-16 11:08 and #1190 — the routing that
sends these pages to the chart floor — merged 2026-10-17 11:25, so in that run
the very pages the floor rescues still carry the 105-byte whole-page marker and
no mark at all. Scoping to the marks excluded exactly the pages PR 2 is for.
Taking the page set from what the floor will fire on after #1190 gives **43
panels over 13 pages in 3 documents, 56 series carrying counts**, and the
reader reproduces #1155's figures exactly on #1155's pages (2015-09-17 p17 5/5,
p18 5/5, p20 4/4; 2010-04-28 p15 3/4, p18 2/3; every 2015 series sums to 17).
Probe: `scratch/routing-baseline/leads/1184/probes/pr2_reconcile.py`,
`pr2_census2.py`.

## PR 2: the counts turned on

A chart panel's counts ship only when BOTH hold, and the two are separate
conditions: the reading is complete — every bin of that series resolved — and
its filled bins span the range this document's own projection table publishes
for that variable, that meeting and that year.

Completeness is not implied by the gate and had to be named
(`REQUIRE_COMPLETE_READING`, orchestrator ruling 2026-10-17). The gate compares
only the lowest and highest bars it can see, so a bar the reader could not
measure in between passes it unnoticed: measured on the Fed pilot, **5 of the
35 series that pass the gate have unread bars**. A histogram shipped with bars
missing reads as complete to whoever cites it, and silent loss ranks below a
missing number. Requiring completeness ships 30 series rather than 35.

Judged per SERIES, not per panel. A panel draws the current meeting's bars and
the previous meeting's in the same axes but each is checked against its own row
of the table, so a bar unread in one says nothing about the other. Built
per-panel first, it shipped 26; per-series ships 30. A panel whose series
disagree ships the good one and SAYS which other series it withheld and why —
shipping the good one silently would let a reader take the panel for whole.

Binding a panel to a table row needs the variable, which only the figure's
caption names. The grammar is identical across the three releases
("Figure 2.A. Distribution of participants' projections for the change in real
GDP, 2008-11"), but the policy-path caption names a variable twice and wraps
mid-phrase, which read line by line yields "federal" and binds to nothing. The
caption is joined until a comma closes the phrase and the LAST "for the ..."
before that comma is taken. Validated on all 13 figure pages of the three
releases: 13 bind, 0 miss.

The witness is found by offering every page of the document to the parser and
taking the first that yields rows; measured over the pilot, exactly one page per
document is accepted. It is looked up inside the lazy builder, not beside the
chart reading, so a page that never reaches the chart floor pays nothing for it.

**Census on the Fed pilot**: 19 panels ship counts and 24 withhold, 30 series
in all. The withheld name their own obstacle — bars unread (the commonest), a
reading that contradicts the published range (2), or no published range for
that year. A further 8 regions never resolve to a panel at all, for want of a
legend binding series names to drawing styles on 2015-09-17 p19 and p22; that
is the chart reader's, filed separately.

29 mutants, all killed by their named test. Suite 8,078 passed, 0 failed;
`ruff format --check` clean.

## PR 2 round 2: the #1199 review

**A hole stopped counting as a group.** PR 2 recorded a bound `n.a.` as a group
that was PRESENT, which reopened #1195's blocker 1: a year whose median and
central tendency print `n.a.` shipped its range with nothing to check it
against, and the test added with it asserted exactly that weaker behaviour.
Only a group carrying a VALUE counts as present now. The whole-column fixture
replaces it — every group of that year printing `n.a.`, which is how these
releases actually print a column they do not collect — and it carries both
rules at once, since the hole is the row's first cell and so also pins the
label cut (#1198).

**Counts could ship under a line saying they had not.** The count tables were
collected from the decisions by index while only one branch of
`region_clause_records` consulted a decision, so a region whose picture failed
to render got "values not shipped" and a count table underneath it — with no
"read from the drawing, not verified" sentence anywhere, because that sentence
lives in the line that was replaced. Grok reproduced it on 2008-10-29 p16 by
pointing `figures_dir` at a file. A block now ships only for a region whose own
RECORD carries the shipped reason, so the page's line and its numbers come from
one place and cannot drift; a test asserts the count of tables equals the count
of records saying they shipped, over every arrangement of pictures.

**The wiring had no test at the real caller.** Emptying `decisions`, dropping
it from either call, and reading the caption off the wrong page all survived
the full suite. A two-page document — a `build_chart` histogram captioned for
the change in real GDP, then a `_side_by_side` Table 1 — now drives
`chart_floor_replacement` and asserts the counts AND the mark together.

**Grok's P1, the axis ticks.** A re-select carrying only the native winner
replaced 2008-10-29 p16 with its raw axis ticks. Every branch of
`structure_class_floor_text` that cannot use the chart floor returns the
whole-page MARKER, not native text, so the construction does not correspond to
a path this code takes; that is argued, not proved, so the invariant is pinned
instead for all three ways the floor can be unavailable (no builder, a builder
that declines, a builder that raises): the page ships the marker and no numeral
drawn inside the chart. Also recorded from Grok: the floor's output is status
error with `audit_passed` false, so **the ledger never keeps these pages** and
every run re-does them.

**Smaller, all from the review.** "No projection table in this document" and
"this caption names no variable" are now different statements, and both are
UNKNOWN rather than FAIL — FAIL is reserved for a panel whose counts were
checked and refused, which is only the gate. So are unread bars and a year the
table does not publish. The counts header names the region and the variable, so
two panels sharing a year label can be told apart. The witness is cached per
document (keyed on path, size and mtime) and skipped entirely when the caption
names no variable, which was 0.51s per 100 dense pages paid on every
chart-floor page and every re-finalize. The exception fallback says the witness
lookup failed instead of claiming the counts are not wired yet.

**The census, reconciled.** The PR body and the previous entry disagreed because
they counted different things. One ordering now, the obstacle each page's line
actually STATES, which is the order `decide_panel` tests in: no table, no
variable, bars unread, no published range, counts outside the range. On the
three SEP releases: 55 chart regions, 43 resolving to a panel, 19 panels
shipping, **30 of 59 series shipping counts** (2015: 26, 2008: 4, 2010: 0).
Withheld: 23 for unread bars, 3 for counts outside the published range, 3 for a
year the table does not publish. Of the 23, **18 would also fail the gate** —
their line says the bars because that is tested first, and the earlier "17" was
that overlap counted as gate failures. The 12 regions that never resolve are
the chart reader's, not this gate's, and are filed separately.
