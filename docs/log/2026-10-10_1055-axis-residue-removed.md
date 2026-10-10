# #1055 — chart-lane number hiding removed (reverses GH-369's fence)

**Decision.** The chart-asset lane ships the native text layer whole and in order. The
GH-369 fence (`split_chart_axis_residue` / `fence_chart_axis_residue`) is deleted.

**Why.** The fence moved every run of two or more bare-number lines into a hidden comment
labelled "axis tick labels ... not data values". It judged text only, and a PDF emits table
cells one per line too. Reproducer: Aruoba & Drechsel (NBER 2024) p29 has no chart, only a
3x2 table of correlations; all six values were hidden under SUCCESS. Corpus run (Rorqual
array 22889958, socr main @ c4937472, 395 papers): 1,503 pages on the chart lane, 6,337
numbers hidden on 321 of them.

**Considered and rejected: geometry-aware hiding** (hide only numbers inside a chart box).
Fable and Astra (gpt-6-astra) reviewed it independently and both said skip it:
the lane's native text carries no positions; `has_chart_marks` returns a bool and the box
function that exists (`chart_region_bboxes`) uses a different gate and extends boxes to the
page edge across the chart's height, so a table beside a chart is still hidden; and a number
inside a chart box can be a data label or an inset table cell.

**Cost accepted.** Axis tick labels show in the body again (the GH-369 symptom). That is
clutter; a hidden data value is loss.

**Do NOT** reinstate a text-only or box-containment number filter on this lane. The real fix
is upstream, tracked under #1053: no-chart pages must not enter the lane (boxed tables pass
`_has_framed_data_cluster`; `_page_has_tables` misses two-column tables via
`_MIN_LANES_PER_ROW = 3`; the GH-994 suspect did not fire), and figures are cropped one by
one so axis text only lives inside a figure crop.

**Guard.** `tests/test_chart_lane.py::TestAgenticChartLaneRouting::test_native_text_ships_whole_and_in_order`.
Mutant (origin/main extractor + orchestrator, new test, copy outside the repo with
`socr.__file__` asserted inside it): `axis-scale` and `table-cells` fail; `lone-year` and
`no-numbers` pass, as the old code never fenced those. Full suite on the branch: 7,083 passed.
