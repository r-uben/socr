# 2026-09-14 — #739: one Mark per drawing item, not per drawing

Branch `fix/739-marks-per-item` off `main@9b15379`, rebased onto `ba46be2`.
On every Fed SEP dot-plot panel the
reader resolved the current-meeting series in full and refused the prior-meeting series
outright: 94 panels, 188 series rows, 95 fully resolved, 90 resolving zero cells, 0
partial — all 90 `dashed_stroke`, each refusing with a stated reason (`page_marks` built
one `Mark` per **drawing** from `d["rect"]`, so a dashed staircase emitted as one compound
path arrived as a single mark whose bbox was neither horizontal nor vertical, and
`read_dashed_series` correctly rejected it — the refusal was working, its input wasn't).

## The change

`page_marks` (`src/socr/figures/chart_reader.py`) now emits one `Mark` per drawing
**item**, not per drawing. A new `_item_bbox(item)` builds each mark's bbox from the
item's own points (line endpoints, curve control points, `re`/`qu` rects) instead of the
parent drawing's `rect`. `width` and `dashed` are stroke properties of the parent drawing,
not of any one item, so they're carried down to every mark cut from it. A curve (`c`) item
still reaches the refusal — it is bounded by all four control points, so a curve whose
endpoints happen to be level is still read as a curve, not silently accepted as a run.

A companion fix was needed alongside it: PDF path builders emit a zero-length `'l'` item
as the path's opening moveto restated as a line to itself (every dashed staircase's first
item, corpus-wide). Without dropping it, the reader would report an unread piece of outline
where there is none. `_item_bbox` returns `None` for a degenerate `'l'` (`x0==x1 and
y0==y1`); `page_marks` skips a `None` bbox rather than emitting a zero-area mark.

Nothing else moved: the refusal path is untouched (unreachable on this corpus now, correct
per the brief — a genuinely diagonal or curved piece still refuses; guarded by a synthetic
fixture since no corpus page can exercise it any more), `re` (fill) items behave exactly as
before, and the solid-series readings, calibration, legend binding and frame finder are
unaffected.

## Corpus, before/after (`~/Data/socr/734b-probes/` harness, `sep-dotplots` corpus)

```
                  before        after
bound             37            37
agreed            106           202
unknown           424           308
contradicted      10            30
verified          0             2
demoted           2             2
```

`page_marks`/`read_chart_page` shape moved from 95 fully resolved / 90 zero-resolved / 0
partial (188 rows total) to 185 fully resolved / 0 zero-resolved / 0 partial — the prior-
meeting series stopped refusing across the corpus, as the brief predicted.

## The unpredicted number, and how it was actually settled

Contradicted rose 10 → 30, not predicted by the brief. Reviewed before commit rather than
absorbed.

**First pass, wrong.** I checked the reader's own per-panel per-series totals against the
FOMC's known participant counts (#737's proposal, done here by hand rather than by a
`published`-total hook) and initially argued `sep-20220316-p09`'s dashed (December-2021)
series summing to 18 while the page's own solid (March-2022) series summed to 16 might
just be two different meetings' genuinely different headcounts. The reviewer independently
computed the same totals and initially read the same 18-vs-16 gap the opposite way — as a
constant `+2` overcounting defect (the staircase's opening/closing risers being read as
extra dots) — and blocked the commit pending resolution.

**What actually settled it: the reviewer pulled the Fed's own published per-bin data
table** (federalreserve.gov publishes an accessible HTML table behind every SEP dot-plot
figure) and checked all 20 new contradictions against it cell by cell, not just the panel
totals. All 20 of 20 resolve in the reader's favour; the model is wrong on every one,
including two many-to-one bins where the Fed's longer-run values don't sit on the same
one-eighth-point grid as the near-term ones (e.g. socr bin `2.13-2.37` on
`sep-20220316-p09` r3 must hold both the Fed's `2.125` and `2.250` values; the reader
returned the exact sum, 6, matching `5+1`) — the binning is right as well as the counting,
in the one place getting it wrong would have been invisible against a total alone. One
model error (`sep-20220316-p09` r3, bin `3.38-3.62`) is a participant invented at a bin the
Fed shows nothing in — not a miscount but a fabrication; one on `sep-20201216-p09` (r3,
bins `0.38-0.62`/`0.63-0.87`) is a matched pair, the same one dot assigned to the wrong
bin on both sides — an error class no panel total, correct or not, could ever have caught.

**Why the panel-total argument that both sides used first was weaker than it looked, and
still the right instinct to reach for before per-bin data was in hand:** a total can be
right while its distribution is wrong (or vice versa — a riser-splitting bug really would
add a constant to a total, indistinguishable at that resolution from a genuine headcount
difference). The reviewer's initial control was itself wrong twice over: it compared the
dashed (prior-meeting) series against the CURRENT meeting's participant count — the dashed
series carries the PRIOR meeting's headcount (December 2021's 18, not March 2022's 16, on
this page) — and it treated the longer-run panels reading one below the near-term
headcount as a pre-existing anomaly to preserve, when the Fed's own footnote says a
participant not submitting a longer-run projection is normal: 16-of-17 (`sep-20201216-p09`)
and 15-of-16 (`sep-20220316-p09` solid) are both simply correct, not off-by-one.

**Takeaway for the next person reading a panel total:** check it against the PRIOR
meeting's participant count for a dashed/prior series, not the page's own current-meeting
count — and prefer the Fed's own per-bin table over any total, panel or corpus-wide, the
moment one is available. Filed as #747 (score the whole corpus against these tables, reader
and model both, per cell) rather than left as a one-off argument on this ticket.

## Tests

Four new tests in `tests/test_gh635_chart_reader.py`, each mutation-verified in a copy
outside the repo (`src` + `tests`, `pyproject.toml`) against all four checks the repo rule
requires — loaded (`socr.__file__` resolves inside the mutant), applied (uncapped
`src.count(anchor) == 1`, aborts on the assertion rather than a capped-then-asserted
tautology), load-bearing (each mutant is shown to change the suite's outcome, not just its
own line count), right suite (`tests/test_gh635_chart_reader.py`, the suite that owns
`page_marks`/`read_dashed_series`) — red on the mutant, green on the fix, for all four:

- `test_page_marks_emits_one_mark_per_item_not_per_drawing`
- `test_a_compound_staircase_path_now_resolves`
- `test_a_curve_inside_a_compound_path_still_refuses`
- `test_a_zero_length_path_stub_is_not_a_mark`

`tests/test_gh734b_wired_grid_reconciliation.py`'s
`test_the_sep_corpus_reproduces_the_measured_shape` pinned the pre-#739 corpus shape,
explicitly gated on this ticket landing. Updated to the post-fix figures above, with the
docstring explaining why the numbers moved (not just what they moved to) — a further move
needs the same kind of stated reason this one gives.

## Gates

Full suite: **5345 passed, 4 xfailed** (`~/venvs/socr/bin/pytest`, `PYTHONPATH=$PWD/src`).
Format gate: `uvx ruff@0.16.0 format --check .` clean, 664 files.

## Not in scope

The synthetic fixture for the refusal path (unreachable on this corpus now) is exercised by
the new tests' hand-built `fitz.Shape` fixtures rather than a standalone golden PDF; no
further guard needed beyond those four. #747 (score reader and model against the Fed's
per-bin tables, corpus-wide) is filed, not built here — this ticket's per-cell check against
the Fed table was done by hand for the 20 contradictions in question, not wired into any
gate.
