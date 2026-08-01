# TICKET-B2 design — 2026-08-01

Read-only design pass on the merged TICKET-B2 (`feat/123-metric-blind-spots` @ `16ba93e`),
after the first attempt (`ad649b5`) was reverted by `717914d`. No source changed.

**Headline: B4 must merge into B2.** Not a preference — every B2-alone comparison contract
I could construct manufactures a *new* wrong-direction accept in `escalation_decision`,
measured below on the plan's own leading-gap fixture. The seam between "lanes exist" and
"lanes are compared" is the same class of seam that killed attempt 1.

---

## 1. What exists today

### The scorer

| fact | location |
|------|----------|
| markdown side compacts empty cells away | `benchmark/table_exactness.py:114` — `values = [c for c in raw_values if _is_value(c)]` |
| GT side is a flat x-ordered list, no gaps | `tables/native_rows.py:237` |
| comparison is by ordinal position among non-empty cells | `table_exactness.py:171-178` |
| `cells` is ground-truth-only | `table_exactness.py:160` — `report.cells += len(want.values)` |
| out-of-range prediction cells read as `""` | `table_exactness.py:172` |
| B1's grid gate | `table_exactness.py:211` → `native_rows.rows_establish_grid:139` |

`cells` being GT-only is load-bearing: it is why `incumbent.cells == candidate.cells`
and why `candidate.exact > incumbent.exact` is a comparison on one denominator.

### The GT parser

- `_rows_in_region` (`native_rows.py:167`) reads `page.get_text("words")` as
  `(x0, y0, x1, y1, text)` at line 186. **`x1` and the centre are both available**; line 237
  keeps only `(x0, text)` and `_drop_footnote_markers` then discards even that.
  → ticket item 5 is implementable with a one-tuple widening. Verified empirically:
  on a right-aligned fixture with varying digit counts, right edges were exactly
  `340.00 / 400.00` (zero variance) while left edges spread `317-327` and `384-387`.
- The label boundary is the **last non-numeric word** (line 219-240), and it already
  survives the landmine: `"Growth Plan after 17 October reversals"` parses with the `17`
  inside the label (x0≈132, far left of the value lanes at 317-400). Confirmed against the
  parser. The ticket's decoupling instruction is therefore satisfiable as written.
- **Stale docstring, worth fixing here.** `_rows_in_region`'s docstring point 2 claims the
  boundary is found "by clustering the x-positions of numeric tokens … (reusing the native
  verifier's lane clusterer)". The code does no such thing — that is the approach that
  *failed*. A reader of exactly this area will be misled.
- `LabeledRow.values` is consumed as plain strings by `escalation_canary.py:78` and by
  `rows_establish_grid` (`len(r.values)`). Changing its element type breaks both.

### Facts the tickets assume that are not true

1. **B4's "per table (per panel)" is not available.**
   `BenchmarkScorer._markdown_table_cells` (`benchmark/scorer.py:443`) flattens *every*
   pipe-table on the page into one grid with no table boundaries, and rows may have
   differing widths. Meanwhile GT lanes are naturally per-region (`native_rows_from_page`
   loops over `locate_tables` boxes and concatenates). There is no table segmentation on
   either side. **Page-level is the only granularity the code supports today.**
2. **The existing lane clusterer cannot be reused.** `native_verifier._cluster_x_positions`
   is driven by `_LANE_X_TOL_PT`, a fixed point-size constant. Importing it imports a magic
   threshold into the metric, which this plan forbids.
3. `orchestrator.py:1469` uses `report.pct is not None and report.pct < 100.0` as the
   escalation **trigger**, not just the gate. See §5 — this is an uncosted side effect.

---

## 2. The comparison contract — options

Notation: `L` = ground-truth lane count, `M` = emitted column count.

### Option A — positional when `M == L`, compacted (today) otherwise · **REJECT**

The natural "honest interim". It is not honest. Simulated against the plan's own
`leading_gap` fixture (right-edge clustering, parameter-free max-gap split, `≥2`-row lane
support; no source touched):

| emitted markdown | Option A | today |
|------------------|----------|-------|
| faithful, width 2 | 5/5 | 5/5 |
| **shifted**, width 2 | **4/5** | 5/5 |
| **same shift + one spurious empty column**, width 3 | **5/5** | 5/5 |

Today those two score 5 and 5 → tie → `escalation_decision` **rejects**. Under Option A they
score 4 and 5 → `candidate.exact > incumbent.exact` → **accepts a candidate carrying the
identical column shift plus a spurious column.** A brand-new wrong-direction accept inside
the production gate, created by the ticket meant to remove one.

Root cause, and the rule the whole design should hang on:

> **Strictness must be a function of the ground truth alone.** Anything that makes the
> comparison stricter or looser according to the *prediction's* shape scores incumbent and
> candidate under different rules, and `exact > exact` stops being a comparison.

Option A gates on `M`. So does every variant of it (compact-on-mismatch, miss-on-mismatch,
not-scorable-on-mismatch). They differ only in which direction they break.

### Option B — total index equality, lane `k` vs column `k` · **REJECT**

Symmetric, deterministic, no gate. But at the documented OBR page 53 shape (L=14, M=6) every
value in lanes 6-13 compares against `""` and lanes 0-5 compare against collapsed columns:
both engines crater to near-zero, ties everywhere, and escalation goes blind on exactly the
class of table it exists for. Plus coincidental alignments produce noise accepts.

### Option C — B2 + B4 together: lanes + positional markdown + one global monotone map · **RECOMMEND**

One monotone, injective map from lanes to emitted columns per page, chosen to maximise total
cell agreement across all matched rows. Same simulation, exhaustive over admissible maps:

| emitted markdown | exact/cells |
|------------------|-------------|
| faithful (w=2) | **5/5 → 100%** |
| shifted (w=2) | 4/5 |
| shifted + spurious column (w=3) | **4/5** — no longer beats the incumbent |
| faithful + spurious column (w=3) | 5/5 — width-invariant, so L≠M does not crater |
| dropped column, "sloppy" (w=2) | 4/5 |

All four required invariants hold, and the Option A asymmetry is gone: the map has to explain
*every* row with one assignment, so a spurious column buys nothing.

Cost: Needleman-Wunsch over ≤ ~20 columns, per page, pure Python, model-free — negligible
next to the OCR call it gates. Determinism/replay: fully deterministic given a fixed tie-break
(ties yield identical `exact` by construction; only `misses` attribution differs — pin the
tie-break anyway).

**Residual risk, stated plainly:** the map is fitted to the prediction being scored, and a
prediction with more columns has strictly more admissible maps (`C(M, L)` of them). The
simulation shows this is not exploitable at M=3, L=2, but the freedom grows with `M`. This is
B4's own documented uniform-shift limitation in a sharper form, and it is consilium question 1.

---

## 3. Recommendation

**Merge B4 into B2.** Yes, that is the second seam in this plan to fail — the orchestrator
should hear it now. The reason is structural, not incidental: *the comparison contract between
an L-lane space and an M-column space **is** the alignment.* There is no state in which lanes
exist but are not compared that does not either crater legitimate L≠M pages (Option B) or vary
strictness with the prediction (Option A). B4's remaining content after the merge is real but
small — the two-block OBR shift test, the module-docstring limitation, and the
unexplained-lane count C1 consumes — and can stay a ticket.

### Concrete shape (lowest blast radius)

- Keep `LabeledRow.values: tuple[str, ...]` **compacted on both sides**, add a parallel
  `lanes: tuple[int, ...]` (GT) / `columns: tuple[int, ...]` (markdown), defaulting to `()`.
  → `escalation_canary.py:78` untouched, `rows_establish_grid` untouched, B1's gate untouched,
  `cells` numerically unchanged. Attempt 1 changed `values` itself; that is what made the two
  sides disagree about what a tuple meant.
- Lane construction: cluster right edges (`x1`); build the partition under `x1` and under the
  centre, pick the one with lower total within-lane dispersion, tie → right edge. This is a
  **per-partition** choice, not the ticket's per-lane choice — per-lane is chicken-and-egg
  (you need the lane to measure its variance). Call the deviation out in the commit.
- Splitting rule, parameter-free: sort the anchor positions, sort the consecutive gaps, cut at
  the largest *ratio jump* in the sorted gap list. Derived from the page's own geometry, no pt
  constant. (This is what the simulations above used; it found 2 lanes correctly, and left the
  in-label `17` unclustered because it never reaches the clusterer.)
- Lane space is **per page**, not per region — see §1, fact 1.

---

## 4. Lane-ambiguous rows — "conservatively" defined

B1 already left two not-scorable contracts (`pct=None, gt_rows=0` for the grid gate;
`pct=0.0, scorable=False` for no-label-matched). **Do not add a third.**

A GT value fails the `≥2`-row lane-support rule only when it is a lone numeric right of the
label boundary. Contract:

- it **stays in `cells`** (no denominator change, no silent loss);
- it is **excluded from the alignment** (it must not consume a column slot in the injective
  map — that is precisely the old "stray numeral drags the boundary" failure);
- it is credited on **presence within its own row's unmapped columns** — i.e. that single cell
  degrades to today's semantics. Positional credit is *withheld*, never guessed;
- it increments a counter (`lane_ambiguous_cells`) for C1 to surface.

Why this and not "count it a miss": weakness is a property of the ground truth, so the
degradation applies identically to incumbent and candidate (§2's rule), and a faithful
transcription can still reach 100% — which `test_a_perfect_transcription_scores_100` requires.
No new ceiling, no new `scorable` variant.

---

## 5. Blast radius on the production gate

**`cells` — unchanged, and must be.** Hard invariant, worth its own test:
`cells == number of ground-truth values`. Requires (i) lanes annotate, never filter;
(ii) blank lanes are **not** counted as cells (counting them would inflate every denominator,
credit "correctly empty", and invalidate every historical comparison); (iii) weak-lane cells
stay in. `escalation_decision.py:116`'s `of {incumbent.cells} cells` stays accurate.

**`exact` — moves in both directions.**
- *Down* where a column shift was previously invisible (the intended fix).
- *Up* where a leading gap previously misaligned an otherwise-correct row: GT value in lane 1
  only, markdown `['x', a]` → today compares `a` against `x` and misses; under the map it
  matches. This is the mechanism by which the leading-gap invariant passes at all.

So **historical escalation accepts flip both ways.** STATUS.md already names this as the real
acceptance test: re-score `~/data/fiscal-ballast/_experiments/2026-07-31_gh96-engine-parity/`
and `2026-08-01_gh96-corpus-rerun/` before and after. Do not re-run OCR.

**The accept rule's meaning shifts — flag it loudly.** The form
`candidate.exact > incumbent.exact` is unchanged, but `exact` now means *right value in the
right lane* rather than *right value at the right ordinal position among non-empty cells*.
Consequence: the calibration table in `escalation_decision.py`'s module docstring
(45.0 / 81.7 / **85.0**) was measured under the old meaning and becomes stale on the day this
lands. Either re-measure from the preserved runs or annotate it; do not leave it standing
unqualified.

**Uncosted side effect nobody has raised.** `orchestrator.py:1469` triggers escalation on
`pct < 100.0`. A stricter metric puts *more* pages below 100 → **more escalations fire → more
second-engine calls**. Direction is knowable from the preserved runs; measure it in the same
re-score pass rather than discovering it as a bill.

---

## 6. Invariants checked on paper (and in simulation)

| invariant | under Option C | note |
|-----------|----------------|------|
| `test_a_perfect_transcription_scores_100` | ✅ 5/5 | leading gap, the case that broke attempt 1 |
| `test_dropping_a_column_never_beats_keeping_the_gap` | ✅ 5 vs 4 | strict, not just the `>=` the test asserts |
| A1 `shift_into_adjacent_empty_cell` (strict xfail → pass) | ✅ | trailing-gap fixture, L=M=2, map forced to identity → the shift is a miss |
| `test_wrapped_label_is_scored_the_same_as_unwrapped` still xfailed | ✅ | B2 touches no label boundary; B5 unaffected |
| benign transforms unchanged | ✅ | all operate on labels/emphasis/whitespace, none on column shape |

Both invariants also hold under Option A *on the fixtures as written* — Option A fails only on
the incumbent-vs-candidate asymmetry, which **no current test covers**. Add one:
a candidate that is a same-shift-plus-spurious-column variant of the incumbent must not be
accepted. Without it, Option A would land green.

---

## 7. Acceptance-criteria readiness

The merged B2 done-when list is implementable **except**:

1. *"asserts the numeral does not form a lane and the sparse row's value snaps to the correct
   lane index"* — needs an observable lane index. Satisfied by the parallel `lanes` field
   above; the criterion should name it so the implementer does not invent a private API.
2. **Missing criterion — add.** The escalation-asymmetry test from §6. It is the only thing
   standing between this design and Option A landing green.
3. **Missing criterion — add.** `cells == number of ground-truth values`, pinned as a test.
   It is the invariant that keeps `exact > exact` a single-denominator comparison.
4. **Missing criterion — add.** The before/after re-score of the two preserved runs, with the
   flip count recorded. STATUS.md calls this "the real acceptance test" but B2's done-when
   does not mention it.
5. B4's *"per table (per panel)"* wording must be relaxed to **per page** in whatever ticket
   carries the alignment — per-panel is not implementable against
   `_markdown_table_cells`'s flattened grid.

---

## 8. Consilium questions

**Q1 (the one I actually want answered — genuinely open).**
A table metric compares ground-truth values carrying anonymous column-lane indices (L lanes)
against an OCR engine's emitted markdown columns (M columns, where L≠M is legitimate and
common), by choosing one global monotone injective map from lanes to columns that maximises
total cell agreement — and the resulting exact-cell count is then used as a production accept
rule, `accept candidate iff candidate.exact > incumbent.exact`, with each side scored
independently against the same ground truth. Because the map is fitted to the prediction being
scored, a prediction with more columns has strictly more admissible maps (C(M,L)); does this
hand systematically wider outputs an advantage that grows with M, and which of these
constant-free corrections is right?
- (a) Accept the freedom and document it as the known uniform-shift limitation.
- (b) Penalise the candidate for unmapped columns that contain values (i.e. columns the map
  could not explain), as a symmetric counterweight to the extra freedom.
- (c) Score both sides under the single map fitted to the *incumbent*, so the candidate is
  never given more freedom than the thing it must beat.

**Q2 (confirm or falsify; I believe the evidence settles it).**
A plan splits one metric change in two: ticket B2 attaches column-lane indices to ground-truth
values and column positions to emitted markdown cells, and ticket B4 later adds the global
monotone map between the two spaces. Measured on a three-row fixture, the only available
interim rule for B2-alone — compare by index when the emitted column count equals the lane
count, fall back to today's compacted comparison otherwise — makes a faithful transcription
score 5/5, a column-shifted one 4/5, and *the same shift plus one spurious empty column* 5/5,
so a production rule of the form `accept iff candidate.exact > incumbent.exact` newly accepts
the shifted-plus-spurious candidate over the shifted incumbent (today both score 5 and tie to
reject). Is there any comparison contract that delivers B2's benefit without B4's map and
without making the metric's strictness a function of the prediction's column count — or must
the two tickets land as one?
