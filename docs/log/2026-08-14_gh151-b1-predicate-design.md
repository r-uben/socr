# GH-151 TICKET-B1 — predicate design: separating a split data row from a header band

Date: 2026-08-14 · Branch at time of writing: `feat/151-b1-structural-gate` (read-only; the only
file written is this note) · Ticket: `docs/plans/gh151-structural-gate/TICKETS.md` TICKET-B1 ·
Supersedes the predicate section of `docs/log/2026-08-13_gh151-b1-design.md` (§2, §3, §10).

Everything numeric below is from probes I ran under `/tmp/b1probe` with `~/venvs/socr/bin/python`
against this branch's source. Corpus: **32 PDFs, 245 born-digital pages with `has_tables`,
422 native table blocks** (every 10th paper from the local papers library, first 400 by name).
That is 14x the 29-block sample the 2026-08-13 note used, which the ticket's *Prevalence caveat*
asked for.

---

## 0. TL;DR

- **There is no invariant in the grid that separates a split data row from a header band.** Every
  candidate discriminator is a heuristic. Two of them are *principled* heuristics — they reuse
  rules the repo already applies elsewhere and introduce no constant — and together they cut the
  gate's firing rate from **26.9% to 14.3% of native table pages** while keeping both of B1's
  acceptance cases green.
- **Geometry does not rescue it either**, and not for the reason anyone assumed: the split shape
  B1 actually ships against is a *full row-pitch* split, geometrically identical to a heading
  band (measured on B1's own fixture: y-gaps `[18,18,18,18,18]`). A genuine sub-pitch superscript
  shift does not even produce the split — the extractor binds it correctly (measured).
- **`ragged` alone is near-inert: 2 of 422 blocks (0.5%), and it does NOT fire on B1's own
  acceptance fixture.** "Fall back to `ragged`" is equivalent to shipping nothing.
- **`_is_canonical_column_band` suppresses 0 pairs in 422 real blocks.** The one hardcoded
  exception in the predicate is dead on real data; it exists to pass one hand-written negative
  control. It has to stay (§2.2), but nothing should be added beside it — the narrowing below is
  what does the work.
- **Recommendation: narrow, don't abandon** (Option B, §5). But the choice between narrowing on
  shape and replacing the shape predicate with the existing geometry signal (Option C) is a
  genuine fork — panel it. Question in §7, ready to paste.

---

## 1. What exists today

### 1.1 The predicate and its call site

`src/socr/tables/structure_check.py:156-172` — `FINDING_DETACHED_LABEL` fires for body row `i`
when `grid[i]` has a non-blank label and no non-blank value cells, `grid[i+1]` has a blank label
and at least one non-blank value cell, and `_is_canonical_column_band(grid[i+1][1:])`
(`structure_check.py:111-125`) is false. The gate is `ragged or detached_label_rows`
(`structure_check.py:210-223`), called once at `src/socr/core/born_digital.py:962-965`, inside the
`has_tables` / not-rotated branch, on the markdown `extract_structured` just produced.

### 1.2 The predicate's input contains no geometry, and cannot be given any without `reconstruct.py`

`check_markdown` takes a **string** (`structure_check.py:198-207`) and delegates to
`reconcile.find_table_blocks` (`src/socr/tables/reconcile.py:87-110`), whose `_Block` carries
`start`, `end`, `grid` — line indices into the markdown, no coordinates. Upstream, the three
markdown producers all discard the row→y mapping:

| Producer | Where | Note |
|---|---|---|
| `find_tables()` passthrough | `born_digital.py:1468-1513` (`_table_to_markdown`) | **pads every row to `max(len(row))`** (`:1495-1499`) — this path can never be `ragged` |
| word-geometry rowizer | `reconstruct.py:1017-1122` | rows are `rows_by_y[round(w[1])]` groups (`:1049-1051`), but only `(rect, markdown)` is returned (`:1116`) |
| text-strategy booktabs | `reconstruct.py:100-227` | same, `_grid_to_markdown` at `:223` |

`_grid_to_markdown` (`reconstruct.py:553-584`) emits the separator with `ncol = len(grid[0])` and
body rows at their own widths, so raggedness survives only when a *cleaned* grid is already
irregular — which `_clean_grid` / `_rowize_segment`'s fixed lane count largely prevent.

**Consequence:** plumbing per-row y-bands into the gate means editing `reconstruct.py`, which this
ticket is barred from, and would still be lossy because `_clean_grid` and
`_collapse_header_prefix` (`reconstruct.py:645-712`) drop and merge rows before markdown is
emitted, destroying the index correspondence.

### 1.3 What the reviewers' four counterexamples actually do — two of them are unreachable on the rowizer path

Round-tripping each hand-built grid through the rowizer's real cleaning chain
(`_clean_grid` → `_collapse_header_prefix` → `_grid_to_markdown` → `find_table_blocks` →
`check_grid`):

| Reviewer shape | After `_collapse_header_prefix` | `detached_label_rows` |
|---|---|---|
| 1. textual column headers | `[['Estimator','Baseline OLS','Alternative IV'], ['log GDP','0.45','0.52']]` | `()` — **collapsed into the header, never reaches the predicate** |
| 2. panel sub-heading | unchanged | `(1,)` fires |
| 3. units row | unchanged | `(1,)` fires |
| 4. bracket numbering | `[['Model','Main [1]','Robustness [2]'], ['Beta','1.2','1.4']]` | `()` — **collapsed** |

Shapes 1 and 4 survive only on the `find_tables` → `_table_to_markdown` path, which does no
header collapsing. Shapes 2 and 3 survive everywhere, because their row 0 is a *data-shaped*
header (`_is_header_row`, `reconstruct.py:614-643`, returns False for a row with a non-blank
col 0 **and** non-blank data cells), so the header scan stops at index 0 and nothing collapses.

This does **not** dismiss the reviewers' finding — the header-band false positive is real and I
measured it on the corpus (§2.3) — but it narrows it: the surviving class is a heading/units band
that appears **below** the first header row, i.e. mid-table. "It's near the top" therefore cannot
be the discriminator.

### 1.4 A geometry-grounded structural check already exists and already fires on B1's own fixture

`BornDigitalDetector._verify_regions` (`born_digital.py:1360-1411`) runs
`native_verifier.verify_native_table_region` over **every** table region on the page
(`born_digital.py:1242-1244`) and sets `PageAssessment.has_unverifiable_table_region`
(`:978`, `:1023`). Its hard-fail is a **numeric-token multiset conservation** check
(`native_verifier.py:848`: `value_guard_multiset_mismatch: N paired row(s) have numeric-token
multiset mismatch (dropped/invented/shifted values)`) — geometry-grounded, no shape inference.

Probed on B1's own shipped acceptance fixture (`tests/test_structural_gate_b1_gh151.py:73-100`):

```
has_tables True  has_unverifiable_table_region True  native_table_structure_defective True
```

That flag is currently **inert**: both the D3 floor in `manifest.py:321-325` and `d3_floor_pages`
in `orchestrator.py:4741-4748` require `native_table_structure_failed` **and**
`native_table_unverifiable`, and `native_table_structure_failed` is set by the old heuristic that
misses this class. So an existing, geometry-grounded detection of exactly B1's defect is already
computed on every table page and consumed by nothing — which is verbatim B1's stated *Problem*
("A defect nothing consumes is not a gate"), reached by a different route.

---

## 2. Measurement (32 papers · 245 native-table pages · 422 blocks)

### 2.1 Base rates, per block and per page

| Signal | blocks / 422 | pages / 245 |
|---|---|---|
| `ragged` | **2 (0.5%)** | 2 (0.8%) |
| `orphan_rows` | 295 (69.9%) | — (confirms the settled exclusion) |
| `detached_label` (as implemented) | 88 (20.9%) | — |
| **gate as implemented** (`ragged or detached`) | 89 | **66 (26.9%)** |
| `has_unverifiable_table_region` (TR-3, independent) | — | **62 (25.3%)** |
| `needs_ocr_enhancement` | — | 10 (4.1%) |

158 individual row-pairs fire across those 88 blocks.

Two facts that change the decision:

1. **`ragged` is 2/422 and does not fire on B1's acceptance fixture** (probed: fixture reports
   `detached=(4,) ragged=False`). It fires on the `GH151_P26_MD` string
   (`tests/test_structure_check_gh151.py:16-25`, `row_widths=(8,6,7,7,7,7,7)`) because that string
   predates the padding in `_table_to_markdown`. Keeping `ragged` in the gate is harmless; relying
   on it is not an option.
2. **`_is_canonical_column_band` suppresses 0 of 422 blocks.** The `(1) (2) (3)` special case never
   triggers on real extracted markdown. Adding `[1]`, `(i)`, `1.` siblings would be adding more
   dead constants.

### 2.2 What the two candidate qualifiers do

- **Q_num — "every non-blank cell of the follower row is a value-shaped token"**, using the repo's
  existing `native_verifier.is_numeric_token`, the same predicate `_is_data_row`
  (`reconstruct.py:520-551`) already uses to decide data-vs-header. Removes 32/158 row-hits.
- **Q_body — "some earlier body row is a labelled data row"** (non-blank col 0 and at least one
  value-shaped cell), i.e. the table's data body has already started, so the pair cannot be part
  of the header stack. Removes 48/158 row-hits. Purely positional, no token vocabulary, no vote.

| Variant | blocks / 422 | pages / 245 |
|---|---|---|
| as implemented | 88 | 66 (26.9%) |
| + Q_num | 69 | 49 (20.0%) |
| + Q_body | 60 | 45 (18.4%) |
| **+ both** | **48** | **35 (14.3%)** |

Both acceptance cases survive the combined narrowing: the `GH151_P26_MD` seam still fires (row 5,
`R2`, all-numeric follower, `ˆγ` data row above it), and the shipped PDF fixture still fires
(row 4, `R2`, all-numeric follower, `alpha`/`beta`/`TERM` above it).

**`_is_canonical_column_band` cannot be deleted — I checked, and it is the one place I was wrong
before running the probe.** Q_body subsumes it only for a band above the first data row; a
*mid-table* `(1) (2) (3)` band survives both qualifiers, because `(1)` is a value-shaped token and
the data body has already started. Probed:

```
[['','col1','col2','col3'],['alpha','1.0','2.0','3.0'],['Panel B','','',''],['','(1)','(2)','(3)'],['beta','4.0','5.0','6.0']]
  Q_num ^ Q_body -> fires at row 2       # the existing mid-table negative control
[['','col1','col2','col3'],['Panel A','','',''],['','(1)','(2)','(3)']]
  Q_num ^ Q_body -> does not fire        # Q_body alone suffices at the top
```

So the exclusion stays, and the honest framing is: it is dead on 422 real blocks and pins one
hand-written control; the qualifiers are what answer the reviewers, and no sibling constants
(`[1]`, `(i)`, `1.`) should be added — a mid-table numeric band remains a known, stated false
positive (§5, Option B).

### 2.3 The false positives are real, and so is the recall cost

Of the 158 firing row-pairs, **17 have an all-textual follower**. Sampled verbatim:

```
2000__romer_romer  p21   label='m'                    follower=('B. Change','Funds-Rate','Target')   <- header band
2013__Snowberg…    p16   label='Notes:∗∗∗,∗∗,∗denote' follower=('statistically','signiﬁcant','at')   <- footnote
2016__eisenbach…   p47   label='nominal terms.'       follower=('L','og(Fees',')')                    <- prose
2017__ozdagli      p37   label='∗p'                   follower=('< 0.10,','< 0.05,','< 0.01')         <- footnote
2015__Hameed…      p44   label='Year Dummies'         follower=('yes','yes','yes','yes')              <- TRUE split
```

The last one is the cost: `Year Dummies → yes yes yes` and `Organization FE → ✓ ✓ ✓` are genuine
split data rows that Q_num discards. That is 2 blocks in 422.

**A second, larger cost that must be stated: `is_numeric_token` has notation gaps on exactly this
corpus.** Probed:

```
is_numeric_token('(.034)')  -> False      is_numeric_token('0.91∗∗')  -> False   (U+2217)
is_numeric_token('.034')    -> False      is_numeric_token('0.91**')  -> True    (ASCII)
```

So a leading-decimal standard error `(.034)` and a significance star typeset as U+2217 make a real
value row look textual. At least 4 of the 17 all-textual-follower hits above are genuine value
rows misread this way (`Difference → (.034) (.043) (.060)`, `Actualt → 0.91∗∗ …`). Q_num is
therefore weaker than it looks until that regex is fixed — and `native_verifier.py` is barred from
this ticket, so it is a separate follow-up.

The pairs **kept** by the combined narrowing are dominated by genuinely damaged pages: labels like
`Siifcare`, `S3. Es`, `Sd. B`, `SILL Error`, `Sçare.` (garbled "Std. Error"/"Significance" rows)
plus `R2 → 0.00 0.04 0.06`. That is the intended target class.

### 2.4 The gate and TR-3 see different things

| | pages |
|---|---|
| gate as implemented, also `unverifiable` | 35 / 66 |
| narrowed gate, also `unverifiable` | 19 / 35 |
| `unverifiable` pages the gate misses entirely | **27** |

Neither subsumes the other. TR-3's independent geometry check has a comparable base rate (25.3%),
so "use geometry instead" is not a cheaper gate — it is a *different, better-grounded* gate at
similar volume.

---

## 3. Question 1 — is there a grid-only formulation? Which candidates are invariants?

| Candidate | Verdict |
|---|---|
| **Position in the table** (header bands near the top / after a rule) | **Heuristic.** `Panel B:` headings occur mid-body after data rows; §1.3 shows the top-of-table cases are already collapsed away, so the surviving false positives are precisely the mid-table ones. Useful only in its weak form (Q_body: *before the first data row* ⇒ not a split), which is sound as an exclusion but says nothing about mid-table pairs. |
| **Label text looks like a heading** ("Panel", "Units", trailing colon, word count) | **Heuristic wearing a disguise**, and a forbidden one: it is a hardcoded vocabulary, the same class of patch as `_is_canonical_column_band`. Reject. |
| **Follower cells numeric vs textual** | **Heuristic, but principled**: it reuses `_is_data_row`'s existing rule and the residual ambiguity it inherits is one the repo has already documented and accepted (`reconstruct.py:539-547`: "a single-line header whose cells are themselves numeric… indistinguishable from data by shape alone and treated as data"). Threshold-free. Measured cost in §2.3. |
| **Pair's column structure matches neighbouring rows** | **Not discriminating at all.** A column-name band and a severed value row both populate the same lanes as the data rows. Verified against every firing block: no separation. |
| **Row-width / arity parity between the follower and the data rows** | Same — both match. No signal. |

**Answer: no invariant exists.** A label-only row above a values-only row is genuinely ambiguous
in the grid; the semantic difference (does the label *belong to* the values, or *introduce* them?)
has no shape footprint. The best available formulation is the conjunction of the two threshold-free
qualifiers in §2.2, which is an honest heuristic with a measured error profile in both directions —
not a disguised invariant, and (importantly) not a vote: both qualifiers are evaluated against the
pair and the rows *above* it, never against a modal row signature.

---

## 4. Question 2 — is the grid even the right input?

**In principle geometry is the right input; in this codebase, for this defect, it does not
separate the two — and it is not reachable anyway.**

Three measured findings:

1. **Not reachable.** §1.2 — `check_markdown` takes a string; no producer returns per-row
   coordinates; the row index correspondence is destroyed by `_clean_grid` /
   `_collapse_header_prefix`; plumbing it means `reconstruct.py`, which is barred.

2. **It would not work on the shape B1 actually ships against.** Probed the shipped fixture
   (`tests/test_structural_gate_b1_gh151.py:73-100`):

   ```
   y-groups [178, 196, 214, 232, 250, 268]  gaps [18, 18, 18, 18, 18]
   detached (4,)  ragged False
   ```

   The `R2` label and its values are a **full, ordinary row pitch** apart — the identical y-signature
   a heading band has. Any "gap smaller than the table's own median pitch ⇒ split row" rule
   (the derived-from-data form, precedent `_SPLIT_GAP_MULT × median_gap`, `reconstruct.py:1063-1066`)
   would classify B1's own acceptance case as a heading band.

3. **The sub-pitch case does not reach the predicate at all.** I built a page with a `R2` label
   baseline-raised 3pt against an 18pt median pitch (word y-groups `[90,108,123,126]`, gaps
   `[18,15,3]`) and ran the real `extract_structured`:

   ```
   raise=3pt   detached=()  grid row 4: ['R2','0.12','0.16','0.61','0.09','0.12','0.61']
   ```

   The extractor binds them correctly. So the "superscript pushes the values onto the next line"
   mechanism named in the 2026-08-13 note (§2) does **not** reproduce on this branch.

**What geometry does buy is a different check, and it already exists**: TR-3's numeric-token
multiset conservation (§1.4), which fires on B1's fixture, is geometry-grounded, and is currently
consumed by nothing because the D3 floor also demands `native_table_structure_failed`.

---

## 5. Options

### Option A — ship the predicate as implemented on PR #200

Gate on `ragged or detached_label_rows` with `_is_canonical_column_band` as the sole exclusion.

- Cost: fires on **26.9% of native-table pages**; 17/158 firing pairs have textual followers, of
  which ~11 are header bands, footnotes or prose swept into a grid.
- Determinism/replay: unchanged, pure function.
- Risk: the reviewers' objection stands, and the one exclusion is dead code (0/422). Every future
  false positive invites another hardcoded sibling — the failure mode the ticket names.

### Option B — narrow to two threshold-free qualifiers; delete the hardcoded exception (recommended)

`FINDING_DETACHED_LABEL` fires at body row `i` iff, in addition to today's four conditions:

- **Q_num** every non-blank cell of `grid[i+1][1:]` is a value-shaped token
  (`native_verifier.is_numeric_token`, the rule `_is_data_row` already uses); **and**
- **Q_body** some body row `j < i` has a non-blank label and at least one value-shaped cell.

`_is_canonical_column_band` is **kept unchanged** — Q_body subsumes it only above the first data
row, and a mid-table `(1)(2)(3)` band survives both qualifiers (§2.2). No sibling constants are
added.

- Cost: fires on **14.3% of native-table pages** (35/245). All four reviewer shapes stop firing
  as written: 2 and 3 by Q_num (textual followers), 4 by Q_body (the numbering band precedes the
  data body), 1 already by `_collapse_header_prefix` on the rowizer path and by Q_num elsewhere.
  Both acceptance cases still fire.
- Residual false positives: a mid-table heading followed by a **numeric** band (a `Panel B:` above
  a `1990 | 2000` year row, or above a `[1] [2]` numbering band, after data rows have started).
  Not eliminable from the grid.
- Residual false negatives: `Year Dummies → yes yes yes`, `Organization FE → ✓ ✓ ✓` (2/422), plus —
  until `is_numeric_token` handles `.034` and U+2217 — value rows in that notation.
- Determinism/replay: unchanged. New coupling: `structure_check` would import `native_verifier`,
  which its module docstring currently disclaims (`structure_check.py:56-57`). No cycle
  (`native_verifier` imports `reconstruct`; neither imports `structure_check`).
- Silent-data-loss risk: **lower than A**, not higher — narrowing removes pages from the flagged
  set, and the removed pages are ones whose grid is correct.

### Option C — replace the shape predicate with the existing geometry signal

Set `native_table_structure_defective` from `PageAssessment.has_unverifiable_table_region` (TR-3)
rather than from grid shape — i.e. B1 becomes "make the already-computed geometry hard-fail
consumable", keeping all the reviewed wiring.

- Cost: fires on **25.3% of native-table pages**, comparable to Option A, higher than B.
- Evidence quality: strictly better — numeric-token multiset conservation against native word
  geometry, not shape inference. Directly serves "no silent content loss": it fires when values
  were **dropped, invented or shifted**.
- Coverage: **not a superset.** It misses 31 of the 66 pages the shape gate flags, and catches 27
  the shape gate misses.
- Scope/risk: does not touch `native_verifier.py` (read the flag only), but it changes B1's
  identity, and the ticket's `Done when` / negative-control tests are all written against grid
  shape and would need rewriting. It also leaves `FINDING_DETACHED_LABEL` in the module as a
  diagnostic with no consumer.

### Option D — fall back to `ragged` alone

Rejected on measurement, recorded for completeness: 2/422 blocks, and it does **not** fire on B1's
own acceptance fixture (§2.1). This ships an inert gate.

---

## 6. Recommendation

**Take Option B, and file two follow-ups.** It is the smallest change that answers the reviewers
on the evidence: it stops all four shapes they demonstrated, it halves the firing rate
on a 14x-larger corpus than the original measurement, and it does so with two constant-free
qualifiers rather than by growing the hardcoded exception list — which is the objection that
actually matters under the house rules.

The honest caveat I will not bury: Q_num is a heuristic, and `is_numeric_token`'s notation gaps
(`.034`, U+2217) make it misfire on real econ notation today. That is a bounded, fixable defect in
a different module, not a reason to keep a broader predicate.

**Option C is a genuine fork and should be panelled**, because it is better *evidence* at worse
*volume*, and because it would re-scope B1 from "new shape predicate" to "consume the geometry
failure we already compute". I could not settle that trade-off from the code: it turns on whether
the owner values evidence quality over firing rate, and on whether the 27 unverifiable-but-not-
ragged pages matter more than the 31 shape-flagged-but-verifiable ones.

Follow-ups to file (not B1):
1. `is_numeric_token` rejects `.034` and U+2217 significance stars (`native_verifier.py`).
2. TR-3's `has_unverifiable_table_region` is computed on every table page and consumed only in
   conjunction with `native_table_structure_failed` (`manifest.py:321-325`,
   `orchestrator.py:4741-4748`), so 62/245 pages carry a detected geometry hard-fail that nothing
   surfaces. This is the same bug shape B1 exists to fix.

---

## 7. The consilium question — paste verbatim

> A deterministic OCR pipeline for academic PDFs decides, per page and with no model in the loop,
> whether a natively-extracted markdown table is structurally damaged enough that it must not ship
> as trusted output. Its current predicate fires when a body row carrying a label and no values is
> immediately followed by a body row carrying values and no label — read as one physically split
> data row. The problem: a multi-tier header band (a panel heading or units row above a
> column-name row) has exactly that shape, and three reviewers showed the predicate fires on
> correct tables. Measured on 32 economics papers / 245 native-table pages / 422 table blocks:
> the predicate fires on 26.9% of pages, and 17 of its 158 firing row-pairs have a purely textual
> follower row (header bands, footnotes and prose swept into a grid). The other half of the gate,
> "rows disagree on width", fires on 2 of 422 blocks and does not fire on the pipeline's own
> acceptance fixture, so it cannot carry the gate alone. Separately, an existing geometry check
> already runs on every table page (it compares the multiset of numeric tokens in the emitted grid
> against the tokens actually present in the PDF text layer under each row, flagging dropped,
> invented or shifted values); it fires on 25.3% of pages, it fires on the acceptance fixture, and
> nothing currently consumes it. Which should the gate key on?
>
> **Option A** — Keep the predicate as it is. Simplest, already implemented and reviewed; fires on
> 26.9% of native-table pages including demonstrably correct ones, and its only exclusion is a
> hardcoded special case for the literal column band "(1) (2) (3)" which matches 0 of the 422 real
> blocks — so every future false positive invites another hardcoded exception.
>
> **Option B** — Narrow it with two qualifiers that introduce no numeric constant and no
> cross-row vote: (i) every non-blank cell of the values-only row must be a value-shaped token by
> the same test the codebase already uses to tell a data row from a header row, and (ii) some
> earlier body row must already be a labelled data row, so the pair cannot be part of the header
> stack. This drops the firing rate to 14.3% of pages and stops all four demonstrated
> counterexamples from firing — but it stops flagging genuinely split rows whose values are words rather than
> numbers (a "Year Dummies | yes | yes | yes" row), and the token test has known gaps on this
> notation (it rejects a leading-decimal standard error ".034" and a significance star typeset as
> U+2217), so it will also miss some real defects until that is fixed in a different module.
>
> **Option C** — Stop gating on grid shape; gate instead on the existing geometry check described
> above, which is already computed on every page and is currently inert. Strictly better evidence
> (conservation of the actual numeric tokens, not an inference from shape), and it fires on the
> acceptance fixture — but at a similar volume (25.3% of pages), and the two signals are not
> nested: it misses 31 of the 66 pages the shape predicate flags and catches 27 it misses. It also
> re-scopes the ticket from "add a shape predicate" to "consume a signal we already compute", and
> the ticket's acceptance tests are all written against grid shape.
>
> Constraints the answer must respect: no majority/modal-consensus reasoning across rows is
> admissible (a previous attempt was rejected for exactly that — in this corpus the minority row is
> frequently the correct one); no numeric threshold may be introduced that is not derived from the
> data; and a gate that fires on a large share of correct tables is considered worse than a
> narrower gate that catches less.

---

## 8. Acceptance-criteria readiness

Against TICKET-B1's current `Done when`:

| Criterion | Verdict under Option B | Under Option C |
|---|---|---|
| Synthetic PDF split into label-row/values-row yields `table_structure_failed` and the page is not `audit_passed=True` | **Implementable as written.** Verified the shipped fixture still fires under the narrowing (`R2`, all-numeric follower, three labelled data rows above). | **Needs rewriting** — the fixture fires on the geometry check too (probed: `has_unverifiable_table_region=True`), but the criterion's wording implies the grid predicate. |
| Seam test over `GH151_P26_MD` asserts the predicate fires | **Implementable as written.** Still fires (row 5). | **Not implementable** — that fixture is a markdown string with no PDF, so no geometry check can run on it. |
| Negative control: SE / t-stat continuation row does not fire | Already holds; unaffected. | Holds vacuously. |
| Negative control: group heading above an unlabelled column-number row does not fire | **Holds only because `_is_canonical_column_band` is kept.** `test_group_heading_above_column_band_does_not_fire_mid_table` (`tests/test_structural_gate_b1_gh151.py:346-352`) places the heading after a data row, where both new qualifiers pass; the exclusion is what carries it (§2.2). Criterion is fine as written; the implementer must not delete the exclusion. | n/a |
| Negative control: footnote mangled into a grid "classified deliberately" | **Needs re-baselining.** `test_split_footnote_pair_fires_as_a_deliberate_classification` (`:354-367`) asserts the footnote pair **fires**; under Q_num it no longer does. The re-baselined assertion (it does not fire) is the better outcome and should say so. | n/a |
| **Missing criterion — should be added** | The four reviewer shapes in §1.3 belong in the test file as explicit negative controls, with shapes 1 and 4 marked as *also* unreachable through the rowizer path. | same |
| **Missing criterion — should be added** | A firing-rate assertion is not testable hermetically, but the measured 14.3% figure and its 32-paper provenance should be recorded in the ticket, replacing the "20/29, self-graded, do not cite" caveat. | same |

The settled wiring criteria (page status, document status, sidecar/resume, `AuditEvent`, CLI, and
the `--native-only` non-override) are unaffected by the predicate choice and remain implementable
as written.

---

## 9. What I could not determine

- **True-positive rate.** I measured how often each variant fires and inspected the labels, but
  "is this page's table actually damaged?" was judged by eye on the printed pairs, not against
  ground truth. The 14.3% is a firing rate, not a defect rate — the same caveat the ticket already
  attaches to the 20/29 figure.
- **Whether reviewer shapes 1 and 4 occur in practice via the `find_tables` path.** They are
  reachable by inspection (`_table_to_markdown` does no header collapsing), and header-band false
  positives are empirically present (§2.3), but I did not attribute any specific corpus hit to a
  specific producer.
- **Whether the p26-class defect has a reproducing PDF at all on this branch.** §4.3 shows the
  superscript mechanism does not reproduce; the only shape I could make reproduce is the shipped
  fixture's full-line split. The 2026-08-13 note already recorded that the real page 26 yields zero
  table blocks today.
- **How Option C behaves once `native_verifier`'s notation gaps are fixed.** The multiset check
  uses the same token machinery, so its 25.3% may itself contain notation-driven false positives.
  I did not separate them.
- **Cost of the flagged pages** in wall-clock or tokens. Unmeasured, as before.
