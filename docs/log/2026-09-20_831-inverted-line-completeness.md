# 2026-09-20 — GH-831: measuring the inverted-line-completeness check

READ-ONLY measurement. No `src/` change. Worktree `/tmp/wt-831m`.
The counts in this file (funnel, INSIDE/OUTSIDE split, 15 per-page verdicts) were
measured at `7cd3752` (`origin/main` after PR #856, before #859/#860 merged;
re-confirmed unchanged from the original `fc33c92` measurement after a clean
rebase — see "Re-run on current main" below). Corpus: the 380 PDFs listed in
`/tmp/gh64_pdfs.txt` (local-only file, not part of this repo — copyrighted,
not committed; only counts and basenames are recorded here).

## The invariant measured

> For every native line a markdown table row anchors to, every numeric token
> on that line must land in that row.

Zero tolerance, no new threshold — see the ticket
(`/tmp/ticket_831m.md`, issue #831) for the full background.

## What was built

A read-only probe, entirely outside `src/`, built from existing helpers only:

- `baseline_bands`, `table_blocks`, `numeric_body_rows`, `match_rows_monotonic`,
  `words_in_region` (all `socr/tables/row_corroboration.py`) — the inverted
  check is: for every candidate row `match_rows_monotonic` binds to a native
  band, does `Counter(band.tokens) - Counter(row_tokens)` have any leftover?
  Each leftover is one evicted token.
- The native `(rect, markdown)` table pairs `extract_structured` would ship
  were obtained by patching `BornDigitalDetector._verify_regions` (the last
  step before the regions are spliced into page text) to raise with the list
  it was called with, instead of re-deriving `find_tables` / the lane-stacked
  rowizer / `reconstruct_table_regions` / the chart-aware rowizer from
  scratch. This reuses the real production control flow unmodified.
- The existing presence oracle for (c) is
  `socr.tables.escalation_canary.presence_verdict_from_text`, called exactly
  as `manifest.py`'s D3 substitution calls it
  (`native_text=page.get_text("text")`,
  `candidate_markdown=extract_structured(page)`); "fires" =
  `verdict.blocks_success` (i.e. `PRESENCE_INVENTED`).

**No new geometry code was needed.** Everything above is an existing,
reused helper.

Scripts (not committed, `/tmp`-only, read-only):
`/tmp/probe_831_lib.py`, `/tmp/probe_831_controls.py`, `/tmp/probe_831_corpus.py`.

## Commands run

```
cd /tmp/wt-831m && git log --oneline -1   # fc33c92

PYTHONPATH=/tmp/wt-831m/src:/tmp ~/venvs/socr/bin/python /tmp/probe_831_controls.py

PYTHONPATH=/tmp/wt-831m/src ~/venvs/socr/bin/python /tmp/probe_831_corpus.py \
    /tmp/gh64_pdfs.txt > /tmp/probe_831_full_out.json 2> /tmp/probe_831_full_err.log
```

Both scripts assert `os.path.realpath(socr.__file__).startswith("/private/tmp/wt-831m")`
before doing anything else (the mandated canary).

The corpus run's stderr log contained real extracted numeric values from the
copyrighted corpus (the pipeline's own `reconstruct_table_regions` debug logs
list destroyed tokens at `logger.debug`/warning level on rejection). That log
was `/tmp`-only, never committed, and has been deleted after this log was
written; no value from it appears anywhere in this file.

## The funnel

| Step | Count |
| --- | --- |
| Documents listed / opened | 380 / 380 (0 failed) |
| Pages total | 4,419 |
| Pages classified born-digital (`PageAssessment.is_born_digital`) | 4,121 |
| **(a)** pages with >= 1 native markdown table region | 2,756 |
| **(b)** pages the inverted check would newly demote | 15 |
| **(c)** pages the EXISTING presence oracle already fires on | 99 |
| **(d)** overlap of (b) and (c) | 1 |
| Genuine coverage gap = (b) − (d) | 14 |

Every step above (a) has a visible, non-trivial denominator, so a small (b)
is not an artefact of a starved funnel: 2,756 pages actually carried a table
the check ran against.

## (e) — top 10 pages by evicted-token count (counts and basenames only)

| Basename : page | evicted tokens | violated rows |
| --- | --- | --- |
| mpr-2008-07.pdf : p46 | 32 | 4 |
| mpr-2021-02.pdf : p55 | 15 | 5 |
| mpr-2022-06.pdf : p69 | 15 | 1 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p1 | 13 | 13 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p2 | 13 | 13 |
| mpr-2019-07.pdf : p52 | 13 | 6 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p3 | 12 | 12 |
| mpr-2020-06.pdf : p60 | 10 | 5 |
| doc01.pdf : p2 | 9 | 4 |
| mpr-2021-02.pdf : p18 | 8 | 8 |

No values or page text are recorded — counts and basenames only, per the
ticket's hard boundary.

## The control (three synthetic fixtures)

Built in-process with PyMuPDF (`fitz`), each a small ruled (top/mid/bottom
rule) booktabs-style table with 1 label + 3 numeric columns (4 data rows),
run through the real `extract_structured`/`reconstruct_table_regions` path.
Ground truth for each is `page.get_text("text")` on the REOPENED PDF, never
the string passed to `insert_text` (per the ticket's PyMuPDF-edge-drop
warning — not triggered here since every mark was placed well inside the
page, but honoured as specified).

| Control | Fires? |
| --- | --- |
| Page number on the same baseline as the last data row | No |
| Footnote marker (right margin) level with a data row | No |
| Running header sharing a band with the first data row | No |

**0/3 false fires.** The presence oracle (`presence_verdict_from_text`) also
did not fire on any of the 3.

**Caveat, not a clean pass.** At the geometries tested (the foreign token
placed inside the reconstructed table's own bbox, on a row's baseline), the
production grid-builder (`reconstruct_table_regions` / the text-strategy
`find_tables` cell assignment) absorbed the foreign token INTO the
neighbouring cell's text rather than leaving it outside the row — e.g. a
page-number `"7"` planted in a column gutter was concatenated into the
adjacent cell as `"6.6 7"`, both tokens then legitimately present in the row.
So the inverted check correctly abstained, but not because it distinguished
"belongs to this row" from "shares this baseline" — the grid-builder had
already folded the foreign token into the row before the check ever ran.
I could not, with the real pipeline's own cell-building geometry, construct
a case where a legitimate same-baseline neighbour stays outside the row's
own token set while remaining inside the table's baseline-band scope. Three
fixtures do not rule out such a case existing elsewhere in the corpus; this
narrows but does not close the false-positive question.

## Is containment implementable under zero tolerance?

Yes, in the narrow sense asked: the check itself needs no new geometry code
and reuses vetted helpers, and 0/3 controls did not false-fire. But the
corpus numbers argue against calling this "done": (b)=15 out of (a)=2,756 is
a 0.5% would-demote rate, and (d)=1 means 14 of those 15 pages are a genuine
NEW coverage gap the existing presence oracle misses today — small in
absolute count, non-zero, and every one of the top-10 (e) pages is from a
real statistical-table-heavy document (MPR, ECB bulletin, a replay-binding
fixture). Whether 15 demotions (with unmeasured false-positive risk beyond
3 controls) is an acceptable trade for closing a 14-page-wide gap is a
policy call, not a mechanical one — this log stops at the measurement.

## What this log does NOT claim (per the ticket's guardrails)

- No frequency for the eviction bug itself is quoted; (b) is "pages the
  check would demote," which includes any false positives, not a measured
  defect rate.
- No claim about which grid builder is responsible.
- No claim about `_run_column_lanes` or any other mechanism — untested here.

## 2026-09-20 — GH-831-AUDIT: gating the 15 fires (independent re-measure + geometry + eyes-on)

Independent audit, per `/tmp/ticket_831audit.md`, run against the SAME worktree/branch
(`fix/831-measure`, `/tmp/wt-831m`) and the SAME corpus list (`/tmp/gh64_pdfs.txt`, 380
PDFs, local-only, not committed). READ-ONLY: no `src/` change. New scripts (not
committed, `/tmp`-only): `/tmp/probe_831_geometry.py`, reusing `/tmp/probe_831_lib.py`
and `socr.tables.row_corroboration` helpers unchanged.

Both scripts assert `os.path.realpath(socr.__file__).startswith("/private/tmp/wt-831m")`
before doing anything else.

### Step 1 — funnel reconfirmed

```
PYTHONPATH=/tmp/wt-831m/src ~/venvs/socr/bin/python /tmp/probe_831_geometry.py \
    /tmp/gh64_pdfs.txt > /tmp/probe_831_geometry_out.json 2> /tmp/probe_831_geometry_err.log
```

| Step | Count |
| --- | --- |
| Documents listed / opened / failed | 380 / 380 / 0 |
| Pages total | 4,419 |
| Pages born-digital | 4,121 |
| (a) pages with >= 1 native table region | 2,756 |
| (b) pages the inverted check would newly demote | 15 |
| (c) pages the existing presence oracle already fires on | 99 |
| (d) overlap of (b) and (c) | 1 |

Exact match to the prior measurement's `(380, 4419, 4121, 2756, 15, 99, 1)`. No
discrepancy to report. (Run took 2,356s; stderr contained real corpus numeric values
from the pipeline's own `reconstruct_table_regions`/verifier debug logs — same leak
class the prior log flagged — and was deleted immediately after this section was
written; nothing from it appears here.)

### Step 2 — the mechanical separator

Three x-extent comparisons were computed per evicted token, all derived from geometry
`row_corroboration.py` already produces (`cluster_band_words`, the matched row's own
span, the table region rect) — no new margin/tolerance/constant:

1. **`region_inside`** — evicted token vs. the table region's own rect (the `(rect,
   markdown)` bbox `extract_structured` ships). **159/159 (100%) INSIDE.** This is
   tautological by construction: `words_in_region(words, region)` is what selects the
   words a band can even be built from, so every evicted token was already inside this
   exact rect before the check ever ran. Reported for completeness; not a usable
   separator, and the ticket's Step 2 instruction to derive an x-extent "from the table
   block geometry the pipeline itself already has" resolves to this rect — which is why
   it does not discriminate.
2. **`row_span_inside`** — evicted token vs. the union bbox of the SAME row's own bound
   native words (the span `match_rows_monotonic` actually matched). **0/159 (0%)
   INSIDE** — every evicted token sits at the extreme left or right of its own row's
   captured span, never interleaved between two of the row's own matched tokens, across
   all 159. Not a logical guarantee (an interleaved eviction would score INSIDE by this
   metric) — an empirical property of this corpus, not the check.
3. **`table_span_inside`** — evicted token vs. the union bbox of EVERY row's matched
   tokens across the WHOLE table block (not just the evicting row) — the columns the
   candidate genuinely populates elsewhere in the same table. This is the one
   informative separator: **126/159 (79.2%) INSIDE, 33/159 (20.8%) OUTSIDE.**
   Per-page: 7 of 15 pages are unanimously INSIDE, 3 of 15 are unanimously OUTSIDE, 5 of
   15 are mixed.

Full per-page counts (basenames and counts only):

| Basename : page | evicted | table_span INSIDE | OUTSIDE |
| --- | --- | --- | --- |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p1 | 13 | 13 | 0 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p2 | 13 | 13 | 0 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p3 | 12 | 12 | 0 |
| ecb-reports-2003-report-p80-82.pdf : p1 | 3 | 0 | 3 |
| ecb-reports-2003-report-p80-82.pdf : p2 | 6 | 6 | 0 |
| ecb-reports-2003-report-p80-82.pdf : p3 | 6 | 2 | 4 |
| mpr-2008-07.pdf : p46 | 32 | 32 | 0 |
| mpr-2019-07.pdf : p52 | 13 | 12 | 1 |
| mpr-2020-06.pdf : p60 | 10 | 8 | 2 |
| mpr-2021-02.pdf : p18 | 8 | 0 | 8 |
| mpr-2021-02.pdf : p55 | 15 | 10 | 5 |
| mpr-2022-06.pdf : p69 | 15 | 15 | 0 |
| doc01.pdf : p1 | 3 | 0 | 3 |
| doc01.pdf : p2 | 9 | 2 | 7 |
| doc02.pdf : p3 | 1 | 1 | 0 |

### Step 3 — eyes on

Rendered (pdftoppm, 200dpi) and visually inspected 13 of the 15 pages: all 3
unanimous-OUTSIDE pages, all 5 mixed ("ambiguous") pages, and 5 of the 7 unanimous-INSIDE
pages (exceeding the >= 3 minimum). The 2 not independently rendered
(`ecb-meetings-...-p127-129.pdf:p3`, `ecb-reports-...-p80-82.pdf:p2`) are continuation
pages of the same multi-page table already inspected on that document's other pages,
with the same evicted-token profile — verdict below is by structural analogy, flagged
as such, not independent inspection.

**Per-page verdict** (does at least one evicted token genuinely belong to the native
row its band was bound to?):

- 12 pages: **genuine** — real statistical tables (ECB economic-bulletin balance-sheet
  tables, Fed MPR SEP median/central-tendency/range tables, two OLS-regression tables)
  where the evicted tokens are either (i) a period/year row-group label dropped from its
  own data row (e.g. "2020", "2021 Q1" printed beside the row's numbers but absent from
  the candidate's row-stub cell), or (ii) an actual numeric cell value dropped from a row
  that otherwise matched (e.g. a SEP "2019"/"2020" Median column, or a regression
  coefficient in a trailing column). Both are real content loss under the check's stated
  invariant.
- 2 pages (`mpr-2008-07.pdf:p46`, `mpr-2022-06.pdf:p69`): **genuine by the check's own
  strict definition, but NOT a table-cell drop** — both pages are Figure panels (a
  histogram of SEP participant projections and a diffusion-index line chart), and the
  "table" the pipeline's rowizer detected is the chart's OWN x-axis tick-label text
  (percent-range bins / year ticks) misread as tabular rows. The evicted tokens are
  printed axis labels absent from the candidate's misparsed "row" — a real absence, but
  a different defect (chart-as-table misclassification), not the "dropped SEP median
  value" class the other 12 pages show. **This is the one finding that must be surfaced
  to the two review seats before shipping**: 2 of the 15 fixture pages are chart pages,
  not statistical tables.
  **FIXTURE LABEL: `mpr-2008-07.pdf:p46` and `mpr-2022-06.pdf:p69` must be labelled
  "chart axis mistaken for a table row" wherever they ship as a fixture — not "dropped
  table cell value".**
- 1 page (`mpr-2021-02.pdf:p18`): genuine by the check's strict definition, lower
  severity — all 8 evicted tokens are ordinal row-numbering markers ("1.", "2." ...
  "12."), a decoration format `_SPEC_NUMBER_RE` (which only matches the parenthesised
  `(1)` form, see #858) does not exclude. They are real printed characters on that row's own
  baseline, genuinely absent from the candidate's row text, so they satisfy the
  invariant — but they are structural numbering, not a data value.

No page's fire was spurious: every one of the 15 carries at least one evicted token
that is a real printed character on the row's own native line, absent from the
candidate text bound to that line. One individual token (not a whole page) is itself a
likely false read: `ecb-reports-2003-report-p80-82.pdf:p1` has a "5)" footnote-marker
glyph (missing its opening paren, so it isn't excluded by `_SPEC_NUMBER_RE`) counted
among its 3 evicted tokens — harmless, since the page's other 2 evictions (dropped
period labels) are genuine.

### Correcting the OUTSIDE-as-false-positive prior

The ticket's Step 2 framed OUTSIDE as "strong prior for a false positive." Measured
against `table_span_inside`, that prior does not hold on this corpus: all 3
unanimous-OUTSIDE pages are genuine (dropped row-group date labels on
`ecb-reports-...:p1`, dropped row-numbering ordinals on `mpr-2021-02.pdf:p18`, and a
dropped row-distinguishing leading digit — "1 YR" vs "2 YR" — on `doc01.pdf:p1`).
OUTSIDE-by-`table_span` more often means "a column position NO row in this table ever
successfully captures" (a systematic drop, still real) than "an unrelated marginal note
sharing a baseline." The distinction the ticket wanted (real cell vs. marginal noise)
is not cleanly recoverable from x-extent geometry alone here; it required the visual
read every time.

### Overall

**PASS.** Every one of the 15 pages carries at least one token genuinely evicted from
its bound row, confirmed by direct visual inspection for 13/15 and structural analogy
(same table, same document) for the remaining 2. Ship-relevant caveats for the review
seats, not blockers:

1. `region_inside` is tautological — do not cite it as evidence either way.
2. 2 of the 15 fixture pages (`mpr-2008-07.pdf:p46`, `mpr-2022-06.pdf:p69`) are chart
   pages, not statistical tables; **fixture label: "chart axis mistaken for a table
   row"** specifically, not "dropped table cell value" and not a generic exemplar of the
   other 13 pages' defect class.
3. `mpr-2021-02.pdf:p18`'s fire is ordinal-marker decoration, not a data value — lowest
   severity of the 15.
4. The false-positive rate is still not rigorously bounded (the corpus supplied zero
   spurious fires among 159 evicted tokens across 15 pages, but 15 pages is a small
   sample and the original 3 synthetic controls remain uninformative per the prior log's
   own caveat) — this audit narrows but does not close that question either.
5. `_SPEC_NUMBER_RE`'s footnote-marker gap (the `5)`-shaped token on
   `ecb-reports-2003-report-p80-82.pdf:p1`, see above) is filed as
   github.com/r-uben/socr/issues/858 — independent of #831, not fixed on this branch.

## Re-run on current main (2026-09-20, follow-up)

Two more PRs landed after `7cd3752`: #859 (issue #855, escalation-latch
change, orchestrator table-scoring path only) and #860 (issue #600,
`binding._assign_bands` in `src/socr/tables/binding.py` gained a fold that
heals printed-line tears plus an x-overlap guard — corpus-wide it refuses
1,023 folds concentrated in ~92 documents, almost entirely
`fomcprojtabl*`/`fomcminutes*`).

**Question:** does the candidate-row path this probe measures
(`get_table_regions` → `extract_structured` → `find_tables` /
`reconstruct_table_regions` / `rowize_from_words_chart_aware` → the patched
`_verify_regions` capture point) ever call `binding.bind()` /
`binding._assign_bands`, such that #860 could move the 15 affected pages'
leftover-token counts?

**Answer: no traversal found on this probe's capture path** (checked by
grep for `_assign_bands|binding\.bind|import.*binding` across every module
on it: `reconstruct.py`, `native_verifier.py`, `label_canonical.py`,
`reconcile.py`, `born_digital.py`, `row_corroboration.py` — none call
`bind()`/`_assign_bands`; `row_corroboration.py`'s only reference is a
docstring warning that `_assign_bands` has a `round(word_y0)` bucketing bug,
i.e. it deliberately does not reuse it).

**Correction:** an earlier draft of this section additionally claimed
"only `adjudication.py` imports from `binding` at all" — that is false and
was found false by grepping the wrong scope (only `src/socr/tables/`, not
the whole tree). On `origin/main`, `src/socr/pipeline/orchestrator.py:6468`
does `from socr.tables.binding import bind` and calls it inside
`_binding_evidence_for_witness` (`orchestrator.py:6444`); `bind()` is also
imported at four more orchestrator sites and in
`judge/table_cell_guard.py:73` and `judge/table_verdict.py:139`. The
conclusion below still holds despite this correction, for a narrower
reason: `_binding_evidence_for_witness` is a verification step that runs on
a witness AFTER the candidate table markdown already exists — it is
downstream of `_verify_regions`, the point this probe raises out of before
any of that code executes. So this call site exists in production and is
real, but it is off this probe's capture path, not absent from the
codebase.

`binding._assign_bands` is only called from `binding.bind()`
(`binding.py:1089`, `:1165`). Nothing upstream of this probe's capture
point (i.e. nothing between `find_tables`/`reconstruct_table_regions` and
`_verify_regions`) calls `bind()`. #860 therefore cannot have moved the
candidate-row token counts for these 15 pages (or any page) through the
path this probe measures.

**Separately, and outside this probe's scope:** `bind()`/`_assign_bands`
IS reached in production downstream, at `_binding_evidence_for_witness`,
feeding `classify_binding_evidence` and the guard/winner-selection chain.
#600's banding change can therefore move binding-evidence verdicts
(PASS/ABSTAIN/CONTRADICT) on real pages, independent of and un-measured by
this probe. That is a separate, queued measurement (see team-lead
follow-up), not part of the #831 check this file is about.

**Conclusion: no re-run against `d07d6e2` is needed.** The funnel and the
15 per-page counts reported above (measured at `7cd3752`, matching the
original `fc33c92` measurement) stand.

Housekeeping fixed in this pass:
- Decision-log header above now states the exact SHA the numbers were
  measured on instead of the stale dispatch-time SHA.
- `mpr-2008-07.pdf:p46` / `mpr-2022-06.pdf:p69` fixture label ("chart axis
  mistaken for a table row") and the `_SPEC_NUMBER_RE` gap
  (github.com/r-uben/socr/issues/858, filed independently, not fixed on
  this branch) were already applied above in the original findings section.

**Overall: PASS stands, unchanged.**
