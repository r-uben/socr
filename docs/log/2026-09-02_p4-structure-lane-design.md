# P4(b) — an acceptance contract for non-table structure pages

2026-09-02. Design pass for programme item **P4(b)** of
`docs/log/2026-09-01_conceptual-revision.md` (owner ruling "Ratified 2026-09-02":
equation and figure pages leave the free native lane and enter the ladder; LaTeX
splicing and figure description stay opt-in). Read-only pass on branch `docs/conceptual-revision-2026-09` (P0 + P2 present).

**Grounding canary.** `_is_trusted_native_without_ocr` is defined at
`src/socr/pipeline/orchestrator.py:1295`; `wc -l src/socr/core/born_digital.py` = **2464**.

## 1. What happens today, step by step

**Signals.** `apply_born_digital` copies `has_tables` / `has_figures` /
`has_equations` / `has_corrupt_math` from the assessment onto `PageState`
(`core/state.py:325-328`). `has_equations` is `_detect_math_fonts(page) or
_detect_equations(raw_text) or has_corrupt_math` (`born_digital.py:1295`), and the
font term returns True on **one** math-font glyph anywhere on the page
(`born_digital.py:1807-1832`, "a single match anywhere on the page is sufficient").
`has_figures` is `_has_images(page)`, i.e. `len(page.get_images()) > 0`
(`born_digital.py:1293`, `:2337-2344`).

**An equation-only page (no table signal).** `_is_trusted_native_without_ocr` ends
at `return not self._page_has_tables(...)` (`orchestrator.py:1336`), so the page is
trusted native. In the agentic loop it takes the `elif is_native:` arm
(`orchestrator.py:3536`) → `_agentic_native_page` (`orchestrator.py:4242`), which
appends one `engine="native"` `PageOutput`, `SUCCESS`, `audit_passed=True`, cost
0.00. No model, no judge, no verifier. If unmapped math glyphs are present it emits
a `native_math_unrecovered` audit event and still ships `SUCCESS`
(`orchestrator.py:4296-4315`). Selection returns at `PASSING_BEST_OUTPUT`
(`manifest.py:1026`).

**A figure-only page.** It never reaches that arm. The chart-asset scan runs over
**all** pages before the loop (`orchestrator.py:3199-3240`); `_is_chart_asset_page`
(`orchestrator.py:1499`) requires native eligibility plus `has_chart_marks`, and
`has_chart_marks` returns True on the **first embedded raster**
(`figures/extractor.py:1066`, "fast path: embedded raster image present"). Since
`has_figures` is derived from the *same* `page.get_images()` call, **every
table-free figure page is already a chart-only page** and routes to
`_agentic_chart_asset_page`: native prose plus a whole-page PNG ref, forced even
with `--save-figures` off. That lane is on by default and needs no flag. So the figure half of
P4(b) is, in substance, already built; what is missing is the equation half.

**If an equation page were routed today.** `route_page` (`agentic.py:147`) walks the
ladder with the judge built by `_build_page_judge` (`orchestrator.py:4647`):
`SourceEvidenceTableJudge` → `NativeTableVerifierJudge` → `VLMPageJudge` /
`HeuristicPageJudge`. Both table judges self-skip on a page with no markdown table
and no table signal (`agentic.py:604-606`), so acceptance for an equation page is
**one VLM look at the page image** ("faithful?", `agentic.py:364-390`) or, with no
vision model, word-count/refusal heuristics (`audit/heuristics.py:38`). Native text
is *not* an attempt on a routed page: a native fallback output is appended only when
the ladder is empty (`orchestrator.py:3323-3336`). Selection then either ships the
accepted model read (`manifest.py:1026`) or, if every rung was refused, falls
through to the born-digital branch and ships `_native_text_with_appends(p)`
(`manifest.py:287`, returned at `manifest.py:1339-1358`) as `NATIVE_CLEAN` /
`SUCCESS`, because `native_is_fallback` needs `needs_ocr_enhancement` or a *table*
defect and an equation page has neither.

## 2. Why R3 failed, precisely

R3 widened two things at once: the routing bypass **and**
`PageState.is_structure_class()` (`core/state.py:157`) from `has_tables` to "tables
or equations". The damage was in the second. `is_structure_class()` is the entry
condition of the S1 branch through `_reaches_structure_class_branch`
(`manifest.py:679-748`), and that branch's only notion of a successful model attempt
is `_grid_authored_attempt` (`manifest.py:632`), which ends in
`has_strict_table_grid(text)` (`tables/reconcile.py:456`) — a GFM header/separator
grid. An equation reading authors no grid. So:

- **False accept.** A model attempt whose text happened to contain pipe-and-dash
  shapes (matrix bars, `|x|`, an aligned display) satisfied `has_strict_table_grid`
  and was selected as the structure-class grid winner over a correct native
  transcription, with nothing verifying it.
- **False reject.** Every other equation page fell to case (iii) with no grid
  winner, demoting a page that had shipped free native `SUCCESS` to `WARNING` and
  flipping the document to `AUDIT_FAILED` for no gain.

That is the reasoning recorded verbatim at `orchestrator.py:1313-1327` and
`state.py:165-178`, and BLOCKING 1 on #269 reverted both terms together.

**P2 has made the false-reject arm strictly worse.** Case (iii) no longer ships
flagged native prose; it ships `structure_class_floor_text`
(`manifest.py:815-853`) — the whole page replaced by
`[page N failed: unverifiable table — see image]` plus the PNG, *no native bytes at
all*, `ERROR`, `STRUCTURE_CLASS_LADDER_EXHAUSTED`. Re-running R3 on top of P2 would
delete the prose of every equation page whose ladder was refused. Any P4(b) build
must therefore keep `is_structure_class()` table-only, or teach the branch a
non-grid notion of "a reading was authored" **before** widening it.

## 3. Candidate acceptance contracts

**A. Route, never demote (native is the floor).** Widen the bypass to
`has_equations` only; leave `is_structure_class()`, `_grid_authored_attempt` and the
P2 floor untouched. An equation page runs the ladder; an accepted model read ships;
a fully refused ladder falls through to the existing born-digital native branch and
ships native prose `SUCCESS`. Deletes nothing.
*False accept:* a VLM-judged hallucinated page read replaces exact born-digital
text, with no deterministic verifier in the path — the single largest new risk, and
exactly the class socr's own docs call worse than a missing number.
*False reject:* none; the floor is the status quo ante.
*Cost:* one local VLM extraction plus one judge call per equation page.

**B. A presence gate on the whole page (deterministic, non-GFM).** As A, plus a
free deterministic verdict before the VLM judge: the model text's numeric multiset
must be contained in the native page's multiset, and the page's non-math word tokens
must be recalled. `escalation_canary.presence_verdict_from_text`
(`escalation_canary.py:289`) already does the numeric half and already has a
whole-page oracle (`native_text_value_counts`, `:234`, falls back to every numeric
token in `native_text`). The one gap is the candidate side: `table_value_tokens`
(`:81`) reads **markdown table blocks only** via `collect_table_tokens`
(`tables/source_evidence.py`), so on an equation page the candidate multiset is
empty and the verdict degenerates to `PRESENCE_LOST`, which does not block. The
build is a whole-text candidate tokenizer plus `PASS | FAIL | ABSTAIN` wiring in the
judge stack — the shape `ARCHITECTURE.md` and the ruling's step 3 already specify.
*False accept:* a model that re-orders or mis-binds correct numbers passes; presence
proves "not invented", never "correctly placed" (the module says so itself).
*False reject:* a model that legitimately normalises a number (`.75` → `0.75`,
minus signs, thousands separators) reads as invented; normalisation exists
(`_normalize_numeric_token`) but is table-tuned. Encoding-suspect and corrupt-math
pages must return ABSTAIN, not FAIL.
*Cost:* zero beyond A; pure string work.
*Deletes:* nothing yet; it is the first non-table member of the free verifier tier,
so it is also the seam P6 needs.

**C. Asset lane for equations, mirroring charts.** Treat an equation page like a
chart page: keep native prose, attach a rendered crop or page PNG, no model rung at
all. Precedent is `_is_chart_asset_page` / GH-150 B1, and the crop machinery already
exists in `math/recover.py` behind `recover_corrupt_math`.
*False accept:* none — nothing is replaced.
*False reject:* none, but the mathematics still ships as whatever PyMuPDF produced;
a human recovers it from the image. This does **not** satisfy the ruling as written
("equation pages go to the ladder"); it satisfies the ruling's *purpose* for figures
only.
*Cost:* one render per equation page, no model.
*Deletes:* nothing.

**Recommendation: B, built as A first.** A is a two-line routing change and is
already safe against P2 because the floor cannot fire on a non-table page. B adds
the deterministic gate the ruling's step 3 demands, in the one place where socr has
a genuine free oracle for a non-table page — the born-digital text layer itself.
C is the right answer for figures and is already shipped there; extending it to
equations contradicts the ruling.

**The cost side is unmeasured, and the trigger is loose.** No log in `docs/log`
records an equation-page or figure-page rate; the only per-page kind manifest
(`docs/log/2026-08-20_lane-comparison-manifest.json`, 15 table / 4 figure / 2
equation pages) is a hand-capped four-pages-per-document selection with tables
sorted first, so it measures nothing about rates. Meanwhile `has_equations` fires on
a single math-font glyph, which in an economics corpus is most body pages. Widening
the bypass on `has_equations` as it stands could move the majority of the 58% free
prose lane onto the ladder — the exact outcome the ruling rejected when it rejected
"native as a $0 rung". This is question 1 below.

## 4. Interaction with P2's floor

Under A or B, an exhausted ladder on an equation page ships **native prose,
`SUCCESS`, `NATIVE_CLEAN`** (`manifest.py:1339-1358`) — the P2 floor is unreachable
because `_reaches_structure_class_branch` requires `p.is_structure_class()`
(`manifest.py:747`), still `has_tables`. That is deliberate and is the whole safety
margin of this design. If a future ticket widens `is_structure_class()`, the floor
becomes reachable and an equation page with a refused ladder loses every byte of its
prose, so that widening must arrive together with a non-grid `_grid_authored_attempt`
equivalent and a floor text that is not whole-page. A mixed equation+table page is
unaffected: `has_tables` already routes it and the floor already owns it.

## 5. The sharp questions for the panel

1. **What signal takes a page out of the free lane — `has_equations` as detected, or
   a narrower one?** For as-detected: it is the signal the ruling names, it is
   already on `PageState`, and any narrowing is a threshold we would have to invent.
   Against: `_detect_math_fonts` returns True on one math-font glyph anywhere on the
   page, so on a maths-typeset corpus this plausibly routes most prose pages, spends
   a VLM call on each, and effectively adopts the "judge every page" option the
   ruling explicitly rejected — with no measurement to bound it.

2. **Should an accepted model reading of an equation page be allowed to replace
   native prose at all, or is the model attempt advisory?** For replacement: the
   whole point of routing is that the model reads the mathematics PyMuPDF flattens,
   and a read that ships nothing changes nothing. Against: acceptance on a non-table
   page is one VLM opinion on an image, with no deterministic term, so the ladder can
   swap exact born-digital text for a fluent hallucination — the failure mode this
   repo ranks worst.

3. **Are figure pages already done?** Every table-free page with an embedded raster
   already leaves the plain native lane into the chart-asset PNG lane, by default,
   because `has_chart_marks` and `has_figures` read the same `page.get_images()`.
   For "done": prose is preserved, the visual is preserved as an asset, cost is one
   render, nothing is replaced. Against: the ruling says figure pages go to *the
   ladder*, and a PNG is not a reading — a figure with in-image text (axis labels,
   legends, embedded tables) still ships that text nowhere in the markdown.

## 6. Acceptance-criteria readiness

There is no `TICKETS.md` entry for P4 yet; the criteria are the programme table row
plus the ratified paragraph. As written they are implementable for option A and
under-specified for B and for the panel's answers. Whatever ticket is cut must pin:
which signal widens the bypass (question 1); that `is_structure_class()` is
**unchanged**, with a test that the P2 floor never fires on a table-free page; and,
per CLAUDE.md and #257, a **difference** rather than a value — the same fixture with
the widening off and on, parametrised over both provider states, asserting that only
the equation page's route moves.

## 7. Panel (Codex gpt-5.6-sol, Gemini) and ruling — 2026-09-02

Both panelists verified the note's two load-bearing claims. Codex added one
nuance: chart-asset routing also requires native eligibility, so a figure page on
a scan or a corrupt-math page already takes the ladder, not the asset lane.

**Q3 — figures: agreed, done for preservation.** Prose and the image survive;
nothing is replaced. What the lane does not do is read in-image text (axis labels,
legends, embedded tables); the lane itself records that data values are not
transcribed. That is a separate, later feature. Follow-up: an explicit
`visual_values_not_transcribed` disposition and a corpus count, so the debt is
visible rather than buried.

**Q2 — replacement: agreed in direction, Codex's stricter form adopted.** A model
reading of an equation page must not replace whole-page native prose on the
strength of a VLM opinion plus a numeric-presence gate. Presence proves "not
invented", never "correctly placed"; the module says so itself. Ruling: **numeric
presence is a rejection guard, not an acceptance contract.** The unit of
replacement is the equation region with its crop attached — the shape the existing
corrupt-math region lane and the LaTeX sidecar already have — and whole-page native
prose is never swapped for a whole-page model read. Routing an equation page to the
ladder therefore means: run the model, keep its reading as a region-scoped,
crop-backed candidate, reject it on presence failure, and otherwise attach it. The
note's option B survives as the guard inside that lane, not as a licence to
replace.

**Q1 — the trigger: split, and the split is settled by measurement, not by
argument.** Codex would route on `has_equations` as detected (over-routing is a
cost problem, replacement is the correctness problem, and Q2 removes the
replacement risk). Gemini would narrow the signal first, because one math-font
glyph fires it and an economics corpus would send most of the free prose lane
through the ladder. Both name the same cheap measurement. Ruling: **no widening
ships before the number exists.** A no-model dry run over the local corpus records,
per page, which detector term fired (`_detect_math_fonts`, `_detect_equations`,
corrupt math) and how many characters sit in math-font spans, and reports the
share of today's free-lane pages each candidate trigger would move. The trigger is
then chosen from that table, with a documented reason, and the ticket pins the
signal by name.

**Order of work for P4(b), replacing the note's "B built as A first":**

1. P4-M — the measurement above (script under `benchmark/`, output to a dated log;
   no model calls).
2. P4-R — region-scoped equation lane on the ladder path: reuse the corrupt-math
   region lane; presence gate as rejection guard; crop attached; `is_structure_class`
   stays table-only, pinned by a test that the P2 floor never fires on a table-free
   page; difference-pinned over both provider states.
3. P4-T — the trigger, chosen from P4-M's table.
4. Follow-up ticket: `visual_values_not_transcribed` for figure pages.

Not done here: the consilium archive entry for this run.
