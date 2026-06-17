# GH-56 — Reliable extraction of dense numeric tables (CE forecaster grids): design

Date: 2026-06-17
Ticket: GH-56 (GitHub issue #56, table-QUALITY slice — distinct from the PP-3 dual-pass
refactor and the GH-56 sidecar-provenance fix in PR #79 / commit 59827ec).
Scope: design only, read-only on source. The provenance/resume bug is already fixed
(`docs/log/2026-06-17_gh56-sidecar-provenance.md`) and is NOT in scope here.

Validated artifact: CE grid `202401.pdf` p.4 (28 forecasters × 10 indicators × 2 years).
Values are captured faithfully (correct, column-ordered, `na` preserved); the page ships
the *worse* of two flawed structural artifacts.

---

## 0. Framing — "VLM for structure, geometry for values"

The principle the panel should reason from, and the lens for every option below.

**Geometry is deterministic and the most document-robust signal for VALUES.** For
born-digital PDFs the raw token positions `(x0, y0, x1, y1, word)` from
`page.get_text("words")` are EXACT and style-independent — they do not depend on whether the
table has ruling lines, what font is used, or how the publisher styles headers. Every
value-bearing path in socr that touches these positions is model-free: `page.find_tables()`
(`born_digital.py:712`), the word-lane clustering and y-row grouping in `reconstruct.py`
(`has_numeric_columns` `reconstruct.py:110`, the x-lane clustering `reconstruct.py:128-135`,
`find_tables(strategy="text")` `reconstruct.py:90`), and the verifier's lane counting
(`_get_native_lane_count` / `_rows_by_y` `native_verifier.py:114,151`). So geometry is the
right engine for the question *"what numeric values are on this page and at what (x, y)?"* —
it is char-exact and cross-document stable.

**What varies wildly across documents is the LOGICAL structure, and that is where the
deterministic heuristics break.** Column boundaries on whitespace-only tables (CE has NO
ruling lines, so the default "lines" strategy collapses — `born_digital.py:712-722`,
producing the 28-names-in-one-cell artifact), header-vs-data-row classification, multi-line
cells (a forecaster name wrapping two lines), and — critically for CE — *where one table ends
and the next begins*. `find_tables()` has no robust notion of these; it either over-merges
(the CE p.4 case: two tables fused, so the verifier compared 27 native lanes to one 11-col
output) or shreds prose. A VLM "sees" the page like a human and is the document-robust engine
for *structure and segmentation* precisely because that judgment is visual, not positional.

**The robust division of labor: VLM LEADS structure/segmentation; deterministic geometry
VERIFIES the values.** The VLM proposes the row×column layout and the region boundaries; the
geometry net catches the VLM silently dropping or shifting a cell — which the VLM cannot be
trusted to self-report. The native-numeric-token **multiset guard** proposed in Options A2/D2
below IS this principle in concrete form: it is geometry-as-safety-net asserting "no value
the page proves exists may vanish or be invented." The existing `native_verifier` hard-fail
(`native_verifier.py:408-447`) is the same idea, but blunt (counts cells, not values); the
multiset guard is the value-level version.

**Hard caveat — geometry exists ONLY for born-digital PDFs.** The deterministic net is
available exactly when `get_text("words")` is non-empty, i.e. `is_born_digital=True`
(detected by `BornDigitalDetector._assess_page`, `born_digital.py:265`; the verifier itself
bypasses cleanly on empty words, `native_verifier.py:380-387`). A scanned / image-only page
has NO geometry → the value-level safety net cannot run → such a page is VLM-only with no
deterministic check and MUST be handled differently (e.g. dual-pass crop re-read, or shipped
flagged as unverifiable). CE `202401.pdf` is born-digital, so the net applies here; the
design must not silently assume it for the scanned corpus.

---

## 1. What exists today — the precise data flow for a failing CE table page

CE p.4 is born-digital, clean text layer, and the detector flags it as a table page
(`born_digital.py:592 _detect_tables` → `page.find_tables()` non-empty OR
`has_numeric_columns`). Because it has tables, it is **not** a trusted-native page:

- `_is_trusted_native_without_ocr` returns `False` for any table page
  (`orchestrator.py:1115-1116`). So the page is NOT in `is_native` and does NOT take the
  free native bypass; it is appended to `ocr_pages` (`orchestrator.py:1300-1302`) and also
  to `native_fallback_pages` (`orchestrator.py:1304-1310`, since it is born-digital with
  native_text).

- **`ps.native_text` for this page is the COLLAPSED artifact.** `apply_born_digital` stores
  `extract_structured(page)` as the native text for table pages
  (`born_digital.py:547-549`). `extract_structured` (`born_digital.py:704-790`) calls
  `page.find_tables()` (default "lines" strategy), and for each table renders it via
  `_table_to_markdown` (`born_digital.py:792-838`), which preserves embedded `\n` inside
  cells (`cell.strip()` only — no newline handling at line 813). On the CE grid
  `find_tables()` returns a single region with 28 names newline-stacked in one cell and
  numbers newline-stacked per column → the fully-collapsed table. **`reconstruct_table_regions`
  runs ONLY when `find_tables()` returns nothing** (`born_digital.py:728-731`). That
  "only-reconstruct-when-empty" gate is the structural bug /codex flagged: a *non-empty but
  lane-stacked* `find_tables()` result never reaches the word-geometry rowizer.

- **OCR routing.** The page goes through `route_page` (`orchestrator.py:1654-1661`,
  `agentic.py:140`). The judge is `NativeTableVerifierJudge` wrapping the inner judge
  (`orchestrator.py:1382 _build_page_judge` → `agentic.py:350`). For each ladder rung's
  output, `verify_native_table` (`native_verifier.py:367`) runs the PyMuPDF geometry check
  BEFORE the inner judge. qwen produces a row-major table that is **ragged** (Goldman row
  has 19 data cells where 20 are required). The Tier-1 hard-fail predicate
  (`native_verifier.py:408-447`) fires: a native data row has K≥2 well-separated numeric
  lanes but the output row has fewer populated cells → `hard_fail=True`,
  `reason="geometry_impossible_collapse … native_lane_count=27 / output_col_count=11"`. The
  judge returns `accept=False` (`agentic.py:414-432`) and does NOT call the inner judge.

- **Every rung re-asks the SAME prompt and gets the SAME ragged output**, so the verifier
  hard-fails each rung. `route_page` walks the whole ladder (escalation is bounded by ladder
  exhaustion, not a retry count — `agentic.py:150-156`) and returns
  `PageDecision(accepted=False, final_output=_best_effort(...).output)`
  (`agentic.py:286`). `_best_effort` (`agentic.py:113-137`) keeps qwen's row-major text.

- **The page is then marked.** `ps.best_output = decision.final_output` (qwen row-major,
  `audit_passed=False` set at `orchestrator.py:1665`); because the page has tables and
  nothing was accepted, `ps.native_table_structure_failed = True`
  (`orchestrator.py:1678-1679`).

- **What ships (the D-policy, in `manifest.py`).** `_winning_page_output`
  (`manifest.py:239-320`): best_output is not `audit_passed`, so line 257 is skipped; the
  page **is** born-digital with native_text, so line 259 fires and it returns the **native
  text** (= the COLLAPSED `find_tables()` table) flagged `WARNING` / `audit_passed=False`
  (`manifest.py:265-276`, `native_is_fallback=True` because `native_table_structure_failed`).

**Net:** qwen's row-major (readable but ragged) attempt is correctly hard-failed and sits in
`best_output`/`attempts`, but the document ships the **collapsed native** table — arguably
the worse artifact — flagged audit-failed. This is the contested selection (option D).

Confirmation of evidence vs. ticket assumptions:
- TRUE: the verifier "correctly hard-fails" qwen's ragged output (`native_verifier.py:427`).
- TRUE: the collapsed native table comes from `find_tables()` + `_table_to_markdown`, and
  `reconstruct_table_regions` only runs on empty `find_tables()` (`born_digital.py:728`).
- NUANCE the ticket understates: the verifier hard-fail is a *blunt* signal. It rejects but
  emits no per-column repair constraint; and because every rung gets the identical ragged
  output, ladder escalation alone cannot fix raggedness. A *constraint* must be injected
  into the re-ask, or the native fallback itself must be made non-collapsed.

---

## 1b. Reframe — the real problem is REGION-AWARE extraction; single-table is the N=1 case

The CE p.4 failure is not fundamentally "a dense table came out ragged." It is that the page
is **already multi-region** and the pipeline extracts and verifies it as ONE thing. p.4
contains, in reading order:

1. the main forecaster grid (28 forecasters × 10 indicators × 2 years);
2. a second **"Historical Data"** table (2020-2023) — a *different* schema and column set;
3. a **Federal Budget Balance chart** (a figure, not a table);
4. a **"Government and Background Data"** text box (prose).

Whole-page extraction is the root cause of the verifier blowup: `native_verifier`
`_get_native_lane_count` clusters **every** numeric token on the page into lanes
(`native_verifier.py:114-148`), so it counted ~27 lanes spanning *both* tables plus the
chart axis, then compared that against ONE merged 11-column output — `geometry_impossible_collapse`
is the inevitable result of comparing two tables' worth of lanes to one table's columns. The
fix is not only "make the one table cleaner"; it is to **stop treating the page as one
region.**

**General fix (the target architecture): segment → extract+verify per region → reassemble.**

1. **Segment** the page into table / figure / text regions with bounding boxes. Geometry can
   propose candidates (`find_tables()` returns a `.bbox` per table; figure/chart bboxes come
   from the figure pipeline — `has_chart_marks` `extractor.py:777` and the per-figure bboxes
   GH-47C added), and/or the VLM can bound them (see Q3).
2. **Extract + verify each region INDEPENDENTLY.** The geometry check is then scoped to ONE
   table's lanes (cluster only the numeric tokens whose `(x, y)` fall inside that region's
   bbox), so the lane count finally matches a single table's column count and the verifier
   becomes meaningful instead of a whole-page false-fail. Table 1 and Table 2 are verified
   against their own schemas; the multiset value guard is applied per region.
3. **Reassemble in reading order.** Tables come back as separate markdown grids; the chart
   region routes to the **image-asset lane** — render the region PNG, embed an image ref, and
   transcribe NO data values (the existing chart lane: `_is_chart_asset_page`
   `orchestrator.py:1128`, `chart_asset` PageOutput `orchestrator.py:1608-1635`, which already
   embeds a PNG ref and records a `chart_asset_page` AuditEvent stating "data values not
   transcribed"). The text box stays native prose.

**The single dense-table repair (Options A/B/D above) is the N=1 special case** of this — one
table region, no figure. Whatever the panel picks for A/B/D **MUST be expressed per
table-region**, not per page, so it composes upward to the multi-region case rather than
fighting it. Concretely: B's rowizer must run on a region bbox, not the whole page; D's
selection policy is the per-region fail-closed floor; A's re-ask must be scoped to one
table's geometry.

**Be critical — two risks with the region architecture:**

- **Reading-order reassembly is the main NEW failure mode.** On a multi-column page layout,
  ordering regions by `bbox.y0` alone (the heuristic `extract_structured` uses today,
  `born_digital.py:737`) interleaves left-column and right-column content wrongly. CE p.4 is
  largely single-column top-to-bottom so y-order is probably safe HERE, but the general fix
  needs a column-aware reading order (x-band then y) or it will scramble two-column papers.
  Flag this as its own design risk, not a free side effect.
- **Do NOT make region-aware the default.** A one-table-per-page document (the academic-paper
  corpus, Fama-French) has N=1; running segmentation there adds cost and a new mis-bounding
  failure mode for zero benefit. Trigger region-aware extraction **on demand** — when
  `find_tables()` returns >1 region, OR a table region and a figure/chart region coexist on
  the page (table present AND `has_chart_marks` true) — and otherwise keep the existing
  single-region path. This keeps the paper corpus on its proven path and scopes the new
  machinery to genuinely multi-region pages like CE.

---

## 2. Options

The four candidates (A re-ask, B fix native fallback, C cloud escalation, D selection
policy) are **not mutually exclusive** — B and D are deterministic and orthogonal to A/C.
The real architectural fork is "deterministic native rowizer vs. model re-ask" for the
authoritative repaired table.

### Option B — Fix the native fallback (deterministic, model-free)
Change `extract_structured` so a *lane-stacked* `find_tables()` cell triggers the
word-geometry rowizer instead of only firing `reconstruct_table_regions` on empty results.
Concretely: detect that a `find_tables()` region has embedded newlines / a name-cell with
≥ a data-derived multiple of the row count, and replace that region's markdown with
`reconstruct_table_regions(page)` output (the same `find_tables(strategy="text")` +
`has_numeric_columns` gate already validated on Fama-French). `reconstruct.py` already
builds a real row×column grid from word x-lanes and y-clusters, char-exact, no model.

- **Cost:** zero (PyMuPDF only). **Determinism/replay:** fully deterministic; identical
  bytes on replay. **Scope:** `born_digital.py` (one gate) + possibly a "is this region
  lane-stacked" predicate. **Silent-loss risk:** LOW *if* the rowizer's existing keep-gates
  (`_looks_tabular`, `_MIN_DATA_ROW_FRAC`) hold; the rowizer either produces a clean grid or
  rejects and the page falls through to plain text — it never ships a shredded grid
  (`reconstruct.py:182-195`). The text-strategy rowizer is a *different* algorithm from the
  "explode collapsed grid on newlines" trap the ticket ruled out: it groups by native word
  geometry (x-lanes, y-rows), not by zipping pre-collapsed newline lists, so multiline
  names, true blanks, wrapped footnotes, and mixed schema rows are handled by geometry, not
  by positional zip. **Residual risk:** the CE summary block (Consensus/High/Low/StdDev) and
  the IMF/OECD comparison block are *different schemas* in one region; `find_tables(strategy="text")`
  may merge or mis-split them. Must be validated on the real p.4 before trusting.

### Option A — Verifier-guided repair re-ask (model in the repair loop)
On hard-fail, re-prompt qwen ONCE with the native geometry as a hard constraint derived from
the verifier: "this page has N data rows and C numeric columns; every data row must have
exactly C cells; emit `na` for blanks." N and C come from the verifier's already-computed
`native_lane_count` and the y-clustered row count (`_rows_by_y`, `native_verifier.py:151`) —
**no magic threshold; the constraint is read from the page's own geometry.** Re-verify; if it
still hard-fails, fall through to B (or D). One retry, not a tunable count (the count itself
would be a magic number — one constrained re-ask is the principled bound).

- **Cost:** +1 VLM call per failing table page (local qwen ~90-125 s on dense tables; cloud
  rung adds $). **Determinism/replay:** a model is now in the repair path — non-deterministic
  output, replay reads the frozen blob so replay is stable, but a *fresh* run can differ.
  This is exactly the "model where determinism is required" the settled architecture warns
  against, but note the *routing* and *verify* gates stay deterministic — only the
  *extraction* re-ask is the model, which is already a model (qwen does the OCR). **Scope:**
  `agentic.py` repair loop + a constrained prompt. **Silent-loss risk:** MEDIUM — a
  constrained re-ask can pad to C cells by *inventing* `na`s to satisfy the cell count,
  converting raggedness into a silent wrong-value. The verifier counts cells, not values, so
  a padded-but-wrong row passes. Needs a value-preservation cross-check (native numeric token
  multiset ⊆ output multiset) to stay fail-closed.

### Option C — Cloud escalation on hard collapse
In non-strict mode, route a hard-collapsed table page to a cloud rung (gemini /
qwen3.5:cloud). The ladder already carries a `TIER_CLOUD` qwen rung
(`providers.py:76-85`); the missing piece is that the *same* prompt to a cloud model is no
more likely to be non-ragged than local qwen — C only helps if combined with A's constraint
prompt. In `--strict-local` the cloud rung is stripped (`orchestrator.py:1339-1342`,
`TIER_LOCAL` filter), so C is a no-op there and the strict-local terminal behavior is whatever
B/D decide. **Cost:** cloud $ per page. **Determinism:** model in path, same caveat as A.
**Silent-loss:** same raggedness/padding risk as A. **Verdict:** C is not a standalone fix; it
is A-with-a-bigger-model and inherits A's risks plus cost. Fold into A as "if local re-ask
still fails AND not strict-local, try the cloud rung with the same constraint."

### Option D — Selection policy (which artifact ships when both local artifacts fail)
Today the collapsed native wins over qwen's row-major (`manifest.py:259` native branch beats
the failed best_output). Flip it: when `native_table_structure_failed` AND a row-major
best_output attempt exists whose native-numeric-token multiset is a *superset-or-equal* of
the native layer's tokens (no value dropped, only a structural raggedness), ship the
**row-major** attempt flagged WARNING instead of the collapsed native. A reader can repair a
ragged row-major table by eye; a fully-collapsed cell is unrecoverable. **Cost:** zero.
**Determinism:** fully deterministic (a selection rule over existing artifacts).
**Silent-loss:** the contested point — qwen's raggedness *could* be a silent column-shift
(the missing cell is mid-row, so every value after it is one column off). Shipping that
flagged is arguably worse than shipping an obviously-collapsed cell, because a column-shifted
number *looks* plausible. This is the genuine fork.

---

## 3. Recommendation

**Hybrid B + D, with A explicitly panelled as the contested escalation.**

1. **B first (deterministic, no model, no cost):** make the lane-stacked `find_tables()`
   region trigger the word-geometry rowizer. This is the highest-leverage, lowest-risk fix
   and is squarely in the "deterministic code + config over an LLM in the path" lane. If B
   produces a clean grid on the real p.4 (must be validated — the summary/comparison
   multi-schema blocks are the risk), the page no longer needs OCR at all and ships a correct
   char-exact native table. This likely resolves the ticket's core "tables" requirement.

2. **D as the fail-closed floor:** when both B's native rowizer AND qwen fail verification,
   the selection policy decides what ships. This is where I do NOT have enough evidence to
   pick unilaterally: "ship qwen's ragged row-major (readable, risk of silent column-shift)"
   vs. "ship the collapsed native (unrecoverable but obviously broken)" is a real
   no-silent-loss judgment call. **Panel it.**

3. **A only if B is insufficient on the real grid.** A puts a model in the repair path and
   carries the padding/silent-value risk; it should be gated behind a value-preservation
   cross-check and one constrained re-ask, and is the *escalation* after B, not the primary
   fix. C is A-with-cloud and folds in.

This is a **genuine fork on D** (and on whether A is acceptable at all given no-silent-loss);
B is largely a straight bug fix but its viability on the CE multi-schema page is unproven and
worth one panel sanity-check.

---

## 4. The consilium question(s)

Read these through section 0 ("VLM for structure, geometry for values") and section 1b
(region-aware extraction). **Q1's selection policy is the per-region fail-closed floor** —
applied to each table region independently, not the whole page.

### Q1 (primary — the D selection fork, applied per table region)

> In socr's agentic table path, a born-digital dense-numeric page (Consensus Economics
> forecaster grid: 28 rows × 20 numeric columns) produces TWO flawed local artifacts, and a
> deterministic PyMuPDF geometry verifier hard-fails both: (a) the local VLM (qwen3-vl:30b-a3b-instruct)
> emits a ROW-MAJOR markdown table that is readable but RAGGED — one data row has 19 cells
> where 20 are required, so a mid-row missing cell silently shifts every later value one
> column left; (b) the native PyMuPDF `find_tables()` fallback emits a FULLY-COLLAPSED table
> (28 names newline-stacked in one cell, numbers newline-stacked per column) that is
> unrecoverable but obviously broken. Both ship flagged `audit_passed=False`. Today the
> collapsed native wins. Under a strict no-silent-content-loss principle (a wrong/shifted
> number is worse than an obviously-missing one), which artifact should the document ship when
> BOTH fail verification, and what cheap deterministic guard (e.g. requiring the row-major
> attempt's native-numeric-token multiset to equal the page's native token multiset before
> preferring it) makes that choice safe?
>
> - **Option D1 — ship collapsed native (status quo):** obviously-broken cell, no false
>   plausibility, but unrecoverable by a downstream reader.
> - **Option D2 — ship qwen row-major flagged:** human-repairable layout, but a ragged row is
>   a silent column-shift; gate it behind a native-token-multiset-equality check so it is only
>   preferred when no value was dropped (raggedness is purely structural, not a lost cell).
> - **Option D3 — ship neither as a table:** emit an explicit per-page failure marker / route
>   the page to the image-asset lane (save the rendered PNG, transcribe nothing), treating an
>   unverifiable dense grid as a figure rather than risking either bad table.

### Q2 (the model-for-structure question — leans YES, hard-gated by a deterministic value check)

> socr's settled architecture keeps deterministic code in the routing/verify path, but the
> "VLM for structure, geometry for values" principle says the VLM is the document-robust
> engine for table *structure and segmentation* (column boundaries on line-less tables,
> header-vs-data rows, multi-line cells) while deterministic geometry is the robust check that
> no value was dropped or shifted. Given that, should socr — when the deterministic word-geometry
> rowizer cannot produce a verified table — attempt ONE re-ask of the local VLM
> (qwen3-vl:30b-a3b-instruct) constrained by the page's OWN native geometry ("this REGION has N
> data rows and C numeric columns derived from the PyMuPDF word lanes inside its bbox; every
> data row must have exactly C cells; emit `na` for blanks"), accepting the result ONLY if a
> deterministic value check passes? The routing and verify gates stay deterministic; only the
> extraction re-ask is the model (and qwen already does the OCR). The trap to gate against: a
> cell-count constraint can be satisfied by PADDING with invented `na`s, turning raggedness
> into a silent wrong-value that a cell-counting verifier would pass — so the accept gate must
> be value-level, not count-level.
>
> - **Option A1 — no model re-ask (deterministic-only):** rely solely on the word-geometry
>   rowizer (Option B: make lane-stacked `find_tables()` regions run `reconstruct_table_regions`)
>   plus the Q1 per-region selection policy; never put a model in the repair loop. Safest on
>   determinism/replay; fails when geometry genuinely cannot infer the line-less column
>   structure.
> - **Option A2 — one constrained re-ask, VALUE-guarded (the principle's embodiment):** allow
>   exactly one geometry-constrained re-ask (VLM leads structure), but accept ONLY if its
>   native-numeric-token multiset is a superset-or-equal of the region's native tokens — no
>   value invented or dropped (geometry verifies values). Otherwise fall through to Q1. This is
>   "VLM for structure, geometry for values" made concrete.
> - **Option A3 — one constrained re-ask, cell-count only:** accept on cell-count match alone
>   (cheapest, but exposes the padding silent-value risk — rejected by the no-silent-loss rule
>   unless the panel disagrees).
>
> Non-strict-mode note: if A2's local re-ask still fails AND `--strict-local` is off, the same
> value-guarded re-ask may escalate to the cloud rung (`qwen3.5:cloud`, `providers.py:76-85`);
> in `--strict-local` the cloud tier is stripped (`orchestrator.py:1339-1342`) so the terminal
> behavior is whatever Q1 decides for that region.

### Q3 (segmentation ownership — who bounds the regions on a multi-table + figure page)

> For a multi-region page like CE p.4 (main forecaster grid + a second "Historical Data" table
> with a different schema + a Federal Budget Balance chart + a text box), region-aware
> extraction must first BOUND the regions before any per-region extract/verify. Who owns
> segmentation, given that geometry is char-exact for values but brittle for *structure* on
> line-less / multi-schema CE layouts (the default `find_tables()` "lines" strategy already
> over-merged the two tables into one region — `born_digital.py:712`), whereas the VLM is
> document-robust for visual segmentation but could mis-bound a region (e.g. swallow the chart
> into the table, or split one table in two)?
>
> - **Option S1 — geometry-led:** derive regions from deterministic bboxes only — `find_tables()`
>   table bboxes + figure/chart bboxes (`has_chart_marks` / GH-47C per-figure bboxes,
>   `extractor.py:777`). Fully deterministic and replay-stable, but brittle exactly where CE
>   breaks: line-less tables over-merge and multi-schema blocks are mis-split.
> - **Option S2 — VLM-led:** ask the VLM to return region boxes + types (table / figure / text)
>   in reading order; extract each, then verify table regions with scoped geometry. Robust
>   across document styles (the VLM sees the four regions a human sees), but the model could
>   mis-bound, and segmentation now depends on a non-deterministic call.
> - **Option S3 — hybrid (geometry proposes, VLM confirms/splits):** geometry emits candidate
>   regions (table bboxes, chart bboxes); the VLM confirms each, splits an over-merged region
>   into its real sub-tables, and supplies reading order — with a deterministic post-check that
>   every native numeric token lands in exactly one region (no token orphaned or double-counted).
>   Most robust, most complex; the post-check is the geometry safety-net for segmentation.

(If the panel endorses A1 + S1, the implementation is pure deterministic geometry + the Q1
per-region selection policy, and no model enters the repair OR segmentation path — at the cost
of the line-less/multi-schema robustness only a VLM provides.)

---

## 5. Acceptance-criteria readiness

The ticket's acceptance criteria (issue #56) are mostly implementable once the design is
chosen, with these caveats:

- "A CE smoke run on `202401.pdf` produces a Markdown file better than native on dense tables
  without degrading chart/front-matter pages" — **implementable and directly testable** once
  B lands; needs the real p.4 as the fixture. Tighten to name the metric: a *cell-by-cell*
  parity check against the validated ground truth, not a subjective "better."
- "Manifest shows expected mixed routing by page type, not blanket native and not blanket
  qwen" — **already largely true** (issue evidence: qwen 31 / native 1); B may move some
  pages back to native (correct char-exact tables), which is the desired direction. Tighten
  to assert specific pages route as expected rather than a blanket count.
- "Audit log flags failed table/figure pages clearly" — **already implemented**
  (`native_table_verifier_hard_fail` AuditEvent, `agentic.py:416`; `native_fallback`
  marker). No new work beyond ensuring the D-selection emits a distinct event for "shipped
  row-major over collapsed" vs. "shipped collapsed."
- **NEW — multi-region parity (region-aware fix, section 1b).** A frozen CE p.4 fixture must
  reconstruct **BOTH** tables as **separate** markdown grids (the main 28×20 forecaster grid
  AND the second "Historical Data" 2020-2023 table, each with its own schema), the Federal
  Budget Balance chart as an **image asset** (PNG ref via the chart lane, NO transcribed data
  values — `orchestrator.py:1608-1635`), and the "Government and Background Data" text box as
  prose, all in reading order. The per-region geometry verifier must pass against each table's
  OWN lanes (not the whole-page lane count that causes today's `geometry_impossible_collapse`).
  This is the criterion that proves the region-aware fix, and it cannot be satisfied by any
  whole-page A/B/D variant — it forces per-region extraction.
- "Dual-pass table extraction cannot hang the run" — **already addressed** (PP-0 deadline +
  cascade guard; PR #79 provenance). Out of this design's scope.
- "Clear recommended CE corpus command before overwriting the production OCR" — **blocked on
  Q1/Q2/Q3 outcome**; cannot be finalized until the per-region selection, repair, and
  segmentation policies are decided.

The one criterion that needs tightening before it is implementable-as-written: "better than
native output on dense tables" must become a concrete cell-level acceptance test on a frozen
p.4 ground-truth fixture, otherwise it is unfalsifiable.

---

## 6. Panel verdict & v1 plan (codex gpt-5.5 + gemini/antigravity, 2026-06-17)

Both panelists read this note and answered Q1-Q3 independently. **Strong consensus on all
three forks; one disagreement on v1 sequencing.**

| Fork | Verdict | Notes |
|------|---------|-------|
| **Q1** — which artifact ships when both local tables fail (per region) | **D3 — ship neither as a table** | Failure marker + route the region/page to the image-asset lane (rendered PNG). Both explicitly **reject D2**: a token-multiset check does NOT prove column *placement*, so a ragged row still risks a silent column-shift (the no-silent-loss violation). |
| **Q2** — may the VLM enter the repair loop | **A2 — one constrained, value-guarded re-ask** | …but **both independently flagged the same hole**: "superset-or-equal" lets the VLM invent *new* numbers into blank cells (instead of `na`) and still pass. **Refinement (adopt): the accept gate must be lane-aware token EQUALITY / no-extra-tokens, not loose superset.** A2 is deferred to v2 regardless. |
| **Q3** — segmentation ownership | **S3 — hybrid** | Geometry proposes regions, VLM confirms/splits, deterministic token-coverage post-check (every native token lands in exactly one region). Pure-geometry already proven broken on CE (`find_tables` over-merges); pure-VLM too risky for no-silent-loss. |

**The one disagreement — what ships first:**
- **Codex: floor first (D3).** Stop shipping plausible-but-wrong data immediately; any region
  that can't be verified becomes an explicit failed-table block + PNG. "Then A2/S3 improve
  recovery without corrupting the corpus."
- **Gemini: root-cause fix first (B + per-region).** The verifier blew up because it judged
  multiple tables + a chart as one monolithic block; route lane-stacked `find_tables()` regions
  to the deterministic rowizer **per region** — "might yield clean grids via Option B alone,
  entirely bypassing the need to inject a VLM into the repair loop."

**Synthesis (adopted) — they are the two ends of one v1, not alternatives:**

> per-region segmentation (fixes the monolithic-verifier root cause) → deterministic rowizer
> (Option B) recovers clean grids at zero model cost → D3 fail-closed floor catches whatever B
> can't → **A2 re-ask and full-S3 VLM-split deferred to v2.**

This captures gemini's "fix the cause" *and* codex's "never ship wrong data." v1 is entirely
**deterministic** (no model in the repair/segmentation path); the VLM only enters in v2 (A2
re-ask, S3 VLM confirm/split), once we have measured whether deterministic per-region + rowizer
already produces clean CE grids. v1 segmentation is therefore **geometry-led with a deterministic
token-coverage post-check** (the S3 hybrid minus the VLM-confirm step, which is v2).

See `docs/plans/table-repair/` for the scoped v1 tickets (TR-0…TR-5).
