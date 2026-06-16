# TICKETS — progressive page processing

Canonical backlog for the **progressive-pages** initiative. See `README.md` for the design
rationale and the `/consilium` decisions (`20260615T181400Z-8388`); see `STATUS.md` for live
execution state.

Status keys: `READY`, `NEEDS-DESIGN`, `BLOCKED`, `WIP`, `DONE`, `DEFERRED`.
Agents (in `.claude/agents/`): `socr-designer` (read-only design pass on NEEDS-DESIGN tickets,
writes a design note + frames the `/consilium` question), `socr-implementer` (one bounded code
ticket), `socr-reviewer` (adversarial review after a completed ticket). `/consilium` is run by the
orchestrator only.

## Dispatch rules

- One implementation ticket per subagent; give each a **disjoint write set**.
- If two tickets touch `orchestrator.py`, they must occupy **distinct method regions** (see each
  ticket's `Write ownership`) or be serialized — never run concurrently otherwise.
- Each implementation ticket ends with a `socr-reviewer` pass before acceptance.
- Use `uv run` (or the direct `~/venvs/socr/bin/*` binaries) for all Python. Never `python script.py`.
- Commit on the initiative branch (see `STATUS.md`), stage files **by name**, never `git add -A`,
  do not push, one commit per ticket. If you hit an architectural fork you cannot resolve from the
  ticket, stop and return `CONSILIUM-GATE` with a one-sentence question.

Line numbers below are **approximate and drift**; method names are the stable anchors.

---

## PP-0 — Crop-reread releasing wall-clock deadline + timeout audit + cascade guard

GitHub: https://github.com/r-uben/socr/issues/55
Status: **READY** · Priority: P0 · Agent: `socr-implementer` · Depends on: none · Wave 0

### Problem
`_phase_dual_pass_tables` crop rereads call `OllamaTableReader.read → httpx.post("…/api/generate",
timeout=crop_timeout)` (`tables/extract.py:74`) with **only an httpx scalar timeout** and no outer
wall-clock guard. httpx read-timeout does not reliably release a wedged Ollama socket, so one crop
holds the whole document hostage — and the only `.md` write is `_phase_assemble`, so nothing is
saved. Shippable **before** any progressive restructure; de-risks the whole effort.

### Plan
1. Wrap the crop reread in a `ThreadPoolExecutor(max_workers=1) + future.result(timeout)` guard
   mirroring `route_page` (`pipeline/agentic.py:213-247`); on `TimeoutError`, release and continue.
2. Crop-tuned timeout, **separate** from the full-page qwen 300 s constant (`orchestrator.py:~1426`)
   — derive from crop count/area, not full-page latency. Make it a named constant with documented basis.
3. On a timed-out or failed crop, append a durable `AuditEvent` (e.g. `dualpass_crop_timeout`) to
   `state.events` with `page_num` + reason, so `audit_log.json` records it (#55 acceptance); then
   degrade to **flag-only reconcile** (keep existing text) — the page still flows to assemble.
4. **Wedged-GPU cascade guard** (consilium refinement): after a VLM/crop timeout, do **not** blindly
   issue the next VLM call. Probe local-backend health (idle/ready); continue only if recovered,
   else mark the local VLM backend degraded for this document and stop further crop rereads.
   Provide the releasing-deadline helper in a form PP-2 can reuse for the document-level
   `PARTIAL_SAVE_VLM_TIMEOUT` halt decision.
5. Close the leaked fitz handle from `get_fitz_page` (`orchestrator.py:~1554`) — `with fitz.open`
   or cache+close per page.
6. Document the residual honestly: closing the client does not abort Ollama server-side generation
   (`stream:false`); the local GPU stays busy until the model finishes, so strict-local stays
   strictly serial. (Ollama-side cancellation is out of scope here — see README open question 3.)

### Write ownership
`src/socr/tables/extract.py`; `src/socr/pipeline/orchestrator.py` (**only** `_phase_dual_pass_tables`
+ `get_fitz_page` regions).

### Acceptance
- A deliberately wedged Ollama crop read releases within the configured deadline and does not block
  the document indefinitely.
- A timed-out crop produces a durable `AuditEvent` in `audit_log.json` with `page_num` + reason.
- After a crop timeout, the pipeline does not immediately fire another VLM call into the wedged
  backend (cascade guard verified).
- No fitz `Doc` handle is leaked per judged/table page (handle count stable over a multi-table doc).
- Existing dual-pass tests pass; reconcile still flags-not-patches on count mismatch.

### Verification
- `uv run pytest tests/test_dual_pass_tables.py tests/test_reconstruct.py -q`
- `uv run ruff check src/socr/tables/extract.py src/socr/pipeline/orchestrator.py`

---

## PP-6 — Fix #54 over-routing: lane-cooccupancy table gate + content-type vector on PageState

GitHub: https://github.com/r-uben/socr/issues/54
Status: **READY** · Priority: P0 · Agent: `socr-implementer` · Depends on: none · Wave 0
Pairs with: **PP-7** (must land together or PP-7 immediately after — see below).

### Problem
The native-vs-ladder routing decision forces a born-digital page onto the Qwen VLM whenever
`has_tables` is set, and `has_tables` is set by the loose `_detect_columnar_numbers` heuristic
(`core/born_digital.py:~610`, single-token columnar-number ratio) that false-fires on CE chart-axis
labels and front-matter (observed: 31 qwen / 1 native on `202401.pdf`). The stronger
`has_numeric_columns` lane-cooccupancy gate (`tables/reconstruct.py:~110`) exists but is used only
for native reconstruction, not routing. Separately `apply_born_digital` (`core/state.py:~149`)
**drops** `has_figures` / `has_equations`, so the per-page gate can't route on content type.

### Plan
1. Route on the stronger `has_numeric_columns` lane-cooccupancy signal for the native-vs-ladder
   decision (`_is_trusted_native_without_ocr`, `orchestrator.py:~1032`), instead of the loose
   single-token-line ratio.
2. Propagate `has_figures` / `has_equations` onto `PageState` (`apply_born_digital`,
   `core/state.py:~149`) so the per-page gate is self-contained, not re-reading `_last_assessment`
   ad hoc (`orchestrator.py:~654, ~711`).
3. Audit the consumers of `has_tables` (`orchestrator.py:~664, ~941, ~1049, ~1132, ~1233, ~1563`;
   `native_verifier`; dual-pass `table_pages` at `~1413`) and confirm narrowing it narrows the
   verifier + dual-pass (intended) **without** breaking native reconstruction.
4. **Do not** let `has_figures` grant routing leniency (a chart page must not silently ship as
   native prose either) — the explicit chart handling is **PP-7**. PP-6 narrows; PP-7 catches.

### Write ownership
`src/socr/core/born_digital.py`; `src/socr/core/state.py` (`apply_born_digital`);
`src/socr/pipeline/orchestrator.py` (**only** `_is_trusted_native_without_ocr`, `_page_has_tables`).

### Acceptance
- CE chart-axis / front-matter pages that previously tripped `has_tables` are no longer routed to
  Qwen (regression fixture from a real CE doc).
- A genuine dense forecast table still routes to the ladder (no false-negative regression).
- `has_figures` / `has_equations` are readable from `PageState`; ad hoc `_last_assessment` re-reads
  in the agentic gate are removed.
- Existing routing tests pass; native reconstruction path unchanged.

### Verification
- `uv run pytest tests/test_born_digital.py tests/test_orchestrator.py tests/test_reconstruct.py -q`
- `uv run ruff check src/socr/core/born_digital.py src/socr/core/state.py src/socr/pipeline/orchestrator.py`

---

## PP-7 — Chart / figure-page route to image-asset extraction (the #54/#56 third lane)

GitHub: https://github.com/r-uben/socr/issues/56 (also #54, #47)
Status: **READY** (design resolved) · Priority: P0 · Agent: `socr-implementer` · Depends on: PP-6 · Wave 2
Design: `docs/log/2026-06-16_chart-route.md` · Decision ratified by `/consilium` 2026-06-16 (Codex + Gemini, **unanimous A2 + B1**).

### Problem
Narrowing `has_tables` (PP-6) without a chart lane means a chart/front-matter page that is no
longer a table — and isn't otherwise flagged — ships as **native prose**: a word-salad of floating
axis labels, legend entries, and data points, with the visual payload lost. For CE this destroys
the data. There must be a **third outcome** neither the loose nor the strict table gate produces:
route such pages to figure-asset extraction.

### Design decision (`/consilium` 2026-06-16, Codex + Gemini — unanimous A2 + B1)

The design pass (`docs/log/2026-06-16_chart-route.md`) found and **verified** a load-bearing error in
this ticket's original assumption: routing on `PageState.has_figures` does NOT work, because
`has_figures = has_images = len(page.get_images()) > 0` (`born_digital.py:287`) is **raster-only** and
CE charts are **vector** (`get_drawings`, zero embedded raster) — so it reproduces the exact #54 bug.
The model-free vector substrate (`_has_vector_data_marks` `figures/extractor.py:692`,
`_looks_like_table_grid:725`, `get_drawings`) already exists in the extractor but is never called from
the router. The panel ratified **A2 + B1**:

- **Detection = A2 — deterministic `has_chart_marks(page)`** reusing the extractor's vector logic, OR-ed
  with raster `has_images`. Route to the chart lane only when
  `born_digital && !table_lane && !clean_prose_lane && has_chart_marks(page)` (positive evidence, not a
  residual — A3 over-routes sparse prose; A1 is blind to vector).
- **False-positive bound (Gemini, load-bearing):** do NOT count marks on the flat `get_drawings()` list
  — 3 scattered decorative rules can sum to `MIN_DATA_MARKS`. **Cluster first** (`_cluster_drawings`) and
  require ≥1 spatially-coherent cluster that (a) meets a min bounding-box area, (b) passes
  `_has_vector_data_marks`, and (c) survives `_looks_like_table_grid` rejection. Keep the prose gate in
  front. This deterministically eliminates decorative furniture (borders, header/footer rules, hairlines,
  near-white fills, single background rects).
- **Representation = B1 — keep native prose + embed the chart PNG + emit an explicit audit flag**
  ("visual chart semantics represented as image asset; data values not transcribed"). B2 (asset-only) is
  rejected: on a false-positive it silently drops real text (titles/source lines), violating the
  no-silent-loss invariant. B1 degrades safely to "redundant PNG + intact prose".
- **Force PNG on chart-lane pages even when `--save-figures` is off** (both panelists). Narrow the flag
  contract: `--save-figures` = opportunistic figure extraction on prose pages; **chart-lane page captures
  are mandatory preservation artifacts.** PNG render failure → **fail-closed / hard audit error**, never
  a quiet markdown page.
- **Residual risk = false NEGATIVES** (monochrome / thin-stroke B&W academic line plots that look like
  rulings). `has_chart_marks` must log mark counts + rejection reason; add B&W academic-plot fixtures.

### Plan
1. Add model-free `has_chart_marks(page)` **in `src/socr/figures/extractor.py`** (where `_cluster_drawings`
   / `_has_vector_data_marks` / `_looks_like_table_grid` already live) — cluster-first per the
   false-positive bound; OR with raster `has_images`. Any new threshold (min cluster area) is a documented
   named constant.
2. Add a **new** routing predicate `_is_chart_asset_page(page_num, ps)` in `orchestrator.py` (do NOT
   rewrite PP-6's `_is_trusted_native_without_ocr` / `_page_has_tables`) firing when
   not-table && not-prose && `has_chart_marks`.
3. Route chart-asset pages to the figure/image-asset pipeline (B1): retain native prose, force PNG
   extraction (overriding `--save-figures` off), embed the chart PNG ref, emit the audit flag. Captions
   stay optional (`--describe-figures`, non-authoritative).
4. Keep it a routing/asset decision; chart data-value verification stays OUT of scope (weak-verify, #47C).

### Write ownership
`src/socr/figures/extractor.py` (`has_chart_marks`), `src/socr/pipeline/orchestrator.py` (new
`_is_chart_asset_page` + chart-lane routing/representation — do NOT touch PP-6's predicates), figure/routing
tests. **Disjoint from PP-6** by construction (PP-6 owns `born_digital.py` + the trusted-native predicates;
PP-7 owns the extractor signal + a new predicate). Land **after** PP-6.

### Acceptance
- A **genuine vector** CE chart page (zero embedded raster) routes to the chart lane (fixture MUST be real
  zero-raster, or the criterion isn't exercised).
- A chart-lane page is represented as **native prose + embedded chart PNG + audit flag** (B1) — not a Qwen
  table reconstruction and not silently-dropped prose.
- The chart PNG is produced **even when `--save-figures` is off**; a render failure fails-closed with a
  hard audit error.
- Decorative-vector prose pages (boxed/ruled prose) do NOT false-trigger the chart lane (cluster-first
  bound); dense data tables still route to the ladder; clean prose still ships native.
- `has_chart_marks` logs mark counts + rejection reason; a B&W academic line-plot fixture is included.

### Verification
- `uv run pytest tests -q -k "figure or born_digital or route or chart"`
- `uv run ruff check src/socr/figures/extractor.py src/socr/pipeline/orchestrator.py`

---

## PP-1 — Per-page fragment flush primitive + atomic sidecar + end-of-run stitch

GitHub: new · Status: **READY** · Priority: P0 · Agent: `socr-implementer` · Depends on: none · Wave 1

### Problem
There is exactly one `.md` write (`_save_markdown`) over the whole-doc body; no append/fragment
primitive exists. Progressive save needs a way to write one finished page and stitch all pages at
the end **bit-consistently** with the contract's `## Page N` structure. Per the consilium
refinement, a fragment alone enables *human* salvage but not *automatic* resume — so a minimal
machine-readable sidecar ships in this ticket, not deferred.

### Plan
1. `_flush_page_fragment(state, page_num, text, output_dir)` → writes `<doc_dir>/pages/NNN.md`
   (zero-padded), atomic temp+rename. Fragment text is **body-only canonical page text** (no page
   header) so the stitch reproduces today's bytes exactly.
2. Atomic sidecar `<doc_dir>/pages/NNN.json` (`.json.tmp` → rename) carrying: `page_num`,
   `status`, `terminal:bool`, `engine`/`provider`, `cost`, **page-fingerprint + run-fingerprint**,
   and the `PageState` decision flags not in `PageOutput.to_dict`
   (`needs_ocr_enhancement`, `native_table_structure_failed`, `judge_rejected`), plus audit result,
   table-pass result, and figure refs. (PP-5 consumes/enriches this; PP-1 just writes it.)
3. `_stitch_fragments(state, output_dir) → str` reads `pages/*.md` in page order and joins via
   `contract.assemble_pages(pages, page_numbers=[…])`, producing the **same** body
   `_canonical_body` produces today.
4. Refactor `_phase_assemble` so the final body = stitched fragments when fragments exist, else
   today's in-memory assembly (non-agentic paths). `_save_markdown` still writes the final stitched
   `<stem>.md` exactly as today (canonical path unchanged). **No second `metadata.json` writer** —
   `RootIndex` stays sole author.

### Write ownership
`src/socr/pipeline/orchestrator.py` (`_phase_assemble`, new `_flush_page_fragment` /
`_flush_page_sidecar` / `_stitch_fragments`).

### Acceptance
- For a doc processed end-to-end, the stitched `<stem>.md` is **byte-identical** to today's
  whole-doc assembly (round-trips `PAGE_MARKER_RE`; `split_native_pages` recovers the same pages).
- `pages/NNN.md` + `pages/NNN.json` exist for every page after a successful run; sidecar writes are
  atomic (no corrupt JSON on a kill mid-write).
- A unit test asserts `assemble_pages(fragments) == in-memory canonical body` for a fixture doc.
- No second `metadata.json` writer introduced; the contract path is unchanged.

### Verification
- `uv run pytest tests/test_orchestrator.py tests/test_canon_round3.py -q`
- `uv run ruff check src/socr/pipeline/orchestrator.py`

---

## PP-3 — Fold dual-pass table reread into the per-page lifecycle

GitHub: https://github.com/r-uben/socr/issues/56 · Status: **NEEDS-DESIGN** · Priority: P1
Agent: `socr-designer` first · Depends on: PP-0 · Wave 2

### Problem
Table reread is a separate late global phase (`_phase_dual_pass_tables`) that mutates
`best_output.text` **after** extraction; in a page-major design a flushed page would need
rewriting. Tables must be reconciled per-page **before** the page flushes. `locate_tables`,
`TableCropExtractor.extract`, `reconcile_page_tables` are already single-page-pure.

### Plan
1. Extract `_reread_page_tables(state, page_num, fitz_page, reader)` from the loop body of
   `_phase_dual_pass_tables` (`locate_tables → extract → reconcile_page_tables → patch
   best_output.text + emit AuditEvents`).
2. Call it inside the per-page lifecycle (PP-2 step 4), after the page's OCR text is
   final-for-that-page, reusing **PP-0's deadline-guarded + cascade-guarded reader** and **one**
   open fitz page (no re-open per call).
3. Hoist crop-model resolution + reader instance (`orchestrator.py:~1407-1436`) to doc scope.
4. Skip native (born-digital) pages exactly as today (`bo.engine == "native"` continue).
5. Preserve the `dualpass_<action>` `AuditEvent` shape so audit/replay assertions and tests hold.
6. Gate Phase 4c off in agentic mode (the global pass is now redundant there) — coordinate the
   actual gate with PP-2 (which owns `process()`'s agentic branch).

### Write ownership
`src/socr/pipeline/orchestrator.py` (`_phase_dual_pass_tables` → `_reread_page_tables` refactor;
call site invoked from `_phase_agentic`). Distinct region from PP-1's `_phase_assemble`.

### Acceptance
- A table page's reread+patch happens **before** its fragment is flushed (the flushed fragment
  already contains patched table text).
- Per-page dual-pass produces the **same** patches/flags as the old global Phase 4c on a fixture
  table doc (parity test).
- `dualpass_patched` / `dualpass_flagged` (+ PP-0 timeout) `AuditEvent`s appear with correct `page_num`.
- The reader + fitz handle are reused per page, not re-instantiated per crop.

### Verification
- `uv run pytest tests/test_dual_pass_tables.py tests/test_orchestrator.py -q`
- `uv run ruff check src/socr/pipeline/orchestrator.py`

---

## PP-4 — Per-page figure extraction + inline embedding (off the doc-tail append)

GitHub: https://github.com/r-uben/socr/issues/47 (coordinates with #50, GH-47B) · Status: **NEEDS-DESIGN**
Priority: P1 · Agent: `socr-designer` first · Depends on: PP-2 · Wave 2
**Blocked on README open question 1 (output-layout contract) before implementation.**

### Problem
Figures are extracted document-wide once (`_describe_and_embed_figures`, `orchestrator.py:~2448`;
`FigureExtractor.extract` over the whole PDF) and appended as a **trailing** section. Per-page
processing needs per-page extraction with **inline** embedding, while preserving the doc-global
`figure_<N>_page<P>.png` filename contract.

### Plan
1. Per-page entry: reuse the existing private `_extract_page_figures` (`figures/extractor.py:~231`)
   taking one fitz page; pass the page already open in the lifecycle (no PDF re-open).
2. Thread doc-scoped state across pages: figure **counter** (`extractor.py:~154`), running
   **`max_total` cap** (`extractor.py:~163`) emitting `figure_cap_reached` as soon as exceeded, and
   the **single hoisted vision engine** (`_get_vision_engine`, built once at doc start, closed at
   doc end — no per-page `/api/tags` re-probe).
3. Embed figure blocks **inline** into the page fragment (after that page's text), not the doc tail;
   preserve the block format (`**Figure N** (page P)` + image ref).
4. Per page, run **save PNG → embed ref → `strip_phantom_images`** in that order so the just-added
   ref survives (`normalizer.py:~160` keeps a ref only if the PNG exists).
5. Skip scanned pages via `page_num in _last_assessment.scanned_pages()`, checked per page.
6. Keep figure numbering **doc-global** so `figure_filename` (`contract.py:~370`) and the
   papers-library layout stay intact.
7. **Equation seam (GH-36, in `main` via #59/#60).** The per-page **equation** splice re-homes at the
   same seam as figures (its design note calls it "PP-4-adjacent"). The actual fuse + the
   agentic-mode no-op gate on the two global equation phases are owned by **PP-2 step 6a** (single
   owner of `_phase_agentic`); PP-4 only needs to ensure the per-page **figure** embed and the
   per-page **equation** sidecar share one open fitz page and the same per-page write seam (no PDF
   re-open, consistent ordering before `strip_phantom_images`). GH-47C's `FigureInfo.bbox` /
   extractor bbox (now in `main` via #58) is the groundwork this per-page extractor builds on — reuse
   it, do not re-add it. Keep the equation path behind its default-off flags; this ticket does not
   enable it.

### Write ownership
`src/socr/figures/extractor.py` (per-page entry); `src/socr/pipeline/orchestrator.py`
(`_describe_and_embed_figures` decomposed into per-page; vision-engine lifecycle). The equation
splice itself stays under PP-2's `_phase_agentic` ownership (step 6a) — PP-4 coordinates the shared
per-page seam only. Distinct region from PP-3.

### Acceptance
- Figures appear **inline** within their page's `## Page N` section for agentic output.
- `figure_<N>_page<P>.png` filenames remain doc-global and monotonic (no renumbering regression).
- The `max_total` cap still fires with a durable `figure_cap_reached` `AuditEvent` at the crossing page.
- Vision engine is constructed once per doc and closed once.
- With `--save-figures` only (no `--describe-figures`), no VLM is called and blocks have empty
  descriptions (parity with today, GH-50).

### Verification
- `uv run pytest tests -q -k figure`
- `uv run ruff check src/socr/figures/extractor.py src/socr/pipeline/orchestrator.py`

---

## PP-2 — Make the agentic loop progressive: per-page lifecycle + immediate flush

GitHub: https://github.com/r-uben/socr/issues/49 · Status: **NEEDS-DESIGN** · Priority: P0
Agent: `socr-designer` first · Depends on: PP-0, PP-1, PP-3, PP-4 · Wave 3

### Problem
The agentic path runs classification (`orchestrator.py:~1085`) and OCR (`~1131`) as two
full-document loops, and the winning text is only materialized into the `.md` at `_phase_assemble`.
Per-page progressive processing requires fusing classify → route → extract → tables → figures →
verify → flush into one loop and flushing each page immediately.

### Plan
1. Fuse the two agentic loops into one per-page driver: classify (`detect_page`), route
   native-vs-ladder, extract via `route_page` (deadline-guarded), per-page tables (PP-3), per-page
   figures (PP-4), **per-page equations (PP-4-adjacent — see step 6a)**, finalize, then flush via
   PP-1's `_flush_page_fragment` + sidecar.
2. **Every page flushes** — native-trusted pages too, not only `ocr_pages` (today's loop iterates
   `ocr_pages`; CE docs have many native pages). Keep the existing `put_page` write-through for
   replay-cache continuity.
3. Keep doc-scoped setup hoisted before the loop (ladder, judge, blob store, vision engine, figure
   counter/cap) — do not rebuild per page.
4. Recompute per-page provenance-failed / `AUDIT_FAILED` accounting (today done at doc end) so a
   flushed page carries correct status in its sidecar.
5. **Document-level cascade halt** (consilium refinement): when PP-0's per-page deadline fires and
   the backend does not recover on probe, flush pages `0..N-1`, mark the document
   `PARTIAL_SAVE_VLM_TIMEOUT`, and halt rather than firing page N+1 into a wedged GPU. Also wrap the
   **judge call** (`agentic.py:~268`, today on the orchestrator thread outside `route_page`'s
   deadline) in the same guard.
6. Gate the global Phase 4c (`orchestrator.py:~395`) to a **no-op in agentic mode** (tables handled
   in-loop per PP-3).
6a. **Equations (GH-36, now in `main` via PRs #59/#60).** GH-36 added two **global** phases —
   `_detect_and_crop_equations` (detect display-equation regions + crop PNGs + provenance) and
   `_attach_equation_latex_sidecars` (local-VLM crop→LaTeX, pylatexenc 1A structural gate, 1C
   non-destructive sidecar). Both are gated behind `--detect-equations` (+ `--recover-clean-equations`)
   and default-OFF. In the fused per-page loop, run the equation step **per page** at the same seam as
   figures (PP-4), and **gate the two global equation phases to no-op in agentic mode** (mirror the
   Phase 4c gate) so equations are not detected/spliced twice. Preserve the 1A-gate + 1C-sidecar
   behavior and the `equation_region_detected` / `equation_latex_accepted` / `equation_latex_rejected_kept_crop`
   `AuditEvent` shapes (replay/test parity). The crop PNG must still be the body's ground truth and bad
   LaTeX must never silently replace native text — same hard AC as the phase-major version.

### Write ownership
`src/socr/pipeline/orchestrator.py` (`_phase_agentic`, `process()` agentic branch + Phase 4c gate
+ the equation-phase agentic-mode gates from step 6a).
**Single owner of `_phase_agentic`** — no other orchestrator ticket runs concurrently in this region.

### Acceptance
- Agentic run flushes `pages/NNN.md` incrementally (observable mid-run; a kill after page K leaves
  K fragments + K sidecars).
- Final stitched `.md` is **byte-identical** to a non-progressive agentic run on the same fixture
  (no output regression).
- Phase 4c does not run twice in agentic mode.
- The GH-36 equation phases do not run twice in agentic mode; with `--detect-equations`
  (+`--recover-clean-equations`) on, equations are detected/spliced **per page** with the crop inline
  and LaTeX in a 1A-validated sidecar (never replacing native text), and the equation `AuditEvent`s
  carry the correct `page_num`. With the equation flags off, output is unchanged.
- A wedged backend after page N halts with `PARTIAL_SAVE_VLM_TIMEOUT` and pages `0..N-1` saved — no
  cascade into N+1.
- Existing agentic tests pass; `total_cost` and per-page audit events unchanged for the happy path.
- Doc-scoped engines built once, not per page.

### Verification
- `uv run pytest tests/test_agentic.py tests/test_orchestrator.py tests/test_p1_cascade_economics.py -q`
- `uv run ruff check src/socr/pipeline/orchestrator.py`

---

## PP-5 — Per-page resume ledger: skip terminal pages on re-run

GitHub: new · Status: **NEEDS-DESIGN** · Priority: P1 · Agent: `socr-designer` first
Depends on: PP-1, PP-2 · Wave 4

### Problem
Even with fragments + the PP-1 sidecar, there is no resume *logic*: `RootIndex` / `DocMetadata`
are document-level only (`contract.py:~509-621`). A crash mid-doc currently reprocesses from page 1.
This ticket adds crash **recovery** on top of crash **visibility**.

### Plan
1. Enrich the PP-1 sidecar if needed and add per-page resume logic in the agentic driver: before
   processing page N, if a **terminal** `pages/NNN.json` + `pages/NNN.md` exist **and the
   run-fingerprint matches**, load the fragment and skip OCR for that page.
2. Keep the outer doc-level `RootIndex` gate (`orchestrator.py:~148`) unchanged as the
   all-or-nothing fast path; the ledger is the new **inner** per-page gate.
3. Reconcile with the provisional `:pre-figures` scheme (`orchestrator.py:~2423`): a partial doc
   with some terminal pages must be **resumable**, not skipped-forever and not fully reprocessed.
4. Use a per-page fingerprint compatible with `_run_fingerprint` (`orchestrator.py:~207`) so a
   model/prompt/render/flag change invalidates fragments (re-OCR), not silently reuses stale pages.
   **Invalidate on run-fingerprint, not just PDF checksum.**

### Write ownership
`src/socr/pipeline/orchestrator.py` (per-page resume gate in `_phase_agentic`; ledger reader);
`src/socr/core/state.py` (optional `PageState ↔ ledger` serialization helper).

### Acceptance
- Killing a run after page K and re-running reprocesses **only** pages > K (verified by logs/timing).
- A run-config/model change invalidates fragments and forces re-OCR of affected pages.
- A non-terminal (mid-lifecycle) page is **not** skipped on resume.
- No second writer of root `metadata.json`; `RootIndex` remains sole author.
- Partial docs resume rather than skip-forever or full-reprocess.

### Verification
- `uv run pytest tests/test_orchestrator.py tests/test_resume.py -q` (or the repo's resume test module)
- `uv run ruff check src/socr/pipeline/orchestrator.py src/socr/core/state.py`
