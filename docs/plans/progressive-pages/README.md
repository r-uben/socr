# Progressive page processing — design & plan

**Initiative goal.** Process each page through its *full lifecycle*
(route → extract text → tables → figures → verify) and **save that page to disk
immediately**, before moving to the next page. Today the pipeline is *phase-major*
and writes exactly once at the very end, so a hang in any late global phase loses
the entire document. This initiative makes the **agentic path page-major with
incremental flush**, and fixes the table/figure correctness gaps that show up on the
Consensus Economics (CE) corpus.

This subfolder is the durable plan for that initiative, in the same format as
`docs/plans/agentic-local-first/`:

- `TICKETS.md` — canonical ticket backlog (PP-0 … PP-7), mapped to GitHub issues.
- `STATUS.md` — live execution: waves, ready/design queues, agent assignments, next action.
- `README.md` (this file) — problem, verified current architecture, target design, the
  decisions taken (with the `/consilium` panel that ratified them), scope boundary, open questions.

Source: open GitHub issues **#54, #55, #56** (the CE cluster), informed by **#49**
(single-pass VLM + free native verify + agentic-on-signal — the general-method ADR),
**#47/#50** (figures), **#39** (cost-aware routing). GH-49-routing and **GH-49A** (native
table verifier) are already DONE on `feat/49-structured-content-routing`.

---

## 1. The problem, verified

A 6-dimension deep read of the 2840-line `src/socr/pipeline/orchestrator.py` (control-flow,
state/persistence, figures, tables, routing, timeouts) established the following, with file
anchors. **Line numbers are approximate and drift; method names are the stable anchors.**

### 1.1 The pipeline is phase-major with a single terminal save point

`OrchestrationPipeline.process()` runs, for one document:

```
Phase 1 Analyze (born-digital detection over ALL pages)
  → ONE of { agentic per-page loop | multi-engine | single-engine backbone } over ALL pages
  → Phase 4c dual-pass table reread (global, mutates page text in place)
  → Phase 5 Assemble  ←── the ONLY .md / metadata / figures write
```

Every non-agentic phase iterates over **all** pages before the next phase begins.
`DocumentState` (the in-memory blackboard: `pages{}`, attempts, audit events) is
**RAM-only until Phase 5**. The single writer is `_phase_assemble → _save_markdown`
(`md_path.write_text(whole_doc_body)`). A crash or hang before Phase 5 discards everything.

### 1.2 The #55 hang is concrete

Phase 4c (`_phase_dual_pass_tables`) crop rereads call `OllamaTableReader.read →
httpx.post("…/api/generate", timeout=crop_timeout)` (`tables/extract.py:74`). The
`crop_timeout` is an **httpx scalar** (300 s for qwen-family, else 120 s) with **no outer
wall-clock guard**. httpx's read-timeout does not reliably release a wedged Ollama socket,
so one crop holds the whole document hostage and nothing is ever written.

By contrast the agentic extraction path **already** wraps each provider call in a releasing
deadline: `route_page` uses `ThreadPoolExecutor(max_workers=1) + future.result(timeout)`
(`pipeline/agentic.py:213`). The dual-pass path simply never got that guard.

### 1.3 The agentic path is already shaped for progressive processing

The single highest-leverage fact across all six maps: **the agentic path already owns the
only per-page loop, the only per-page write-through persistence, and the only releasing
deadline.**

- Only genuine per-page loop: `_phase_agentic` (`orchestrator.py:~1131`, `for page_num in ocr_pages`).
- Only per-page write-through: `_page_blob_store.put_page(ps.best_output)` after each page
  (`orchestrator.py:~1166`). **But** the blob store is content-addressed (key = sha256 of
  content, `core/cache.py`), has no `page_num → hash` index, is **never read back on resume**,
  and produces **no partial `.md`**. It gives crash *visibility*, not crash *recovery*.
- Only releasing deadline: `route_page` (`agentic.py:213`).

The non-agentic paths include **consensus** (cross-attempt voting, `pipeline/consensus.py`)
and a **repair router** that plans over whole-document state (`RepairRouter.plan_repairs(state)`,
`pipeline/repair.py`). These are inherently cross-page and do **not** decompose per-page cheaply.

### 1.4 Figures and tables are doc-tail / late-global, not per-page

- **Figures** are produced once for the whole document at assemble time
  (`_describe_and_embed_figures`, `orchestrator.py:~2448–2563`), then **appended as a trailing
  section** (`_build_figure_blocks`, append after the last page's text). `FigureExtractor.extract`
  re-opens the PDF and loops all pages; figure numbering and the `figures_max_total` cap are
  doc-global; the vision engine is built/closed once per doc.
- **Tables** are reread in the global Phase 4c (§1.2), which patches `best_output.text` in
  place **after** the per-page extraction is "done". The native verifier
  (`NativeTableVerifierJudge`, GH-49A) already runs per-page at the routing gate; only the
  dual-pass reread is still a late global phase.

### 1.5 #54 over-routing

CE chart/front-matter pages over-route to Qwen (observed: 31 qwen / 1 native on `202401.pdf`)
because the routing `has_tables` flag is set by the loose `_detect_columnar_numbers` heuristic
(`core/born_digital.py:~610`, a single-token columnar-number ratio) that false-fires on chart-axis
labels and front-matter. A stronger lane-cooccupancy gate (`has_numeric_columns`,
`tables/reconstruct.py:~110`) exists but is used only for native reconstruction, not routing.
Separately, `apply_born_digital` (`core/state.py:~149`) **drops** `has_figures` / `has_equations`
from the content-type vector, so a per-page gate can't route on chart-vs-table-vs-equation.

---

## 2. Target architecture (consilium-ratified)

Make **only the agentic path** progressive; leave the non-agentic backbone/multi/consensus/repair
paths phase-major and bit-stable. Persist as **human-readable `pages/NNN.md` fragments** stitched
at the end. The per-page agentic lifecycle becomes:

```
for page in pages (native AND ocr):
  1. classify         detect_page (born_digital.py:246, already page-pure)
  2. route            native-vs-ladder (_is_trusted_native_without_ocr)
  3. extract          route_page (already per-page, deadline-guarded)
  4. tables           per-page dual-pass reread + reconcile (PP-3), deadline-guarded (PP-0)
  5. figures          per-page extract + describe + INLINE embed (PP-4)
  6. verify/finalize  audit, failure markers, provenance accounting
  7. FLUSH            write pages/NNN.md + atomic pages/NNN.json sidecar (PP-1)
```

Doc-scoped setup (engine ladder, judge model, vision engine, blob store, figure counter,
running figure cap) is hoisted **once** before the loop and threaded in. At document end,
`_phase_assemble` becomes a **stitch + finalize** step: it concatenates the fragments via
`contract.assemble_pages(pages, page_numbers=[…])` so the final `<stem>.md` is **byte-identical**
to today's whole-doc assembly (round-trips `PAGE_MARKER_RE`). Phase 4c becomes a **no-op in
agentic mode** (still runs globally for non-agentic modes).

### 2.1 Decisions taken — `/consilium` panel `20260615T181400Z-8388`

Codex (gpt-5.5) + Gemini (antigravity) **independently ratified** all three leans and
**independently challenged** the chart-handling binary. High confidence; sided with **both**.

| # | Decision | Verdict | Load-bearing reason |
|---|----------|---------|---------------------|
| 1 | **Scope: agentic-only, staged** (not a unified rewrite) | RATIFY | Consensus/repair are whole-document by design; a unified rewrite re-buffers state (killing the progressive benefit) or breaks global heuristics + detonates ~168 phase-shape tests. The agentic path already *is* a per-page loop with a deadline. |
| 2 | **Persistence: `pages/NNN.md` fragments + sidecar** (not blob-ledger) | RATIFY | The blob store is replay machinery, not a crash-recovery artifact. Fragments give `cat pages/*.md` salvage; the ledger pays O(N²) rewrites and freezes the wrong text layer. |
| 3a | **#55 fix: releasing thread-deadline is an acceptable v1 ceiling** | RATIFY (+ guard) | It uncouples orchestrator survival from the VLM state machine and saves pages `0..N-1`. Ollama-side `stream:true` cancellation is "notoriously finicky" and not required to stop total data loss. |
| 3b | **Chart handling: native prose is NOT enough** | CHALLENGE → third lane | Native text on a chart is "word-salad of floating axis labels / legend / data points" with the visual lost — "for Consensus Economics this destroys the data." Charts must route to **figure-asset extraction**. |

### 2.2 The three refinements the panel forced (now baked into the tickets)

1. **Wedged-GPU cascade guard (PP-0 + PP-2).** Both panelists, unprompted, raised the same
   failure: a releasing deadline frees the *orchestrator thread*, but Ollama with `stream:false`
   does **not** abort generation on disconnect — the GPU stays busy on abandoned page N, so
   firing page N+1's VLM request cascades into timeouts or OOM. The verified timeout layering
   confirms it: the inner subprocess bound (1800 s) dwarfs the outer soft timeout (300 s), so the
   outer "timeout" is cosmetic for *releasing resources*. **Guard:** after a VLM timeout, never
   blindly issue the next VLM call — probe backend health; continue only if recovered, else flush
   `0..N-1` and halt the document with a durable `PARTIAL_SAVE_VLM_TIMEOUT` marker. (This unifies
   Codex's degrade/probe-and-continue with Gemini's halt-partial: **health-probe-gated continue,
   defaulting to halt-partial on non-recovery**.)

2. **Don't fully defer the machine-readable sidecar (PP-1).** A fragment proves *text exists*; it
   does not prove the page completed route/table/figure/audit/failure decisions. Human salvage
   works from fragments alone; **automatic resume cannot**. So PP-1 writes a minimal **atomic**
   `pages/NNN.json` (`.tmp` → rename) carrying page-fingerprint, run-fingerprint, lifecycle status,
   engine/provider, audit result, table-pass result, and figure refs. PP-5 enriches + consumes it.

3. **New PP-7 — chart → figure-asset lane (paired with #54).** Narrowing `has_tables` (PP-6)
   *without* this creates the word-salad gap. Pages that are neither dense-table nor prose route
   to **figure-asset extraction covering BOTH vector and raster charts** (CE dashboards are vector)
   — not Qwen-table-VLM, not native prose.

Folded-in risks: native-trusted pages must flush fragments too (today's loop only iterates
`ocr_pages`); fragment text must be **body-only canonical** for byte-identity; figure numbering
stays **doc-global**; resume invalidates on **run-fingerprint**, not just PDF checksum; the
**judge call** (`agentic.py:~268`) runs on the orchestrator thread outside `route_page`'s deadline
and should reuse PP-0's guard; `get_fitz_page` (`orchestrator.py:~1554`) leaks a fitz handle per
judged page and must be closed.

---

## 3. Scope boundary (explicit)

**In scope (this initiative):** the agentic path becomes progressive (per-page lifecycle +
incremental `pages/NNN.md` flush + sidecar), dual-pass tables and figures move into the per-page
step, the #55 crop-reread deadline + cascade guard, the #54 routing narrowing, and the #56 chart
lane.

**Out of scope (stay phase-major, bit-stable):** the non-agentic single-engine backbone,
multi-engine, **consensus** (cross-attempt), and **repair router** (whole-state) paths. They keep
doc-tail figures and the global Phase 4c. A unified rewrite is deferred until the per-page agentic
lifecycle is proven in production on CE.

**Accepted consequence:** two divergent code paths (agentic = progressive; others = phase-major)
and a figure-layout that differs by mode (agentic = inline per page; others = doc-tail). The panel
ratified this divergence as a necessary tax. The output-layout change is flagged as an open
question (§5) before PP-4 lands, because it touches the papers-library `.md` contract.

---

## 4. Sequencing (waves)

| Wave | Tickets | Rationale |
|------|---------|-----------|
| **0 — de-risk** | PP-0 (#55), PP-6 (#54) | Independent, READY, shippable without the restructure. PP-0 removes the acute data-loss risk *now*. Disjoint-enough write sets (PP-0: `extract.py` + dual-pass/`get_fitz_page` region; PP-6: `born_digital.py` + `state.py` + routing-gate region). |
| **1 — scaffold** | PP-1 | The load-bearing flush + stitch + sidecar primitive everything depends on. Prove byte-parity before building on it. |
| **2 — per-page steps** | PP-3 (tables), PP-4 (figures), PP-7 (chart lane) | PP-3 needs PP-0's deadlined reader. PP-4/PP-7 touch the figures subsystem. Land before PP-2 fuses them. |
| **3 — fuse** | PP-2 | Wires the per-page steps + flush into one agentic loop; gates Phase 4c off in agentic mode. Single owner of `_phase_agentic`. |
| **4 — recovery** | PP-5 | Per-page resume ledger on top of crash-visibility. Designed against the finished lifecycle, not a moving target. |

**Write-set discipline:** PP-0, PP-2, PP-3, PP-4, PP-5 all touch `orchestrator.py`. Keep them in
**distinct method regions** and land in wave order to minimize rebase pain; never run two
orchestrator-touching tickets concurrently unless their method regions are confirmed disjoint.
See `STATUS.md` for the live assignment table.

---

## 5. Open questions (resolve before the dependent ticket lands)

1. **Output-layout contract (before PP-4):** ~~inline-per-page figures change the agentic `.md`
   layout vs the doc-tail appendix.~~ **RESOLVED 2026-06-16 (user decision): figures embed INLINE
   per page** — each figure appears inside its `## Page N` section, right after that page's text, so
   a flushed page fragment is self-contained. PP-4 implements this; downstream papers-library
   consumers read the inline layout.
2. **Two divergent paths (long-term):** is agentic=progressive / others=phase-major acceptable
   indefinitely, or is a unified path eventually required (needs a consensus/repair decomposition plan)?
3. **Ollama server-side cancellation:** PP-0 releases the thread but the GPU stays busy
   (`stream:false`). In strict-local this forces strictly-serial per-page processing. Is that the
   accepted v1, or do we pursue `stream:true` + connection-abort / `keep_alive` tuning later?
4. **Replay vs fragments:** v1 keeps the replay manifest built once at the end (unchanged) and uses
   `pages/NNN.md` + `pages/NNN.json` for progressive save/resume. Acceptable for replay to remain
   doc-end-only, or must replay also become incremental?
5. **Chart route depth (PP-7):** a label cross-check for chart captions is the #47C weak-verify
   layer — in scope here, or kept as a separate figure-verification ticket?
6. **Crop-timeout calibration (PP-0/PP-3):** the 300 s constant is tuned for full-PAGE qwen, not
   table crops, so it is over-generous. Derive the per-crop deadline from crop pixel-area / count
   rather than a model-name prefix?
