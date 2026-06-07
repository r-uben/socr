# socr v1.0 Refactor — Tickets

## CLI Command Reference

| Engine | Command | Key Flags |
|--------|---------|-----------|
| gemini-ocr | `gemini-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--task`, `--model`, `-w` |
| deepseek-ocr | `deepseek-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--backend ollama\|vllm`, `--vllm-url`, `-w`, `--analyze-figures` |
| mistral-ocr | `mistral-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--max-pages`, `-w`, `--table-format` |
| nougat-ocr | `nougat-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--pages 0-5`, `--device`, `--batch-size` |
| marker-ocr | `marker-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--pages 0-5`, `--device`, `--force-ocr` |

---

## Tickets

### [TICKET-1] Core data model — PipelineConfig, DocumentHandle, DocumentResult
- **Status:** done
- **Priority:** high
- **Files:** `src/socr/core/config.py`, `src/socr/core/document.py`, `src/socr/core/result.py`
- **Description:**
  - Replace 8 nested dataclasses with single `PipelineConfig` (engine selection, output dir, timeouts, flags)
  - Replace eager `Document` (renders ALL pages to PIL) with lazy `DocumentHandle` (holds path + page list, no rendering)
  - Simplify `OCRResult` → `DocumentResult` (document-level, not per-page in standard mode)
  - Keep `EngineType` enum, add `MARKER`
- **Acceptance Criteria:**
  - [ ] `PipelineConfig` replaces all 8 config dataclasses
  - [ ] `DocumentHandle` holds PDF path, page count, file hash — no PIL rendering
  - [ ] `DocumentResult` stores whole-document markdown + metadata
  - [ ] No backwards-compat shims for old classes

### [TICKET-2] Engine interface + fix CLI commands
- **Status:** done
- **Priority:** high
- **Files:** `src/socr/engines/base.py`, `src/socr/engines/gemini.py`, `src/socr/engines/nougat.py`, `src/socr/engines/deepseek.py`, `src/socr/engines/mistral.py`
- **Description:**
  - New `BaseEngine.process_document(pdf_path: Path, output_dir: Path, config: PipelineConfig) -> DocumentResult`
  - Remove `process_image()` / `process_pdf_page()` (per-page interface)
  - Each engine calls its CLI once per document via subprocess, reads output dir
  - Fix commands: `gemini-ocr <path>` (not `gemini-ocr process`), `nougat-ocr` (not `nougat-ocr-cli`)
  - `deepseek-ocr <path>` works as-is (auto-inserts `process`)
  - Handle CLI output structure: read the generated `.md` file from output dir
- **Acceptance Criteria:**
  - [ ] All engines call CLI once per PDF (one subprocess per document)
  - [ ] Correct CLI commands for all 5 engines
  - [ ] Engines return `DocumentResult` with markdown content
  - [ ] Subprocess timeout from config
  - [ ] Old per-page interface fully removed

### [TICKET-3] Add Marker engine
- **Status:** done
- **Priority:** medium
- **Files:** `src/socr/engines/marker.py` (new)
- **Description:**
  - New `MarkerEngine` following the same interface as TICKET-2
  - Calls `marker-ocr <path> -o <dir>` via subprocess
  - Supports `--pages`, `--device`, `--force-ocr` passthrough
  - Register in engine router
- **Acceptance Criteria:**
  - [ ] `MarkerEngine.process_document()` works
  - [ ] Registered in `EngineType` enum and router
  - [ ] Passes device/pages flags when configured

### [TICKET-4] Deduplicate figure extraction — shared FigureExtractor
- **Status:** done
- **Priority:** high
- **Files:** `src/socr/figures/extractor.py` (new), `src/socr/figures/__init__.py` (new)
- **Description:**
  - Extract ~400 lines of figure extraction from `processor.py` and ~400 from `hpc_sequential_pipeline.py`
  - Shared `FigureExtractor` class with 3 strategies: vector clustering, IMAGE blocks, raw embedded
  - `_cluster_drawings_into_figures()` union-find lives here once
  - Both StandardPipeline and HPCPipeline call `FigureExtractor.extract(pdf_path, output_dir)`
- **Acceptance Criteria:**
  - [ ] Single `FigureExtractor` class with all 3 extraction strategies
  - [ ] No figure extraction code in pipeline files
  - [ ] Union-find clustering exists in one place only
  - [ ] Both pipelines use the shared module

### [TICKET-5] Rewrite StandardPipeline — document-level stages
- **Status:** done
- **Priority:** high
- **Files:** `src/socr/pipeline/processor.py`
- **Description:**
  - Rewrite `OCRPipeline` as `StandardPipeline` with document-level stages:
    1. Primary OCR: `engine.process_document(pdf_path, output_dir)`
    2. Audit: document-level quality check on full markdown (word count, garbage ratio, hallucination)
    3. Fallback: if audit fails, re-run with fallback engine on whole document
    4. Figures: `FigureExtractor.extract()` (from TICKET-4)
  - Uses `MetadataManager` (TICKET-7) for incremental processing
  - Takes `PipelineConfig` (TICKET-1)
  - ~935 lines → target ~300 lines
- **Acceptance Criteria:**
  - [ ] 4-stage document-level pipeline
  - [ ] Audit runs on whole-document markdown
  - [ ] Fallback re-runs whole document with different engine
  - [ ] No per-page processing logic
  - [ ] Uses FigureExtractor, MetadataManager, PipelineConfig

### [TICKET-6] Simplify HPC pipeline — shared figures, simplified config
- **Status:** done
- **Priority:** medium
- **Files:** `src/socr/pipeline/hpc_pipeline.py`, `src/socr/engines/base.py`, `src/socr/engines/deepseek_vllm.py`, `src/socr/engines/vllm.py`, `src/socr/pipeline/router.py`, `src/socr/pipeline/reconciler.py`
- **Description:**
  - Added `BaseHTTPEngine` abstract class for vLLM/HPC per-page engines (separate from CLI-based `BaseEngine`)
  - Rewrote `DeepSeekVLLMEngine` and `VLLMEngine` using `BaseHTTPEngine` + local config dataclasses
  - Merged `HPCPipeline` + `HPCSequentialPipeline` into single `HPCPipeline` class
  - Deleted `hpc_sequential_pipeline.py`
  - Ported router to `PipelineConfig` (removed `AgentConfig` dependency)
  - Fixed reconciler (`PageResult` no longer has `cost` field)
  - Extended `DocumentHandle` with lazy `render_page()` / `render_all_pages()` for HPC per-page rendering
  - Added `EngineType.VLLM` to enum, `confidence` to `PageResult`, `engine` to `FigureInfo`
  - Added `--hpc-sequential` flag to CLI
  - Fixed all tests for new interfaces
- **Acceptance Criteria:**
  - [x] Single `HPCPipeline` class (no more separate sequential pipeline)
  - [x] Uses shared `FigureExtractor`
  - [x] Uses `PipelineConfig`
  - [x] vLLM server lifecycle management preserved
  - [x] `hpc_sequential_pipeline.py` deleted
  - [x] `BaseHTTPEngine` for per-page HTTP API engines
  - [x] Router and reconciler ported to new data model
  - [x] All 8 tests pass

### [TICKET-7] MetadataManager — incremental batch processing
- **Status:** done
- **Priority:** high
- **Files:** `src/socr/core/metadata.py` (new)
- **Description:**
  - Port `MetadataManager` pattern from sibling CLIs (gemini-ocr, marker-ocr, nougat-ocr)
  - SHA256 file checksums for change detection
  - `is_processed(file_path) -> bool`, `record(file_path, **kwargs)`
  - Stores `metadata.json` in output directory
  - Used by StandardPipeline for skip/reprocess logic
- **Acceptance Criteria:**
  - [ ] `MetadataManager` with `is_processed()` and `record()`
  - [ ] SHA256 checksums
  - [ ] `metadata.json` output
  - [ ] Integrated into StandardPipeline

### [TICKET-8] CLI cleanup — flags, deduplication
- **Status:** done
- **Priority:** medium
- **Files:** `src/socr/cli.py`
- **Description:**
  - Add `--dry-run`, `--quiet`, `--reprocess` to top-level (pass through to pipeline)
  - Remove 12x duplicated timeout settings (single `--timeout` flag)
  - Deduplicate shared options between `process` and `batch` commands
  - Wire up `PipelineConfig` construction from CLI args
  - Add `marker` to engine choices
- **Acceptance Criteria:**
  - [ ] `--dry-run` lists files without processing
  - [ ] `--quiet` suppresses non-error output
  - [ ] `--reprocess` forces re-OCR of already-processed files
  - [ ] Single `--timeout` flag (not 12)
  - [ ] Marker available as engine choice

### [TICKET-9] Version bump, pyproject cleanup, tests
- **Status:** done (version bump + pyproject; tests deferred)
- **Priority:** low
- **Files:** `pyproject.toml`, `src/socr/__init__.py`, `tests/`
- **Description:**
  - Bump version to 1.0.0
  - Add `marker-ocr-cli` to optional dependencies
  - Update engine list in README
  - Add/update unit tests for new interfaces
- **Acceptance Criteria:**
  - [ ] Version 1.0.0
  - [ ] `marker-ocr-cli` in optional deps
  - [ ] Tests pass for new engine interface and pipeline

---

## v2.5 Consolidation — Hybrid architecture (deterministic backbone + agentic repair)

See decision record: `docs/log/2026-05-29_hybrid-architecture-decision.md`.
Three-way agreement (Claude + Codex + Gemini): consolidate to ONE orchestrator
on the existing `DocumentState` blackboard; make only the repair stage agentic,
on flagged pages, with cached decisions. Do these in order — each ticket should
be one commit on `refactor/unified-page-contract`.

> **REVISED after the go-team panel (see decision-record addendum).** Two
> load-bearing corrections, both adopted:
> 1. **Python-on-top, not agent-on-top.** Python owns the loop/budget/manifest/
>    error-handling and checkpoints per page; the LLM is a *stateless* per-page
>    `decide(image, current_ocr) -> action` function. The `.md` is the judge
>    PROMPT, never the orchestrator. Entry points stay Python: `socr agent`,
>    `socr replay`, `socr batch`.
> 2. **Manifest = artifact cache, NOT a re-execution recipe.** VLM OCR is
>    non-deterministic even at temp 0; `socr replay` serves frozen output blobs
>    and invokes no engine. Fingerprint keys off the RENDERED-IMAGE hash + render
>    params + engine/model + prompt + normalizer/assembly versions.
>
> TICKET-15 (cache/manifest/replay) and TICKET-16 (judge benchmark) supersede the
> earlier 11–14 framing; 12–14 still apply but as Python modules, not an agent.

### [TICKET-15] Content-addressed cache + manifest + `socr replay`
- **Status:** done
- **Priority:** critical (foundation — everything else sits on this)
- **Files:** `src/socr/core/cache.py` (new), `src/socr/core/manifest.py` (new),
  `src/socr/core/result.py` (PageOutput/FigureInfo `to_dict`/`from_dict`),
  `src/socr/cli.py` (`replay` command), `tests/test_manifest_replay.py` (new)
- **Description:**
  - `BlobStore`: filesystem content-addressed store (SHA-256 of canonical JSON,
    sharded, atomic writes).
  - `PageFingerprint`: pdf_hash + page_num + render_dpi + engine + model_version
    + rendered-image hash + prompt_hash + normalizer/assembly versions; `.key()`
    is the invalidation identity.
  - `Manifest` (per-doc): page -> (blob_ref, fingerprint, journal); JSON save/load.
  - `build_manifest(state, blobs)`: freeze winning PageOutput per page; image hash
    computed only for rasterized (non-native) pages.
  - `replay(manifest, blobs)`: reconstruct markdown from blobs, ZERO engine calls;
    `stale_pages` flags missing blobs; `socr replay <manifest> [-o out.md]`.
- **Acceptance Criteria:**
  - [x] `replay` rebuilds identical markdown from disk with no model calls
  - [x] Fingerprint changes on engine/model/image/DPI drift (invalidation)
  - [x] Broken/empty cache raises rather than emitting a holed document
  - [x] 8 new tests pass; full suite 442 passed; ruff clean

### [TICKET-16] Judge benchmark — accuracy AND iterations-to-fix
- **Status:** harness done; awaiting labeled data + live run
- **Priority:** high (gates whether the repair loop is worth building)
- **Files:** `src/socr/prompts/judge_page.md` (new), `src/socr/judge/{__init__,judge,ollama_judge,benchmark}.py` (new), `src/socr/cli.py` (`judge-benchmark`), `tests/test_judge_benchmark.py` (new)
- **Description:**
  - Judge prompt is POLICY-as-data in `prompts/judge_page.md` (no numeric cutoffs;
    model reasons about page-vs-transcription faithfulness). This is the only
    place the `.md` belongs — Python owns control flow.
  - `JudgeVerdict` + `parse_verdict` (tolerates fences/prose); `Judge` protocol;
    `OllamaVisionJudge` (local VLM, temp 0, headless — the zero-cost path).
  - `benchmark.py`: `load_dataset` (labels.json + images/ + ocr/), `run_benchmark`,
    `BenchmarkReport` with the two headline rates — FN (corpus poisoning) and FP
    (budget burning).
  - `socr judge-benchmark <dataset>` runs it against a local Ollama vision model.
- **Done:**
  - [x] Harness + prompt + scorer built; 9 tests pass (stub judge); full suite 451
  - [x] FN/FP reported by `BenchmarkReport.summary()`
- **Remaining (empirical — needs real data):**
  - [ ] Label ~50 good / ~50 mangled pages from the corpus into a dataset dir
  - [ ] Run `socr judge-benchmark` with a real VLM; record FP/FN
  - [ ] Add iterations-to-fix experiment (run escalation chain, judge after each);
        decide repair depth from the distribution (if ~1 → gate + single escalation,
        not a multi-iteration loop)
  - [ ] Verify `prompts/*.md` ships in the built wheel (hatchling package data)

### [TICKET-10] Move venv off iCloud (environment fix)
- **Status:** done
- **Priority:** critical
- **Files:** none (environment only)
- **Description:**
  - The in-iCloud `.venv` corrupts (`RECORD file is invalid ... os error 60`).
    This was ~half of "can't run it".
  - `uv venv ~/venvs/socr --python 3.11`; `export UV_PROJECT_ENVIRONMENT=~/venvs/socr`; `uv pip install -e .`
  - Make sticky (direnv `.envrc` or shell export); confirm `.venv*` is gitignored.
- **Acceptance Criteria:**
  - [x] venv lives at `~/venvs/socr`, not in iCloud
  - [x] `socr engines` runs without RECORD error (broken in-iCloud `.venv` removed)
  - [x] `UV_PROJECT_ENVIRONMENT` sticky via `.envrc` (gitignored); `~/venvs/socr/bin/socr` works unconditionally
  - Note: `uv pip install` ignores `UV_PROJECT_ENVIRONMENT` — install with `--python ~/venvs/socr/bin/python`

### [TICKET-11] Enforce per-page PageOutput for whole-doc engines (the key seam)
- **Status:** todo
- **Priority:** high
- **Files:** `src/socr/engines/base.py`, whole-doc CLI adapters (`gemini.py`, `marker.py`, `nougat.py`, `mistral.py`, `deepseek.py`), `src/socr/core/result.py`
- **Description:**
  - Whole-doc CLI engines currently return a single `PageOutput(page_num=0)`
    holding the entire document. Split their monolithic output into per-page
    `PageOutput`s AFTER OCR.
  - **Do NOT slice the input PDF before OCR** (Gemini guardrail): Marker/Nougat
    need full-document context (font dicts, headers/footers, bibliography).
    Feed the whole PDF; slice the *output*, not the input.
  - Use page markers / form-feed / page-count alignment to map output→pages.
    Where an engine gives no reliable page boundaries, store as a single
    PageOutput but mark `page_split=False` so triage treats it whole-doc.
- **Acceptance Criteria:**
  - [ ] Whole-doc engine output is split into per-page `PageOutput`s when boundaries are recoverable
  - [ ] Input PDF is never pre-sliced before a whole-doc engine
  - [ ] Per-page API engines (gemini-api) and whole-doc CLIs both populate `DocumentState.pages` uniformly

### [TICKET-12] Collapse 5 orchestrators into ONE blackboard pipeline
- **Status:** Increment 1 done; Increment 2 (HPC) deferred
- **Priority:** high
- **Files:** keep `pipeline/orchestrator.py` (UnifiedPipeline) as THE pipeline; deleted `pipeline/processor.py`
- **Codex review (session 13):** "safe to route, not safe to delete blind." UnifiedPipeline
  is NOT behaviorally identical to StandardPipeline (whole-doc vs per-page OCR; native-text
  substitution; synthesized `page_num=0` result). Required gate before flipping the default:
  parity tests + the invariant "prose-only born-digital docs do zero OCR and never enter
  repair." Keep `--hpc-sequential` as a thin dedicated path (different runtime: vLLM lifecycle,
  sequential model swap, Nougat reconciliation, frontmatter) — converge its internals onto
  DocumentState LATER.
- **Increment 1 (done):**
  - [x] Verified the fast-path invariant already holds (prose-only → Tier-1 native, no OCR,
        `needs_repair` False). The benchmark's 90s was *table* pages (legit OCR), not the prose path.
  - [x] Added parity characterization tests (`TestDefaultPathParity`): scanned→OCR+write,
        prose-only→zero OCR, unavailable engine→ERROR.
  - [x] CLI default (`process`/`batch`) now routes to UnifiedPipeline; `--unified` kept as no-op.
  - [x] Deleted `processor.py` (StandardPipeline); `pipeline/__init__` exports only UnifiedPipeline.
  - [x] Full suite 454 passed; no new lint.
- **Increment 2 (todo, higher risk — touches HPC workflow):**
  - [ ] Refactor `HPCPipeline` internals to reuse DocumentState + scoring + assemble helpers
  - [ ] Then fold consensus/reconciler selection into a single `best_output` step
  - [ ] Decide if `hpc_pipeline.py` + `reconciler.py` can be deleted (keep the `--hpc-sequential` flag)

### [TICKET-13] Triage gate — calibrated, avoids both failure modes
- **Status:** todo
- **Priority:** medium
- **Files:** `src/socr/audit/*`, `src/socr/core/difficulty.py`
- **Description:**
  - Reduce the heuristic stack to a single triage gate that flags suspect pages.
  - Calibrate against the benchmark so it neither silently passes mangled
    tables/equations (corpus poisoning) nor over-triggers the LLM (agentic
    bottleneck) — the two traps Gemini named.
  - Prefer a hard-data trigger: flag when two engines on the same page diverge
    (edit distance > threshold) in addition to audit-failure flags.
- **Acceptance Criteria:**
  - [ ] Single triage entry point returns the flagged-page set
  - [ ] Flagged fraction is single-digit-percent on the benchmark corpus
  - [ ] Trigger includes engine-disagreement (diff) signal, not heuristics alone

### [TICKET-14] Agentic repair stage — flagged pages only, reproducible
- **Status:** todo
- **Priority:** medium
- **Files:** new `src/socr/pipeline/agentic_repair.py`
- **Description:**
  - For flagged pages only: an LLM/VLM sees the page image + candidate OCR(s),
    judges quality, selects the next engine, re-runs. Diff-reconciliation role
    when engines disagree.
  - **Reproducibility:** temperature 0; persist every decision artifact keyed by
    page content hash; re-runs replay the cache instead of re-querying.
  - **HPC:** stage is skippable/gated; cluster shards run the deterministic
    backbone, repair runs as a separate pass with API access (or off).
- **Acceptance Criteria:**
  - [ ] LLM invoked only on flagged pages
  - [ ] Decisions cached by page hash; re-run with cache = identical output
  - [ ] Stage can be disabled for pure-deterministic HPC runs

### [TICKET-17] Cost-aware agentic OCR — best provider on the go
- **Status:** done (core + integration + CLI; validated end-to-end on real engines)
- **Priority:** high (the headline goal)
- **Files:** `core/providers.py` (new), `pipeline/agentic.py` (new), `pipeline/orchestrator.py` (`_phase_agentic` + helpers), `core/result.py` (`cost_usd`), `core/config.py` (agentic flags), `cli.py` (`--agentic` + flags), `tests/test_providers.py`, `tests/test_agentic.py` (new), `tests/test_orchestrator.py` (TestAgenticIntegration)
- **What it does:** per OCR page, try the cheapest available provider; an injected
  judge (VLM via Ollama, or heuristic fallback) accepts or escalates up a
  cost-ordered ladder; bounded by `max_retries` and `cost_budget`. Born-digital
  prose takes free native text (unless `--no-native-first`). Winning provider +
  cost recorded on `PageState`; `DocumentState.total_cost` reflects spend;
  agentic runs auto-write a replayable manifest.
- **Design notes:** Python owns the loop (LLM is a stateless per-page decider);
  prices are tunable DEFAULTS in `core/providers.py`, routing uses RELATIVE
  cost ordering (no capability tables — the judge escalates dynamically).
- **Validated:** `socr demo.pdf --agentic --no-native-first` ran the real ladder
  glm->deepseek->marker (gemini capped out by max_attempts), recorded the
  attempt chain in the manifest journal, fell back to free native text when OCR
  was rejected (cost $0), and `socr replay` reconstructed the doc bit-identically
  with 0 model calls.
- **Acceptance Criteria:**
  - [x] Cost-ordered ladder, cheapest-first, judge-driven escalation
  - [x] Bounded cost (max_attempts, cost_budget, max_cost_per_page)
  - [x] Cost recorded; manifest written + replayable
  - [x] CLI `--agentic` (+ judge/cost flags); default-off preserves legacy
  - [x] Full suite green (475); no new lint
- **Next (optional):** judge benchmark (TICKET-16) to tune accept thresholds;
  real per-provider prices; populate model_version/prompt_hash in fingerprints.

### [TICKET-18] Table quality + text-layer integrity
- **Status:** done (core program; see `docs/log/2026-06-05_*` and `2026-06-06_table-quality-and-text-integrity.md`)
- **Priority:** high (corpus correctness — owner reads these for research)
- **Branch:** `feat/audit-log` (9 commits, off `main`)
- **Files:** `tables/{reconstruct,locate,image_locate,extract,reconcile}.py`, `core/{audit_log,born_digital,config,state}.py`, `pipeline/orchestrator.py`, `cli.py`, `tests/test_{reconstruct,encoding_corruption,dual_pass_tables,audit_log}.py`
- **What it does:** evidence-driven program that hardened table extraction and
  native-text trust after a real-corpus audit. The corpus is overwhelmingly
  born-digital, so the work landed on deterministic, no-model fixes.
  - **Structure-restore:** born-digital booktabs tables (no vertical rules) made
    `find_tables(lines)` return 0 and were dumped as a flat token stream; recover
    the grid via PyMuPDF text strategy + cleanup, char-exact, no model.
  - **Numeric-column gate:** only reconstruct where numbers co-occupy >=3 x-lanes
    per row (a grid); skips references pages that hung text-strategy for minutes.
  - **Text-layer corruption detection:** two-tier (flag mild / escalate pervasive
    to OCR) for broken font maps ("Journal"->"Joumal") the garbage check misses.
  - **Dual-pass tables:** crop + re-read located tables, reconcile vs page OCR;
    default flag-only (`--auto-patch-tables` to opt in), never silently edits.
  - **Image-based localization:** connected-component rule detection for scanned
    pages (vector detectors are blind there).
  - **Durable audit log:** `audit_log.json` of RECITATION / judge / dual-pass events.
- **Design notes:** auto-patch reversal and the gate design cross-checked with codex
  (gpt-5.5); rejected clipping text-strategy to the rule band (silently truncated a
  table when the bottom rule was undetected — unacceptable for a research corpus).
- **Acceptance Criteria:**
  - [x] Born-digital booktabs tables emit a markdown grid, not a flat dump
  - [x] References/prose never trigger reconstruction (numeric-column gate)
  - [x] Corrupted text layers flagged; pervasive ones routed to OCR
  - [x] Dual-pass flag-only by default; opt-in auto-patch
  - [x] Full suite green (541); new code lint-clean
- **Next (see TODO.md):** per-page provenance by default; firing-rate validation;
  `model_version` in fingerprints; judge spot-check on native table pages.

---

## Tickets — equation-heavy textbook failures (see docs/log/2026-06-07_*)

Opened 2026-06-07 from the first textbook-class run (Dougherty 2013, 203 pp,
born-digital, clean text layer; local-first, no cloud). Evidence and reproducers
in `docs/log/2026-06-07_math-textbook-failures.md`.

### [TICKET-19] `extract_structured()` shreds prose/reference pages into fake grids
- **Status:** open
- **Priority:** highest (corrupts clean native text)
- **Files:** `core/born_digital.py` (`extract_structured`), `tables/reconstruct.py`, `tables/locate.py`
- **Description:** A born-digital exercises+references page with one small embedded
  data table (Dougherty p19) is rendered *entirely* as a 9-column markdown grid,
  splitting words mid-token ("Consider th | e | following"). ~62 corrupted lines
  doc-wide. Reproduced byte-identically on native-only AND VLM runs → root cause is
  the native structured-table path (`find_tables` text-strategy over-firing), not the
  VLM. Directly contradicts TICKET-18's met criterion "References/prose never trigger
  reconstruction (numeric-column gate)" — the gate does not catch this mixed
  prose+small-table page.
- **Acceptance Criteria:**
  - [ ] p19 emits clean prose + the small table only (or linearized table), no full-page grid
  - [ ] Regression test: a prose page with one embedded ≤N-row table is not gridded
  - [ ] Firing-rate check confirms no new false-negatives on real booktabs tables

### [TICKET-20] Born-digital pages over-classified as "complex"
- **Status:** open
- **Priority:** high
- **Files:** `core/born_digital.py` (`needs_ocr_enhancement`), `pipeline/orchestrator.py`
- **Description:** `has_tables or has_figures or has_equations` flagged 129/203 pages
  for OCR enhancement, routing clean native text onto slower, worse local VLMs. On a
  clean text layer this pays model time to degrade correct text. Enhancement should be
  gated on evidence the native text is actually deficient, not merely on
  presence-of-complex-content.
- **Acceptance Criteria:**
  - [ ] Clean-text-layer pages with figures/equations stay native unless native is shown deficient
  - [ ] Figures still extracted without forcing whole-page OCR

### [TICKET-21] Local-only run hard-errors on a fully-written output
- **Status:** open
- **Priority:** high (offline UX)
- **Files:** `pipeline/orchestrator.py` (escalation + exit-status logic)
- **Description:** With no cloud key, a page that fails local audit has escalation
  target = cloud → impossible → run ends `Error: Processing failed` (non-zero exit)
  although the full `.md` is written. Also: p3 logged "recovered by deepseek" but the
  page is blank ("recovered" to empty). No "local-only: accept best local output,
  exit 0" mode.
- **Acceptance Criteria:**
  - [ ] Local-only mode exits 0 when a complete `.md` is produced, with failed pages flagged in audit_log
  - [ ] "recovered" requires non-empty output; empty fallback is reported as a failure not a recovery

### [TICKET-22] "Scanned" over-count on full-page-figure / sparse pages
- **Status:** open
- **Priority:** medium
- **Files:** `core/born_digital.py` (`MIN_WORDS_PER_PAGE`, `MIN_CHARS_FOR_TEXT_LAYER`)
- **Description:** Detector reported 13 scanned pages; only 4 are truly image-only
  (`get_text()` < 40 chars: front matter pp 1,2,3,9). The other 9 are full-page-figure
  or sparse pages tripping the min-words/min-chars gates and getting needlessly OCR'd.
- **Acceptance Criteria:**
  - [ ] A born-digital page dominated by one full-page figure is not classed "scanned"
  - [ ] Scanned-count on Dougherty matches ground truth (≈4)

### [TICKET-23] No math-aware (equation→LaTeX) extraction path
- **Status:** open
- **Priority:** high (the central gap for math textbooks)
- **Files:** new `engines/` math step; `pipeline/orchestrator.py` routing
- **Description:** Detected math pages (`_MATH_FONT_RE`) route only to general VLMs
  (no LaTeX) or native `get_text()` (linearizes math). No equation→LaTeX route. Per a
  /consilium panel (Codex): add a local math-aware step — marker-pdf / Texify on MPS —
  run on detected equation regions only, always storing the crop PNG beside any LaTeX
  so bad LaTeX never silently replaces a faithful image. Validate LaTeX renders; on
  failure keep crop + linearized text.
- **Acceptance Criteria:**
  - [ ] Display equations on a sample chapter emit LaTeX that renders, with crop PNG retained
  - [ ] Throughput measured on M3 Max MPS for 200 pp and 1220 pp (region-only vs full-page)

### [TICKET-24] No native-only / enhancement-threshold CLI knob
- **Status:** open
- **Priority:** medium
- **Files:** `cli.py`, `core/config.py`, `core/born_digital.py`
- **Description:** To trust a clean text layer and OCR only genuine scans we had to
  patch source (a `SOCR_NATIVE_ONLY` env gate). No supported flag expresses "this is
  born-digital with a clean layer; do not enhance prose." `--no-native-first` is the
  opposite lever.
- **Acceptance Criteria:**
  - [ ] `--native-only` (or `--enhancement-threshold`) flag, documented, with a test
