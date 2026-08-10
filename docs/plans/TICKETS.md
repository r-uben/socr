# TICKETS - GitHub issue action plan

Source reviewed: open GitHub issues in `r-uben/socr`, fetched 2026-06-15.
Base commit reviewed: `7541175` (`main`, also branch start for `feat/001-issue-plans`).

Status keys: `READY`, `NEEDS-DESIGN`, `BLOCKED`, `WIP`, `DONE`, `DEFERRED`.
Agent keys (defined in `.claude/agents/`): `socr-designer` for read-only design passes on
NEEDS-DESIGN tickets (writes a design note, frames the `/consilium` question), `socr-implementer`
for bounded code tickets, `socr-reviewer` for review after a completed ticket. `/consilium` is run
by the orchestrator only — see the per-issue workflow in `docs/plans/STATUS.md`.

These are the active issue-derived tickets. Older root `TICKETS.md` entries remain useful
history, but this file is the current GitHub issue board.

## Dispatch Rules

- One implementation ticket per subagent.
- Give each agent a disjoint write set. If two tickets both touch a file, serialize them or use
  separate worktrees and merge deliberately.
- Each implementation ticket ends with a `socr-reviewer` pass before acceptance.
- Use `uv run` for all Python commands. Do not run `python script.py` directly.
- Do not commit or push from a subagent unless the parent task explicitly asks for it.

## GH-51 - Qwen model resolution is ambiguous

GitHub: https://github.com/r-uben/socr/issues/51
Status: DONE
Priority: P0
Suggested agent: `socr-implementer`
Depends on: #46 provider identity work already merged
Write ownership: `src/socr/core/config.py`, `src/socr/cli.py`, `src/socr/engines/qwen.py`,
`tests/test_qwen_engine.py`, narrowly related CLI/config tests.

### Problem

`PipelineConfig.qwen_model` defaults to `qwen3.5:cloud`, while the local qwen tier validated for
tables is `qwen3-vl:30b-a3b-instruct`. `QwenEngine._build_command()` only substitutes the local
model when `qwen_backend == "ollama"`, so the default `qwen_backend == "auto"` can silently pass a
cloud model string on the non-agentic `--primary qwen` path.

### Plan

1. Introduce one explicit resolver for qwen backend/model intent, rather than scattering exact
   string checks.
2. Preserve explicit user pins: if a user passes `--qwen-model`, do not rewrite it.
3. Make local qwen resolution mean the validated local instruct model whenever the resolved backend
   is local/ollama-class, including the auto-local case.
4. Add an operator-visible log/console line with resolved qwen backend and model for unattended
   runs.
5. Rename or clarify the math model default path so `qwen3-vl:8b` cannot be mistaken for the OCR
   qwen tier.

### Acceptance Criteria

- `--primary qwen` with default/auto local resolution does not pass a cloud model string to the
  local qwen backend.
- Explicit `--qwen-model ...` still reaches `qwen-ocr` unchanged.
- Agentic `qwen-local-instruct` still pins the provider profile already used in the cost ladder.
- Tests cover auto backend, explicit ollama backend, explicit model pin, blank model, and agentic
  local profile behavior.

### Verification

- `uv run pytest tests/test_qwen_engine.py tests/test_providers.py tests/test_agentic.py -q`
- `uv run ruff check src/socr/engines/qwen.py src/socr/core/config.py src/socr/cli.py tests/test_qwen_engine.py`

## GH-50 - Split figure extraction from VLM figure descriptions

GitHub: https://github.com/r-uben/socr/issues/50
Status: DONE
Priority: P0
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/core/config.py`, `src/socr/cli.py`, `src/socr/pipeline/orchestrator.py`,
figure-related tests.

### Problem

`--save-figures` currently extracts PNGs and also generates VLM descriptions. PNG extraction is a
deterministic artifact; captions are non-authoritative model prose and can hallucinate details.
The current flag couples a safe archival action to a risky interpretive action.

### Plan

1. Keep `--save-figures` for PNG extraction only.
2. Add `--describe-figures` for opt-in VLM captions.
3. Add `PipelineConfig.describe_figures`.
4. Make the figure phase run extraction when either saving or describing is requested, but only call
   `_get_vision_engine()` and `describe_figure()` when `describe_figures` is true.
5. Ensure run fingerprints distinguish save-only from describe-enabled runs.
6. Keep backward compatibility deliberate: if existing behavior must be preserved under a temporary
   alias, document it in the CLI help rather than silently coupling the actions.

### Acceptance Criteria

- `--save-figures` writes PNGs and appends image references, but produces no VLM caption prose.
- `--describe-figures` runs captions only when explicitly selected.
- A failure in the caption phase cannot destroy already written OCR text.
- Resume metadata/fingerprint invalidates correctly when `describe_figures` changes.

### Verification

- `uv run pytest tests/test_orchestrator.py tests/test_canon_round3.py tests/test_silent_content_destruction.py -q`
- `uv run ruff check src/socr/core/config.py src/socr/cli.py src/socr/pipeline/orchestrator.py`

## GH-34 - Recovered-to-empty must not count as recovered

GitHub: https://github.com/r-uben/socr/issues/34
Status: DONE
Priority: P0
Suggested agent: `socr-implementer`
Depends on: #38 partial fix already merged
Write ownership: `src/socr/pipeline/orchestrator.py`, `src/socr/core/state.py`,
`src/socr/core/audit_log.py`, relevant silent-content tests.

### Problem

The issue is narrowed by later comments. Exit status for a complete `.md` is partly fixed, but a
repair attempt can still be recorded/promoted as "recovered" even when the chosen output is empty or
contentless.

### Plan

1. Audit the promotion path from repair attempt to page winner.
2. Define "usable output" through an existing structured helper where possible, not by a new ad-hoc
   string check.
3. Reject or mark empty repair outputs as failed attempts, preserving the best non-empty prior
   attempt or explicit page failure marker.
4. Ensure audit events do not say "recovered by X" for an empty result.

### Acceptance Criteria

- Empty repair output is never promoted over non-empty rejected text.
- Empty repair output cannot generate a misleading `recovered_by` event.
- Complete documents with flagged non-fatal audit failures still exit through the already-fixed
  partial/success semantics.

### Verification

- `uv run pytest tests/test_silent_content_destruction.py tests/test_orchestrator.py -q`
- `uv run ruff check src/socr/pipeline/orchestrator.py src/socr/core/state.py src/socr/core/audit_log.py`

## GH-46-D2 - Sparse comparison-row lane drift

GitHub: https://github.com/r-uben/socr/issues/46
Local plan: `docs/plans/agentic-local-first/TICKETS.md` ticket D2
Status: DONE
Priority: P1
Suggested agent: `socr-implementer`
Depends on: #46 D1 validation
Write ownership: `src/socr/prompts/table_extract.md`, table prompt tests/fixtures if present.

### Problem

Dense table extraction is now validated, but sparse comparison rows with long blank runs can shift a
single value into an adjacent column lane. The known example is the CBO row in the Consensus
Forecasts validation page. This is a column-lane anchoring problem, not a digit-recognition problem.

### Plan

1. First try the cheapest fix: update the table prompt so sparse rows preserve header lanes and do
   not pack values left or right across blanks.
2. Add a regression fixture or prompt-level test if the repo has an established pattern for prompt
   tests.
3. If prompt-only is insufficient, create a follow-up for deterministic header-lane reconciliation
   instead of hiding logic inside the prompt.

### Acceptance Criteria

- Sparse rows retain blank cells under empty header lanes.
- The CBO comparison row no longer shifts its lone value pair into the wrong lane.
- Dense rows and summary rows from D1 remain unchanged.

### Verification

- `uv run pytest tests/test_dual_pass_tables.py tests/test_reconstruct.py -q`
- Manual validation on the CE page fixture if available locally.

## GH-47A - Figure extraction safety: logo false positives and silent cap

GitHub: https://github.com/r-uben/socr/issues/47
Status: DONE
Priority: P1
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/figures/extractor.py`, figure extractor tests.

### Problem

The figure investigation found two extraction defects: title-page letterhead/logo extraction as a
figure, and a silent stop at `figures_max_total` that can drop later figures without telling the
operator.

### Plan

1. Add an explicit "cap reached" signal to the extractor result or audit/log path.
2. Make hitting the cap visible in console output and durable metadata/audit where the pipeline
   already records non-fatal notable events.
3. Improve logo/header false-positive filtering using page position and semantic evidence already
   available from PyMuPDF, while preserving real top-of-page figures.
4. Add tests for cap visibility and letterhead/logo suppression.

### Acceptance Criteria

- If figure extraction stops because a cap is reached, the run records that fact.
- A title-page logo/banner is not emitted as `figure_1` unless it satisfies the same evidence gates
  as substantive figures.
- Existing real chart extraction tests, if any, still pass.

### Verification

- `uv run pytest tests -q -k figure`
- `uv run ruff check src/socr/figures/extractor.py`

## GH-47B - Figure caption anti-fabrication prompt and warning

GitHub: https://github.com/r-uben/socr/issues/47
Status: DONE
Priority: P1
Suggested agent: `socr-implementer`
Depends on: GH-50 preferred, but can be developed independently
Write ownership: `src/socr/engines/gemini_api.py`, figure prompt tests if present.

### Problem

The figure description prompt asks for specifics and can induce fabricated axis ranges, thresholds,
or arrow labels when the image does not support them. Captions should be searchable gist, not a
verbatim source.

### Plan

1. Rewrite the figure prompt to prefer visible, qualified observations.
2. Tell the model to omit unreadable numeric values and relationships instead of guessing.
3. Add a standard markdown warning around generated descriptions if GH-50 has not yet made captions
   opt-in.
4. Keep the raw PNG path adjacent to any caption.

### Acceptance Criteria

- Captions explicitly avoid unverifiable numeric/detail claims.
- Generated description blocks identify themselves as model-generated, non-authoritative gist.
- Prompt tests or snapshot tests cover the no-guessing instruction.

### Verification

- `uv run pytest tests -q -k figure`
- `uv run ruff check src/socr/engines/gemini_api.py`

## GH-47C - Free label cross-check for figure captions

GitHub: https://github.com/r-uben/socr/issues/47
Status: DONE (Option C, consilium run 20260615T203853Z-9682)
Priority: P2
Suggested agent: `socr-designer` first
Depends on: GH-50, GH-47B
Write ownership: design first; likely `src/socr/figures/*`, `src/socr/pipeline/orchestrator.py`.

### Problem

Native text can verify labels found in a figure, but it cannot verify plotted data values or arrow
directions. The verification layer for figures must be weaker and more explicit than the native
table verifier proposed in #49.

### Plan

1. Explore which labels PyMuPDF can recover from vector charts and embedded figures in the known
   corpus.
2. Define a label-presence check that can flag caption claims unsupported by extracted page labels.
3. Keep this as a warning signal, not an automatic correction.

### Acceptance Criteria

- Design note identifies which figure claim types are cheaply verifiable and which are not.
- Any implementation records label-check warnings without rewriting captions silently.

### Verification

- Design ticket: inspect representative figure pages and record findings in `docs/log/`.
- Implementation ticket later: targeted figure tests.

## GH-49A - Native table verifier before VLM judge on born-digital pages

GitHub: https://github.com/r-uben/socr/issues/49
Status: DONE
Priority: P1
Suggested agent: `socr-designer` first, then `socr-implementer`
Depends on: GH-46-D2 findings; coordinates with #39 calibration
Write ownership: design first; likely `src/socr/tables/*`, `src/socr/audit/*`,
`src/socr/pipeline/orchestrator.py`.

### Problem

The architecture decision is already documented: extract, verify, and escalate are separate layers.
For born-digital tables, native text geometry can cheaply verify column positions and header lanes
before paying for a second VLM/judge call.

### Plan

1. Write a short design note with the native signals to use: header lane count, x-position
   consistency, numeric-cell alignment, and text-layer availability.
2. Implement the verifier as a deterministic warning/fail signal ahead of the VLM judge for
   born-digital table pages.
3. Keep scans out of scope because they lack the native geometry layer.
4. Route failures to existing agentic/escalation paths; do not invent a parallel router.

### Acceptance Criteria

- Born-digital table pages get a deterministic verification result before VLM judge escalation.
- Sparse lane drift is catchable without a second model call.
- Scanned pages bypass this verifier cleanly.
- Audit log records verifier failures with enough context to inspect the page.

### Verification

- `uv run pytest tests/test_audit_heuristics.py tests/test_dual_pass_tables.py tests/test_orchestrator.py -q`
- Add focused tests for a synthetic header-lane mismatch.

## GH-39A - Human-verified ground truth for benchmark table/equation pages

GitHub: https://github.com/r-uben/socr/issues/39
Status: BLOCKED
Priority: P1
Suggested agent: none until human labels exist
Depends on: human verification
Write ownership: benchmark fixture/data area to be identified before work starts.

### Problem

#39 Stage 1 has landed. Stage 2 requires hand-verified ground truth for table/equation pages in the
10-paper benchmark set. This cannot be safely completed by an unattended agent because the point is
human verification of research-critical numbers and equations.

### Plan

1. Identify the benchmark dataset directory and current seed outputs.
2. Create a labeling checklist and file naming convention if missing.
3. Human-check table cells/equations against page images.
4. Record provenance: source PDF, page image, seed source, checker, date.

### Acceptance Criteria

- Each selected table/equation page has verified ground truth and provenance.
- The benchmark loader can distinguish verified GT from seed/unverified output.
- No calibration code consumes unverified seed data as ground truth.

### Verification

- `uv run pytest tests/test_benchmark.py tests/test_benchmark_runner.py tests/test_benchmark_scoring_p1.py -q`

## GH-39B - Calibration artifact and ladder unification

GitHub: https://github.com/r-uben/socr/issues/39
Status: BLOCKED
Priority: P1
Suggested agent: `socr-implementer` after GH-39A
Depends on: GH-39A
Write ownership: `src/socr/benchmark/*`, `src/socr/core/providers.py`,
`src/socr/pipeline/repair.py`, `src/socr/pipeline/agentic.py`, routing tests.

### Problem

Routing ladders are still partly encoded in source lists and defaults. #39 Stage 3 calls for a
versioned calibration artifact that captures benchmark hash, page-type ladders, model/backend
identity, metric summaries, and price assumptions.

### Plan

1. Add `socr benchmark calibrate --apply` around the existing benchmark/calibrate modules.
2. Write `calibration.lock.json` from verified benchmark results.
3. Make auto engine order, local engine order, repair router, and provider ladder consume the
   artifact at runtime, filtered by availability, key state, and budget.
4. Keep source code as study design and artifact reader, not a store of empirical benchmark data.

### Acceptance Criteria

- Calibration artifact is deterministic and versioned.
- Runtime routing uses the artifact when present and has a clear fallback when absent.
- Tests prove source order lists and agentic provider ladder delegate consistently.

### Verification

- `uv run pytest tests/test_benchmark_scoring_p1.py tests/test_providers.py tests/test_repair_router.py tests/test_p1_cascade_economics.py -q`

## GH-37 - Add native-only or enhancement-threshold CLI control

GitHub: https://github.com/r-uben/socr/issues/37
Status: DONE
Priority: P1
Suggested agent: `socr-implementer`
Depends on: none, but coordinate with GH-35
Write ownership: `src/socr/cli.py`, `src/socr/core/config.py`, `src/socr/core/born_digital.py`,
tests for CLI/config/born-digital routing.

### Problem

There is no supported flag for "trust this clean born-digital text layer; only OCR genuine scans".
`--no-native-first` is the opposite control.

### Plan

1. Add a CLI/config control for native-only behavior or configurable enhancement policy.
2. In native-only mode, suppress OCR enhancement for clean born-digital pages while preserving scan
   handling and optional figure extraction.
3. Make the control visible in fingerprints so resume caches do not mix policy modes.
4. Document the flag in CLI help and README or architecture docs.

### Acceptance Criteria

- Clean born-digital pages stay native under the new flag.
- Genuine scans still route to OCR.
- Figure extraction can still run without forcing whole-page OCR.
- Tests cover config defaults, CLI flag parsing, and routing behavior.

### Verification

- `uv run pytest tests/test_born_digital.py tests/test_orchestrator.py tests/test_canon_round3.py -q`
- `uv run ruff check src/socr/cli.py src/socr/core/config.py src/socr/core/born_digital.py`

## GH-35-FU - Gate clean-short-text born-digital exception by raster image coverage

Status: DONE
Priority: P1
Suggested agent: `socr-implementer`
Depends on: GH-35 (DONE)
Write ownership: `src/socr/core/born_digital.py`, `tests/test_born_digital.py`.

### Problem

GH-35 introduced a clean-short-text exception: a sparse but clean native text layer
(e.g. figure caption, section heading) would skip the word-count gate and classify the
page as born-digital.  A /consilium panel (Codex + Gemini, decision id
20260615T104828Z-1577) found this is UNSAFE on image-dominant pages: a full-page-raster
scan with a baked-in OCR caption passes all text-quality checks and is
INDISTINGUISHABLE from a genuine born-digital figure page by text quality alone.
Skipping OCR on such a page causes permanent content loss in a citation corpus.

### Plan

1. Add `_raster_coverage(page)` helper using `page.get_image_info()` bbox data.
2. Add named constant `RASTER_DOMINANCE_RATIO = 0.90` with documented basis.
3. In the clean-short-text pass-through, check raster coverage first: if
   `coverage >= RASTER_DOMINANCE_RATIO` route to OCR; otherwise pass through to
   born-digital (preserving the GH-35 win).
4. Update the pinned "accepted tradeoff" test to assert the corrected behaviour.
5. Add a guard test ensuring a non-image-dominant sparse page still classifies born-digital.

The text render-mode (Tr 3 invisible text) discriminator was evaluated and SKIPPED:
PyMuPDF's `get_texttrace()` does expose render mode but the signal is fragile for
synthetic test fixtures (PyMuPDF's `insert_text` does not set Tr explicitly) and the
coverage gate is sufficient to catch the problematic case robustly.

### Acceptance Criteria

- Full-page raster + short clean baked-in OCR → SCANNED (is_born_digital=False).
- Genuine born-digital figure page with small/partial chart → born-digital (GH-35 preserved).
- Audit notes surface the image-dominance gate reason.
- RASTER_DOMINANCE_RATIO is a named constant with documented basis.

### Verification

- `~/venvs/socr/bin/pytest tests/test_born_digital.py tests/test_orchestrator.py -q`
- `~/venvs/socr/bin/ruff check src/socr/core/born_digital.py tests/test_born_digital.py`
- `~/venvs/socr/bin/ruff format --check src/socr/core/born_digital.py tests/test_born_digital.py`

## GH-35 - Recheck scanned over-count on sparse and full-page-figure pages

GitHub: https://github.com/r-uben/socr/issues/35
Status: DONE
Priority: P2
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/core/born_digital.py`, `tests/test_born_digital.py`.

### Problem

The detector can classify full-page-figure or sparse born-digital pages as scanned because the text
layer is short. Current tests already assert clean figure pages do not require OCR enhancement, but
the issue is specifically about scanned-page classification and corpus-level over-count.

### Plan

1. Add characterization tests for full-page-figure and sparse born-digital pages that still have a
   valid text layer.
2. Separate "short text layer" from "no usable native layer" using available PDF image/vector/text
   evidence.
3. Ensure decorative front matter or true image-only pages still classify as scanned.
4. Record remaining corpus-specific validation in `docs/log/` if fixtures are not checked in.

### Acceptance Criteria

- A born-digital full-page-figure page is not classified as scanned solely because text is short.
- True image-only pages remain scanned.
- The classifier explains sparse/figure pages in a way the audit log can surface.

### Verification

- `uv run pytest tests/test_born_digital.py -q`
- `uv run ruff check src/socr/core/born_digital.py tests/test_born_digital.py`

## GH-36 - General clean-equation to LaTeX path

GitHub: https://github.com/r-uben/socr/issues/36
Status: SPLIT (GH-36a DONE; GH-36b DONE — consilium 20260615T210537Z-6621; branch feat/36b-equation-latex)
Priority: P1
Suggested agent: `socr-designer` first
Depends on: corrupt-math recovery already merged
Write ownership: design first; later likely `src/socr/math/*`, `src/socr/pipeline/orchestrator.py`,
tests for math recovery.

### Problem

The corrupt-font math subcase is implemented, but there is still no general clean-equation route to
LaTeX. Native extraction can linearize math, flatten superscripts/subscripts, and lose symbols.

### Design decision (consilium 20260615T210537Z-6621)

Validation-before-splice policy (Fork 1): **Hybrid 1A + 1C, unanimous**.
- 1A structural gate: pylatexenc pure-Python structural validation (offline, deterministic).
- 1C non-destructive presentation: NEVER silently replace crop or native text; inline crop +
  attach 1A-validated LaTeX adjacently (sidecar / comment / alt-text).
- 1B (full render / image-compare) REJECTED — dependency/replay/throughput hazard.
Engine: reuse local qwen3-vl:30b-a3b-instruct (NOT marker-pdf/Texify — too heavy).

### Phase split

**GH-36a — model-free foundation (DONE, branch feat/36a-equation-detection)**

1. Deterministic display-equation REGION DETECTION (math-font spans + centering), no OCR.
2. Crop-PNG storage: `equation_{n}_page{N}.png` under `equations/` beside figures.
3. Manifest/audit provenance: AuditEvent kind `equation_region_detected` in `state.events`.
4. `--detect-equations` CLI flag / `PipelineConfig.detect_equations = False` default, in fingerprint.
5. Throughput harness in `tests/test_equation_detection.py` (geometry-only, deterministic).
6. Fixed: `recover.py` DEFAULT_MODEL changed from forbidden `qwen3-vl:8b` to `qwen3-vl:30b-a3b-instruct`.
7. Documented: `splice_math` has no LaTeX validation (OK for corrupt-font case; GH-36b must gate clean case).

**GH-36b — engine + validation + splice (READY, blocked on GH-36a throughput review)**

1. Wire engine: reuse `latex_for_image` (local qwen3-vl:30b-a3b-instruct) on detected regions.
2. 1A validation gate: pylatexenc structural check before any splice.
3. 1C splice policy: inline crop PNG always; attach 1A-validated LaTeX adjacently; keep native text on failure.
4. New audit event kinds: `equation_latex_accepted` / `equation_latex_rejected_kept_crop`.
5. Add `recover_clean_equations: bool = False` config flag (parallel to `recover_corrupt_math`).
6. Measure per-region model latency; update throughput numbers from GH-36a harness.

### Plan (original)

1. Design the regional equation detection boundary: when to crop, how to avoid whole-page OCR, and
   where to store crop PNGs.
2. Evaluate local candidates such as marker-pdf or Texify on representative equation regions.
3. Validate rendered LaTeX before splicing; on failure, keep the crop plus native linearized text.
4. Add manifest/audit provenance for any equation replacement.

### Acceptance Criteria

- Display equations on a sample chapter produce renderable LaTeX with retained crop PNGs.
- Bad LaTeX never silently replaces a faithful image/crop.
- Region-only throughput is measured before any default-on decision.

### Verification

- Design phase: write `docs/log/YYYY-MM-DD_math-latex-route.md`.
- GH-36a: `uv run pytest tests/test_equation_detection.py tests/test_math_recover.py tests/test_orchestrator.py -q`
- GH-36b: `uv run pytest tests/test_math_recover.py tests/test_orchestrator.py tests/test_equation_detection.py -q`

## GH-46-E1 - Optional OCR skill/profile update

GitHub: https://github.com/r-uben/socr/issues/46
Local plan: `docs/plans/agentic-local-first/TICKETS.md` ticket E1
Status: DEFERRED
Priority: P3
Suggested agent: none until real usage shows the needed profiles
Depends on: real agentic-default usage on papers or Consensus Forecasts batch
Write ownership: likely outside this repo if updating the `/ocr` skill.

### Problem

The existing `/ocr` workflow should eventually expose the new agentic-default and strict-local
controls, but a frozen-profile advisor is premature until real usage shows which profiles are worth
codifying.

### Plan

1. Do not build a live routing agent.
2. After real jobs, decide whether simple skill flag exposure is enough.
3. If a planner is needed, make it emit a frozen profile into the manifest; Python remains the
   executor.

### Acceptance Criteria

- No profile planner exists until backed by observed workflow needs.
- Any future planner writes reproducible, manifest-visible choices.

---

## PP-6 — Fix #54 over-routing: lane-cooccupancy table gate + content-type vector on PageState

GitHub: https://github.com/r-uben/socr/issues/54
Status: DONE
Priority: P1
Branch: fix/54-routing-overroute
Depends on: none

### Problem

The native-vs-ladder routing decision forces a born-digital page onto the Qwen VLM whenever
`has_tables` is set, and `has_tables` was set by the loose `_detect_columnar_numbers` heuristic
(single-token columnar-number ratio) that false-fires on CE chart-axis labels and front-matter.
The stronger `has_numeric_columns` lane-cooccupancy gate existed but was only used for native
reconstruction, not routing. Separately, `apply_born_digital` dropped `has_figures`/`has_equations`
so the per-page gate couldn't route on content type.

### Plan (completed)

1. Route on `has_numeric_columns` instead of `_detect_columnar_numbers` in `born_digital._detect_tables`.
2. Propagate `has_figures`/`has_equations` onto `PageState` via `apply_born_digital`.
3. Remove ad hoc `_last_assessment` re-reads in the tiered routing page_hints block (now uses PageState).
4. Update `_phase_judge_hard_pages` and `_phase_dual_pass_tables` to prefer PageState, with
   `_last_assessment` fallback for partial pipeline runs.

### Acceptance Criteria (met)

- Chart-axis tick values (single x-lane) no longer trigger `has_tables`. (synthetic test)
- Genuine multi-column forecast table still routes to the ladder. (synthetic test)
- `has_figures`/`has_equations` readable from `PageState`; `apply_born_digital` propagates them.
- Existing routing/born-digital tests pass; native reconstruction path unchanged.

### Write ownership

- `src/socr/core/born_digital.py`
- `src/socr/core/state.py`
- `src/socr/pipeline/orchestrator.py`
- `tests/test_born_digital.py`
- `tests/test_document_state.py`

---

## PP-4 — per-page figure extraction + inline embedding

GitHub: https://github.com/r-uben/socr/issues/69
Status: DONE
Priority: P1
Branch: feat/69-pp4-inline-figures
Depends on: PP-1 (GH-65, fragment/stitch scaffold), GH-50 (save/describe split)

### Problem

`_describe_and_embed_figures` appended all figure blocks at the document tail as a
flat list, making the .md non-self-contained per page. PP-4 embeds each figure inline
within the `## Page N` section it belongs to.

### Plan (completed)

1. Added `cap_page: int | None` to `ExtractionResult` so the cap AuditEvent can be
   page-scoped (not always page_num=0).
2. Rewrote `_describe_and_embed_figures` to:
   a. Keep doc-wide extraction (FigureExtractor handles global counter).
   b. Build vision engine once; close once.
   c. Group figures by page.
   d. Parse phantom-stripped `text` with `split_native_pages`.
   e. Append figure blocks to per-page body texts.
   f. Update fragment files atomically (non-fatal on failure).
   g. Reassemble with `assemble_pages(updated_bodies, page_numbers=...)`.
3. Figure-free docs return `text` unchanged (PP-1 byte-identity preserved).

### Acceptance Criteria (met)

- Figures appear inline within `## Page N` sections.
- Figure numbering is doc-global and monotonic across pages.
- Cap AuditEvent uses `cap_page` from ExtractionResult (stopping page, not 0).
- Vision engine constructed once per doc and closed once.
- `--save-figures` only → no VLM call, empty descriptions (GH-50 parity).
- Figure-free doc → byte-identical .md (PP-1 preserved).

### Write ownership

- `src/socr/figures/extractor.py`
- `src/socr/pipeline/orchestrator.py`
- `tests/test_pp4_inline_figures.py`

---

## PP-7 (GH-73) — Chart/figure-page routing lane

GitHub: https://github.com/r-uben/socr/issues/73
Status: DONE
Priority: P1
Branch: feat/73-pp7-chart-lane
Depends on: PP-6 (GH-54)
Settled design: A2 (cluster-first vector detector) + B1 (native prose + PNG ref + audit flag).
Consilium note: `docs/log/2026-06-16_chart-route.md`

### Problem

PP-6 narrowed `has_tables`, so CE chart/front-matter pages that no longer qualify as tables fall
through to native prose — producing word-salad for vector dashboards whose visual payload is in
`get_drawings()`, not `get_images()`. A third routing lane is needed: detect vector charts
deterministically and route them to a saved chart image asset.

### Plan (completed)

1. Added `CHART_MIN_CLUSTER_AREA` named constant and `has_chart_marks(page)` to
   `src/socr/figures/extractor.py`.  Cluster-first: requires ≥1 spatially-coherent drawing
   cluster that (a) meets `CHART_MIN_CLUSTER_AREA`, (b) passes `_has_vector_data_marks`, (c)
   survives `_looks_like_table_grid` rejection.  OR-s with raster `page.get_images()`.
   Logs mark counts + rejection reasons at DEBUG.
2. Added `_is_chart_asset_page(page_num, ps, pdf_path)` predicate to `orchestrator.py`.  Fires
   only when `_is_trusted_native_without_ocr` would have returned True AND not table AND
   `has_chart_marks`.  Does NOT modify PP-6's predicates.
3. Added `_render_chart_page_png(pdf_path, page_num, figures_dir)` helper — renders a full-page
   PNG at `RENDER_DPI`, saves to `chart_page_{N}.png`.  Raises `RuntimeError` on failure.
4. Hooked chart-lane routing into the PP-2 fused agentic loop as an `elif is_native and
   _is_chart_asset_page(...)` branch before the normal native-bypass branch.  B1 representation:
   native prose retained + chart PNG embedded (`![Chart page N](figures/...png)`) + explicit
   AuditEvent(kind="chart_asset_page").  Force PNG regardless of `--save-figures`.
5. Fail-closed on render failure: AuditEvent(kind="chart_asset_render_failed") + status=WARNING +
   audit_passed=False; never silent.

### Acceptance Criteria (met)

- A genuine zero-raster vector chart page routes to chart-asset lane (not OCR ladder).
- Chart-lane page = native prose + chart PNG ref + `chart_asset_page` audit event.
- PNG saved even with `--save-figures=off`; render failure → hard audit error (fail-closed).
- Decorated-vector prose pages (thin neutral rules) do NOT trigger chart lane.
- Dense data tables still route to ladder; clean prose still ships native.
- Monochrome B&W line-plots: documented false-negative (thin neutral strokes only; no VLM needed).
- `has_chart_marks` logs mark counts + rejection reasons.

### Write ownership

- `src/socr/figures/extractor.py`
- `src/socr/pipeline/orchestrator.py`
- `tests/test_chart_lane.py` (new)

---

# Open-issue backlog — 2026-08-09 reconciliation

Added after a two-model triage of every open issue (evidence-gated) plus a three-lens
adversarial review of this decomposition (coverage / gating-safety / ticket-size).
#50 and #51 were closed as already-fixed on 2026-08-09 and are out of scope.

**Design tickets write only `docs/log/*.md`, so they are file-disjoint from all
implementation work and parallelize freely.**

## Stream A — routing / cost

### GH-46-E2 — Make the Ollama-Cloud rung reachable in the agentic ladder · DONE (2026-08-10) · depends-on: none · wave 1
**Problem:** the declared local → Ollama-Cloud → Gemini ladder has no middle rung, for two
independent reasons:
1. `Orchestrator._available_engines_for_agentic()` (`src/socr/pipeline/orchestrator.py:2607`)
   builds from `DEFAULT_PROVIDERS.get(engine_type)`, and `DEFAULT_PROVIDERS[EngineType.QWEN]`
   is `PROFILE_QWEN_LOCAL` (`src/socr/core/providers.py:166`), so `PROFILE_QWEN_CLOUD` can
   never be emitted.
2. Even if emitted, it would be gated out: `QwenEngine.is_available()`
   (`src/socr/engines/qwen.py:96-112`) returns True only via `VLLM_BASE_URL` or
   `_check_ollama_model(OLLAMA_MODEL)` — the *local* instruct build. Cloud availability is
   never probed.

The function's docstring and `providers.py:162` both claim two QWEN rungs are supplied.
`docs/MODELS.md:121-123` honestly records this as open. Tests pass over the gap because
`tests/test_b2_routing.py` and `tests/test_providers.py` hand-construct or patch the profile
list instead of calling the real function.

**Do:** add a cloud-availability probe distinct from the local-model check; emit both QWEN
profiles as distinct rungs when each backend is reachable. Do NOT change `DEFAULT_PROVIDERS`
(the same-EngineType collision is deliberate and documented). Do NOT move the `strict_local`
tier filter — it lives in the caller (`orchestrator.py:1953-1957`) and stays there.
Fold in the `docs/MODELS.md` corrections: `:76-79` and `:118` claim figure description is
Gemini-default and local-first is pending, but `_get_vision_engine()` already returns
`LocalFirstFigureEngine`; and `:121-123` is what this ticket closes.

**Files:** `src/socr/pipeline/orchestrator.py`, `src/socr/core/providers.py`,
`src/socr/engines/qwen.py`, `src/socr/core/ollama_utils.py`, `docs/MODELS.md`,
`tests/test_b2_routing.py`, `tests/test_providers.py`

**Done when:**
- A test constructs `PipelineConfig(agentic=True, enabled_engines=[EngineType.QWEN], quiet=True)`,
  patches `socr.pipeline.orchestrator.get_engine` (module-level import at `orchestrator.py:43`
  — patch the orchestrator namespace, not the engine module) AND the new cloud probe at its
  definition site, then calls the real `_available_engines_for_agentic()` and asserts
  `qwen-local-instruct` and `qwen-cloud` both appear as distinct rungs.
- A test asserts that with `qwen-cloud` present, `_resolve_table_escalation_provider`
  (`orchestrator.py:1409-1426`, `min` by cost) selects `qwen-cloud` ($0.0) over `gemini`
  ($0.0002) — this is a real behavior change to the GH-96 escalation lane.
- `grep -n "one place still cloud-default" docs/MODELS.md` returns nothing.
- `~/venvs/socr/bin/pytest tests/test_b2_routing.py tests/test_providers.py -q` exits 0
  **and** `uvx ruff@0.16.0 format --check .` is clean.

**CI trap:** CI has no ollama and no `qwen-ocr` CLI, so an unstubbed probe returns False and
the first assertion fails there while passing locally. Both seams must be patched by name.
`strict_local` tier-drop is already covered hermetically by `tests/test_b2_routing.py:132-159`;
do not re-assert it against the real function, which returns profiles unfiltered.

**Shipped (`ae68364`, 2026-08-10).** `cloud_model_available()` in `engines/qwen.py` — a
module-level function, not a `QwenEngine` method, because `is_available()` probes the LOCAL tier
and one engine object cannot answer for two backends sharing `EngineType.QWEN`; keeping it off the
class also stops a `get_engine` mock (every attribute truthy) from satisfying it vacuously.
`DEFAULT_PROVIDERS` untouched as instructed; tier filter left in `_phase_agentic`. One correction
found while implementing: the local probe's `except` used `continue`, which would have suppressed
the cloud rung whenever the local probe raised — the two rungs are now genuinely independent, with
a test for it. Escalation-lane behaviour change verified: `qwen-cloud` ($0.0000) now wins over
`gemini` ($0.0002). Hermeticity proven, not assumed — 19/19 pass with `ollama` and `qwen-ocr`
absent from `PATH`. Full suite 1413 passed / 1 xfailed; `ruff format --check` and `ruff check` clean.
`docs/MODELS.md` corrections folded in; the ticket's `grep` gate returns nothing.
`core/providers.py` and `core/ollama_utils.py` needed no change — `providers.py:162` and the
`_available_engines_for_agentic` docstring were already describing this design, they were just
describing something that did not exist yet.

### GH-46-E4 — Generalize the thinking-model prohibition · READY · depends-on: GH-46-E2 · wave 2
**Problem:** the ban on thinking builds is hardcoded to specific model strings
(`src/socr/engines/qwen.py:32-35`, `src/socr/core/config.py:166`). A later #46 comment asks
that it extend to any Ollama model tagged `thinking`. Serialized behind GH-46-E2 (shared `qwen.py`).
**Files:** `src/socr/engines/qwen.py`, `src/socr/core/config.py`, `tests/test_qwen_engine.py`
**Done when:** a model tagged `thinking` is rejected by the resolver without naming it literally;
`~/venvs/socr/bin/pytest tests/test_qwen_engine.py -q` exits 0.

### GH-39A — Human-verified ground truth · BLOCKED (needs human labels) · unchanged
### GH-39B — Calibration artifact + ladder unification · depends-on: GH-39A, GH-46-E2
**Reconciliation (2026-08-09):** still valid. `src/socr/benchmark/calibrate.py` produces a
`CalibrationReport` (save/load + `apply_to_config`), NOT the versioned `calibration.lock.json`
this ticket specifies. `ENGINE_PRIORITY` (`core/config.py:33`), `AUTO_ENGINE_ORDER`
(`core/config.py:53`) and `_LOCAL_ENGINE_ORDER` (`engines/registry.py:64`) remain three
independent hardcoded lists. Gated on GH-46-E2 so the ladder is correct before it is unified.

## Stream B — born-digital fidelity

All implementation tickets here serialize on `src/socr/core/born_digital.py`.

### GH-127-P — Make structured extraction reachable on prose-only pages · READY · depends-on: none · wave 1
**Problem:** two independent bypasses keep prose pages away from the span loop
(`born_digital.py:971-978`) where formatting metadata is available:
(a) `_assess_page:648-656` calls `extract_structured` **only** when `has_tables`; prose pages
get `raw_text.strip()`. (b) `extract_structured:908-909` returns `page.get_text("text").strip()`
when no table regions are found. Any Markdown-structure work applied only to the span loop
would silently affect table-bearing pages alone.
**Do:** restructure both branches so span-level extraction runs on prose pages, with output
byte-identical to today until a later ticket emits structure. Pure reachability refactor.
**Files:** `src/socr/core/born_digital.py`, `tests/test_born_digital.py`
**Done when:** a prose-only fixture page (no tables) provably traverses the span loop; existing
born-digital output is unchanged (golden assertion); `~/venvs/socr/bin/pytest tests/test_born_digital.py -q` exits 0.

### GH-127-A — Emphasis from span flags · depends-on: GH-127-P · wave 2
### GH-127-B — Links from `page.get_links()` rect↔span overlap · depends-on: GH-127-A · wave 3
### GH-127-C — List markers from line prefix + indent · depends-on: GH-127-B · wave 4
Each: **Files** `src/socr/core/born_digital.py`, `tests/test_born_digital.py`. Serial because
they share the file, not because of logic. Each **Done when:** its own fixture round-trips to
`**`/`[text](url)`/`- ` respectively and `~/venvs/socr/bin/pytest tests/test_born_digital.py -q` exits 0.

### GH-127-D-DESIGN — Heading-level derivation · NEEDS-DESIGN · `socr-designer` · depends-on: none · wave 1
**Problem:** "heading level from span size relative to the page's body-text mode" is not an
implementation detail — the mode is polluted by captions, running heads, table headers and
footnotes, and "how much larger than mode" is an undecided rule. The repo's no-magic-numbers
rule relocates this design question rather than eliminating it.
**Sharp question:** what data-derived, documented rule maps a page's span-size distribution to
H1/H2/H3 without a fixed pt threshold, and what does it do with multi-size lines?
**Files:** `docs/log/2026-08-09_heading-derivation.md` (new)
**Done when:** the design note exists, measures size histograms over a named fixture set, and
states the rule precisely enough for a Done-when to be written for GH-127-D.

### GH-64 — Audit-flag tabular-looking pages that fall to native text · depends-on: GH-127-C, GH-46-E2 · wave 5
**Problem:** `_detect_tables` (`born_digital.py:720-745`) requires `has_numeric_columns`, needing
`_MIN_LANES_PER_ROW = 3` (`tables/reconstruct.py:79`). A borderless two-column label|value table
matches neither `find_tables()` nor the lane test, so it falls to native text unflagged; the
structure-loss audit is gated behind `_page_has_tables` and cannot fire. Recorded as PP-6's
residual in `docs/plans/progressive-pages/STATUS.md`.
**Shared-fate verification (2026-08-10 panel):** the single false `has_tables` boolean switches off
*every* table-conditional audit on the page at once — this is why the class ships SUCCESS at $0.00
with no trace, not merely why detection misses it.
**Second leak — verification depth depends on provider availability (verified `orchestrator.py:2510-2530`):**
the `if` at `:2510` is gated only on `_escalation_profile is not None and not _escalation_degraded
and bo.text` — **no** `_page_has_tables` — so with an escalation provider reachable, every
text-bearing page (including free `engine="native"`) reaches `_table_page_needs_escalation`. The
`elif` at `:2520` **is** `_page_has_tables`-gated. So the same PDF is audited to different depths
depending on whether a provider happens to be up: local-only runs, `--strict-local`, CI, and anyone
hit by the unreachable middle rung (GH-46-E2) fall to the gated branch and get nothing. Note the
`elif` gate is a deliberate priced trade-off (its comment: prose pages skip ~137ms scoring because
"they have no table to lose") — this ticket does not remove it, it adds a cheaper probe beside it.
Even on the escalation-enabled branch a GH-64 page returns early at `:1480-1491` before `score_page`
(no located region / no grid), so it earns at most `table_not_scorable`, never an exactness compare.
**Do:** geometry probe for repeated 2-lane alignment; emit an audit event on a page that routed
to native text. Flag only — no routing change. **The probe must run on every page shipping
`engine="native"`, ignoring BOTH gates** — regardless of `_page_has_tables` and regardless of
whether an escalation provider exists — otherwise it inherits the shared fate it exists to break.
Deterministic only: no model call, and reuse the word geometry already read during born-digital
assessment rather than re-reading the page.

**The trust criterion is CROSS-ROW ALIGNMENT STABILITY, not lane count.** This is the design
question the ticket previously left open ("repeated 2-lane alignment" never said what makes the
repetition trustworthy), and getting it wrong reproduces the bug: socr's existing guard against
false-positive table detection *is* the lane count (`_MIN_LANES_PER_ROW = 3`), so a probe built on
lane count structurally cannot see the two-column case. Ask a different question instead —
**a table's column positions are stable across rows; prose word positions are not.** Shape of the
test, in order:
1. group words into rows by y-proximity;
2. within a row, cluster x-positions into lanes by gap;
3. keep rows with **2+** lanes;
4. require several such rows contiguously;
5. **verify the lane start positions agree across those rows** — this step, not the lane count,
   is what separates a borderless label|value table from a paragraph.

This is the criterion that admits two-column tables while still rejecting prose, and it is what
bounds the false-positive risk the panel flagged (bibliographies, glossaries, key/value lists,
leader-dot contents). Expect a residue that alignment alone cannot separate — table-of-contents
pages in particular — and handle it explicitly rather than by tightening the gap constants.

**Derive every constant from the page** (median inter-row spacing, median inter-word gap, font
size), never a tuned literal — the no-magic-thresholds rule applies with full force here, and it
is the main reason not to lift an existing implementation verbatim.

**Prior art — `firecrawl/pdf-inspector` (MIT), read 2026-08-10 at `src/tables/detect_heuristic.rs`.**
An independent deterministic extractor that solves this exact problem with the criterion above
(`find_table_regions_strict`, `:768-830`; `min_cols = 2` at `:945`). Two things to know if you open
it: its doc comment claims "3+ distinct X-position clusters" while the code requires
`cluster_starts.len() >= 2` — the code is correct and the comment is stale; and its constants are
tuned literals (8pt row tolerance, 20pt cluster gap, 25pt gap floor) that must NOT be copied. Cite
the provenance in the implementing code comment. Same repo also carries a **script-attachment
filter** (`:465-479`): sub/superscripts in display equations otherwise cluster into phantom table
regions on TeX-style pages. socr processes that corpus and has no such filter — likely a live
false-positive source for this probe. Out of scope here; file it if the fixture work confirms it.
**Files:** `src/socr/core/born_digital.py`, `src/socr/pipeline/orchestrator.py`, `tests/test_born_digital.py`
**Done when:** `grep -rn "possible_table_structure_not_reconstructed" src/` returns a hit; a fixture
page emits it; the event is in `TABLE_DISTRUST_KINDS` (`core/tables_trust.py:47-73`) so it reaches
the page sidecar, `tables_trust.json`, the document metadata trust note and the CLI summary; the
fixture emits it **with no escalation provider configured** (the local-only branch is the one that
currently has no coverage); **a plain multi-paragraph prose fixture does NOT emit it, and neither
does a leader-dot table-of-contents page** — a probe with only a positive fixture proves nothing,
since the failure mode being guarded against is over-firing on prose; no existing routing test
changes outcome.
**Gating:** GH-127-C for `born_digital.py`, GH-46-E2 for `orchestrator.py`.
**Adjacent, not in scope:** `_check_token_coverage` (`born_digital.py:1042-1094`) is an existing
deterministic post-hoc coverage diagnostic that is region-gated and DEBUG-only, so it never surfaces.
Promoting it to an audit event is a separate cheap win — file it if this ticket confirms the seam.

### GH-56-DESIGN — Settle the residual #56 fork · NEEDS-DESIGN · `socr-designer` · depends-on: none · wave 1
**Problem:** the draft ticket "deterministic multi-section / nested-column reconstructor"
contradicts measured history: `docs/plans/table-repair/` records TR-0…TR-6 done and pure
deterministic geometry proven **insufficient** on the real CE page, with TR-5 VLM segmentation
deferred. An implementer given that ticket would re-litigate a settled failure.
**Sharp question:** is the residual #56 gap a deterministic reconstructor in `reconstruct.py`,
or the deferred VLM-for-structure + geometry value-guard path (TR-5/TR-7)?
**Files:** `docs/log/2026-08-09_56-residual-fork.md` (new)
**Done when:** the note picks one fork with evidence and names the wiring call site.

### GH-49B-DESIGN — Native label→value binding · DONE (2026-08-10) · depends-on: none · wave 1
**Decision:** `docs/log/2026-08-09_native-binding.md`. Numeric-cell rebind over the VLM's
Markdown skeleton — VLM keeps authorship of all structure; native overwrites numbers only,
all-or-nothing per table, gated on one-to-one region match + **full hierarchy path** keys
(not leaf labels) + bijective document order + no `-1` lanes; submitted as a zero-cost
candidate through `decide_escalation`. Ambiguity fails closed and flags the page.
**Scope limit:** born-digital only. On a scan `native_rows_from_page` returns nothing,
`rows_establish_grid` fails, and `orchestrator.py:1477-1479` bails with `table_not_scorable`.
Scanned tables stay on the weaker canary gate (`escalation_decision.py:29-36`).
**Blocks the implementation ticket — two owner decisions open:** (1) ship default-on or
flag-gated for one corpus-measurement cycle; (2) absorb the two-sided cost (per-region
matching also requires the Markdown parser to learn table boundaries — region flattening is
deliberate, `native_rows.py:141-145`) or split it into its own gating ticket.
**Problem:** #49's later comment (from GH-96) asks that a trustworthy native reconstruction
*bind* the label→value mapping. Today `native_rows_from_page` (`tables/native_rows.py:125`)
feeds only `benchmark/table_exactness.py` (grading) and `orchestrator.py:1474-1477` (escalation
gate predicate) — it never corrects or replaces VLM output. GH-49A (the verifier) is genuinely DONE.
**Sharp questions:** (1) replace page markdown from native rows, or only accept/reject a VLM
candidate when labels bind? (2) what makes a native reconstruction "trustworthy" enough to
override? (3) how do hierarchical paths, empty parent rows and multi-table pages become markdown
without inventing columns?
**Files:** `docs/log/2026-08-09_native-binding.md` — written 2026-08-10.
**Done when:** ~~the note answers all three and states an outsider-checkable acceptance test.~~
Met. The note answers all three sharp questions and specifies
`tests/test_native_binding.py`, including a guard asserting a scanned page is a no-op **and**
is flagged (so the dead path cannot later be "fixed" by loosening the gate).
**Note:** GH-49B does NOT depend on GH-56-R — `native_rows.py` imports only `benchmark.scorer`
and `tables.native_verifier`, never `tables.reconstruct`. Its real gate is `orchestrator.py`
serialization against GH-64.

### GH-114-DESIGN — Post-hoc `socr escalate` · DONE (2026-08-10) · depends-on: GH-49B-DESIGN · wave 2
**Decision:** `docs/log/2026-08-09_post-hoc-escalate.md`. **In place — the escalated version
becomes the official copy** (owner call, 2026-08-10). A derived sibling would fork every
document and leave the *better* copy outside the resume path, so future `socr agent` runs
would silently continue from the worse text. Shape: `socr escalate <doc_dir> --pdf <pdf>` as a
staged transaction — refuse on `input_checksum` mismatch or missing artifacts, re-run the grid
trigger, gate every candidate through `decide_escalation` **verbatim** (no separate post-hoc
policy), commit blob + fragment + sidecar + manifest entry + restitched `.md` + `tables_trust`
together.
**Blocks the implementation ticket — one constraint open:** deciding in-place did not dissolve
the fingerprint objection, it converted it into a requirement. `_run_fingerprint` includes
`escalate_ambiguous_tables` (`orchestrator.py:298-301`) precisely so a resumed run cannot
"silently ship a mix of escalated and non-escalated pages" — which is what an in-place pass
under the original fingerprint produces. Choose (a) bump the fingerprint and record the old one
as lineage, or (b) keep it and record escalation state per page in the sidecar. Needs the code
read first. Either way, `Manifest.save` (`manifest.py:207-209`) is a bare `write_text` with no
tmp+rename and must be made atomic — in-place has no untouched original to fall back on.
**Superseded record:** an earlier panel (2026-08-09) logged this as settled on *immutable
sibling* with one model conceding. That round is not citable — a proposal agent was soliciting
confirmatory evidence for a position it had already picked. Re-run clean, the panel split.
**Problem:** escalation happens only inside the live page-major loop (`orchestrator.py:1562-1703`,
mutating `DocumentState`). No CLI path re-escalates an existing document directory; `replay`
(`cli.py:588-628`) deliberately makes zero engine calls, and `manifest.py` rebuilds markdown from
stored page blobs, so rewriting fragments alone would not update replay output.
**Scope correction:** the original HPC-egress premise was invalidated by measurement in the
issue's own comment (egress 403 from lnode01/cnode05 — the in-process lane works on HPC).
Surviving scope is corpus reprocessing for pre-#96 documents and deliberate strict-local runs,
plus manifest consistency and fingerprint semantics. **The issue body needs rewriting to match.**
**Files:** `docs/log/2026-08-09_post-hoc-escalate.md` — written 2026-08-10.
**Done when:** ~~the note defines reprocess identity, the CLI surface, and manifest-consistency
rules.~~ Partially met — CLI surface and manifest-consistency rules are defined; **reprocess
identity is deliberately left as the one open fork (a)/(b) above**, because picking it needs the
resume-gate code read rather than a panel opinion.

## Resolved without a ticket

**#56 resume/PARTIAL semantics.** Triage disagreed on whether an unrecoverable page prevents a
document from sticking on resume. Settled by evidence: a matching checksum + fingerprint makes a
`PARTIAL` document resume-skippable (`orchestrator.py:80-107`); a non-success result carrying text
is recorded `PARTIAL` (`:4655-4661`) and persisted (`:4691-4692`); when reprocessing IS forced,
failed pages are not accepted as terminal and are retried (`:4106-4121`). Covered by
`tests/test_silent_content_destruction.py:619-637` and `tests/test_pp5_resume_ledger.py:755-780`.
Working as designed. No ticket.
