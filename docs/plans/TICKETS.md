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
Status: READY
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
Status: NEEDS-DESIGN
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
Status: NEEDS-DESIGN
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
Status: NEEDS-DESIGN
Priority: P1
Suggested agent: `socr-designer` first
Depends on: corrupt-math recovery already merged
Write ownership: design first; later likely `src/socr/math/*`, `src/socr/pipeline/orchestrator.py`,
tests for math recovery.

### Problem

The corrupt-font math subcase is implemented, but there is still no general clean-equation route to
LaTeX. Native extraction can linearize math, flatten superscripts/subscripts, and lose symbols.

### Plan

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
- Implementation phase: `uv run pytest tests/test_math_recover.py tests/test_orchestrator.py -q`

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
