# TICKETS — Open-issue priority backlog (Waves 1–8)

Source: 41 open GitHub issues in `r-uben/socr`, ranked 2026-08-12.
Graph: [`../issue-priority-graph.md`](../issue-priority-graph.md) · Obsidian: [`../issue-priority-graph.canvas`](../issue-priority-graph.canvas)
Index: root [`TODO.md`](../../../TODO.md) (pointers only). GitHub is source of truth.

Status keys: `READY`, `NEEDS-DESIGN`, `BLOCKED`, `WIP`, `DONE`, `DEFERRED`.
Agents: `socr-designer`, `socr-implementer`, `socr-reviewer`.

## Dispatch rules

- One implementation ticket per agent/worktree.
- Disjoint write sets; serialize if two tickets touch the same file.
- Lower wave first; within a wave, lower rank first.
- Waves 1–3 may run in parallel on separate lanes (content / trust / routing).
- Use `uv run` / `~/venvs/socr/bin/pytest`. Format gate: `uvx ruff@0.16.0 format --check .`.
- Agentic tests must patch `_available_engines_for_agentic` (CI has no ollama).
- Do not estimate calendar time; size work by modules + risk.


---

# Wave 1 — Destroy content (FIRST)

## GH-150 — figures extracted as tables (worst corpus pages)

GitHub: https://github.com/r-uben/socr/issues/150
Status: **READY** · Priority: P0 · Wave: 1 · Rank: 1
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/figures/extractor.py, src/socr/pipeline/orchestrator.py (_is_chart_asset_page), tests for chart vs table routing`

### Problem

has_chart_marks misses thin-line/vector charts (Heston p10, Drechsler p55); table lane wins whenever has_tables fires, so charts ship as broken tables.

### Plan

1. Add a failing fixture/regression for Heston-like high drawing-count thin-line chart pages and Drechsler-like bar charts.
2. Extend chart detection beyond coloured fills/thick strokes: drawing-density / axis-like geometry signals derived from page data (no magic thresholds).
3. Change precedence so strong chart evidence beats tabular-looking axis ticks; keep real tables in the table lane.
4. Ensure chart lane emits image assets (not rowized markdown grids) for those pages.

### Acceptance Criteria

- [ ] Heston p10 / Drechsler p55 class pages no longer ship axis ticks as tables.
- [ ] True table pages still rowize.
- [ ] Audit/manifest shows chart-asset routing for the fixed pages.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'chart or has_chart or figure' ; uvx ruff@0.16.0 format --check src/socr/figures/extractor.py`

## GH-144 — word-geometry rowizer drops numeric values

GitHub: https://github.com/r-uben/socr/issues/144
Status: **READY** · Priority: P0 · Wave: 1 · Rank: 2
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/tables/reconstruct.py (rowize_from_words*), native table assembly, focused table tests`

### Problem

Numeric tokens present in the PDF text layer disappear from reconstructed markdown; page still ships SUCCESS.

### Plan

1. Reproduce with the known page (49/152 missing) as a golden fixture comparing text-layer numbers to markdown cells.
2. Find the column-boundary / clustering step that drops tokens; fix without inventing cells.
3. Fail closed: if post-rowize numeric multiset is smaller than text-layer table numbers, demote/audit rather than SUCCESS.

### Acceptance Criteria

- [ ] Known page retains the previously dropped numeric values.
- [ ] No silent numeric loss on the fixture set; missing numbers surface as audit failure.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'rowize or reconstruct or native_table'`

## GH-147 — landscape pages rowized on wrong axis

GitHub: https://github.com/r-uben/socr/issues/147
Status: **READY** · Priority: P0 · Wave: 1 · Rank: 3
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/tables/reconstruct.py, born_digital dominant_text_direction, table routing refuse-or-rotate`

### Problem

dir=(0,-1) landscape pages cluster by y and emit transposed nonsense with audit_passed=True.

### Plan

1. Gate: if dominant_text_direction is rotated, do not rowize in upright axes.
2. Preferred fix A: refuse native table lane and route to VLM/OCR (image is upright).
3. Optional fix B: transform word coords into reading frame, rowize, transform back.
4. Never ship transposed grids as trusted native SUCCESS.

### Acceptance Criteria

- [ ] Nakamura-Steinsson landscape appendix class pages are coherent or explicitly escalated.
- [ ] Upright pages unchanged.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'landscape or dominant_text or rowize'`

## GH-146 — first data row emitted as table header

GitHub: https://github.com/r-uben/socr/issues/146
Status: **READY** · Priority: P0 · Wave: 1 · Rank: 4
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/tables/reconstruct.py (_is_header_row / header merge)`

### Problem

When grid[0] is not a header, code still promotes a data row into the header band.

### Plan

1. Immediate: if not _is_header_row(grid[0]), emit empty markdown header and keep row 0 in body.
2. Then: extend region upward to capture a real header band when present (same class as #145 boundaries).
3. Regression: numeric-looking first row must not become column names.

### Acceptance Criteria

- [ ] Data rows never become headers.
- [ ] Real header bands still detected when present.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'header_row or reconstruct'`

## GH-152 — side-by-side tables merged into one region

GitHub: https://github.com/r-uben/socr/issues/152
Status: **READY** · Priority: P0 · Wave: 1 · Rank: 5
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/tables/reconstruct.py (region segmentation / x-bands), dual-pass localization`

### Problem

Two column-wise tables merge into one region and flatten vertically.

### Plan

1. Detect x-band gaps between table clusters before rowizing.
2. Rowize each band independently (clip-then-rowize).
3. Fixture: Haim p31-style side-by-side tables + figure.

### Acceptance Criteria

- [ ] Two side-by-side tables emit two markdown tables with structure preserved.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'side_by_side or multi_table or reconstruct'`

## GH-145 — one-point table overlap deletes whole text block

GitHub: https://github.com/r-uben/socr/issues/145
Status: **READY** · Priority: P0 · Wave: 1 · Rank: 6
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/core/born_digital.py (extract_structured overlap clipping)`

### Problem

`extract_structured` discards an entire text block on any intersection with a table region (even one-point), deleting prose around tables while still shipping SUCCESS.

### Plan

1. Replace all-or-nothing `intersects` discard with subtract/clip: keep non-overlapping fragment text.
2. Prefer duplicate-line risk over silent deletion (fail closed on content loss).
3. Regression fixtures from the issue’s before→after recall pages; assert prose near tables survives.

### Acceptance Criteria

- [ ] One-point overlaps no longer delete whole prose blocks.
- [ ] Table regions still exclude in-table text from prose stream.
- [ ] Pages that previously lost caption/note lines retain them or audit-fail loudly.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'born_digital or extract_structured or overlap'`


---

# Wave 2 — Fail closed

## GH-162 — table verifier exceptions fail open

GitHub: https://github.com/r-uben/socr/issues/162
Status: **READY** · Priority: P0 · Wave: 2 · Rank: 1
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/pipeline/agentic.py (SourceEvidenceTableJudge, NativeTableVerifierJudge)`

### Problem

Verifier exceptions delegate to accepting inner judge → unverified tables ship.

### Plan

1. On verifier exception: reject or mark unverifiable and escalate; never accept.
2. Emit durable audit event with verifier type + exception class.
3. Regression: accepting inner judge + raising verifier must not accept.

### Acceptance Criteria

- [ ] Verifier exception cannot produce accepted verdict.
- [ ] Audit event records failure reason.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'SourceEvidence or NativeTableVerifier or agentic'`

## GH-166 — all-failed crop rereads look clean

GitHub: https://github.com/r-uben/socr/issues/166
Status: **READY** · Priority: P0 · Wave: 2 · Rank: 2
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/tables/* crop extractor, orchestrator _reread_page_tables, tables_trust.json`

### Problem

Non-timeout crop failures drop silently; all-failed returns (0,0) with no distrust.

### Plan

1. Every crop yields success or typed failure sentinel.
2. All-failed reread emits page-level distrust in tables_trust.
3. Treat dualpass_crop_timeout as distrust until resolved.

### Acceptance Criteria

- [ ] Render/reader/empty/timeout failures are visible and block clean trust.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'dual_pass or reread_page or tables_trust'`

## GH-161 — resume treats judge-rejected SUCCESS as terminal

GitHub: https://github.com/r-uben/socr/issues/161
Status: **READY** · Priority: P0 · Wave: 2 · Rank: 3
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/pipeline/orchestrator.py (_load_terminal_page)`

### Problem

Resume requires SUCCESS but not audit_passed; best-effort rejected pages are skipped forever.

### Plan

1. Require audit_passed=true (and SUCCESS) for resume skip.
2. Reprocess SUCCESS+audit_passed=false sidecars.
3. Tests for both statuses.

### Acceptance Criteria

- [ ] Only clean audited SUCCESS pages are resume-skippable.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'terminal_page or resume or progressive'`

## GH-140 — math-font pages trusted native without audit

GitHub: https://github.com/r-uben/socr/issues/140
Status: **READY** · Priority: P0 · Wave: 2 · Rank: 4
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/core/born_digital.py, equation detection routing, audit events`

### Problem

Math-font pages ship native SUCCESS while PyMuPDF mangles math; recovery default-off; no audit event.

### Plan

1. Detect math-font / corrupt-math pages and refuse silent native trust.
2. Emit durable audit event when math is present and recovery is off/unrun.
3. Wire optional recovery flags without forcing expensive default if product chooses opt-in — but never silent SUCCESS with known mangling.

### Acceptance Criteria

- [ ] Math-font pages cannot ship as clean native SUCCESS without recovery or an explicit unrecovered audit signal.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'math_font or equation or born_digital'`


---

# Wave 3 — Routing identity

## GH-159 — ProviderProfile identity discarded (cloud runs local)

GitHub: https://github.com/r-uben/socr/issues/159
Status: **READY** · Priority: P0 · Wave: 3 · Rank: 1
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/pipeline/agentic.py (route_page), src/socr/engines/qwen.py, src/socr/core/providers.py`

### Problem

Local and cloud Qwen share EngineType.QWEN; route_page passes only engine → cloud rung executes local backend.

### Plan

1. Pass full ProviderProfile (provider_id/backend/model) into run_provider.
2. Prove local vs cloud attempts use different configs in tests.
3. Provenance must match the profile that ran.

### Acceptance Criteria

- [ ] Each ladder rung executes its declared provider_id/backend/model.
- [ ] Cloud-only env can use qwen-cloud rung.

### Verification

- `~/venvs/socr/bin/pytest tests/test_providers.py tests/test_agentic.py tests/test_qwen_engine.py -q`


---

# Wave 4 — Gates and honesty

## GH-151 — recall is not a sufficient table gate

GitHub: https://github.com/r-uben/socr/issues/151
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 1
Suggested agent: `socr-implementer`
Depends on: #144, #146 (soft)
Write ownership: `src/socr/tables/* metrics/gates, audit heuristics, agentic accept path`

### Problem

100% word recall with destroyed grid still ships; no structure check.

### Plan

1. Add structure signals: column-count consistency, header alignment, empty-pipe rows, numeric lane coherence.
2. Reject/escalate when structure fails even if recall is perfect.
3. Fixture: Bauer-Pflueger-Sunderam p26 class.

### Acceptance Criteria

- [ ] Destroyed-structure tables cannot accept on recall alone.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'table_structure or recall or audit'`

## GH-167 — any embedded raster routes to chart lane

GitHub: https://github.com/r-uben/socr/issues/167
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 2
Suggested agent: `socr-implementer`
Depends on: #150
Write ownership: `src/socr/figures/extractor.py (has_chart_marks)`

### Problem

Any page.get_images() hit forces chart lane — logos/photos/signatures false-positive.

### Plan

1. Require size/placement/semantic filters; image presence alone insufficient.
2. Keep true raster charts in chart lane.
3. Regressions for both chart and non-chart rasters.

### Acceptance Criteria

- [ ] Logos/photos/signatures stay native/ordinary figures; real charts still route.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'has_chart or chart_asset'`

## GH-163 — any OCR word defers scanned source-evidence gate

GitHub: https://github.com/r-uben/socr/issues/163
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 3
Suggested agent: `socr-implementer`
Depends on: #162
Write ownership: `src/socr/pipeline/agentic.py / table verifiers (page_has_native_words)`

### Problem

Any non-empty word defers scanned verification; corrupt OCR layers skip needed checks.

### Plan

1. Use trusted-native classification, not mere word presence.
2. Untrusted OCR layers still run raster/classical verification.
3. Regression: untrusted words + hallucinated table → hard reject.

### Acceptance Criteria

- [ ] Scanned pages with rejected OCR layers still verify.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'source_evidence or scanned_table or native_words'`

## GH-154 — --max-cost-per-page does not constrain qwen-cloud ($0)

GitHub: https://github.com/r-uben/socr/issues/154
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 4
Suggested agent: `socr-implementer`
Depends on: #159
Write ownership: `src/socr/core/providers.py (PROFILE_QWEN_CLOUD cost), provider_ladder filtering, CLI help`

### Problem

Cloud rung priced $0.00 bypasses max_cost_per_page; only --strict-local actually blocks cloud.

### Plan

1. Give qwen-cloud a real positive cost_per_page_usd (or treat $0 paid rungs as uncapped-only with explicit policy).
2. Document how caps treat cloud rungs.
3. Test: max_cost_per_page=0 excludes cloud unless opted in.

### Acceptance Criteria

- [ ] max_cost_per_page can keep runs local when cloud is available.

### Verification

- `~/venvs/socr/bin/pytest tests/test_providers.py -q -k 'cost or ladder or cloud'`

## GH-160 — table escalation ignores cost caps/budget

GitHub: https://github.com/r-uben/socr/issues/160
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 5
Suggested agent: `socr-implementer`
Depends on: #154
Write ownership: `src/socr/pipeline/orchestrator.py (_resolve_table_escalation_provider, _escalate_table_page)`

### Problem

Escalation picks from raw engines, ignores max_cost_per_page and remaining cost_budget.

### Plan

1. Select escalation from the fully filtered ladder.
2. Pre-check remaining document budget before starting a call.
3. Count timeout/refusal/empty as attempted cost where policy requires.

### Acceptance Criteria

- [ ] No escalation call starts that violates per-page or document budget.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'escalat or cost_budget or table_escalation'`

## GH-139 — --no-audit inert on agentic path

GitHub: https://github.com/r-uben/socr/issues/139
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 6
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/cli.py, src/socr/pipeline/orchestrator.py (_phase_agentic)`

### Problem

audit_enabled only gates non-agentic branches; agentic ignores --no-audit.

### Plan

1. Either wire audit_enabled into agentic accept/judge path, or reject --agentic --no-audit with a clear error.
2. Prefer reject-combination if agentic-without-audit is incoherent for citation corpus.
3. Update help text.

### Acceptance Criteria

- [ ] Flag is either honored or hard-errors; never silently inert.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'no_audit or audit_enabled or cli'`

## GH-168 — --config/--profile values silently dropped

GitHub: https://github.com/r-uben/socr/issues/168
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 7
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/core/config.py (from_file), src/socr/cli.py precedence`

### Problem

from_file omits many fields; CLI defaults overwrite loaded profile values.

### Plan

1. Complete schema/allowlist for all PipelineConfig fields.
2. Absent CLI options preserve loaded values; explicit CLI overrides.
3. CliRunner tests for budgets, judge, timeout, figures, write_manifest.

### Acceptance Criteria

- [ ] Loaded config fields survive unless explicitly overridden.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'config or profile or from_file or cli'`

## GH-172 — soft timeouts leave workers that block CLI exit

GitHub: https://github.com/r-uben/socr/issues/172
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 8
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `timeout wrappers around provider/judge/table-escalation/crop (process boundary)`

### Problem

future.cancel()+shutdown(wait=False) cannot stop running non-daemon threads.

### Plan

1. Run killable work in subprocess/process pool, or daemonize with explicit kill path.
2. Cascade halt must leave no live worker delaying shutdown.
3. Integration: never-returning stub exits within bounded wall clock.

### Acceptance Criteria

- [ ] Blocked provider cannot keep CLI alive past deadline.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'timeout or halt or cascade'`

## GH-177 — single-file exit codes disagree with partial=nonzero

GitHub: https://github.com/r-uben/socr/issues/177
Status: **READY** · Priority: P1 · Wave: 4 · Rank: 9
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `CLI process vs batch exit mapping, RunOutcome policy module`

### Problem

Batch exits nonzero on partial; single-file process() can exit 0 on AUDIT_FAILED/PARTIAL.

### Plan

1. One shared exit-code policy for process and batch.
2. Document codes for AUDIT_FAILED / PARTIAL_SAVE / lost-content.
3. CliRunner parity tests.

### Acceptance Criteria

- [ ] Identical EngineResult.status → identical exit code on both entrypoints.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'exit_code or RunOutcome or cli'`


---

# Wave 5 — Equations

## GH-157 — recover-clean-equations skips pages without PageOutput

GitHub: https://github.com/r-uben/socr/issues/157
Status: **READY** · Priority: P1 · Wave: 5 · Rank: 1
Suggested agent: `socr-implementer`
Depends on: none (soft: #140)
Write ownership: `src/socr/pipeline/orchestrator.py (_attach_equation_latex_sidecars)`

### Problem

Native-trusted pages with detected equation crops skip attach: 'no PageOutput'.

### Plan

1. Ensure every equation_region_detected page has a mutable PageOutput before attach (create native one if needed).
2. Share one attach helper across agentic and legacy paths.
3. Regression: --detect-equations --recover-clean-equations never logs no PageOutput skip for detected regions.

### Acceptance Criteria

- [ ] Detected crops get sidecar/LaTeX or explicit rejected-kept-crop audit — never silent skip.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'equation or recover_clean or sidecar'`

## GH-165 — PUA-only math pages skip recovery routing

GitHub: https://github.com/r-uben/socr/issues/165
Status: **READY** · Priority: P1 · Wave: 5 · Rank: 2
Suggested agent: `socr-implementer`
Depends on: #140
Write ownership: `has_equations / has_unmapped_math_glyphs, equation routing, unrecovered audit`

### Problem

has_equations omits has_unmapped_math_glyphs; PUA pages miss detection; unrecovered audits can be suppressed wrongly.

### Plan

1. Include PUA/unmapped glyphs in equation detection when flags enabled.
2. Suppress native_math_unrecovered only after successful region recovery.
3. STIXGeneral+PUA end-to-end regression.

### Acceptance Criteria

- [ ] PUA-only math enters recovery; failures retain durable unrecovered event.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'unmapped_math or PUA or equation'`

## GH-164 — rejected recovery appends full-page native text per region

GitHub: https://github.com/r-uben/socr/issues/164
Status: **READY** · Priority: P1 · Wave: 5 · Rank: 3
Suggested agent: `socr-implementer`
Depends on: #157
Write ownership: `_attach_equation_latex_sidecars / build_equation_sidecar`

### Problem

On validation failure, full page native_text is appended per region, duplicating prose.

### Plan

1. Crop fallback native text to each equation bbox.
2. Do not append text already present in PageOutput.
3. Tests with one and multiple rejected regions.

### Acceptance Criteria

- [ ] Surrounding prose stays single-copy; fallback is region-local.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'equation_sidecar or recover'`


---

# Wave 6 — Provenance

## GH-158 — populate model_version in fingerprints / provenance

GitHub: https://github.com/r-uben/socr/issues/158
Status: **READY** · Priority: P2 · Wave: 6 · Rank: 1
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `manifest/fingerprint writers, EngineResult provenance, resume identity`

### Problem

model_version often blank; resume/replay identity weaker than designed.

### Plan

1. Write non-empty model/model_version for non-native pages.
2. Native pages record engine=native explicitly.
3. Changing model tag changes fingerprint / invalidates resume.

### Acceptance Criteria

- [ ] Provenance fields populated; docs paragraph on fields.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'fingerprint or model_version or manifest'`

## GH-173 — resume fingerprint omits auto_patch_tables and equation models

GitHub: https://github.com/r-uben/socr/issues/173
Status: **READY** · Priority: P2 · Wave: 6 · Rank: 2
Suggested agent: `socr-implementer`
Depends on: #158
Write ownership: `orchestrator._run_fingerprint`

### Problem

Fingerprint has detect/recover equation flags but not clean_equation_model/math_model/auto_patch_tables.

### Plan

1. Include auto_patch_tables + resolved equation model identities.
2. Changing any invalidates terminal resume.
3. One regression per omitted knob.

### Acceptance Criteria

- [ ] Fingerprint inequality for each newly included knob.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'run_fingerprint or resume'`

## GH-171 — terminal sidecars finalized before figure provenance

GitHub: https://github.com/r-uben/socr/issues/171
Status: **READY** · Priority: P2 · Wave: 6 · Rank: 3
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `orchestrator assemble/flush vs _describe_and_embed_figures`

### Problem

Terminal sidecars flush before figures exist; figure metadata missing from authoritative sidecars.

### Plan

1. Rewrite authoritative terminal sidecars after figure extraction.
2. Include paths/bbox/type/caption engine/page-local figure audit events.
3. Resume tests for figure-phase retry.

### Acceptance Criteria

- [ ] Sidecar provenance matches final Markdown and manifest.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'sidecar or figure or progressive'`

## GH-170 — replay ignores figure/chart/equation assets

GitHub: https://github.com/r-uben/socr/issues/170
Status: **READY** · Priority: P2 · Wave: 6 · Rank: 4
Suggested agent: `socr-implementer`
Depends on: #171
Write ownership: `manifest provenance + replay asset verification`

### Problem

Replay checks page blobs only; missing images still reassemble markdown paths.

### Plan

1. Hash and record every emitted visual asset in manifest.
2. Replay fails loudly on missing/corrupt assets; copy or rewrite on relocate.

### Acceptance Criteria

- [ ] Missing/modified/relocated assets covered by tests.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'replay or manifest'`

## GH-169 — manifests drop judge rejection reasons for non-empty attempts

GitHub: https://github.com/r-uben/socr/issues/169
Status: **READY** · Priority: P2 · Wave: 6 · Rank: 5
Suggested agent: `socr-implementer`
Depends on: none (soft: #161)
Write ownership: `_phase_agentic attempt recording / manifest builder`

### Problem

skip_reason copied only when rejected output is empty; non-empty rejects lose AcceptDecision.reason.

### Plan

1. Persist judge reason for every attempt (accept + reject).
2. Serialize raw verdict fields JSON-safely.
3. Round-trip tests: accept, semantic reject, judge failure, timeout, budget skip.

### Acceptance Criteria

- [ ] Non-empty rejected attempts retain reasons in manifests.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'manifest or skip_reason or attempt'`

## GH-142 — audit every CLI flag against agentic path

GitHub: https://github.com/r-uben/socr/issues/142
Status: **READY** · Priority: P2 · Wave: 6 · Rank: 6
Suggested agent: `socr-implementer`
Depends on: #139, #154, #168 (soft — do after known liars)
Write ownership: `cli help + agentic config consumption matrix + tests`

### Problem

Several flags lie on default agentic path; need systematic classify works / non-agentic-by-design / reject-combo.

### Plan

1. Inventory ~50 CLI flags vs fields _phase_agentic actually reads.
2. Fix or document each liar; reject incoherent combos.
3. Checkable matrix in docs + CliRunner smoke.

### Acceptance Criteria

- [ ] No silently inert flag on agentic without help-text saying so or hard error.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'cli' ; manual matrix review`

## GH-64 — audit-flag tabular-looking native fallthrough (PP-6 follow-up)

GitHub: https://github.com/r-uben/socr/issues/64
Status: **READY** · Priority: P2 · Wave: 6 · Rank: 7
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `born-digital table gate residuals + audit event emission`

### Problem

Borderless 2-column tables fail detection and fall to native unflagged.

### Plan

1. Detect tabular-looking native fallthrough without changing #54 routing gate.
2. Emit durable audit event naming the page.
3. Prose pages must not trip the flag.

### Acceptance Criteria

- [ ] 2-column borderless fixture flagged; prose clean; routing unchanged.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'native_fallthrough or has_numeric or audit'`


---

# Wave 7 — Architecture

## GH-178 — ADR: stay Python; optional native kernels only after profiling

GitHub: https://github.com/r-uben/socr/issues/178
Status: **READY** · Priority: P3 · Wave: 7 · Rank: 1
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `docs/ ADR + ARCHITECTURE.md pointer`

### Problem

Need recorded decision against Rust rewrite; optional native only for profiled pure kernels.

### Plan

1. Write ADR: Python orchestration stays; native kernels only after profiling names a pure hot function.
2. Link from ARCHITECTURE.md and #155/#174.

### Acceptance Criteria

- [ ] ADR merged; no rewrite of CLI/engines/orchestrator without measured bottleneck.

### Verification

- `docs review only`

## GH-174 — quarantine legacy backbone; agentic first-class only

GitHub: https://github.com/r-uben/socr/issues/174
Status: **NEEDS-DESIGN** · Priority: P3 · Wave: 7 · Rank: 2
Suggested agent: `socr-designer`
Depends on: none (before #155)
Write ownership: `orchestrator process() branches, ARCHITECTURE.md, CLI`

### Problem

Default is agentic but full deterministic stack still maintained; flag semantics diverge.

### Plan

1. Design: delete vs --legacy-routing allowlist.
2. Docs/CLI state agentic as sole product path.
3. Collapse dual dual-pass/judge wiring to one call site.
4. CI guard: agentic-only flags cannot be wired only in legacy branches.

### Acceptance Criteria

- [ ] Legacy deleted or single explicit flag; ARCHITECTURE.md updated.

### Verification

- `design note in docs/log/ then implementer pass`

## GH-175 — break inverted package layering

GitHub: https://github.com/r-uben/socr/issues/175
Status: **READY** · Priority: P3 · Wave: 7 · Rank: 3
Suggested agent: `socr-implementer`
Depends on: none (parallel with #155)
Write ownership: `socr.tables ↔ socr.benchmark imports; born_digital → tables privates`

### Problem

Production tables import benchmark; core imports private table regexes.

### Plan

1. Move shared helpers to a public owned module.
2. Enforce DAG: benchmark→core/tables only.
3. Import-linter or test for the rule.

### Acceptance Criteria

- [ ] No tables/core → benchmark imports; no cross-package private imports.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'import or layer' ; import graph check`

## GH-176 — dumb DocumentState + one authoritative page-text selector

GitHub: https://github.com/r-uben/socr/issues/176
Status: **READY** · Priority: P3 · Wave: 7 · Rank: 4
Suggested agent: `socr-implementer`
Depends on: #174 (soft)
Write ownership: `src/socr/core/state.py, assemble/manifest/sidecar readers`

### Problem

PageState.needs_repair encodes policy; DocumentState.text stitches authoritatively in conflict with assemble.

### Plan

1. Move repair/routing policy out of PageState into pipeline policy objects.
2. One canonical_page_texts selector for assemble/manifest/sidecars/tests.
3. Document facts vs decisions on the blackboard.

### Acceptance Criteria

- [ ] Single authoritative text selector; dumb blackboard restored.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'state or assemble or canonical'`

## GH-155 — split orchestrator.py god-module (~5.5k LOC)

GitHub: https://github.com/r-uben/socr/issues/155
Status: **NEEDS-DESIGN** · Priority: P3 · Wave: 7 · Rank: 5
Suggested agent: `socr-designer then socr-implementer`
Depends on: #174, #178 (soft)
Write ownership: `pipeline/{preflight,agentic_loop,legacy_tiers,assemble}.py, lanes/{tables,equations,figures}.py`

### Problem

One file owns routing, agentic loop, tables, equations, figures, audit, resume — blocks review and dual-path honesty.

### Plan

1. Design module map (behavior-preserving moves).
2. Extract lanes and assemble first; keep DocumentState blackboard.
3. Facade <~800 LOC target (<1.5k hard).
4. Unit tests per lane without importing whole orchestrator.

### Acceptance Criteria

- [ ] Split landed green; progressive+replay fixtures pass; ARCHITECTURE section names canonical path.

### Verification

- `~/venvs/socr/bin/pytest -q ; uvx ruff@0.16.0 format --check .`

## GH-156 — TODO.md / TICKETS.md drift vs GitHub

GitHub: https://github.com/r-uben/socr/issues/156
Status: **READY** · Priority: P3 · Wave: 7 · Rank: 6
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `TODO.md, docs/plans/TICKETS.md, this backlog`

### Problem

TODO lists closed work as open; live bugs only in prose.

### Plan

1. Rewrite TODO.md as pointer board to open issues (this change).
2. Mark superseded TICKET-19..24 rows with links.
3. Policy line: GitHub is SoT; TODO is an index.

### Acceptance Criteria

- [ ] Every Now/next bullet links an open issue or is removed.

### Verification

- `manual link audit`


---

# Wave 8 — North-star / design (LAST)

## GH-49 — three-layer method ADR (extract / verify / escalate)

GitHub: https://github.com/r-uben/socr/issues/49
Status: **NEEDS-DESIGN** · Priority: P4 · Wave: 8 · Rank: 1
Suggested agent: `socr-designer`
Depends on: none
Write ownership: `docs ADR + ARCHITECTURE.md; guides #151/#162`

### Problem

Need recorded general method: single-pass VLM + free native verify + agentic-on-signal.

### Plan

1. ADR documenting three layers and born-digital vs scan caveats.
2. Apply lens to figures; keep implementation tickets separate.

### Acceptance Criteria

- [ ] ADR merged; referenced from table/figure gate work.

### Verification

- `docs review`

## GH-39 — quality-per-dollar calibrated ladders

GitHub: https://github.com/r-uben/socr/issues/39
Status: **NEEDS-DESIGN** · Priority: P4 · Wave: 8 · Rank: 2
Suggested agent: `socr-designer`
Depends on: Stage2 GT human
Write ownership: `benchmark calibrate, calibration.lock.json, ladder unification`

### Problem

Stage 1 partial; no calibration.lock.json; ladders still independent; needs human GT for Stage 2.

### Plan

1. Finish remaining Stage 1 mechanical gaps if any.
2. Stage 2: hand-verified GT for table/equation pages.
3. Stage 3: calibrate --apply + all ladders delegate to artifact.

### Acceptance Criteria

- [ ] Versioned calibration.lock.json drives AUTO/_LOCAL/RepairRouter/provider_ladder.

### Verification

- `socr benchmark run / calibrate; unit tests for artifact load`

## GH-114 — socr escalate post-hoc pass

GitHub: https://github.com/r-uben/socr/issues/114
Status: **NEEDS-DESIGN** · Priority: P4 · Wave: 8 · Rank: 3
Suggested agent: `socr-designer`
Depends on: #159,#154 (soft)
Write ownership: `new CLI subcommand design; resume/fingerprint interaction`

### Problem

Inline escalation couples local GPU and cloud egress in one process.

### Plan

1. Design post-hoc escalate reading tables_trust / audit flags.
2. Byte-identical fragment rewrite + replay of escalated doc.
3. Rewrite issue body for current HPC premise.

### Acceptance Criteria

- [ ] Design note + acceptance from issue; implement later.

### Verification

- `design note in docs/log/`

## GH-127 — native path discards heading/emphasis/list/link structure

GitHub: https://github.com/r-uben/socr/issues/127
Status: **READY** · Priority: P4 · Wave: 8 · Rank: 4
Suggested agent: `socr-implementer`
Depends on: none
Write ownership: `src/socr/core/born_digital.py extract_structured`

### Problem

Native emits flat prose; font size/flags/links discarded.

### Plan

1. Heading tiers from document font-size distribution (no hardcoded pt).
2. Links from page.get_links(); bold/italic from span flags; list markers at line start.
3. No change to born-digital classification/routing.

### Acceptance Criteria

- [ ] Structured markdown for headings/emphasis/lists/links on fixtures.

### Verification

- `~/venvs/socr/bin/pytest tests -q -k 'born_digital or extract_structured'`

## GH-56 — CE OCR umbrella — reliable tables and figures

GitHub: https://github.com/r-uben/socr/issues/56
Status: **NEEDS-DESIGN** · Priority: P4 · Wave: 8 · Rank: 5
Suggested agent: `socr-designer`
Depends on: Waves 1–4 product fixes
Write ownership: `tracker only; points at Wave 1–5 tickets`

### Problem

CE corpus not production-ready; tables/figures remain the bottleneck.

### Plan

1. Keep as umbrella: do not implement directly.
2. CE smoke acceptance after Waves 1–4 land.
3. Recommended CE command before overwriting production OCR.

### Acceptance Criteria

- [ ] Smoke on 202401.pdf meets issue acceptance once child tickets done.

### Verification

- `manual CE smoke after child tickets`

