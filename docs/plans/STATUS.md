# STATUS - GitHub issue action plan

Last updated: 2026-08-09
Branch: `chore/triage-open-issues`
Base reviewed: `29dc6f0` (`main`)
GitHub source: open issues #39, #46, #49, #56, #64, #114, #127
(#34, #35, #37, #50, #51 closed; #36, #47 shipped)

## Stage

**Backlog reconciled 2026-08-09 after ~8 weeks of drift.** Every open issue was triaged by two
independent models with mandatory file:line evidence, then the resulting decomposition was
attacked by three orthogonal review lenses (coverage / gating-safety / ticket-size).

Closed as already-fixed, each confirmed independently by both triage models citing the same commit:
- **#51** → `88b6cc8` (`resolve_qwen_intent` makes local resolution backend-agnostic)
- **#50** → `67edaca` (`--save-figures` PNG-only; `--describe-figures` separate opt-in)

New tickets are in `TICKETS.md` under "Open-issue backlog — 2026-08-09 reconciliation".

**Two board claims were stale and are now corrected:**
- `GH-49A` is genuinely DONE (`NativeTableVerifierJudge`, `pipeline/agentic.py:443`), but #49
  grew a new deliverable *after* the ticket closed — native label→value binding. Now `GH-49B-DESIGN`.
- `GH-46-E1` was deferred "until real usage shows needed profiles". Wrong frame: the Ollama-Cloud
  rung is **unreachable**, not unrequested. Now `GH-46-E2`, and E1 remains separate (it is the
  `/ocr` skill/profile interface, not the runtime ladder).

**Verified defect found during triage (`GH-46-E2`):** the declared local → Ollama-Cloud → Gemini
ladder has no middle rung, for two independent reasons — `_available_engines_for_agentic`
(`orchestrator.py:2607`) can only emit `PROFILE_QWEN_LOCAL`, and `QwenEngine.is_available()`
(`engines/qwen.py:96-112`) never probes cloud availability at all. The function's docstring and
`providers.py:162` both claim otherwise; tests pass because they hand-construct the profile list
rather than calling the real function. `docs/MODELS.md:121-123` honestly records it as open.

Prior wave (PR #52, `2409192`) remains as recorded below: GH-51, GH-50, GH-34, GH-46-D2, GH-47A,
GH-37, GH-35, GH-35-FU shipped and were accepted after implementer/reviewer passes.

The durable planning format is:

- `docs/plans/TICKETS.md` - canonical issue-derived backlog.
- `docs/plans/STATUS.md` - live execution, assignment, dependency, and agent state.
- `docs/plans/agentic-local-first/` - FROZEN #46-phase-2 subplan (historical reference + logs).
  Its open items D2/E1 are now owned here as GH-46-D2 / GH-46-E1.

## Issue Review

Refreshed 2026-08-09 from a two-model evidence-gated triage of every open issue.

| Issue | Current read | Ticket mapping |
|-------|--------------|----------------|
| #127 | still-valid (unanimous). Native path discards size/flags/font and never calls `page.get_links()`. Filed 2026-08-09. | GH-127-P, -A, -B, -C, -D-DESIGN |
| #114 | still-valid, **re-scoped**. HPC-egress premise invalidated by measurement in the issue's own comment; corpus-reprocessing scope survives and is unimplemented. Issue body needs rewriting. | GH-114-DESIGN |
| #64 | still-valid (unanimous). Borderless 2-column tables fail both detection passes and fall to native text unflagged. Already recorded as PP-6's residual. | GH-64 |
| #56 | partially-addressed (unanimous). #79/#87 landed; multi-section/nested-column reconstruction did not. Fork needs settling first. | GH-56-DESIGN |
| #49 | partially-addressed (unanimous). GH-49A verifier DONE; native label→value *binding* is net-new. | GH-49B-DESIGN |
| #46 | partially-addressed. Lineup constants landed (`6d6ee79`); the Ollama-Cloud rung is unreachable. | GH-46-E2, GH-46-E4, (GH-46-E1 separate) |
| #39 | partially-addressed (unanimous). Stage 1 landed; no `calibration.lock.json` exists and the three ladder lists remain independent. | GH-39A (blocked), GH-39B |
| #50, #51 | **CLOSED 2026-08-09** as already-fixed. | — |

## Dispatch waves (2026-08-09 backlog)

Design tickets write only `docs/log/*.md` and are file-disjoint from everything, so all four
NEEDS-DESIGN rows can run concurrently with wave-1 implementation.

| Wave | Tickets | Shared-file note |
|------|---------|------------------|
| 1 | GH-46-E2 · GH-127-P · GH-127-D-DESIGN · GH-56-DESIGN · GH-49B-DESIGN | disjoint |
| 2 | GH-46-E4 · GH-127-A · GH-114-DESIGN | E4 serializes behind E2 on `engines/qwen.py` |
| 3 | GH-127-B | `born_digital.py` serialization |
| 4 | GH-127-C | `born_digital.py` serialization |
| 5 | GH-64 | needs GH-127-C (`born_digital.py`) + GH-46-E2 (`orchestrator.py`) |
| — | GH-39B | after GH-39A (human labels) + GH-46-E2 |

**Stream B parallelism is low and that is real, not a modelling artifact:** GH-127-A/B/C and GH-64
all own `src/socr/core/born_digital.py`. They are split for reviewability, not for concurrency.

## Next action

Dispatch wave 1. The four design tickets are free (docs-only); `GH-46-E2` is the only wave-1
implementation ticket with a fully written Done-when, and it carries a named CI-hermeticity
recipe — CI has no ollama, so both probe seams must be patched by name or the test passes locally
and fails in CI.


## Completed in PR #52

| Ticket | Commit | Notes |
|--------|--------|-------|
| GH-51 | `88b6cc8` | Unambiguous qwen backend/model resolution. |
| GH-50 | `67edaca` | Split `--save-figures` from opt-in `--describe-figures`. |
| GH-34 | `2c486cf` | Empty repair output cannot become `best_output` or `recovered_by`. |
| GH-46-D2 | `2d3dadb` | Prompt-only sparse-row header-lane anchoring; manual VLM validation pending. |
| GH-47A | `b20521c` | Figure cap signal and title-page logo/letterhead filter. |
| GH-37 | `779f512` | `--native-only` policy control and fingerprints. |
| GH-35 | `842ab26` | Rescue sparse/full-page-figure born-digital pages from the word-count gate. |
| GH-35-FU | `2409192` | Raster-coverage gate for image-dominant clean-short-text pages. |

## Structured Content Routing (GH-49-routing)

| Ticket | Status | Notes |
|--------|--------|-------|
| GH-49-routing | DONE | Born-digital table pages routed through OCR ladder; provenance-masking guard added; 844 tests pass |
| GH-49A | DONE | Two-tier deterministic verifier (NativeTableVerifierJudge) wrapping every judge; geometry_impossible_collapse hard-fail + warn-and-defer tiers; 864 tests pass |

## Ready Queue

Last non-design implementable ticket:

| Ticket | Agent | Ownership | Notes |
|--------|-------|-----------|-------|
| GH-47B | DONE | figure caption prompt/tests | Anti-fabrication prompt and warning; unblocked by GH-50. |

Design/research queue:

| Ticket | Agent | Blocker |
|--------|-------|---------|
| GH-47C | `socr-designer` first | Needs GH-50/GH-47B shape before implementation. |
| GH-49A | `socr-designer` first | Needs design note for deterministic native table verifier. |
| GH-36 | `socr-designer` first | Needs route design and local engine evaluation. |
| GH-39A | none | Requires human-verified labels. |
| GH-39B | `socr-implementer` later | Blocked on GH-39A. |
| GH-46-E1 | none | Deferred until real usage shows needed profiles. |

## Active Agents

| Ticket | Agent id/name | Started | Status | Owned files | Notes |
|--------|---------------|---------|--------|-------------|-------|
| PP-6 (GH-54) | socr-implementer (claude-sonnet-4-6) | 2026-06-16 | DONE | born_digital.py, state.py, orchestrator.py, test_born_digital.py, test_document_state.py | Lane-cooccupancy gate + content-type vector; 968 tests pass |
| GH-49-routing | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | orchestrator.py, test_orchestrator.py, STATUS.md, log/2026-06-15_structured-content-routing.md | Provenance-masking fix: native_table_structure_failed set on agentic rejection; 844 tests pass |
| GH-49A | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | native_verifier.py, agentic.py, orchestrator.py, test_native_table_verifier.py, test_p1_cascade_economics.py | Two-tier verifier; hard-fail=geometry_impossible_collapse; warn-and-defer for ambiguous mismatches; 864 tests pass |
| GH-51 | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | config.py, qwen.py, cli.py, test_qwen_engine.py | Resolver added; 791 tests pass |
| GH-50 | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | config.py, cli.py, orchestrator.py, test_orchestrator.py | save/describe split; 801 tests pass |
| GH-34 | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | state.py, audit_log.py, test_silent_content_destruction.py, test_audit_log.py | Empty-repair guard + empty-recovery event guard; 808 tests pass |
| GH-46-D2 | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | prompts/table_extract.md | Prompt-only fix: column-lane anchoring bullet added; 39 tests pass; empirical CBO-row VLM validation PENDING (manual) |
| GH-47A | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | extractor.py, orchestrator.py, test_figure_pass.py, test_orchestrator.py | Cap signal (cap_reached flag + audit event + console warn) + logo filter (3-condition geometry); 815 tests pass |
| GH-37 | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | cli.py, config.py, born_digital.py, orchestrator.py, test_orchestrator.py | --native-only flag + PipelineConfig.native_only + fingerprint + routing in backbone+agentic; 828 tests pass |
| GH-35 | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | born_digital.py, test_born_digital.py | Word-count gate made quality-aware; sparse/figure pages rescued; 7 new tests; 835 total pass |
| GH-35-FU | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | born_digital.py, test_born_digital.py | Raster-coverage gate (RASTER_DOMINANCE_RATIO=0.90) added; scan+baked-OCR false-positive fixed; Tr-mode discriminator skipped as fragile; 160 tests pass |
| GH-47B | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | _figure_prompt.py (new), gemini_api.py, vllm.py, test_gemini_api.py | Anti-fabrication prompt extracted to shared module; all 3 engine paths (Gemini/Ollama/vLLM-HPC) use hardened prompt + wrap_caption; 867 tests pass; empirical VLM validation PENDING (manual) |
| GH-47C | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | extractor.py, result.py, orchestrator.py, test_figure_pass.py, test_orchestrator.py | Option C log-only: bbox persisted on ExtractedFigure+FigureInfo; recoverable-label AuditEvent(kind="figure_recoverable_labels") for described figures; 878 tests pass |
| GH-36a | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | math/detect_equations.py (new), math/recover.py, core/config.py, cli.py, pipeline/orchestrator.py, tests/test_equation_detection.py | Model-free equation region detector + crop-PNG + provenance + detect_equations flag + fingerprint + throughput harness + :8b defect fix; 855 tests pass |
| GH-36b | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | math/equation_latex.py (new), math/validate_latex.py (new), pipeline/orchestrator.py, core/config.py, cli.py, pyproject.toml, tests/test_equation_latex.py (new) | VLM engine + pylatexenc 1A gate + 1C non-destructive sidecar + provenance events + recover_clean_equations flag (default off); 886 tests pass |
| PP-0 (GH-55) | socr-implementer (claude-sonnet-4-6) | 2026-06-16 | DONE | tables/extract.py, pipeline/orchestrator.py (_phase_dual_pass_tables + get_fitz_page), tests/test_dual_pass_tables.py | ThreadPoolExecutor wall-clock deadline guard + cascade guard + dualpass_crop_timeout AuditEvent + fitz single-slot cache; 969 tests pass |
| PP-1 (GH-65) | socr-implementer (claude-sonnet-4-6) | 2026-06-16 | DONE | pipeline/orchestrator.py (_flush_page_fragment, _flush_page_sidecar, _stitch_fragments, _phase_assemble refactor), tests/test_pp1_fragment_flush.py | Fragment flush + atomic sidecar + end-of-run stitch; byte-identity verified by test; 994 tests pass |
| PP-3 (GH-67) | socr-implementer (claude-sonnet-4-6) | 2026-06-16 | DONE | pipeline/orchestrator.py (_reread_page_tables extracted, _phase_dual_pass_tables refactored), tests/test_dual_pass_tables.py | Behavior-preserving refactor; reader built once at doc scope; 4 parity tests added; 999 tests pass |
| PP-4 (GH-69) | socr-implementer (claude-sonnet-4-6) | 2026-06-16 | DONE | figures/extractor.py (cap_page field), pipeline/orchestrator.py (_describe_and_embed_figures pure text transformer + _rewrite_all_fragments unconditional in _phase_assemble), tests/test_pp4_inline_figures.py (new, 18 tests) | Inline embedding; _rewrite_all_fragments unconditional in _phase_assemble (covers figure-free phantom docs — all 3 REVISE rounds resolved); vision engine once/close once; cap AuditEvent at crossing page; 255 targeted tests pass |
| PP-2 (GH-71) | socr-implementer (claude-sonnet-4-6) | 2026-06-16 | DONE | pipeline/orchestrator.py (_phase_agentic full rewrite, _TimeoutJudge, _flush_page_sidecar terminal kwarg, phase-4c gate), tests/test_pp2_agentic_fuse.py (new, 13 tests) | Fused all-pages loop; provisional flush (terminal=False); cascade-halt on wedged backend; byte-identity via fork A; 1031 tests pass |
| PP-7 (GH-73) | socr-implementer (claude-sonnet-4-6) | 2026-06-16 | DONE | figures/extractor.py (CHART_MIN_CLUSTER_AREA + has_chart_marks), pipeline/orchestrator.py (_is_chart_asset_page, _render_chart_page_png, chart-lane hook in _phase_agentic), tests/test_chart_lane.py (new, 20 tests) | Cluster-first vector detector + B1 representation (native prose + PNG ref + audit flag); force-PNG; fail-closed; monochrome false-negative documented; 327 targeted tests pass |

## Per-issue workflow (orchestrator-driven, consilium-gated)

We process issues one at a time (or in disjoint-write-set waves) through a fixed pipeline. The
**orchestrator** (the main Claude session) drives it; agents do bounded work. `/consilium` is a
main-thread tool — only the orchestrator runs it, never a subagent.

```
                        ┌──────────────────────────────────────────┐
   READY ticket ────────┤ 2. socr-implementer  →  3. socr-reviewer  ├──→ ACCEPT → next issue
                        └──────────────────────────────────────────┘
                              ▲ CONSILIUM-GATE          │ CONSILIUM-GATE
                              │                         ▼
   NEEDS-DESIGN ──→ 1. socr-designer ──→ [orchestrator runs /consilium] ──→ decision into ticket
```

1. **Design gate (NEEDS-DESIGN tickets: GH-49A, GH-47C, GH-36).** Dispatch `socr-designer`
   (read-only). It writes a design note to `docs/log/` and returns a sharp, self-contained
   question. The **orchestrator** then runs `/consilium <that question>`, synthesizes, and records
   the chosen design in the ticket before any code is written. For READY tickets with a non-trivial
   root cause (e.g. GH-34), the orchestrator MAY run a quick `/consilium` on root cause first per
   the global bug-fix protocol; skip it for mechanical tickets.
2. **Implement.** Dispatch `socr-implementer` with exactly one ticket id. If it returns
   `CONSILIUM-GATE` (hit an architectural fork mid-work), the orchestrator runs `/consilium` on the
   stated question, feeds the decision back, and re-dispatches.
3. **Review.** Dispatch `socr-reviewer` on the ticket's commit hash. `ACCEPT` → done.
   `REVISE` → orchestrator re-dispatches `socr-implementer` with the numbered fix list.
   `CONSILIUM-GATE` (contested judgment call) → orchestrator runs `/consilium` to break the tie,
   then directs the outcome.

`/consilium` routing: design/architecture/trade-off forks → default panel (Codex + Gemini);
genuinely split after iteration → surface both sides, `--tiebreak kimi` only on request. Skip the
panel for mechanical tickets — two opinions add nothing to a one-line config fix.

## Dispatch Contract

Prompt each agent with exactly one ticket section from `docs/plans/TICKETS.md`, plus:

- You are not alone in the codebase; do not revert unrelated edits.
- Own only the files listed in the ticket's `Write ownership` unless you first report why more scope
  is required.
- Use `uv run` for Python (or the direct `~/venvs/socr/bin` binaries); never `python script.py`.
- Report changed files, tests run, failures, and residual risks.
- Commit on the ticket's own `feat/NN-…` / `fix/NN-…` branch (never `main`); stage by name; do not
  push; one commit per ticket.
- If you hit an architectural fork you cannot resolve from the ticket, stop and return
  `CONSILIUM-GATE` with a one-sentence question — do not guess past it.

## Next Action (superseded — historical)

Recorded 2026-06-16, kept for provenance. All of it has since resolved: PR #52 merged, GH-47B
and GH-49A are DONE, GH-36 shipped as GH-36a/GH-36b. The live next action is at the top of this
file under "Next action".

- ~~GH-47B as the last ready implementation ticket.~~
- ~~GH-49A, GH-36, and GH-47C as a design-gated batch.~~
- GH-39A/B only after human-verified benchmark ground truth exists. *(still true)*
