# STATUS - GitHub issue action plan

Last updated: 2026-06-15
Branch: `feat/001-issue-plans`
Base reviewed: `7541175`
GitHub source: open issues #34, #35, #36, #37, #39, #46, #47, #49, #50, #51

## Stage

Implementation wave complete on PR #52. Eight tickets have shipped on
`feat/001-issue-plans` and were accepted after implementer/reviewer passes:
GH-51, GH-50, GH-34, GH-46-D2, GH-47A, GH-37, GH-35, and GH-35-FU.

Latest pushed head: `2409192` (`fix(35): GH-35-FU gate clean-short-text by raster coverage`).
CI is green on GitHub: `test (3.11)` and `typecheck` passed. Local full-suite result reported by
the ticket workflow: 837 tests passing, ruff clean.

The durable planning format is:

- `docs/plans/TICKETS.md` - canonical issue-derived backlog.
- `docs/plans/STATUS.md` - live execution, assignment, dependency, and agent state.
- `docs/plans/agentic-local-first/` - FROZEN #46-phase-2 subplan (historical reference + logs).
  Its open items D2/E1 are now owned here as GH-46-D2 / GH-46-E1.

## Issue Review

| Issue | Current read | Ticket mapping |
|-------|--------------|----------------|
| #51 | Open, actionable P0. Non-agentic qwen model resolution can silently mean cloud. | GH-51 |
| #50 | Open, actionable P0. Split deterministic PNG extraction from VLM captions. | GH-50 |
| #49 | ADR/docs mostly done on main; implementation still needed for native table verifier. | GH-49A |
| #47 | Investigation done; follow-ups are concrete figure extraction/caption/verification tickets. | GH-47A, GH-47B, GH-47C |
| #46 | Main #46 implementation mostly merged; sparse-row lane drift has a prompt-only fix, with manual CBO-row VLM validation still pending. | GH-46-D2, GH-46-E1 |
| #39 | Stage 1 landed; Stage 2 human GT and Stage 3 calibration remain. | GH-39A, GH-39B |
| #37 | Done in PR #52. User-facing `--native-only` / enhancement policy control landed. | GH-37 |
| #36 | Partially addressed by corrupt-math recovery; clean equation-to-LaTeX route remains. | GH-36 |
| #35 | Done in PR #52 plus GH-35-FU. Sparse/figure pages rescued, then image-dominant short-text pages gated by raster coverage. | GH-35, GH-35-FU |
| #34 | Done in PR #52. Empty repair output is no longer promoted or logged as recovered. | GH-34 |

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
- Commit on `feat/001-issue-plans` (never `main`); stage by name; do not push; one commit per ticket.
- If you hit an architectural fork you cannot resolve from the ticket, stop and return
  `CONSILIUM-GATE` with a one-sentence question — do not guess past it.

## Next Action

Merge PR #52 after this bookkeeping update lands. Next PR candidates:

- GH-47B as the last ready implementation ticket.
- GH-49A, GH-36, and GH-47C as a design-gated batch.
- GH-39A/B only after human-verified benchmark ground truth exists.
