# STATUS - GitHub issue action plan

Last updated: 2026-06-15
Branch: `feat/001-issue-plans`
Base reviewed: `7541175`
GitHub source: open issues #34, #35, #36, #37, #39, #46, #47, #49, #50, #51

## Stage

Planning complete in this branch. No implementation subagents have been spawned yet.

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
| #46 | Main #46 implementation mostly merged; remaining repo-level item is sparse-row lane drift. | GH-46-D2, GH-46-E1 |
| #39 | Stage 1 landed; Stage 2 human GT and Stage 3 calibration remain. | GH-39A, GH-39B |
| #37 | Still actionable. Needs user-facing native-only/enhancement policy control. | GH-37 |
| #36 | Partially addressed by corrupt-math recovery; clean equation-to-LaTeX route remains. | GH-36 |
| #35 | Still needs scanned-classifier validation for sparse/full-page-figure pages. | GH-35 |
| #34 | Partially fixed by #38; remaining scope is recovered-to-empty guard. | GH-34 |

## Ready Queue

Recommended first wave, safe to run in parallel because write sets are mostly disjoint:

| Ticket | Agent | Ownership | Notes |
|--------|-------|-----------|-------|
| GH-51 | `socr-implementer` | qwen config/CLI/engine/tests | P0, fixes silent routing confusion. |
| GH-50 | `socr-implementer` | figure flags/config/orchestrator/tests | P0, separates safe artifacts from captions. |
| GH-34 | `socr-implementer` | repair promotion/audit/tests | P0, prevents misleading recovery records. |

Second wave:

| Ticket | Agent | Ownership | Notes |
|--------|-------|-----------|-------|
| GH-46-D2 | `socr-implementer` | table prompt/tests | May stay prompt-only; escalate to verifier only if needed. |
| GH-47A | `socr-implementer` | figure extractor/tests | Cap visibility and logo false-positive handling. |
| GH-37 | `socr-implementer` | CLI/config/born-digital/tests | Coordinate with GH-35 to avoid competing policy changes. |
| GH-35 | `socr-implementer` | born-digital classifier/tests | Characterize before changing broad classification logic. |

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
| GH-51 | socr-implementer (claude-sonnet-4-6) | 2026-06-15 | DONE | config.py, qwen.py, cli.py, test_qwen_engine.py | Resolver added; 791 tests pass |

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

Start with the first wave (GH-51, GH-50, GH-34) — disjoint write sets, safe in parallel. GH-51 and
GH-50 are mechanical (skip the design gate); GH-34 is a correctness/promotion fix where a root-cause
`/consilium` pass before implementing is worthwhile. Record each dispatched agent in **Active
Agents** above.
