# STATUS — agentic local-first routing (#46 phase 2)

Last updated: 2026-06-13 (planning session, pre-execution)

## Stage
Scaffolding written. **Not yet executed.** Awaiting fresh-session parallel dispatch.

## Base state (clean before tickets)
- Branch `feat/46-model-lineup-refresh`, last commit `6d6ee79` (deterministic lineup refresh).
- Instruct-encoding committed on top (qwen.py local tier = `qwen3-vl:30b-a3b-instruct`,
  MODELS.md trap documented). Full suite green via `~/venvs/socr/bin/pytest`.
- `scratch/` gitignored.

## Ticket board
| Ticket | Stream | Status | depends-on | Parallel group |
|--------|--------|--------|------------|----------------|
| A1 | provider identity | TODO | — | wave 1 (alone) |
| B1 | drop DeepSeek/demote Mistral | TODO | A1 | wave 2 |
| B2 | agentic default + flags | TODO | A1 | wave 2 |
| B3 | enrich manifest | TODO | A1 | wave 2 |
| C1 | thinking/stall guard | TODO | — | wave 1 (parallel) |
| C2 | local-first figure desc | TODO | — | wave 1 (parallel) |
| D1 | dense-table summary-row prompt | TODO | — | wave 1 (parallel) |

## Dispatch waves
- **Wave 1 (parallel now):** A1, C1, C2, D1 — no cross-deps, mostly disjoint files
  (A1=providers/agentic/orchestrator; C1=agentic.py — coordinate with A1 if both touch
  agentic.py: run A1 first or use worktrees; C2=gemini_api.py; D1=prompts only).
- **Wave 2 (after A1):** B1, B2, B3.

> Note: A1 and C1 both touch `pipeline/agentic.py`. Either serialize (A1 → C1) or give each
> agent an isolated worktree and merge. Reviewer must check for collision.

## Next action
Fresh session: read this file + TICKETS.md, dispatch wave-1 implementer agents, review, commit
per ticket, then wave 2.

## Open questions deferred to execution
- Preflight skill (frozen-profile advisor) — Codex says optional; decide AFTER the refactor.
- Exact soft-timeout values for C1 — derive from observed per-provider latencies, don't hardcode.
