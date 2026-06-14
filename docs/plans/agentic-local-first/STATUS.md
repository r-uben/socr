# STATUS — agentic local-first routing (#46 phase 2)

Last updated: 2026-06-14

## Stage
Wave 1+2 DONE (A1, B1, B2, B3, C1, C2, D1). C1b DONE. Next: M1 (final verify + merge).

## Base state (clean before tickets)
- **Repo now at `~/repos/socr`** (moved off iCloud 2026-06-14; git/uv/pytest native here).
- Branch `feat/46-model-lineup-refresh`. Full suite green (716 after A1).
- `scratch/` gitignored (D1's `scratch/bench/ce/202606_p4.png` carried over for validation).
- A1 committed: `ProviderProfile` now carries `id/backend/model/auto_eligible`; named profile
  constants defined; `provider_ladder()` gained `include_ineligible` param and direct-profile path.

## Ticket board
| Ticket | Stream | Status | depends-on | Parallel group |
|--------|--------|--------|------------|----------------|
| A1 | provider identity | DONE | — | wave 1 (alone) |
| B1 | drop DeepSeek/demote Mistral | DONE | A1 | wave 2 |
| B2 | agentic default + flags | DONE | A1 | wave 2 |
| B3 | enrich manifest | DONE | A1 | wave 2 |
| C1 | thinking/stall guard | DONE | — | wave 1 (parallel) |
| C2 | local-first figure desc | DONE | — | wave 1 (parallel) |
| D1 | dense-table summary-row prompt | DONE | — | wave 1 (parallel) |
| C1b | calibrate stall-guard timeouts | DONE | C1 | follow-up |
| M1 | final verify + merge feat/46 | TODO | C1b | follow-up |

## Dispatch waves
- **Wave 1 (parallel now):** A1, C1, C2, D1 — no cross-deps, mostly disjoint files
  (A1=providers/agentic/orchestrator; C1=agentic.py — coordinate with A1 if both touch
  agentic.py: run A1 first or use worktrees; C2=gemini_api.py; D1=prompts only).
- **Wave 2 (after A1):** B1, B2, B3.

> Note: A1 and C1 both touch `pipeline/agentic.py`. Either serialize (A1 → C1) or give each
> agent an isolated worktree and merge. Reviewer must check for collision.

## Next action
Run M1: full suite + ruff clean on the whole branch, then merge feat/46-model-lineup-refresh.

## Open questions deferred to execution
- Preflight skill (frozen-profile advisor) — Codex says optional; decide AFTER the refactor.
- Exact soft-timeout values: calibrated in C1b (QWEN=300s, GEMINI=240s from bench data).
