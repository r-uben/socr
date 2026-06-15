# STATUS - agentic local-first routing (#46 phase 2)

Last updated: 2026-06-15

## Stage
#46 implementation waves are merged to `main` through `7541175`. The old M1 merge/verify item is
DONE. Remaining actionable #46 scope is now tracked from the top-level GitHub issue plan:

- D2 sparse comparison-row lane drift - still TODO.
- E1 preflight/profile skill - optional/deferred.
- Z1 Consensus Forecasts batch - downstream, not a socr repo implementation ticket.

## Base state (clean before tickets)
- **Repo now at `~/repos/socr`** (moved off iCloud 2026-06-14; git/uv/pytest native here).
- Branch `feat/46-model-lineup-refresh` was merged before this status update.
- `scratch/` gitignored (D1's `scratch/bench/ce/202606_p4.png` carried over for validation).
- A1 landed: `ProviderProfile` now carries `id/backend/model/auto_eligible`; named profile
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
| D2 | sparse comparison-row lane drift | TODO | — | follow-up |
| M1 | final verify + merge feat/46 | DONE | C1b | follow-up |
| E1 | preflight profile skill | DEFERRED | M1 | optional |

## Dispatch waves
- Historical waves 1 and 2 are complete.
- Dispatch D2 as a prompt/table validation ticket from `docs/plans/TICKETS.md` GH-46-D2.

## Next action
Use the top-level `docs/plans/STATUS.md` queue for any new subagent dispatch.

## Open questions deferred to execution
- Preflight skill (frozen-profile advisor) — Codex says optional; decide AFTER the refactor.
- Exact soft-timeout values: calibrated in C1b (QWEN=300s, GEMINI=240s from bench data).
