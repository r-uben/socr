# STATUS — defect-census fixes

Last updated: 2026-09-06

## Stage
Plan drafted from the two-institution census; advisory panel (codex, gemini, grok) pending;
nothing dispatched.

## Base state (clean before tickets)
- `main@eb14c82`; census log committed on `docs/fed-ecb-census` (`7015f46`, `d00fb11`).
- Pinned measurement checkout `~/repos/.worktrees/socr-census` (detached at `eb14c82`).
- Fixtures under `~/Data/socr/census-ecb-2026-09-06/`, `~/Data/socr/census-591-recheck/`.

## Ticket board
| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1 | selection | TODO | — | 1 |
| C1 | native geometry | TODO | — | 1 |
| D1 | throughput | TODO | — | 1 |
| A2 | selection | TODO | A1 | 2 |
| B1 | marker scope | TODO | A1 | 2 |
| D2 | throughput (measure) | TODO | — | 3 |
| E1 | noise | TODO | B1 | 3 |
| E2 | noise | TODO | E1 | 4 |
| F1 | Fed one-table | TODO | A2 | 4 |
| F2 | Fed one-table | TODO | F1 | 4 |

## Dispatch waves
- Wave 1: A1 (orchestrator), C1 (born_digital), D1 (providers) — disjoint files.
- Wave 2: A2, B1 — B1 touches orchestrator; A2 does not. Disjoint.
- Wave 3: D2, E1 — both touch orchestrator → serialize (E1 first).
- Wave 4: E2, F1, F2 — F1/F2 serial on normalizer.py; E2 disjoint.

## Next action
Owner: approve the plan (email sent 2026-09-06). Then `/plan next` wave 1 on
`feat/<nn>-row-corroboration` etc., one branch per ticket, from main.
