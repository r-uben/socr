# STATUS — table judge ladder (GH-353)

Last updated: 2026-08-30

## Stage
Plan created and panel-reviewed (codex gpt-5.6-sol / gemini / grok-4.6 advisory,
2026-08-30). Design ratified in `docs/log/2026-08-30_table-judge-ladder.md`; CLI₁ seat
decided by the GH-356 bake-off (`docs/log/2026-08-30_gh356-bakeoff.md`): rung 1 =
glm-5.3-flash:cloud via ollama `/api/chat`, rung 2 = gemini CLI. No implementation
dispatched yet.

## Base state (clean before tickets)
- Repo: socr, branch per ticket off `origin/main`; flag `table_judge_ladder` defaults
  OFF so golden/byte-identity tests stay untouched until the flip.
- CI has no ollama/gemini; every ticket's tests are hermetic (see TICKETS.md standing
  constraints).
- Panel findings folded in: gate sits post-repetition-guard (~`orchestrator.py:3099`);
  manifest winner-selection preservation is its own ticket (C3); witness prep is B0;
  native lane folded into B1's acceptance tests; fenced JSON is not ¬S1.

## Ticket board
| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A0 | prompt | DONE | — | 1 |
| A1 | judge core | DONE | — | 1 |
| C1 | status | DONE | — | 1 |
| G1 | config | DONE | — | 1 |
| A2 | judge core | DONE | A0, A1, G1 | 2 |
| A3 | judge core | DONE | A0, A1, G1 | 2 |
| A4 | judge core | DONE | A1 | 2 |
| B0 | witnesses | DONE | A1 | 2 |
| B2 | trust | DONE | A1 | 2 |
| C3 | status | DONE | C1 | 2 |
| C2 | status | DONE | C1, C3, G1 | 3 |
| B1 | gate | TODO | A2, A3, A4, B0, B2, C1, C3, G1 | 4 |
| D1a | resume | TODO | B1 | 5 |
| D1b | resume | TODO | D1a | 6 |
| E1 | binding | TODO | B1 | 7 |
| H1 | e2e | TODO | D1b, E1 | 8 |

## Dispatch waves
- Wave 1: A0, A1, C1, G1 (disjoint files, no deps)
- Wave 2: A2, A3, A4, B0, B2, C3 (disjoint files)
- Wave 3: C2 (orchestrator.py + cli.py)
- Wave 4: B1 (orchestrator.py — the gate)
- Waves 5–8: D1a → D1b → E1 → H1 (serialized on orchestrator.py)

## Active Agents

| Ticket | Agent | Status |
|--------|-------|--------|
| C3 | impl-C1 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_ticket-c3.md` |
| C2 | impl-C2 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_ticket-c2.md` |

## Next action
Wave 3 (C2) DONE. Dispatch wave 4 (B1, the gate, `orchestrator.py`) next — its own
deps (A2/A3/A4/B0/B2/C1/C3/G1) are all satisfied.
