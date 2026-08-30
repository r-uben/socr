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
| B1 | gate | DONE | A2, A3, A4, B0, B2, C1, C3, G1 | 4 |
| D1a | resume | DONE | B1 | 5 |
| D1b | resume | DONE | D1a | 6 |
| E1 | binding | DONE | B1 | 7 |
| H1 | e2e | DONE | D1b, E1 | 8 |

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
| B1 | impl-B1 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_TICKET-B1.md` |
| D1a | impl-B1 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_TICKET-D1a.md` |
| D1b | impl-D1a (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_TICKET-D1b.md` |
| E1 | impl-E1 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_TICKET-E1.md` |
| H1 | impl-H1 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_TICKET-H1.md` |

## Next action
TICKET-H1 DONE — **the board is CLOSED**: all 16 tickets across 8 waves are
DONE. `tests/fixtures/table_ladder/` (+ `generate_fixture.py`) is a committed,
deterministic two-page fixture (a clean control table + a GH-273-shape
row-label-rotation table) driving `tests/test_ladder_e2e.py`'s full
`UnifiedPipeline.process()` sweep of all three terminals the ladder can reach
(REJECTED, plain infra UNVERIFIED, and E1's CLAMPED-UNVERIFIED), each
asserted flag-off-vs-on at all four no-silent-loss surfaces (page
disposition, document status, `metadata.json` note, CLI summary). Resume
interaction was deliberately left to D1b's own tests (`test_ladder_resume.py`)
rather than duplicated — see `docs/log/2026-08-30_TICKET-H1.md`.
Outstanding, not owned by any single ticket (repeated from every prior
ticket's log): a live smoke test against the real `gemini` CLI and a real
ollama host, which belongs to whoever merges `feat/353-judge-ladder`.

