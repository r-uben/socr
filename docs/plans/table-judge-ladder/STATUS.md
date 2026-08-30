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
| B1 | impl-B1 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_TICKET-B1.md` |
| D1a | impl-B1 (this session, table-judge-ladder team) | DONE — `docs/log/2026-08-30_TICKET-D1a.md` |

## Next action
Wave 5 (D1a, sidecar persist/restore) DONE. Dispatch wave 6 (D1b, resume skip policy,
`orchestrator.py`) next — its dep (D1a) is now satisfied. D1a made
`ps.table_ladder_disposition` and the ladder's audit events survive resume; D1b still
needs to add the positive early-return in `_load_terminal_page` (~`:4315`) so a
REJECTED page (status=WARNING via C3's guard) is skipped-and-kept rather than falling
through the existing `status == SUCCESS` gate, while UNVERIFIED never skips. Note for
the reviewer/D1a dispatcher (unchanged from B1's log): a live smoke against the real
`gemini` CLI and a real ollama host is still outstanding and belongs to whoever merges
this branch, not to any single ticket.
