# STATUS — GH-151 structure lost at full recall

Last updated: 2026-08-15

## B1 — DONE (2026-08-15)

Shipped as the #200 escalation gate on `feat/200-structural-escalation-gate` per the
2026-08-15 panel ruling. See TICKETS.md's B1 entry for the disjunction detail and
`docs/log/2026-08-15_200-open-decision-1-resolved.md` for the pre-merge SOFT/HARD
measurement the ratified spec required before merge. Full suite 1648 passed / 1 xfailed;
lint clean. Not merged to `main` by this session — awaiting CI/review.

## Stage
Scaffolded, not dispatched. The motivating page is verified: p26 now measures 100%
recall / 0 missing tokens after the GH-146 work, and its table is still structurally
wrong. That is the cleanest possible demonstration that recall is not sufficient.

## Base state (clean before tickets)
- Repo `tools/socr`, branch from `main`
- PR #149 (GH-146) must merge before B1's recall evidence holds — p26's 100% figure
  depends on it. A1 and A2 do not.

## Ticket board
Wave numbers are **global** and owned by
[`../extraction-defects/STATUS.md`](../extraction-defects/STATUS.md), because three of the
five defect plans write the same files. Do not schedule from this table alone.

| Ticket | Stream | Status | depends-on | Global wave |
|--------|--------|--------|------------|-------------|
| A1 | structural signals | TODO | — | 1 |
| A2 | structural signals | TODO | — | 1 |
| B1 | consequence | TODO | A1, A2, **GH-147 A2** | 3 |
| B2 | consequence | TODO | B1 | 4 |

B1 slipped from local wave 2 to global wave 3: it writes `core/born_digital.py` and
`pipeline/orchestrator.py`, which GH-147 A2 holds in wave 2. It is also the ticket that
should mirror GH-147 A2's refusal path rather than invent one concurrently.

## Next action
Dispatch A1 and A2 in parallel — both are in global wave 1. A1's p26 fixture is a
checked-in grid string, so it does not wait on the wave-0 merges; B1's recall evidence does.
