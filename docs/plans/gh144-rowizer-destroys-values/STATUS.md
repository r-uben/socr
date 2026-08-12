# STATUS — GH-144 rowizer destroys numeric values

Last updated: 2026-08-11

## Stage
Scaffolded, not dispatched. Worst of the extraction defects by severity — it loses
NUMBERS from regression tables in a citation corpus. A minimal synthetic repro
exists and is pinned as a strict xfail.

## Base state (clean before tickets)
- Repo `tools/socr`, branch from `main`
- `tests/test_region_overlap_gh145.py::test_no_table_value_is_lost` is xfail(strict)
- **Precondition for A2:** GH-146's `reconstruct.py` work committed and merged

## Ticket board
Wave numbers are **global** and owned by
[`../extraction-defects/STATUS.md`](../extraction-defects/STATUS.md), because three of the
five defect plans write the same files. Do not schedule from this table alone.

| Ticket | Stream | Status | depends-on | Global wave |
|--------|--------|--------|------------|-------------|
| A1 | diagnose | TODO | — | 1 |
| A2 | fix | BLOCKED | A1 + PR #149 (wave 0) | 2 |
| A3 | evidence | TODO | A2 | 4 |

**This plan owns the critical path.** `A1 → A2 → GH-152 A1 → GH-152 A2 → GH-152 B1` is the
longest chain in the program, all on `tables/reconstruct.py`. A2 goes before GH-152 A1 —
rationale and the condition that would invert it are in the coordinator.

**A1 must answer one extra question** beyond its own `Done when`: does full-page-width row
clustering *cause* the false gutter? If yes, GH-152's x-banding is the root cause and the
`reconstruct.py` order inverts. Say so explicitly either way.

## Next action
Dispatch A1 now — it is read-only, writes only to `logs/`, and does not touch
`reconstruct.py`, so it is unaffected by the open PR #149.
