# STATUS — GH-152 side-by-side tables merged

Last updated: 2026-08-11

## Stage
Scaffolded, BLOCKED twice over. `reconstruct.py` carries the GH-146 work as open PR #149,
and is then held by GH-144 A2 through wave 2. This plan is the tail of the program's
critical path.

## Base state (clean before tickets)
- Repo `tools/socr`, branch from `main`
- **Precondition:** GH-146's `reconstruct.py` work committed and merged

## Ticket board
Wave numbers are **global** and owned by
[`../extraction-defects/STATUS.md`](../extraction-defects/STATUS.md), because three of the
five defect plans write the same files. Do not schedule from this table alone.

| Ticket | Stream | Status | depends-on | Global wave |
|--------|--------|--------|------------|-------------|
| A1 | column segmentation | BLOCKED | PR #149 (wave 0) + **GH-144 A2** | 3 |
| A2 | column segmentation | TODO | A1 | 4 |
| B1 | evidence | TODO | A2 | 5 |

A1 slipped from local wave 1 to global wave 3: `tables/reconstruct.py` is held by GH-144 A2
in wave 2, and this plan had no way to see that. All three tickets sit on the program's
critical path — they cannot be compressed, only started early.

## Next action
Nothing to dispatch. Wave 0 (merge PR #149) and GH-144 A2 must land first. If GH-144 A1's
diagnosis finds that full-page-width clustering causes the boundary defect, A1 becomes the
root-cause fix and moves ahead of GH-144 A2 — the coordinator records that condition.
