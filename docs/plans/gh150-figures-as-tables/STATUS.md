# STATUS — GH-150 figures extracted as tables

Last updated: 2026-08-12

## Stage
Scaffolded, not dispatched. Diagnosis complete and evidenced: Heston p10 (17% recall)
and Drechsler p55 (41%) are the two lowest-recall pages in the corpus and both are
figures. Measured: p10 has 932 drawing ops, 0 images, `has_chart_marks=False`,
`has_tables=True`.

## Base state (clean before tickets)
- Repo `tools/socr`, branch from `main`
- GH-150 filed with the measurement table and the docstring quote naming the precedence rule

## Ticket board
Wave numbers are **global** and owned by
[`../extraction-defects/STATUS.md`](../extraction-defects/STATUS.md), because three of the
five defect plans write the same files. Do not schedule from this table alone.

| Ticket | Stream | Status | depends-on | Global wave |
|--------|--------|--------|------------|-------------|
| A1 | detection recall | DONE | — | 1 |
| A2 | detection recall | TODO | A1 | 2 |
| B1 | precedence | TODO | — | 1 |
| B2 | precedence | TODO | A1, B1 | 3 |

A2 and B2 both write `tests/test_chart_detection_gh150.py` and must not run concurrently.

## Next action
A1 landed on fix/150-a1-framed-chart-marks; A2 (wave 2, depends-on A1) next —
fixture constraint (enclosing draw_rect) recorded in TICKETS.md A2. B1 remains
independent wave-1 work on orchestrator.py only.
