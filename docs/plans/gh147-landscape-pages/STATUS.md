# STATUS — GH-147 landscape pages

Last updated: 2026-08-11

## Stage
Scaffolded, not dispatched. Corpus measurement complete and decisive: 441 landscape
pages (1.92%), but 20 of the 40 pages below 80% recall. Design decision recorded:
refuse and route to OCR rather than build coordinate transforms.

## Base state (clean before tickets)
- Repo `tools/socr`, branch from `main`
- Note: `dominant_text_direction()` already exists on the GH-145 branch (PR #148); if
  that merges first, A1 reduces to exposing it on `PageAssessment`.

## Ticket board
Wave numbers are **global** and owned by
[`../extraction-defects/STATUS.md`](../extraction-defects/STATUS.md), because three of the
five defect plans write the same files. Do not schedule from this table alone.

| Ticket | Stream | Status | depends-on | Global wave |
|--------|--------|--------|------------|-------------|
| A1 | refuse | TODO | PR #148 (wave 0) | 1 |
| A2 | refuse | TODO | A1 | 2 |
| B1 | evidence | TODO | A2 | 3 |

A2 holds `core/born_digital.py` and `pipeline/orchestrator.py` for the whole of wave 2;
GH-151 B1 waits on it.

## Next action
Wave 0: merge PR #148, which adds `dominant_text_direction()`. Then dispatch A1 — it
reduces to exposing that function on `PageAssessment`. Dispatching before the merge means
writing it twice.
