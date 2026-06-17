# STATUS — table-repair

Last updated: 2026-06-17

## Stage

**Wave 1 in progress.** TR-0 and TR-1 DONE. TR-2 (per-region verifier scoping + hist-table
split from chart) is next. The e2e xfail remains: TR-1 extracts the main forecaster grid but
the historical table still appears as flat text (chart+hist segment fails `_looks_tabular`).

## Branch

`feat/56-tr1-rowizer` (TR-1 commit). TR-2 onwards to commit on same initiative branch (or
`feat/56-tr2-…` — decide at dispatch).

## Decisions (settled)

- v1 is deterministic-only: per-region segmentation → rowizer (B) → D3 fail-closed floor.
- Q1=D3, Q2=A2 (deferred v2, lane-aware token EQUALITY not superset), Q3=S3 (geometry-led in v1).
- Fixture must be **synthesized** (CE PDF is licensed — never commit it).
- TR-1 finding: `find_tables()` returns ZERO tables for the fixture (no ruling lines). The
  text-strategy over-merges the whole page into one region that fails `_looks_tabular`. The
  word-geometry rowizer correctly extracts the main table but the historical table is blocked
  by the chart+hist segment merge — that's TR-2's job.

## Active Agents

| Ticket | Agent | Status |
|--------|-------|--------|
| TR-0 | socr-implementer (claude-sonnet-4-6) | DONE |
| TR-1 | socr-implementer (claude-sonnet-4-6) | DONE |

## Ticket state

| Ticket | What | Status | Depends on |
|--------|------|--------|------------|
| TR-0 | License-clean CE-like fixture + cell-parity harness | DONE | — |
| TR-1 | Deterministic rowizer on lane-stacked `find_tables()` regions (Option B) | DONE | TR-0 |
| TR-2 | Per-region verifier scoping + reading-order reassembly | NEEDS-DESIGN (split spike) | TR-1 |
| TR-3 | D3 fail-closed floor + selection-policy fix | READY | TR-2 |
| TR-4 | A2 value-guarded VLM re-ask | DEFERRED (v2) | TR-3 + evidence |
| TR-5 | S3 VLM confirm/split segmentation | DEFERRED (v2) | TR-2 spike |

## Outstanding / open questions

- TR-2 spike: the chart+hist merged segment (y=205-335) in the fixture needs splitting.
  `has_chart_marks` bboxes can clip the chart sub-region; then the historical table can be
  rowized independently. TR-2 implementer should use chart bboxes from `has_chart_marks` /
  `extractor.py:777` as the split signal.
- Whether deterministic v1 (per-region + rowizer) alone yields clean CE grids — measured by
  the TR-0 parity test once TR-1…TR-3 land. After TR-1: main table correct; hist table and
  chart/image-asset handling are TR-2/TR-3.

## Next action

Dispatch **TR-2** (`socr-designer` for spike, then `socr-implementer`) — per-region verifier
scoping + reading-order reassembly. TR-1 finding: use `has_chart_marks` bboxes to split the
chart from the hist table in the word-geometry rowizer path.
