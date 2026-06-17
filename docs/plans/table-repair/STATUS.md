# STATUS — table-repair

Last updated: 2026-06-17

## Stage

**Wave 1 COMPLETE.** TR-0, TR-1, and TR-2 DONE. The TR-0 parity test
(`TestEndToEndParity::test_agentic_parity_on_ce_like_fixture`) passes — xfail mark removed.
Both tables, chart image-ref, and prose all verified with correct reading order.

## Branch

`feat/56-tr2-segmentation` — TR-2 committed here.

## Decisions (settled)

- v1 is deterministic-only: per-region segmentation → rowizer (B) → D3 fail-closed floor.
- Q1=D3, Q2=A2 (deferred v2, lane-aware token EQUALITY not superset), Q3=S3 (geometry-led in v1).
- Fixture must be **synthesized** (CE PDF is licensed — never commit it).
- TR-1 finding: `find_tables()` returns ZERO tables for the fixture (no ruling lines). The
  text-strategy over-merges the whole page into one region that fails `_looks_tabular`. The
  word-geometry rowizer correctly extracts the main table but the historical table was blocked
  by the chart+hist segment merge — fixed by TR-2 chart-clip.
- TR-2 finding: chart tick labels at x=54 (left of vertical axis at x=73) were diluting the
  historical table's `data_row_frac`. Fix: extend chart bbox x0 to 0 (page left edge) to
  capture all axis tick labels. Multi-line headers (indicator row + year row) collapsed into
  a single header row by `_collapse_header_prefix`. Chart PNG rendered to `figures/` so
  `strip_phantom_images` preserves the ref.

## Active Agents

| Ticket | Agent | Status |
|--------|-------|--------|
| TR-0 | socr-implementer (claude-sonnet-4-6) | DONE |
| TR-1 | socr-implementer (claude-sonnet-4-6) | DONE |
| TR-2 | socr-implementer (claude-sonnet-4-6) | DONE |

## Ticket state

| Ticket | What | Status | Depends on |
|--------|------|--------|------------|
| TR-0 | License-clean CE-like fixture + cell-parity harness | DONE | — |
| TR-1 | Deterministic rowizer on lane-stacked `find_tables()` regions (Option B) | DONE | TR-0 |
| TR-2 | Per-region verifier scoping + reading-order reassembly | DONE | TR-1 |
| TR-3 | D3 fail-closed floor + selection-policy fix | READY | TR-2 |
| TR-4 | A2 value-guarded VLM re-ask | DEFERRED (v2) | TR-3 + evidence |
| TR-5 | S3 VLM confirm/split segmentation | DEFERRED (v2) | TR-2 spike |

## Outstanding / open questions

- TR-3 (D3 fail-closed floor): now unblocked. Dispatch when ready.
- The e2e parity test is now PASSING — no remaining blockers in v1 wave 1.

## Next action

Dispatch **TR-3** — D3 fail-closed floor + selection-policy fix. TR-2 is fully committed and
the parity gate is green.
