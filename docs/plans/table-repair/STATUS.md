# STATUS — table-repair

Last updated: 2026-06-17

## Stage

**Wave 1 (v1 deterministic) MERGED — but real CE proved it INSUFFICIENT.** TR-0…TR-3 are in
`main`; the SYNTHETIC fixture passes. Validating on the REAL CE `202401.pdf` p.4 (2026-06-17):
v1 gets VALUES + COLUMNS right and flags itself honestly, but **breaks the ROW structure** (top
block merged, names offset from values by one row — gap-segmentation can't handle CE's dense/
variable spacing). The clean fixture gave false confidence. See
`docs/log/2026-06-17_real-CE-v1-finding.md`.

**→ Moving to v2: VLM for STRUCTURE, geometry value-guard for VALUES** (the panel's A2/S3, now
unblocked by measured evidence). TR-4a (real-geometry fixture = the failing gate) first, then TR-4
(value-guarded VLM-for-structure) after a design pass.

## Branch

`feat/56-tr3-fail-closed` — TR-3 committed here (branched from `feat/56-tr2-segmentation`).

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
| TR-3 | socr-implementer (claude-sonnet-4-6) | DONE |

## Ticket state

| Ticket | What | Status | Depends on |
|--------|------|--------|------------|
| TR-0 | License-clean CE-like fixture + cell-parity harness | DONE | — |
| TR-1 | Deterministic rowizer on lane-stacked `find_tables()` regions (Option B) | DONE | TR-0 |
| TR-2 | Per-region verifier scoping + reading-order reassembly | DONE | TR-1 |
| TR-3 | D3 fail-closed floor + selection-policy fix | DONE | TR-2 |
| TR-4 | A2 value-guarded VLM re-ask | DEFERRED (v2) | TR-3 + evidence |
| TR-5 | S3 VLM confirm/split segmentation | DEFERRED (v2) | TR-2 spike |

## Outstanding / open questions

- v1 is complete. TR-4 and TR-5 are deferred to v2 (require measured evidence that
  deterministic per-region + rowizer is insufficient on real CE grids).
- The agentic-path whole-page ladder verifier (`agentic.py:~405`) remains whole-page;
  converting it to per-region is explicitly a v2 item (TR-4/TR-5 scope).

## Next action

Real-CE measurement done → v1 insufficient on row structure. Dispatch **TR-4a** (real-geometry
failing fixture) and the **TR-4 design pass** (value-guarded VLM-for-structure) concurrently.
