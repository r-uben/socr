# STATUS — table-repair

Last updated: 2026-06-17 (TR-4a DONE)

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
| TR-4a | socr-implementer (claude-sonnet-4-6) | DONE |

## Ticket state

| Ticket | What | Status | Depends on |
|--------|------|--------|------------|
| TR-0 | License-clean CE-like fixture + cell-parity harness | DONE | — |
| TR-1 | Deterministic rowizer on lane-stacked `find_tables()` regions (Option B) | DONE | TR-0 |
| TR-2 | Per-region verifier scoping + reading-order reassembly | DONE | TR-1 |
| TR-3 | D3 fail-closed floor + selection-policy fix | DONE | TR-2 |
| TR-4a | Real-CE-geometry dense fixture (failing gate for TR-4) | DONE | — |
| TR-4 | A2 value-guarded VLM re-ask | NEEDS-DESIGN | TR-4a |
| TR-5 | S3 VLM confirm/split segmentation | DEFERRED (v2) | TR-2 spike |

## TR-4a findings (2026-06-17)

Real root-cause established from `202401.pdf` p.4 geometry:
- `find_tables(strategy="lines")` detects one table (triggered by the outer border rect
  + vertical column-separator strokes in the real CE PDF).
- The detected table is lane-stacked (`_is_lane_stacked=True`) because the name/value
  y-offset (~1 pt) causes embedded `\n`-stacked tokens in cells.
- Routes to `rowize_from_word_list` which groups words by `round(y0)`: value row at
  y=143 and name row at y=144 are DISTINCT y-groups, producing interleaved output
  (value row: empty label + full data; name row: label + no data).
- The synthetic dense fixture (`ce_like_p4_dense.pdf`) reproduces this exactly:
  outer border + 4 column separators → find_tables returns lane-stacked table → same
  interleaved failure on `extract_structured`.

## Outstanding / open questions

- TR-4a DONE: dense fixture committed, xfail parity test in place.
- TR-4 NEEDS-DESIGN: value-guarded VLM-for-structure (see TICKETS.md design questions).
- TR-5 DEFERRED (v2).

## Next action

TR-4 design pass (value-guard algorithm, escalation trigger, scanned vs born-digital,
wiring into the agentic ladder) → then TR-4 implementation against the xfail gate.
