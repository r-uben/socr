# STATUS — table-repair

Last updated: 2026-06-17

## Stage

**Scoping complete.** Design note written + revised (VLM-for-structure / geometry-for-values,
region-aware framing), `/consilium` panel run (codex + gemini), verdict recorded. v1 tickets
scoped (TR-0…TR-5). Ready to start implementation Wave 1.

## Branch

`feat/56-table-repair-plan` (this scoping commit). Implementation tickets commit on the same
initiative branch (or a fresh `feat/56-…` per wave — decide at dispatch).

## Decisions (settled)

- v1 is deterministic-only: per-region segmentation → rowizer (B) → D3 fail-closed floor.
- Q1=D3, Q2=A2 (deferred v2, lane-aware token EQUALITY not superset), Q3=S3 (geometry-led in v1).
- Fixture must be **synthesized** (CE PDF is licensed — never commit it).

## Ticket state

| Ticket | What | Status | Depends on |
|--------|------|--------|------------|
| TR-0 | License-clean CE-like fixture + cell-parity harness | READY | — |
| TR-1 | Deterministic rowizer on lane-stacked `find_tables()` regions (Option B) | READY | TR-0 |
| TR-2 | Per-region verifier scoping + reading-order reassembly | NEEDS-DESIGN (split spike) | TR-1 |
| TR-3 | D3 fail-closed floor + selection-policy fix | READY | TR-2 |
| TR-4 | A2 value-guarded VLM re-ask | DEFERRED (v2) | TR-3 + evidence |
| TR-5 | S3 VLM confirm/split segmentation | DEFERRED (v2) | TR-2 spike |

## Outstanding / open questions

- TR-2 spike: does `find_tables()` return CE's two tables as separate regions or one merged
  region? Determines whether a deterministic split is needed in v1 or escalates to TR-5 (v2).
- Whether deterministic v1 (per-region + rowizer) alone yields clean CE grids — measured by the
  TR-0 parity test once TR-1…TR-3 land. If not, TR-4/TR-5 (v2) unlock.

## Next action

Dispatch **TR-0** (`socr-implementer`) — the fixture + parity harness is the acceptance gate every
other ticket tests against. Then TR-1.
