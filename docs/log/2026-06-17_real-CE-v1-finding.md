# Real-CE validation of v1 → pivot to VLM-for-structure (GH-56)

Date: 2026-06-17
Context: after merging the deterministic v1 chain (TR-0…TR-3) to `main`, ran socr on the REAL
Consensus Economics `202401.pdf` page 4 (the dense forecaster grid the initiative started from),
`--agentic --strict-local`. (CE data is licensed — nothing from it is committed; this note
describes the result in prose only.)

## Result: v1 is a real improvement but does NOT solve real CE

What v1 got RIGHT on the real page:
- **Columns correct** — every indicator column holds the right values.
- **Values correct** — spot-checked top/bottom forecaster rows + the High/Low summary rows against
  the rendered source; all match. No invented or dropped numbers.
- **Honestly flagged** — `native_table_verifier_hard_fail`, `audit_failed`.

What v1 got WRONG — the ROW structure:
1. The top block of forecasters **merged into a single row** (names crammed in one cell, values
   space-stacked per column).
2. Lower forecasters have **names and values offset by one row** (the values land in the row above
   the name; the name row is blank). The data is all present and column-correct, but the
   **name↔value binding is broken**.

## Root cause

The TR-1 rowizer clusters tokens into rows by **vertical-gap segmentation** (split when the gap >
1.5× the page's median inter-row gap). Real CE rows are **densely and unevenly packed**: the top
block is tighter than the median (so it gets merged), and lower rows carry a small name/value
vertical offset (so they get split). The TR-0 synthetic fixture used **clean, evenly-spaced,
single-line rows**, so it never exercised this — it went green while the real page stays broken.

## Process lesson

A synthetic fixture that does not reproduce the target's real geometry gives **false confidence**.
The v1.1+ fixture MUST be built from a real CE page's **actual row y-positions** (license-clean:
copy the geometry, synthesize fake names/numbers at those positions) so "green" means green on the
real failure mode.

## Decision: move to v2 — VLM for structure, geometry as the value-guard

Deterministic geometry alone caps out: it cannot generalise across the variety of born-digital
layouts, and it cannot touch **scanned** PDFs (no geometry at all). Real CE is the measured evidence
the `/consilium` panel (design note §6) said would justify the v2 escalation (Q2 = A2, Q3 = S3).

BUT a raw VLM is exactly what produced the original ragged/collapsed table — an unguarded model
trades "rows misaligned, numbers correct" for "looks clean, a number silently in the wrong column,"
which is the WORSE failure this initiative exists to prevent. So v2 is **not** "VLM instead of
geometry"; it is:

- The **VLM proposes the structure** (rows/columns/which-table) — it generalises across layouts and
  is the only option for scanned pages.
- **Geometry value-guards** it: every numeric token the VLM emits must match a real native token on
  the page in that column — **no invented, no dropped, no shifted** (lane-aware token EQUALITY, the
  panel's A2 refinement, NOT loose superset). Pass → ship; fail → fail closed to the image (D3).
- The deterministic rowizer stays the **free first pass**; the VLM is the **escalation** when the
  rowizer's output fails its verifier (real CE) or when there's no geometry (scanned).

## Next

Re-scope **TR-4** (value-guarded VLM-for-structure) + **TR-4a** (real-CE-geometry fixture, the
failing gate) — see `docs/plans/table-repair/TICKETS.md`. TR-4 is NEEDS-DESIGN (value-guard
algorithm, escalation trigger, scanned vs born-digital, wiring into the agentic ladder) → design
pass before implementation.
