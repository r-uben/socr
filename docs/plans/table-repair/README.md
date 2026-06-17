# table-repair — reliable extraction of dense numeric tables (GH-56)

Initiative to make socr extract dense, line-less, multi-table pages (Consensus Economics
forecaster grids) correctly — or **fail closed** rather than ship plausible-but-wrong data.

## Why

Cell-by-cell validation on the real CE grid (`202401.pdf` p.4) found that the numeric values
extract faithfully, but the page ships the **worse of two flawed tables**: the local VLM emits a
readable-but-ragged row-major table (a missing mid-row cell silently shifts later values into the
wrong column), the native `find_tables()` fallback emits a fully-collapsed table, and the
deterministic geometry verifier correctly rejects both — then the collapsed native ships anyway.
A separate provenance/resume bug exposed during that validation is already fixed (PR #79,
`docs/log/2026-06-17_gh56-sidecar-provenance.md`). This initiative is the **table-quality** fix.

## Design + decision

- Full design + the `/consilium` verdict: `docs/log/2026-06-17_gh56-table-repair-design.md`.
- Governing principle: **NO SILENT CONTENT LOSS** — a wrong/shifted number is worse than an
  obviously-missing one.
- Design doctrine: **"VLM for structure, geometry for values."** For born-digital PDFs the
  deterministic token (x,y) geometry is the most document-robust signal for VALUES; the VLM is the
  robust engine for STRUCTURE/segmentation. Geometry exists only for born-digital pages (scanned
  pages have no deterministic net).

## Panel verdict (codex + gemini, 2026-06-17)

- **Q1 (which artifact ships when both fail):** D3 — ship neither as a table; failure marker +
  image-asset lane.
- **Q2 (VLM in the repair loop):** A2 — one constrained re-ask, gated by **lane-aware token
  equality** (not loose superset) — **deferred to v2**.
- **Q3 (segmentation ownership):** S3 — hybrid (geometry proposes, VLM confirms/splits,
  deterministic token-coverage post-check); v1 ships the geometry-led half, VLM-split is v2.

## v1 = deterministic only

per-region segmentation → deterministic rowizer (Option B) → D3 fail-closed floor. No model in the
v1 repair/segmentation path. v2 (TR-4 A2 re-ask, TR-5 VLM split) only if v1 measurably falls short
on real CE grids.

See `TICKETS.md` for TR-0…TR-5 and `STATUS.md` for live state.
