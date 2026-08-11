# Page-lane router — design

Date: 2026-08-11
Branch: `cursor/page-lane-router-8050`

## Problem

OCR (VLM / CLI engines) is overkill for many PDF pages. Born-digital prose already
has a trustworthy text layer; calling a model wastes cost and can degrade clean
text. The pipeline already decides native vs OCR, but the decision is scattered
across `BornDigitalDetector`, `UnifiedPipeline._is_trusted_native_without_ocr`,
`_is_chart_asset_page`, and the confusingly named `route_page()` (which is the
**OCR provider ladder**, not native-vs-OCR).

That split made the optimisation hard to see, hard to audit, and easy to
re-break.

## Goal

Make **page-by-page modality routing** first-class:

```
for each page:
  classify text-layer quality (born-digital detect)
  decide lane: NATIVE | CHART_ASSET | OCR
  if NATIVE / CHART_ASSET → ship PyMuPDF text (no OCR LLM)
  if OCR                 → climb provider ladder (route_page / route_ocr_provider)
```

OCR remains the exception; native PDF reading is the default for trusted pages.

## Non-goals

- No change to born-digital quality thresholds (GH-35, encoding bands, etc.).
- No change to table-page policy (tables still leave the native bypass — #49).
- No move of `_phase_analyze` into the per-page loop (PP-2 fork C2 stands).
- No new PDF backend (PyMuPDF stays).

## Design

### New module: `src/socr/pipeline/page_router.py`

Pure decision function + types:

| Symbol | Role |
|--------|------|
| `PageLane` | `NATIVE`, `CHART_ASSET`, `OCR` |
| `PageRouteDecision` | `lane`, `reason`, optional detail |
| `decide_page_lane(...)` | modality router — OCR LLM vs native PDF reading |
| Reason constants | stable strings for audit / tests |

Inputs are plain values (config flags + page assessment flags + optional
`has_chart_marks`). No I/O. Chart detection stays in the orchestrator (needs
fitz); the router only consumes the boolean.

### Orchestrator wiring

- `_is_trusted_native_without_ocr` / `_is_chart_asset_page` become thin wrappers
  over `decide_page_lane` (behavior-preserving).
- `_phase_agentic` branches on `PageLane` once per page and emits a
  `page_lane` audit event (`lane`, `reason`) so provenance answers
  "why did page N skip / take OCR?".

### Naming

- Keep `route_page` (provider ladder) for compatibility.
- Add alias `route_ocr_provider = route_page` and document the split in
  `ARCHITECTURE.md` / `README.md`.

## Policy (unchanged, now explicit)

| Condition | Lane |
|-----------|------|
| `native_first=False` | OCR |
| not born-digital / no native text | OCR |
| `native_only` + born-digital + text | NATIVE (or CHART if marks) |
| `needs_ocr_enhancement` | OCR |
| `has_tables` | OCR |
| trusted native + chart marks | CHART_ASSET |
| trusted native otherwise | NATIVE |

## Success criteria

- Unit tests lock every policy row above.
- Existing chart-lane / native-first / agentic fuse tests stay green.
- Agentic runs emit one `page_lane` event per processed page.
- Docs describe two routing levels: modality (this) then OCR provider ladder.
