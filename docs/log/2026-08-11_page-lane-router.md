# Page-lane modality router

Date: 2026-08-11
Branch: `cursor/page-lane-router-8050`

## Why

OCR LLMs are overkill for many PDF pages. The pipeline already preferred native
PyMuPDF text for trusted born-digital prose, but the decision lived across
orchestrator predicates and was easy to confuse with `route_page()` (the OCR
*provider* ladder).

## What landed

- `src/socr/pipeline/page_router.py` — pure `decide_page_lane` with
  `PageLane.{NATIVE,CHART_ASSET,OCR}` and stable reason codes.
- Orchestrator `_decide_page_lane` wraps that policy + optional chart-mark I/O.
- `_is_trusted_native_without_ocr` / `_is_chart_asset_page` delegate to it
  (behavior-preserving).
- Agentic loop branches on `PageLane` and emits a `page_lane` audit event per
  processed page.
- Alias `route_ocr_provider = route_page` for naming clarity.
- README / ARCHITECTURE document the two routing levels.

## Policy (unchanged)

Native first for clean born-digital prose; tables / enhancement / scans → OCR;
trusted native + chart marks → chart-asset. `--native-only` / `--no-native-first`
still override.

## Tests

`tests/test_page_router.py` plus existing chart-lane / PP-2 fuse suites.
