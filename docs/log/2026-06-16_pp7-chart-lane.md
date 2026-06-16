# PP-7 implementation log — Chart/figure-page routing lane

Date: 2026-06-16
Ticket: PP-7 (GH-73)
Branch: feat/73-pp7-chart-lane
Agent: socr-implementer (claude-sonnet-4-6)
Design settled: A2 + B1 (consilium note: 2026-06-16_chart-route.md)

---

## What changed

### `src/socr/figures/extractor.py`

1. Added named constant `CHART_MIN_CLUSTER_AREA: float = 120.0 * 120.0` (14 400 pts²).
   Documented: derived from `MIN_AREA` (80×80 = 6400 pts²), scaled up to suppress decorative
   marks while catching the smallest chart that can carry data.  Not a magic literal.

2. Added public function `has_chart_marks(page) -> bool`.
   - Cluster-first (load-bearing): calls `_cluster_drawings(drawings, ..., CLUSTER_GAP)` on
     `page.get_drawings()`, then for each cluster checks:
     (a) cluster bounding-box area >= `CHART_MIN_CLUSTER_AREA`
     (b) `_has_vector_data_marks(cluster_drawings)` — coloured fills / thick strokes
     (c) NOT `_looks_like_table_grid(...)` — rejects ruling-only grids
   - OR-s with `len(page.get_images()) > 0` (raster fast-path).
   - Logs DEBUG: drawing count, cluster count, per-cluster rejection reason.
   - Reuses all existing named-constant thresholds (`MIN_DATA_MARKS`, `DATA_STROKE_MIN_WIDTH`,
     `CLUSTER_GAP`, etc.) — no new magic numbers beyond `CHART_MIN_CLUSTER_AREA`.

### `src/socr/pipeline/orchestrator.py`

1. Added import: `from socr.figures.extractor import ..., has_chart_marks`.

2. Added `_is_chart_asset_page(page_num, ps, pdf_path)` predicate (new).
   - Returns True only when `_is_trusted_native_without_ocr` would also be True (so the chart
     lane is a sub-case of the native lane — never fires on OCR pages).
   - Tables are explicitly excluded: `_page_has_tables` gates it off.
   - Opens the PDF with fitz, calls `has_chart_marks(page)`, returns the result.
   - Failures (fitz open error, etc.) return False with DEBUG log.

3. Added `_render_chart_page_png(pdf_path, page_num, figures_dir)` helper (new).
   - Renders the full page at `RENDER_DPI` (same as `FigureExtractor`).
   - Saves `chart_page_{page_num}.png` in `figures_dir`.
   - Raises `RuntimeError` on any failure — caller is responsible for fail-closed handling.
   - Does NOT respect the `save_figures` flag (chart PNGs are mandatory preservation artifacts).

4. Added pre-loop `_chart_figures_dir` computation in `_phase_agentic` before the main loop.
   Uses `ocr_output_contract.figures_dir_for` when available; falls back to `output_dir/figures`.

5. Hooked chart-lane routing into the PP-2 fused agentic loop as:
   ```python
   if is_native and self._is_chart_asset_page(page_num, ps, state.handle.path):
       # chart lane: B1 — native prose + PNG ref + audit event
   elif is_native:
       # original native-bypass branch (unchanged)
   else:
       # OCR ladder (unchanged)
   ```
   PP-6's `_is_trusted_native_without_ocr` and `_page_has_tables` are NOT modified.

6. B1 representation:
   - `chart_body = native_prose.rstrip() + "\n\n" + "![Chart page N](figures/chart_page_N.png)"`
   - `PageOutput(engine="chart_asset", audit_passed=not chart_render_failed, ...)`
   - `AuditEvent(kind="chart_asset_page", detail="visual chart semantics represented as image
     asset; data values not transcribed", data={"png_saved": ..., "png_path": ...})`

7. Fail-closed on render failure:
   - `AuditEvent(kind="chart_asset_render_failed", ...)`
   - `PageOutput(status=PageStatus.WARNING, audit_passed=False)`
   - Never silent; error logged at ERROR level.

### `tests/test_chart_lane.py` (new, 20 tests)

Fixtures (all pure PyMuPDF, no embedded rasters unless stated):
- `_make_vector_chart_pdf`: coloured bars + thick line strokes, ZERO embedded images.
- `_make_decorated_prose_pdf`: prose + 3 thin neutral horizontal rules (false-trigger guard).
- `_make_raster_chart_pdf`: embedded PNG, zero vector drawings.
- `_make_prose_only_pdf`: dense text, no drawings.
- `_make_monochrome_lineplot_pdf`: B&W academic line-plot (documented false-negative fixture).
- `_make_dense_table_pdf`: ruling grid (table fixture, ladder non-regression).

Test classes:
- `TestHasChartMarks` (6 tests): vector detected, raster fast-path, decorated prose rejected,
  prose-only rejected, monochrome false-negative documented, debug logging asserted.
- `TestIsChartAssetPage` (5 tests): vector fires, table blocked, prose blocked, native_first=False
  blocked, raster fires.
- `TestRenderChartPagePng` (3 tests): PNG saved, force-save without flag, failure raises RuntimeError.
- `TestAgenticChartLaneRouting` (5 tests): routes to chart_asset lane, PNG without save_figures,
  clean prose stays native, table goes to OCR ladder, render failure is fail-closed.
- Standalone: `test_chart_min_cluster_area_is_named_constant`.

---

## Test result

`pytest tests -q -k "figure or chart or born_digital or route or orchestrator"`:
327 passed, 726 deselected (0 failures).

`pytest tests/test_chart_lane.py tests/test_figure_pass.py tests/test_orchestrator.py tests/test_pp4_inline_figures.py -q`:
175 passed (0 failures).

`ruff format --check` + `ruff check`: all files clean.

---

## Deviations from ticket spec

None. The implementation follows the settled A2+B1 design exactly.

---

## Known limitations / residual risks

1. **Monochrome false-negative**: B&W academic line-plots with only thin neutral strokes are not
   detected as charts (`_has_vector_data_marks` returns False). Documented in the test fixture
   `_make_monochrome_lineplot_pdf` and asserted as an expected false-negative. Catching these
   requires either a VLM pass or raster-contrast heuristics — out of scope for PP-7.

2. **Chart lane in non-agentic paths**: the chart-lane routing hook lives only in `_phase_agentic`.
   The non-agentic backbone/score/repair path (used when `agentic=False`) does not intercept chart
   pages. This is consistent with the PP-6 design note which put content-type routing in the agentic
   path. A follow-up can extend to the non-agentic path if needed.

3. **Per-page `_is_chart_asset_page` cost**: opens the PDF with fitz and calls `get_drawings()` for
   each born-digital non-table native page. This is the same call the figure-extraction phase already
   makes — no net new I/O on documents that run the figure phase. On documents without `--save-figures`
   or `--describe-figures`, this is the first time `get_drawings()` is called for those pages.
   Cost is bounded (one fitz.open per chart-candidate page) and acceptable for the routing benefit.

4. **Agentic ladder skipped for chart pages**: a chart page that also carries some tabular data
   (not detected as a table by PP-6's lane-cooccupancy gate) will route to the chart lane, not the
   OCR ladder. This is correct by design — if PP-6 did not detect a table, the page is not a table.
