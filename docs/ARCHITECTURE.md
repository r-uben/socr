# socr Architecture

## Modules
- `socr/cli.py`: Click commands (`process`, `engines`, `audit-status`, `describe_figures`, shorthand `p`).
- `core/`: shared types and configuration.
  - `config.py`: `AgentConfig`, per-engine configs, audit settings, routing overrides, optional cross-check toggle.
  - `document.py`: PDF loading/rendering, basic document classification.
  - `result.py`: page/figure/results, stats, markdown export.
- `engines/`: one adapter per engine implementing `BaseEngine` (`nougat`, `deepseek`, `gemini`, `mistral`).
- `audit/`: heuristic checks (`HeuristicsChecker`) and failure-mode scoring (`FailureModeScorer`).
- `pipeline/`:
  - `processor.py`: `StandardPipeline` -- orchestrates stages, output writer, figure pass.
  - `orchestrator.py`: `UnifiedPipeline` -- 5-phase pipeline (analyze, backbone, score, repair, assemble).
- `ui/`: Rich-based console/progress/panels/theme.
- `tests/`: routing/output, figure pass, and heuristic tests (require dev extras).

## Pipeline Stages
1) **Analyze** -- born-digital detection.
2) **Backbone OCR** -- primary engine extraction.
3) **Score** -- heuristic quality audit (failure-mode scoring).
4) **Repair** -- selective fallback on failed pages with alternative engines.
5) **Assemble** -- stitch final output, figure extraction, save markdown + metadata.

## Verification
- Heuristics: word count, garbage ratio, structure, repeated patterns.
- Failure-mode scoring: classifies audit failures (hallucination, truncation, garbled, etc.).

## Figure Pass
- Uses PyMuPDF to extract embedded images per page, skips tiny/extreme-aspect assets, downscales large images, calls `describe_figure` on first available vision engine (Gemini/DeepSeek/Mistral). Results are attached to `PageResult.figures` and printed.

## Testing
- Install (editable): `uv pip install -e .`
- Run: `pytest -q --disable-warnings --maxfail=1`.
- Coverage: figure extraction (`tests/test_figure_pass.py`), heuristics/reprocessing (`tests/test_audit_heuristics.py`). Tests skip if rich/fitz/Pillow aren’t installed.
