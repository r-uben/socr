# socr Architecture

## Modules
- `socr/cli.py`: Click commands (`process`, `engines`, `audit-status`, `describe_figures`, shorthand `p`).
- `core/`: shared types and configuration.
  - `config.py`: `AgentConfig`, per-engine configs, audit settings, routing overrides, optional cross-check toggle.
  - `document.py`: PDF loading/rendering, basic document classification.
  - `result.py`: page/figure/results, stats, markdown export.
- `engines/`: one adapter per engine implementing `BaseEngine` (`nougat`, `deepseek`, `gemini`, `mistral`).
- `audit/`: heuristic checks (`HeuristicsChecker`) and optional Ollama LLM audit (`LLMAuditor`).
- `pipeline/`:
  - `router.py`: engine selection (primary/fallback/cross-check).
  - `processor.py`: orchestrates stages, output writer, figure pass.
- `ui/`: Rich-based console/progress/panels/theme.
- `tests/`: routing/output, figure pass, and heuristic tests (require dev extras).

## Pipeline Stages
1) **Primary OCR** — pick engine via `EngineRouter` (honors overrides, prefers local).  
2) **Verifier** — heuristics; optional cross-check on flagged pages using the other local engine; optional Ollama LLM audit.  
3) **Fallback OCR** — reprocess flagged pages with a different engine (prefers cheaper cloud).  
4) **Figure Pass** — experimental: extract embedded page images, filter out tiny/extreme assets, send to vision-capable engine `describe_figure`.  
5) **Output** — write `output/<doc_stem>/<doc_stem>.<ext>` plus `metadata.json` (stats, engines used, pages needing rerun, doc metadata).

## Verification (local-first)
- Heuristics: word count, garbage ratio, structure, repeated patterns.
- Cross-check (optional): re-run a few flagged pages on the other local engine (Nougat↔DeepSeek) before cloud fallback.
- LLM audit: optional Ollama-based review on flagged pages.

## Figure Pass
- Uses PyMuPDF to extract embedded images per page, skips tiny/extreme-aspect assets, downscales large images, calls `describe_figure` on first available vision engine (Gemini/DeepSeek/Mistral). Results are attached to `PageResult.figures` and printed.

## Testing
- Install (editable): `uv pip install -e .`
- Run: `pytest -q --disable-warnings --maxfail=1`.
- Coverage: routing/output (`tests/test_pipeline_routing.py`), figure extraction (`tests/test_figure_pass.py`), heuristics/reprocessing (`tests/test_audit_heuristics.py`). Tests skip if rich/fitz/Pillow aren’t installed.
