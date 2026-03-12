# smart-ocr v1.0 Refactor — Tickets

## CLI Command Reference

| Engine | Command | Key Flags |
|--------|---------|-----------|
| gemini-ocr | `gemini-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--task`, `--model`, `-w` |
| deepseek-ocr | `deepseek-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--backend ollama\|vllm`, `--vllm-url`, `-w`, `--analyze-figures` |
| mistral-ocr | `mistral-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--max-pages`, `-w`, `--table-format` |
| nougat-ocr | `nougat-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--pages 0-5`, `--device`, `--batch-size` |
| marker-ocr | `marker-ocr <path> -o <dir>` | `--reprocess`, `--dry-run`, `-q`, `--pages 0-5`, `--device`, `--force-ocr` |

---

## Tickets

### [TICKET-1] Core data model — PipelineConfig, DocumentHandle, DocumentResult
- **Status:** done
- **Priority:** high
- **Files:** `src/smart_ocr/core/config.py`, `src/smart_ocr/core/document.py`, `src/smart_ocr/core/result.py`
- **Description:**
  - Replace 8 nested dataclasses with single `PipelineConfig` (engine selection, output dir, timeouts, flags)
  - Replace eager `Document` (renders ALL pages to PIL) with lazy `DocumentHandle` (holds path + page list, no rendering)
  - Simplify `OCRResult` → `DocumentResult` (document-level, not per-page in standard mode)
  - Keep `EngineType` enum, add `MARKER`
- **Acceptance Criteria:**
  - [ ] `PipelineConfig` replaces all 8 config dataclasses
  - [ ] `DocumentHandle` holds PDF path, page count, file hash — no PIL rendering
  - [ ] `DocumentResult` stores whole-document markdown + metadata
  - [ ] No backwards-compat shims for old classes

### [TICKET-2] Engine interface + fix CLI commands
- **Status:** done
- **Priority:** high
- **Files:** `src/smart_ocr/engines/base.py`, `src/smart_ocr/engines/gemini.py`, `src/smart_ocr/engines/nougat.py`, `src/smart_ocr/engines/deepseek.py`, `src/smart_ocr/engines/mistral.py`
- **Description:**
  - New `BaseEngine.process_document(pdf_path: Path, output_dir: Path, config: PipelineConfig) -> DocumentResult`
  - Remove `process_image()` / `process_pdf_page()` (per-page interface)
  - Each engine calls its CLI once per document via subprocess, reads output dir
  - Fix commands: `gemini-ocr <path>` (not `gemini-ocr process`), `nougat-ocr` (not `nougat-ocr-cli`)
  - `deepseek-ocr <path>` works as-is (auto-inserts `process`)
  - Handle CLI output structure: read the generated `.md` file from output dir
- **Acceptance Criteria:**
  - [ ] All engines call CLI once per PDF (one subprocess per document)
  - [ ] Correct CLI commands for all 5 engines
  - [ ] Engines return `DocumentResult` with markdown content
  - [ ] Subprocess timeout from config
  - [ ] Old per-page interface fully removed

### [TICKET-3] Add Marker engine
- **Status:** done
- **Priority:** medium
- **Files:** `src/smart_ocr/engines/marker.py` (new)
- **Description:**
  - New `MarkerEngine` following the same interface as TICKET-2
  - Calls `marker-ocr <path> -o <dir>` via subprocess
  - Supports `--pages`, `--device`, `--force-ocr` passthrough
  - Register in engine router
- **Acceptance Criteria:**
  - [ ] `MarkerEngine.process_document()` works
  - [ ] Registered in `EngineType` enum and router
  - [ ] Passes device/pages flags when configured

### [TICKET-4] Deduplicate figure extraction — shared FigureExtractor
- **Status:** done
- **Priority:** high
- **Files:** `src/smart_ocr/figures/extractor.py` (new), `src/smart_ocr/figures/__init__.py` (new)
- **Description:**
  - Extract ~400 lines of figure extraction from `processor.py` and ~400 from `hpc_sequential_pipeline.py`
  - Shared `FigureExtractor` class with 3 strategies: vector clustering, IMAGE blocks, raw embedded
  - `_cluster_drawings_into_figures()` union-find lives here once
  - Both StandardPipeline and HPCPipeline call `FigureExtractor.extract(pdf_path, output_dir)`
- **Acceptance Criteria:**
  - [ ] Single `FigureExtractor` class with all 3 extraction strategies
  - [ ] No figure extraction code in pipeline files
  - [ ] Union-find clustering exists in one place only
  - [ ] Both pipelines use the shared module

### [TICKET-5] Rewrite StandardPipeline — document-level stages
- **Status:** done
- **Priority:** high
- **Files:** `src/smart_ocr/pipeline/processor.py`
- **Description:**
  - Rewrite `OCRPipeline` as `StandardPipeline` with document-level stages:
    1. Primary OCR: `engine.process_document(pdf_path, output_dir)`
    2. Audit: document-level quality check on full markdown (word count, garbage ratio, hallucination)
    3. Fallback: if audit fails, re-run with fallback engine on whole document
    4. Figures: `FigureExtractor.extract()` (from TICKET-4)
  - Uses `MetadataManager` (TICKET-7) for incremental processing
  - Takes `PipelineConfig` (TICKET-1)
  - ~935 lines → target ~300 lines
- **Acceptance Criteria:**
  - [ ] 4-stage document-level pipeline
  - [ ] Audit runs on whole-document markdown
  - [ ] Fallback re-runs whole document with different engine
  - [ ] No per-page processing logic
  - [ ] Uses FigureExtractor, MetadataManager, PipelineConfig

### [TICKET-6] Simplify HPC pipeline — shared figures, simplified config
- **Status:** pending
- **Priority:** medium
- **Files:** `src/smart_ocr/pipeline/hpc_pipeline.py`
- **Description:**
  - Keep vLLM HTTP API approach (per-page, direct API — not CLI)
  - Remove duplicated figure extraction, use `FigureExtractor` from TICKET-4
  - Use `PipelineConfig` from TICKET-1
  - Merge `hpc_sequential_pipeline.py` into single `HPCPipeline` (delete `hpc_sequential_pipeline.py`)
  - Keep reconciler for DeepSeek+Nougat merging
- **Acceptance Criteria:**
  - [ ] Single `HPCPipeline` class (no more separate sequential pipeline)
  - [ ] Uses shared `FigureExtractor`
  - [ ] Uses `PipelineConfig`
  - [ ] vLLM server lifecycle management preserved
  - [ ] `hpc_sequential_pipeline.py` deleted

### [TICKET-7] MetadataManager — incremental batch processing
- **Status:** done
- **Priority:** high
- **Files:** `src/smart_ocr/core/metadata.py` (new)
- **Description:**
  - Port `MetadataManager` pattern from sibling CLIs (gemini-ocr, marker-ocr, nougat-ocr)
  - SHA256 file checksums for change detection
  - `is_processed(file_path) -> bool`, `record(file_path, **kwargs)`
  - Stores `metadata.json` in output directory
  - Used by StandardPipeline for skip/reprocess logic
- **Acceptance Criteria:**
  - [ ] `MetadataManager` with `is_processed()` and `record()`
  - [ ] SHA256 checksums
  - [ ] `metadata.json` output
  - [ ] Integrated into StandardPipeline

### [TICKET-8] CLI cleanup — flags, deduplication
- **Status:** done
- **Priority:** medium
- **Files:** `src/smart_ocr/cli.py`
- **Description:**
  - Add `--dry-run`, `--quiet`, `--reprocess` to top-level (pass through to pipeline)
  - Remove 12x duplicated timeout settings (single `--timeout` flag)
  - Deduplicate shared options between `process` and `batch` commands
  - Wire up `PipelineConfig` construction from CLI args
  - Add `marker` to engine choices
- **Acceptance Criteria:**
  - [ ] `--dry-run` lists files without processing
  - [ ] `--quiet` suppresses non-error output
  - [ ] `--reprocess` forces re-OCR of already-processed files
  - [ ] Single `--timeout` flag (not 12)
  - [ ] Marker available as engine choice

### [TICKET-9] Version bump, pyproject cleanup, tests
- **Status:** pending
- **Priority:** low
- **Files:** `pyproject.toml`, `src/smart_ocr/__init__.py`, `tests/`
- **Description:**
  - Bump version to 1.0.0
  - Add `marker-ocr-cli` to optional dependencies
  - Update engine list in README
  - Add/update unit tests for new interfaces
- **Acceptance Criteria:**
  - [ ] Version 1.0.0
  - [ ] `marker-ocr-cli` in optional deps
  - [ ] Tests pass for new engine interface and pipeline
