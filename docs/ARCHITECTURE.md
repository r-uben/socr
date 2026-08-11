# socr Architecture

socr turns a PDF into Markdown by routing each page first to a *modality*
(native PDF text vs OCR LLM vs chart asset), then — only when OCR is needed —
to an OCR engine, checking the result, and re-trying on a different engine
when the result is poor. It runs in two modes that differ only in **how the
OCR engine for a page is chosen**.

## Two routing levels (every page)

1. **Modality** (`pipeline/page_router.py`): should this page use native PDF
   reading, the chart-asset lane, or an OCR LLM? Trusted born-digital prose
   skips OCR entirely — OCR is overkill for clean text layers. Each decision
   is recorded as a `page_lane` audit event (`lane` + `reason`).
2. **OCR provider** (`pipeline/agentic.route_page` / `route_ocr_provider`): when
   OCR is required, which engine on the cost ladder?

## Two OCR-engine selection modes

### Deterministic (legacy routing — `--legacy-routing`)
The engine is chosen **up front** by predicting page difficulty:
1. Born-digital prose -> native text (no OCR, free).
2. "Easy" pages -> the cheap local engine (`config.local_engine`).
3. "Hard" pages -> the primary engine (`config.primary_engine`, e.g. cloud).

Easy/hard comes from `core/difficulty.py` (tables, equations, multi-column
layout, drawings, image density). Quality is checked by heuristics
(`audit/heuristics.py`); failed pages are re-OCR'd by `pipeline/repair.py`
(`RepairRouter`), which picks the next engine by **failure mode**.

### Agentic, cost-aware (default)
The engine is chosen **dynamically** by cost while judging the real output:
1. Born-digital prose -> native text (free; skip with `--no-native-first`).
2. Chart pages -> native prose + PNG asset (PP-7).
3. Every other page -> a **cost-ordered provider ladder** (cheapest first). Run
   the cheapest; a **judge** accepts the output or escalates to the next-cheapest;
   stop at the first accepted output. Bounded by `max_retries` / `cost_budget`.

This mode records the winning provider + cost per page and writes a replayable
manifest. See `docs/log/2026-05-30_cost-aware-agentic-ocr.md`.

## Extraction method: extract / verify / escalate
Both modes are instances of one general method for getting structured content
(tables, figures) out of a page. The three layers are **separable** — conflating
them is what makes high-fidelity extraction look like it needs an expensive agentic
loop on every page. It does not.

| Layer | Question | Cost |
|-------|----------|------|
| **Extract** | how do I get the element out? | a **single VLM pass** (local `qwen3-vl:30b-a3b-instruct` / cloud Gemini), or native text where the text layer is trustworthy |
| **Verify** | how do I know it's right? | a **free** native cross-check — text-layer geometry + header column count — *before* any paid model |
| **Escalate** | what to do when verify fails | agentic crop-reconcile / second VLM — **only on a fired signal** |

- **Single-pass is the default extract step.** Validated 2026-06-14 on a dense
  forecaster table (`qwen3-vl:30b-a3b-instruct`, one call): 120/120 summary cells
  exact. Agentic crop-reconcile is the tail (escalation), not the trunk.
- **Verify is free, not a second model.** On born-digital pages PyMuPDF knows the
  column x-positions and the header fixes the column count, so a value outside its
  lane is a zero-cost red flag. This deterministic check sits *ahead* of the VLM
  judge (`judge/ollama_judge.py`), which is itself a paid call — so the judge and any
  escalation fire only when the cheap check disagrees.
- **Scope:** holds for **born-digital** PDFs (≈the whole corpus). **Pure scans** have
  no text layer to verify against, so there the only checks are a second VLM pass or
  self-consistency voting — closer to agentic. Default: single-pass VLM + free native
  verification; agentic reserved for scans-with-disagreement, never every page.

See `docs/log/2026-06-14_general-extraction-method.md` (issue #49).

## Modules
- `cli.py`: Click commands — `process` (default, PDF-path shorthand), `batch`,
  `engines`, `replay`, `judge-benchmark`. Agentic flags: `--agentic`,
  `--judge-backend`, `--judge-model`, `--max-cost-per-page`, `--cost-budget`,
  `--write-manifest`.
- `core/`:
  - `config.py`: `PipelineConfig` (single flat config), `EngineType`,
    `ENGINE_PRIORITY`, agentic flags.
  - `document.py`: `DocumentHandle` — lazy PDF handle, per-page rendering, hash.
  - `result.py`: `PageOutput` (now with `cost_usd`), `EngineResult`, enums,
    `to_dict`/`from_dict` for caching.
  - `state.py`: `DocumentState` blackboard — per-page `PageState` with `attempts`
    and `best_output`. The spine both modes mutate.
  - `providers.py`: provider cost registry + `provider_ladder()` (cheapest-first).
  - `manifest.py` + `cache.py`: content-addressed blob store + per-document
    manifest. `build_manifest()` freezes the winning output per page; `replay()`
    reconstructs the document from cache with **no engine calls**.
  - `difficulty.py`, `born_digital.py`: page classification used by deterministic
    routing and native-text extraction.
- `engines/`: one adapter per CLI engine implementing `BaseEngine`
  (`gemini`, `deepseek`, `marker`, `glm`, `nougat`, `mistral`) + HPC vLLM engines.
  `registry.py` resolves/probes engines.
- `judge/`: the OCR-faithfulness judge. `judge.py` (`JudgeVerdict`, prompt loader,
  parser), `ollama_judge.py` (local VLM), `benchmark.py` (score a judge against
  labeled pages). Prompt lives in `prompts/judge_page.md` (policy as data).
- `audit/`: heuristic checks (`HeuristicsChecker`) + failure-mode scoring.
- `pipeline/`:
  - `orchestrator.py`: `UnifiedPipeline` — the single pipeline. Phases:
    analyze -> backbone -> score -> repair -> assemble (deterministic), or
    analyze -> `_phase_agentic` -> assemble (agentic). Writes the manifest.
  - `page_router.py`: modality router — `decide_page_lane` chooses
    `NATIVE` / `CHART_ASSET` / `OCR` before any OCR LLM call.
  - `agentic.py`: `route_page()` / `route_ocr_provider` (OCR provider ladder) +
    `PageJudge` adapters (`VLMPageJudge`, `HeuristicPageJudge`).
  - `repair.py`, `consensus.py`, `reconciler.py`, `hpc_pipeline.py`: repair
    routing, multi-engine consensus, and the HPC/vLLM path.
- `figures/`: `FigureExtractor` (PyMuPDF embedded-image extraction + VLM captions).
- `ui/`: Rich console/progress/panels.

## Reproducibility
The judge / VLM OCR is non-deterministic, so reproducibility is **not** "re-run
and hope for the same bytes." Instead the winning `PageOutput` per page is frozen
as a content-addressed blob; the manifest maps each page (by a fingerprint over
the rendered-image hash + engine + render params) to its blob. `socr replay
<manifest>` serves those blobs — zero model calls, bit-identical output, safe to
run headless/HPC.

## Design principles
- **Python owns the loop; the LLM is a stateless per-page decider.** The judge
  prompt is data (`prompts/judge_page.md`), not the orchestrator.
- **Cost ordering is relative; prices are tunable defaults.** Routing tries
  cheapest-first and lets the judge escalate — no static "engine X handles math"
  capability tables.
- **One pipeline.** `UnifiedPipeline` is the only orchestrator
  (`StandardPipeline` was removed). `--hpc-sequential` is a thin dedicated path.

## Testing
- Install (editable, venv off iCloud): `uv pip install -e ".[dev]" --python ~/venvs/socr/bin/python`
- Run: `pytest -q`. Key suites: `test_providers.py`, `test_agentic.py`,
  `test_manifest_replay.py`, `test_judge_benchmark.py`, `test_orchestrator.py`.
