# socr Architecture

socr turns a PDF into Markdown by routing each page to an OCR engine, checking
the result, and re-trying on a different engine when the result is poor. The
default product path is the agentic, cost-aware router.

## Agentic, cost-aware routing

Running `socr paper.pdf` chooses providers **dynamically** by cost while judging
the real output:
1. Born-digital prose -> native text (free; skip with `--no-native-first`).
2. Every other page -> a **cost-ordered provider ladder** (cheapest first). Run
   the cheapest; a **judge** accepts the output or escalates to the next-cheapest;
   stop at the first accepted output. Bounded by `max_retries` / `cost_budget`.

The router records the winning provider + cost per page and writes a replayable
manifest. See `docs/log/2026-05-30_cost-aware-agentic-ocr.md`.

A deprecated deterministic path (backbone -> score -> judge -> repair) remains
reachable through the hidden `--legacy-routing` flag pending deletion. It
chooses the initial engine from page difficulty, checks output with heuristics,
then uses `RepairRouter` to select another engine by failure mode.

## Extraction method: extract / verify / escalate
Agentic routing uses one general method for getting structured content (tables,
figures) out of a page. The three layers are **separable** — the agentic path
does not require an expensive model loop on every page.

| Layer | Question | Cost |
|-------|----------|------|
| **Extract** | how do I get the element out? | a **single VLM pass** (local `qwen3-vl:30b-a3b-instruct` / cloud Gemini), or native text where the text layer is trustworthy |
| **Verify** | how do I know it's right? | a **free** native cross-check — text-layer geometry + header column count — *before* any paid model |
| **Escalate** | what to do when verify fails | agentic crop-reconcile / second VLM — **only on a fired signal** |

- **Single-pass is the default extract step.** Validated 2026-06-14 on a dense
  forecaster table (`qwen3-vl:30b-a3b-instruct`, one call): 120/120 summary cells
  exact. Crop-reconcile is the tail (escalation), not the trunk.
- **Verify is free, not a second model.** On born-digital pages PyMuPDF knows the
  column x-positions and the header fixes the column count, so a value outside its
  lane is a zero-cost red flag. This deterministic check sits *ahead* of the VLM
  judge (`judge/ollama_judge.py`), which can run locally. The judge and any provider
  escalation fire only when the cheap check disagrees.
- **Scope:** holds for **born-digital** PDFs (≈the whole corpus). **Pure scans** have
  no text layer to verify against, so there the only checks are a second VLM pass or
  self-consistency voting. The agentic router defaults to a single-pass VLM + free
  native verification and escalates only when a signal fires.

See `docs/log/2026-06-14_general-extraction-method.md` (issue #49).

## Modules
- `cli.py`: Click commands — `process` (default, PDF-path shorthand), `batch`,
  `engines`, `replay`, `judge-benchmark`. Agentic routing controls:
  `--strict-local`, `--judge-backend`, `--judge-model`, `--max-cost-per-page`,
  `--cost-budget`, `--write-manifest`.
- `core/`:
  - `config.py`: `PipelineConfig` (single flat config), `EngineType`,
    `ENGINE_PRIORITY`, agentic flags.
  - `document.py`: `DocumentHandle` — lazy PDF handle, per-page rendering, hash.
  - `result.py`: `PageOutput` (now with `cost_usd`), `EngineResult`, enums,
    `to_dict`/`from_dict` for caching.
  - `state.py`: `DocumentState` blackboard — per-page `PageState` with `attempts`
    and `best_output`. All routing branches mutate this shared state.
  - `providers.py`: provider cost registry + `provider_ladder()` (cheapest-first).
  - `manifest.py` + `cache.py`: content-addressed blob store + per-document
    manifest. `build_manifest()` freezes the winning output per page; `replay()`
    reconstructs the document from cache with **no engine calls**.
  - `difficulty.py`: classifies tables, equations, multi-column layouts,
    drawings, and image density during `UnifiedPipeline._phase_analyze`; every
    routing branch consumes that analysis. `born_digital.py` handles native-text
    extraction.
- `engines/`: one adapter per CLI engine implementing `BaseEngine`
  (`qwen`, `gemini`, `deepseek`, `marker`, `glm`, `nougat`, `mistral`) + HPC vLLM
  engines. `qwen` is listed first because it is the primary local workhorse --
  `qwen3-vl:30b-a3b-instruct`, the model `CLAUDE.md` names as *the* local OCR model,
  and the engine most runs actually use. `registry.py` resolves/probes engines.
- `judge/`: the OCR-faithfulness judge. `judge.py` (`JudgeVerdict`, prompt loader,
  parser), `ollama_judge.py` (local VLM), `benchmark.py` (score a judge against
  labeled pages). Prompt lives in `prompts/judge_page.md` (policy as data).
- `audit/`: heuristic checks (`HeuristicsChecker`) + failure-mode scoring.
- `pipeline/`:
  - `orchestrator.py`: `UnifiedPipeline` — the shared orchestrator. Every run
    analyzes first, then dispatches to the default agentic branch, the
    multi-engine branch, or the deprecated deterministic branch before
    assembly. Writes the manifest.
  - `agentic.py`: `route_page()` (Python-owned per-page loop) + `PageJudge`
    adapters (`VLMPageJudge`, `HeuristicPageJudge`).
  - `repair.py`: `RepairRouter` selects another engine by failure mode in the
    deprecated deterministic branch; pending deletion.
  - `consensus.py`, `reconciler.py`, `hpc_pipeline.py`: multi-engine consensus,
    reconciliation, and the HPC/vLLM path.
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
- Run: `~/venvs/socr/bin/pytest -q` (the canonical command in `CLAUDE.md`; a bare
  `pytest` may resolve to a different interpreter). Key suites:
  `test_providers.py`, `test_agentic.py`,
  `test_manifest_replay.py`, `test_judge_benchmark.py`, `test_orchestrator.py`.
