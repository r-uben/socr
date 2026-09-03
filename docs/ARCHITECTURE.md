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

The deterministic backbone -> score -> judge -> repair path was deleted on
2026-08-25 (#174, `docs/log/2026-08-25_174-legacy-fork.md`). There is one control
loop.

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
  exact. Crop-reconcile is an opt-in escalation tool (`--dual-pass-tables`, default
  off), never a trunk pass over all accepted table pages. When enabled, it fires
  only after a table verifier/routing signal. It runs after `route_page` has
  reached its verdict, so its reconciled text is a NEW CANDIDATE: it goes back
  through the same judge before it can ship, and the previously accepted bytes
  ship if the judge refuses it. An accepted re-judge promotes the page as a
  first-time acceptance would -- clearing the verdict and exhaustion state the
  refused ladder left -- so a crop can recover a page whose whole ladder was
  refused. It cannot recover a page whose winning attempt was an OPERATIONAL
  failure (an errored or truncated read): a crop repairs a table, not a read that
  never finished, so that page is refused before a judge call is spent. The crop
  still precedes the GH-96 escalation candidate and the table-judge terminal
  verdict.
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

## The table-judge ladder — ON by default, fail-closed

Every table page socr emits is judged before it ships. The ladder is two
independent readers, cheapest first: an ollama-cloud vision judge, then a
Gemini-family CLI. Each answers PASS or FAIL with a confidence. It is **on by
default** since 2026-09-03 (owner ruling Q3,
`docs/log/2026-09-02_gh359-ladder-terminals-design.md`); opt out with
`--no-table-judge-ladder`.

**Fail-closed** is the contract the default encodes: a table socr cannot
verify never ships SUCCESS. On a machine with no reachable rung — air-gapped,
no subscription, daemon down — every table page ships UNVERIFIED and the
document is PARTIAL. That is deliberate. Shipping an unwitnessed table as
clean is the bug the ladder exists to fix. The CLI says so at startup, names
the cause, and prints the opt-out.

Four terminals, and the difference between the last two is the shipped bytes:

| Terminal | What it means | What ships |
|---|---|---|
| ACCEPTED | a reader approved it, or a guard overruled a reader | the table |
| TABLE_UNVERIFIED | nobody could answer (outage, timeout, unparseable) | the table, page demoted; retryable on resume |
| TABLE_REJECTED | legacy label, kept for replay of older runs | the table, page demoted |
| TABLE_WITHHELD | a reader rejected it and no guard cleared it | **no table bytes** — a failure marker plus the page image; prose outside the region is kept |

**The two guards.** Neither of the two soft endings is settled by the readers
alone. When the ladder ends with two low-confidence PASSes, and when it ends
with a rejection, the same chain runs in the same order:

1. **Native geometry** (free, local). `bind()` checks rows AND columns against
   the page's own word layer. A pass overrules the readers
   (`verified_by_geometry`). A matching set of numbers is never enough on its
   own — matching numbers prove "not invented", never "correctly placed".
2. **Blind cell transcription** (one call, a third vendor). The cells the
   readers themselves doubted are transcribed from the crop by a model that is
   shown neither the emitted table nor the readers' opinions. Every doubted
   cell agreeing with the extraction clears it
   (`verified_by_blind_cell_transcription`); anything less does not.

A table is hidden only when the readers AND the guards agree. Every guard call
is metered and is refused before it is made when the per-page cap or the
document budget cannot cover it.

## Modules
- `cli.py`: Click commands — `process` (default, PDF-path shorthand), `batch`,
  `engines`, `replay`, `judge-benchmark`. Agentic routing controls:
  `--strict-local`, `--judge-backend`, `--judge-model`, `--max-cost-per-page`,
  `--cost-budget`, `--write-manifest`, `--dual-pass-tables/--no-dual-pass-tables`.
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
  - `reconciler.py`, `hpc_pipeline.py`: reconciliation and the HPC/vLLM path.
- `figures/`: `FigureExtractor` (PyMuPDF embedded-image extraction + VLM captions).
- `ui/`: Rich console/progress/panels.

## Reproducibility
The judge / VLM OCR is non-deterministic, so reproducibility is **not** "re-run
and hope for the same bytes." Instead the winning `PageOutput` per page is frozen
as a content-addressed blob; the manifest maps each page (by a fingerprint over
the rendered-image hash + engine + render params) to its blob. `socr replay
<manifest>` serves those blobs — zero model calls, bit-identical output, safe to
run headless/HPC.

### Provenance fields
Each page's fingerprint records **who read it**, so a change of reader
invalidates the cached page instead of silently reusing it:

- `engine` — the lane that won the page. `native` means the born-digital text
  layer; no model ran.
- `model_version` — the resolved model tag (e.g. `qwen3-vl:30b-a3b-instruct`),
  taken from the caller's run determinants, else the engine's `EngineResult`,
  else the page's own `provider_model`. **Empty for a native page**, and that
  emptiness is meaningful: it distinguishes "no model ran" from "a model ran",
  which a placeholder string would erase.
- `prompt_hash` — set when the caller supplied run determinants; a
  model/backend/task/prompt swap changes it.

The manifest entry's `journal` carries the same identity for every *attempt*,
not just the winner: `engine`, `provider_id`, `model`, `backend`, `cost_usd`,
`accepted`, `confidence`, `failure_mode`, the rejection `reason`, and
`judge_model`. The per-page sidecar (`pages/NNN.json`) carries the winner's full
serialised output, `provider_model` included, so "which model read page N?" is
answerable without the manifest.

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
