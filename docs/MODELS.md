# Model lineup & routing policy

> **For anyone (human or LLM) editing socr's model choices.** This file is the
> single human-readable explanation of *which model does what, and why*. The
> machine source of truth is `src/socr/core/config.py` (`AUTO_ENGINE_ORDER`,
> `ENGINE_PRIORITY`, default model strings) and `src/socr/engines/registry.py`
> (`_LOCAL_ENGINE_ORDER`). If you change routing, change both the code and this
> doc in the same commit.

## The policy in one line

**Native text first (free) → local VLM → paid cloud (Gemini) only for edge cases.**
(The Ollama-Cloud Qwen rung was removed from the default ladder in #905; see below.)

Best *cheap* combination, not best absolute quality. The workload is overwhelmingly
**born-digital academic PDFs**, where PyMuPDF native-text extraction already handles
most prose pages for free. OCR/VLM engines only earn their cost on scanned pages,
equations, tables, and figures.

## Why these models (measured, not benchmarked in the abstract)

Numbers below are **measured on the owner's 64GB M-series Mac on the real workload**,
not generic leaderboard scores. See [Checking for newer models](#checking-for-newer-models)
for the external leaderboard, and the design logs under `docs/log/` for the raw data.

> **`qwen3.5:cloud` was retired by Ollama Cloud on 2026-09-25.** Every call now returns
> `410 Gone`, but `ollama list` still shows it. Its rows below are historical until a
> replacement is measured (#903). #905 removed it from the default provider ladder
> (now local qwen → marker → gemini) and moved the equation-crop `math_model` default
> to the local `qwen3-vl:30b-a3b-instruct`. No cloud replacement is chosen: none is measured.

| Model | Where | Quality | Speed | Cost | Verdict |
|-------|-------|--------:|-------|------|---------|
| **native (PyMuPDF)** | local | exact text | instant | free | Default for born-digital prose + table *values* |
| ~~`qwen3.5:cloud`~~ (retired 2026-09-25) | Ollama Cloud | ~0.57 | ~49s/pg | free* | **HISTORICAL.** Was the workhorse VLM (cleared all 3 hard page types). Every call now 410s |
| `qwen3-vl:8b` | local Ollama | ~0.47 | ~135s/pg | free | Offline / simple-page fallback. **Times out (>300s) on dense pages** |
| **Gemini 3.x** | cloud API | 0.60–0.64 | fast | ~$0.0002/pg | **Edge-case escalation.** Best quality on the board; occasionally returns empty |
| Mistral OCR | cloud API | 0.45 | fast | ~$0.001/pg | **Manual only.** Worse *and* ~5x pricier than Gemini → strictly dominated |
| GLM-OCR | local Ollama | 0.37 | ~10s/pg | free | Fast local emergency fallback |
| DeepSeek-OCR | local Ollama | 0.085 | — | free | **Dead weight.** Dropped from auto/local ladders; reach via `--primary deepseek` only |
| `minicpm-v:8b` | local Ollama | — | ~27s/pg | free | Coarse offline captions only; **collapses table sub-columns** |

\* (Historical) `qwen3.5:cloud` ran on the Ollama Cloud account — no extra API key, billed as
part of the Ollama subscription, treated as the cheap "cloud" rung.

## Routing per sub-task

socr produces **markdown**. Three sub-tasks, three routing rules:

### Local tier model — `qwen3-vl:30b-a3b-instruct` (validated 2026-06-13)

The `QWEN` engine's **local** (ollama) backend uses `qwen3-vl:30b-a3b-instruct` — the
Qwen3-VL-30B **A3B MoE** (~3B active/token). On the owner's 64GB Mac it reconstructs dense
multi-column tables with exact digits (verified against native ground truth), recovers math
the native text layer mangles, and runs at ~1-2 min/page. It is the local frontier for this
hardware (Gemini web research: everything better is too big or API-only).

> **Trap — use the INSTRUCT build, never the default `qwen3-vl:30b`.** The default `:30b`
> is the *thinking* build; on dense 11-column tables it emits 5000+ thinking tokens and
> never reaches the transcription within any sane timeout. Neither `think:false` (API) nor
> `/no_think` (prompt) suppress it on Ollama 0.30.8 — only the `-instruct` *variant* does.
> The dense `qwen3-vl:8b` collapses dense tables and is slow; not a local-tier option.
> `minicpm-v4.5:8b` emits a degenerate loop on Ollama (broken). See
> `[[reference-local-ocr-benchmark-jun2026]]`.

**Stall-guard soft-timeout defaults** (`DEFAULT_PROVIDER_TIMEOUTS` in `pipeline/agentic.py`):
measured latencies from `scratch/bench/out200/results.tsv` (2026-06-13):
`qwen3-vl:30b-a3b-instruct` (local QWEN) peaks at ~125s on dense tables → soft timeout 300s;
Gemini API (GEMINI rung) latency not measured in bench data; 240s is a conservative upper-bound.
The thinking build never terminates — the timeout guard is its only defence.

### 1. Text & formulas (LaTeX in markdown)
- **Default:** native PyMuPDF text for born-digital prose (free).
- **Hard / scanned / math pages:** local `qwen3-vl:30b-a3b-instruct` (free). A cloud Qwen
  model is reached only by an explicit `--qwen-model <tag>:cloud` pin, which is treated as
  cloud egress: refused (console line plus a `qwen_cloud_pin_refused` audit event) under
  `--strict-local` or an explicit `--max-cost-per-page 0`.
- **Escalation:** Gemini when Qwen is unavailable or returns empty.
- **Font-corrupted equations** (`recover_corrupt_math`): `config.math_model` =
  `qwen3-vl:30b-a3b-instruct` (`DEFAULT_MATH_MODEL`, local, free; changed from the retired
  `qwen3.5:cloud` in #905). Override with `--math-model`; a name containing `cloud` is
  refused under `--strict-local` / `--max-cost-per-page 0`.

### 2. Figures (images)
- **Extraction is model-free:** PyMuPDF locates figures, crops the frame, writes the
  PNG to `figures/`. No model involved in *finding* the figure.
- **Description** is **local-first** (`_get_vision_engine`, `pipeline/orchestrator.py:5279`):
  when Ollama is reachable it returns a `LocalFirstFigureEngine` that tries the local
  VLM and falls back to the **Gemini vision API** (`engines/gemini_api.py`) per call on
  an empty or failed result; Gemini is used directly only when Ollama is unavailable.
  Figure descriptions are lower-stakes than text, so a cloud fallback here is the least
  costly one to keep.

### 3. Tables (markdown grids)
- **Values come from native text — native is ground truth.** A VLM is *never* the
  authority for a cell value (silent corruption of a research number is the worst
  failure mode). `auto_patch_tables` stays **off** by default.
- **Layout only:** the dual-pass crop reader uses the resolved judge model
  (`qwen3.8:27b` first, since GH-903) to restore row×column *structure*, reconciled
  against the native values.
- Crop-read VLMs vary in reliability on dense tables (`qwen3-vl:8b` times out,
  `minicpm-v:8b` collapses sub-columns). `qwen3.5:cloud` is no longer in this
  ladder at all — Ollama Cloud retired it 2026-09-25 (GH-903). Gemini is the edge
  fallback for dense *scanned* tables.

### Judge (quality gate / escalation decider)
- `_JUDGE_MODEL_CANDIDATES = [qwen3.8:27b, minicpm-v:8b, qwen3-vl:8b]` — local-first
  since GH-903 (Ollama Cloud retired `qwen3.5:cloud`, the former first candidate,
  2026-09-25); first available wins. Availability is a real 1-token generation on
  the exact model (`OllamaVisionJudge.is_available`), not a `/api/tags` listing —
  a retired cloud model can still be *listed* while every generation 410s.
  Override via `config.judge_model`.

## The ladders in code

```python
# config.py — deterministic auto-probe (default mode), best-cheap-first
AUTO_ENGINE_ORDER = [QWEN, GEMINI, MARKER, GLM, NOUGAT]  # DeepSeek + Mistral dropped

# registry.py — local-only tier for tiered routing
_LOCAL_ENGINE_ORDER = [QWEN, GLM, NOUGAT, MARKER]  # DeepSeek dropped
```

`resolve_auto_engine()` returns the first *available* engine in `AUTO_ENGINE_ORDER`
(installed + dependencies satisfied). Default mode never reaches DeepSeek or Mistral.

## What is intentionally NOT changed here (tracked follow-ups)

These belong to the **agentic** cost-ladder (`core/providers.py` `provider_ladder`),
which is wired into `test_providers.py` / `test_agentic.py` / `test_p1_cascade_economics.py`
and is a larger, separately-tested change:

- Dropping DeepSeek / demoting Mistral from the **agentic** escalation ladder
  (`DEFAULT_PROVIDERS` / `ENGINE_PRIORITY`). Today they remain in the cost registry for
  replay and `cost_of`.
- Splitting provider identity by **engine + backend + model** at the *engine* layer.
  GH-46-E2 closed the routing half of this by emitting `PROFILE_QWEN_LOCAL` and
  `PROFILE_QWEN_CLOUD` as independently probed rungs. #905 reversed the emission:
  `qwen3.5:cloud` is retired, so the default ladder is local qwen → marker → gemini and
  `PROFILE_QWEN_CLOUD` stays registered only so pre-retirement manifests still resolve
  (its `cloud_model_available` probe is now a real generation, not a listing). `EngineType.QWEN` still
  names two backends, and `DEFAULT_PROVIDERS` still holds only one profile per engine —
  a deliberate collision, worked around rather than removed.

## Checking for newer models

Check these periodically, and always before re-ranking an engine:

- **[socOCRbench](https://noahdasanaike.github.io/posts/sococrbench.html)** (N. Dasanaike):
  an OCR leaderboard for social-science documents, covering text (NES, chrF) and tables
  (TEDS), with open-weight and API models side by side. It uses historical, handwritten
  and degraded scans, while socr's workload is born-digital papers. Use it to shortlist
  candidates, never to rank them.
- **Ollama library pages** (`https://ollama.com/library/<name>`): which tags exist,
  which are `:cloud`, and which read images. Cloud models can be **retired without
  notice**, as `qwen3.5:cloud` was on 2026-09-25.

Before switching a default:
1. Probe the exact model with a 1-token generation. Do not trust `ollama list` or
   `/api/tags`: they keep listing retired models.
2. Measure it head-to-head against the current default on the owner's pages. Score
   decimals against the native text layer, and record seconds per page and whether it
   finished. For a thinking build, verify that thinking is actually off: the run must
   finish, and the output must contain no reasoning text. Asking is not enough. On
   Ollama 0.30.8, `qwen3-vl:30b` ignored both `think:false` and `/no_think` (see the
   trap above), so only an instruct variant was usable there.

## How to add or re-rank an engine

1. `core/config.py`: add to `EngineType`, place in `AUTO_ENGINE_ORDER` / `ENGINE_PRIORITY`.
2. `engines/`: implement a `BaseEngine` adapter; register in `engines/registry.py`.
3. `core/providers.py`: add a `ProviderProfile` (cost + tier) for agentic routing.
4. `pyproject.toml`: add the optional CLI dependency.
5. Update this doc + the README engine table in the same commit.
