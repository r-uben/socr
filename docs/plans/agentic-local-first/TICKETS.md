# TICKETS — agentic local-first routing (#46 phase 2)

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Parallelizable = no shared files / no dep. Each ticket = one `socr-implementer` agent,
then one `socr-reviewer` pass before commit.

---

## Foundation (do first — others build on it)

### TICKET-A1 — Provider identity = engine + backend + model  · TODO · depends-on: none
**Problem:** `ProviderProfile` keys only on `EngineType`, so `QWEN` ambiguously means local
`qwen3-vl:30b-a3b-instruct` OR cloud `qwen3.5:cloud` — different cost/latency/availability.
**Do:** add `id`, `backend`, `model` to `ProviderProfile`. Define distinct profiles:
`qwen-local-instruct` (ollama, qwen3-vl:30b-a3b-instruct, free), `qwen-cloud` (ollama-cloud,
qwen3.5:cloud, ~free), `gemini`, `marker`, `glm`, `nougat`; `mistral`/`deepseek` present but
`auto_eligible=False`. Keep `cost_of`/replay working. Files: `core/providers.py`,
`pipeline/agentic.py` (ProviderAttempt stores id/model/backend), `pipeline/orchestrator.py`
(`_available_engines_for_agentic` returns profiles). Update `tests/test_providers.py`,
`tests/test_agentic.py`, `tests/test_p1_cascade_economics.py`.
**Done when:** ladder builds from profiles; local vs cloud qwen are distinct rungs; tests green.

---

## Stream B — agentic default (depends on A1)

### TICKET-B1 — Drop DeepSeek / demote Mistral from the agentic ladder · TODO · depends-on: A1
Use `auto_eligible=False` so `provider_ladder()` excludes them by default (still reachable via
explicit `--primary`, still in `cost_of`/replay). Update `tests/test_providers.py`.

### TICKET-B2 — Agentic as default mode · TODO · depends-on: A1
`config.agentic` default → True. Add CLI `--legacy-routing` (old deterministic path) and
`--strict-local` (no cloud rungs). Files: `pipeline/orchestrator.py`, `cli.py`. Tests for flag
wiring. Keep native-first intact.

### TICKET-B3 — Enrich agentic manifest for bit-exact replay · TODO · depends-on: A1
Manifest entries record: ladder snapshot, provider id/model/backend, every attempt, costs,
judge model + prompt_hash + raw verdict/confidence, accepted flag, skip reasons. `socr replay`
must reconstruct with 0 model calls. Files: manifest builder, `pipeline/agentic.py`. Tests.

---

## Stream C — robustness (parallel with A/B; minimal shared files)

### TICKET-C1 — Thinking-aware / stall guard · TODO · depends-on: none
A provider that stalls (forced-thinking runaway, e.g. thinking-build qwen on a dense 11-col
table — empty `response` while `thinking` grows; or wall-clock > budget) must **escalate up the
ladder**, never hang the batch. Add a per-provider soft timeout + (where streaming) a
no-`response`-progress detector. Files: `pipeline/agentic.py`. Tests with a stub slow provider.

### TICKET-C2 — Local-first figure description · TODO · depends-on: none
`engines/gemini_api.py` `describe_figure` currently Gemini-only. Make it local-first
(`qwen3-vl:30b-a3b-instruct` via Ollama) with Gemini fallback on empty/error. Files:
`engines/gemini_api.py` + caller in `orchestrator.py`. Tests.

---

## Stream D — quality (parallel; independent)

### TICKET-D1 — Dense-table prompt: keep summary rows paired · TODO · depends-on: none
On dense forecaster tables the VLM un-pairs the 2026/2027 columns for summary rows
(Consensus/High/Low/Std Dev) while data rows stay paired. Tune `prompts/table_extract.md`
(and/or the page OCR prompt) so summary rows use the same paired-column structure. Validate on
`scratch/bench/ce/202606_p4.png` (US forecaster table). No code logic change expected.

---

## Downstream (out of socr-repo scope — note only)

### TICKET-Z1 — Consensus Forecasts batch via qwen3-vl:30b-a3b-instruct · TODO
The ce_ocr job (`~/.claude/jobs/.../ce_ocr/`, 27 PDFs) can now run local-free with the instruct
MoE instead of raising the Gemini cap. Separate job; track elsewhere. Reference only.
