# TICKETS — agentic local-first routing (#46 phase 2)  · FROZEN — historical

> **FROZEN (2026-06-15).** Live ownership moved to `docs/plans/TICKETS.md`. The only open items,
> **D2** (sparse-row lane drift) and **E1** (preflight skill), are tracked there as **GH-46-D2**
> and **GH-46-E1** — edit those, not this file. Everything below is kept as the historical #46
> record (what shipped, field notes, validation evidence).

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Parallelizable = no shared files / no dep. Each ticket = one `socr-implementer` agent,
then one `socr-reviewer` pass before commit.

**Status reconciled 2026-06-15** against `main`: A1/B1/B2/B3/C1/C1b/C2/D1 all have
`feat(46)` commits merged to `main` (several with reviewer-blocker revisions) — flipped
from stale `TODO` to `DONE`. M1 (merge feat/46) landed: the feat/46 commits are on `main`.
Still open: **D2** (sparse-row lane drift, new), **Z1** (downstream CE batch, WIP),
**E1** (optional skill).

---

## Foundation (do first — others build on it)

### TICKET-A1 — Provider identity = engine + backend + model  · DONE · depends-on: none
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

### TICKET-B1 — Drop DeepSeek / demote Mistral from the agentic ladder · DONE · depends-on: A1
Use `auto_eligible=False` so `provider_ladder()` excludes them by default (still reachable via
explicit `--primary`, still in `cost_of`/replay). Update `tests/test_providers.py`.

### TICKET-B2 — Agentic as default mode · DONE · depends-on: A1
`config.agentic` default → True. Add CLI `--legacy-routing` (old deterministic path) and
`--strict-local` (no cloud rungs). Files: `pipeline/orchestrator.py`, `cli.py`. Tests for flag
wiring. Keep native-first intact.

### TICKET-B3 — Enrich agentic manifest for bit-exact replay · DONE · depends-on: A1
Manifest entries record: ladder snapshot, provider id/model/backend, every attempt, costs,
judge model + prompt_hash + raw verdict/confidence, accepted flag, skip reasons. `socr replay`
must reconstruct with 0 model calls. Files: manifest builder, `pipeline/agentic.py`. Tests.

---

## Stream C — robustness (parallel with A/B; minimal shared files)

### TICKET-C1 — Thinking-aware / stall guard · DONE · depends-on: none
A provider that stalls (forced-thinking runaway, e.g. thinking-build qwen on a dense 11-col
table — empty `response` while `thinking` grows; or wall-clock > budget) must **escalate up the
ladder**, never hang the batch. Add a per-provider soft timeout + (where streaming) a
no-`response`-progress detector. Files: `pipeline/agentic.py`. Tests with a stub slow provider.

### TICKET-C2 — Local-first figure description · DONE · depends-on: none
`engines/gemini_api.py` `describe_figure` currently Gemini-only. Make it local-first
(`qwen3-vl:30b-a3b-instruct` via Ollama) with Gemini fallback on empty/error. Files:
`engines/gemini_api.py` + caller in `orchestrator.py`. Tests.

---

## Stream D — quality (parallel; independent)

### TICKET-D1 — Dense-table prompt: keep summary rows paired · DONE (validated) · depends-on: none
On dense forecaster tables the VLM un-pairs the 2026/2027 columns for summary rows
(Consensus/High/Low/Std Dev) while data rows stay paired. Tune `prompts/table_extract.md`
(and/or the page OCR prompt) so summary rows use the same paired-column structure. Validate on
`scratch/bench/ce/202606_p4.png` (US forecaster table). No code logic change expected.

**Field note (2026-06-14):** qwen3-vl:30b-a3b-instruct first-pass output on 202606 p4
confirmed GOOD — each forecaster on its own row, all 10 columns with 2026/2027 pairs
aligned, `na na` for missing cells. Main table structure is solved.

**Validation (2026-06-14, `fix/46-ce-summary-rows`):** statistical summary rows
(Consensus / Last Month / 3 Months Ago / High / Low / Std Dev) inspected cell-by-cell
against high-DPI crops — **120/120 cells exact, all paired-column aligned.** The D1 core
concern (un-pairing on summary rows) does **not** occur with the instruct MoE; objective
met, no prompt change needed. See `docs/log/2026-06-14_ce-summary-row-validation.md`.
Residual narrower issue surfaced (sparse-row drift) -> TICKET-D2.

### TICKET-D2 — Sparse comparison-row lane drift · TODO · depends-on: none
On rows with long runs of blank cells (institutional comparison rows: CBO/IMF/OECD), the
VLM can slide a lone value into the wrong column lane. Observed on 202606 p4: `CBO (Feb. '26)`
shifted its `Employment Costs` pair `3.4 3.4` one lane right into the blank `Auto & Light
Truck Sales` column (digits correct, lane wrong); IMF/OECD on the same page were correct.
Dense rows unaffected — this is a sparse-row (few anchors), not summary-row, problem.
Cheapest fix first: prompt nudge in `prompts/table_extract.md` — count column positions
from the header, keep each value under its own header lane, never left/right-pack across
blanks. If that doesn't hold, consider the structural reconcile path (header fixes column
count; flag values on ambiguous lanes). Validate on the CBO row; quantify firing rate via Z1
(one CBO row is not a rate).

---

## Downstream (out of socr-repo scope — note only)

### TICKET-Z1 — Consensus Forecasts batch via qwen3-vl:30b-a3b-instruct · WIP
The ce_ocr job (`~/.claude/jobs/.../ce_ocr/`, 27 PDFs) can now run local-free with the instruct
MoE instead of raising the Gemini cap. Separate job; track elsewhere. Reference only.

---

## Follow-ups (post-refactor — close out feat/46, then optional skill)

### TICKET-C1b — Calibrate stall-guard soft-timeouts from measured data · DONE · depends-on: C1
C1 wired a per-provider soft-timeout dict but left the **values** unset. Do NOT run a new
benchmark — derive defaults from the 2026-06-13 measured latencies already on disk
(`scratch/bench/out200/results.tsv` + the CE runs). Observed worst-cases: local
`qwen3-vl:30b-a3b-instruct` ~50-60s prose/math, ~91-125s dense tables; `qwen3.5:cloud`
~100-127s; the thinking build never terminates (the case the guard exists to catch). Set
defaults comfortably above real worst-case, below runaway — e.g. local-instruct ~250-300s,
cloud ~240s — as the dict's DEFAULTS (keep it tunable; no magic constants buried in logic).
Add a one-line note in MODELS.md / a code comment citing the data source. Files: wherever C1's
timeout dict lives (`pipeline/agentic.py` or config). Test: a stub provider exceeding its
soft-timeout escalates. **Not merge-blocking** — reasonable defaults are enough to merge.

### TICKET-M1 — Final verify + merge feat/46 · DONE · depends-on: C1b (and all DONE tickets)
Full suite (`uv run pytest -q`) green + `ruff format --check .` clean on the whole branch.
Then merge `feat/46-model-lineup-refresh` (already pushed). Confirm `socr replay` still
reconstructs a prior agentic run bit-for-bit (reproducibility gate) before merging.

### TICKET-E1 — Preflight profile skill (OPTIONAL, post-merge) · TODO · depends-on: M1
Codex's frozen-preflight advisor: a skill that pre-classifies a doc and emits a frozen routing
**profile** (`strict-local` / `balanced` / `high-accuracy` + cost budget + judge model +
figures on/off), written into the manifest; the deterministic engine then runs from it. **The
skill is a PLANNER, never the live router** (reproducibility). Decide whether to build only
AFTER using agentic-default on a real job (Consensus batch / papers) — that reveals which
profiles are actually worth having. Cheaper precursor: just extend the existing `/ocr` skill to
expose the new `--strict-local` / agentic-default flags.
