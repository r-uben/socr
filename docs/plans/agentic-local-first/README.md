# Plan: Agentic local-first OCR routing (issue #46, phase 2)

**Goal.** Make socr's OCR routing **agentic-by-default** with a deterministic, replayable
"cheapest-local-capable-first, cloud-only-on-failure" policy, and make the validated local
tier (`qwen3-vl:30b-a3b-instruct`) a first-class rung. Execute via parallel subagents.

## Why (settled by research — see `logs/`)

- **Codex (gpt-5.5), two rounds:** agentic should be the default, but the cheap-first policy
  must live in **deterministic code + config**, not an LLM. A skill is optional and only as a
  *frozen preflight*. Before flipping the default: split provider identity (engine+backend+model)
  and enrich the manifest so replay stays bit-exact.
- **Own-hardware benchmark (2026-06-13):** `qwen3-vl:30b-a3b-instruct` (A3B MoE, non-thinking)
  is the local frontier on a 64GB Mac — reconstructs dense multi-column tables with exact
  digits, recovers signs native mangles, ~1-2 min/page. The **thinking** build `qwen3-vl:30b`
  runs away on dense 11-col tables (never terminates; `think:false`/`/no_think` ignored on
  Ollama 0.30.8). `qwen3-vl:8b` collapses dense tables; `minicpm-v4.5` broken on Ollama.
- **Gemini web research:** nothing better than `qwen3-vl:30b` fits locally on 64GB; cloud
  frontier = Qwen3-VL-235B / Gemini; dedicated table models (TableFormer/Docling) are subsumed
  by modern VLMs — but cropping the table *before* the VLM helps (validates dual-pass).

Full findings: `logs/2026-06-13_research-findings.md`.

## Target end-state ladder (all manifest-frozen, deterministic)

```
native (free) → qwen3-vl:30b-a3b-instruct (local MoE) → qwen3.5:cloud (Ollama Cloud) → Gemini (paid edge)
```
Mistral/DeepSeek manual-only. Figure description local-first. A thinking-aware timeout guard
escalates any provider that stalls (so a dense table can never hang a batch).

## Execution model — parallel subagents

Work is decomposed into independent TICKETS (`TICKETS.md`). A fresh session orchestrates:
dispatch one `socr-implementer` subagent per ready ticket (respecting `depends-on`), then a
`socr-reviewer` subagent per completed ticket before commit. State lives in `STATUS.md`
(volatile) and per-ticket logs in `logs/`.

**Hard conventions (every agent MUST follow — baked into the subagent defs):**
- **NEVER `uv run`** in this repo — it hangs on the iCloud venv. Use the venv binaries directly:
  `~/venvs/socr/bin/python`, `~/venvs/socr/bin/pytest`, `~/venvs/socr/bin/ruff`.
- `ruff format --check` is a **BLOCKING CI gate** — run it before declaring a ticket done.
- Branch `feat/46-model-lineup-refresh`. Stage files by name (never `git add -A`). One commit
  per ticket, message references the ticket id. Don't push unless asked.
- Local model = `qwen3-vl:30b-a3b-instruct` (instruct/non-thinking). Never the thinking `:30b`.
- Update `STATUS.md` and append a `logs/<date>_<ticket>.md` note when finishing a ticket.
