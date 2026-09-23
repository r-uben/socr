# GH-842 — engine types the agentic ladder cannot run (2026-09-23)

Branch `fix/842-report-unservable-engines`.

## Step 0, measured as the issue asked

- `PipelineConfig().enabled_engines` is every `EngineType`, so **every default run** puts
  `VLLM` and `DEEPSEEK_VLLM` through `_available_engines_for_agentic`.
- Both have `DEFAULT_PROVIDERS` entries (`vllm`, `deepseek-vllm`) but no `_ENGINES`
  registration; `get_engine` raises `ValueError("No CLI engine for …")`, and the reachability
  `except` swallowed it exactly as it swallows an outage.
- No content is lost: those rungs could never serve on this path (`hpc_pipeline.py` builds
  them directly; the agentic path reaches a vLLM server through `--qwen-backend vllm`). So
  this is a **clarity** fix, scoped as the issue anticipated.

## What

- `registry.has_cli_engine(engine_type)` — the structural question, answered from the
  registry rather than learned by catching an exception.
- `_available_engines_for_agentic` skips no-engine types explicitly and collects them.
- `_report_unservable_engines` warns **once per pipeline**, and **only when
  `enabled_engines` was narrowed on purpose** and names one. The default list contains them
  because it contains everything; warning then would fire on every run about engines nobody
  asked for — the GH-525 noise lesson.

Reachability failures of registered engines keep their old handling.

## Tests

`tests/test_gh842_unservable_engines.py`, 7 hermetic tests (`get_engine` and the cloud probe
patched). Mutations seen to fail in an out-of-repo copy with the import canary: removing the
structural skip fails 2; removing the default-list quiet guard fails 1.

## Not done

`DEFAULT_PROVIDERS` still carries the two profiles. Removing them would touch
pricing/provenance lookups keyed on them and the HPC path; with the pruning now explicit
and reported, that is a separate cleanup, not a correctness fix.
