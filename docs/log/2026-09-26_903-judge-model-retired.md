# 2026-09-26 — GH-903: retired page judge, generation-based availability probe

## What changed

Ollama Cloud retired `qwen3.5:cloud` on 2026-09-25 (`POST /api/generate` -> 410
Gone, `error: "qwen3.5:397b was retired at ..."`), but `/api/tags` kept listing
it, so `_resolve_judge_model`'s tags-based probe kept selecting a judge that
raised on every call (measured across #901's rotated-page runs and the pinned
archive re-OCR — 296 log/output files mention the 410).

Three changes, all judge-scope:

1. **`think: false` on every judge request.**
   - `socr/judge/ollama_judge.py::_post_generate` (the real page-judge call)
     and the new availability probe both send `"think": false`.
   - `socr/judge/table_rung_ollama.py::_build_payload` (the table judge rung 1
     `/api/chat` body, shared verbatim by the cell-transcription adjudicator
     via `cell_transcribe.transcribe_cell`) also sends it.
   - Without this, a thinking-model candidate (the new default, `qwen3.8:27b`)
     puts its answer in `thinking`/`message.thinking` and leaves the field
     socr parses empty — reported as "no JSON object found in judge output"
     or a 120s timeout, per the issue's 2026-09-26 measurement.
   - `check_ollama_model` (engine availability, `socr/core/ollama_utils.py`)
     and the OCR engines / math model / clean-equation paths are untouched —
     out of scope per the issue.

2. **New default page-judge candidate.** `orchestrator.py`:
   `UnifiedPipeline.JUDGE_MODEL_DEFAULT = "qwen3.8:27b"` (named, documented
   constant), and `_JUDGE_MODEL_CANDIDATES = [JUDGE_MODEL_DEFAULT,
   "minicpm-v:8b", "qwen3-vl:8b"]`. The ladder is now local-only by default —
   a cloud judge is only ever reached via an explicit `--judge-model`
   override, still subject to the existing `strict_local` /
   `zero_cap_pinned_forbids_cloud` policy check.
   `docs/MODELS.md`'s "Judge" section and the crop-reader paragraph that names
   the resolved judge model are updated to match; CLI help text for
   `--judge-model`/`--judge-backend` names no default and needed no change.

3. **Generation-based availability probe.**
   `OllamaVisionJudge.is_available()` now POSTs a real 1-token generation
   (`num_predict: 1`, `think: false`, `PROBE_TIMEOUT_SEC = 10.0` — same order
   as `check_ollama_model`'s 10s subprocess timeout, distinct from the full
   judge call's 120s `timeout`) instead of listing `/api/tags`. A 4xx (410
   retired, 404 never pulled) or a transport failure sets
   `self.unavailable_reason` (a short, human-readable string — HTTP status +
   the response body's `error` field when present) and returns `False`.
   `_resolve_judge_model` records the last candidate's reason on
   `self._judge_unavailable_reason` (new class-level cache, same
   `object.__new__`-safe pattern as `_judge_model_cache`); `_build_page_judge`
   surfaces it in the existing `judge_degraded_to_heuristic` audit event's
   `detail` string and a new `data["unavailable_reason"]` field — no new
   persisted record, reusing the one that already exists for this purpose.
   Probing stays memoized to once per run via the existing
   `_judge_model_cache`; `_build_page_judge`'s previous SECOND
   `vj.is_available()` call (after `_resolve_judge_model` had already
   verified the same model) was removed as redundant now that a "probe" is a
   real model call, not a cheap GET.
   `ollama_rung_reachable` (table judge circuit breaker, tags-based) and
   `check_ollama_model` (engines) are unchanged — only page-judge resolution
   uses the generation probe, per scope.

## Files

- `src/socr/judge/ollama_judge.py` — `think: false`, generation-based
  `is_available`, `unavailable_reason`, `PROBE_TIMEOUT_SEC`.
- `src/socr/judge/table_rung_ollama.py` — `think: false` in `_build_payload`.
- `src/socr/pipeline/orchestrator.py` — `JUDGE_MODEL_DEFAULT` constant,
  updated `_JUDGE_MODEL_CANDIDATES`, `_judge_unavailable_reason` cache,
  `_resolve_judge_model` reason capture, `_build_page_judge` detail/audit
  update and redundant-probe removal.
- `docs/MODELS.md` — Judge section + crop-reader paragraph.
- `tests/test_judge_wiring_gh133.py` — rewritten `_stub_tags` ->
  `_stub_generate` (POST /api/generate, not GET /api/tags) across all 10
  affected tests; behaviour pinned unchanged.
- `tests/test_gh154_remote_call_entry_points.py` — one test
  (`test_unpinned_zero_still_resolves_cloud_page_judge_by_default`) repointed
  to an explicit `judge_model="qwen3.5:cloud"` override, since the default
  ladder no longer has a cloud entry to demonstrate the policy against.
- `tests/test_gh903_judge_model_retired.py` — new: think:false on both wire
  formats, 410 fallthrough + reason capture, the tags-vs-generation
  difference pin, strict_local still forbidding an explicit cloud override.

## Verification

- `~/venvs/socr/bin/pytest tests/test_judge_wiring_gh133.py
  tests/test_gh903_judge_model_retired.py
  tests/test_gh154_remote_call_entry_points.py tests/test_table_rung_ollama.py
  tests/test_gh873_judge_vllm_backend.py -q` — 137 passed.
- Full suite: `~/venvs/socr/bin/pytest tests/ -q` — **5819 passed, 4 xfailed**,
  430s.
- `uvx ruff@0.16.0 format --check .` — clean (2 files needed
  `ruff format`, applied).

## Mutation check (per CLAUDE.md)

Both mutants made in a copy at `/tmp/socr-mut-903-{a,b}` (`src` + `tests` +
`pyproject.toml`, canary `socr.__file__` resolved inside the copy before
testing):

- **(a) drop `think: false`** from `_post_generate`: killed —
  `test_page_judge_generate_call_sends_think_false` failed
  (`assert None is False`). 1 failed, 8 passed.
- **(b) revert the probe to tags-only**: killed — 10 of 19 tests in
  `test_gh903_judge_model_retired.py` + `test_judge_wiring_gh133.py` failed,
  including the dedicated difference pin
  (`test_tags_listing_does_not_override_a_failing_generation`).

Both mutation copies were deleted after the check.

## Deviations / follow-ups

- `_build_page_judge`'s second `is_available()` call (on the freshly
  constructed judge, after `_resolve_judge_model` had already verified the
  same identity) was removed rather than kept: with a tags-listing probe it
  was a cheap, harmless double-check; with a real generation it would be a
  second model call per run for no benefit the memoization comment doesn't
  already forbid. Documented inline at both the resolver and the builder.
- `test_gh154_remote_call_entry_points.py`'s
  `test_unpinned_zero_still_resolves_cloud_page_judge_by_default` asserted
  the DEFAULT ladder resolves to a cloud model — no longer possible since the
  ladder is local-only now. Repointed to an explicit override, which still
  exercises the same policy (unpinned zero permits cloud).
- All #901-era real-page measurements taken since 2026-09-25 were made with a
  broken page judge (per the issue) and still need to be redone with this fix
  — out of scope here, left for the next measurement pass.
