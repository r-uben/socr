# GH-984: test hermeticity against a live, busy Ollama

## Problem
`_run_fingerprint` -> `_resolve_judge_model` -> `OllamaVisionJudge.is_available` ->
`probe_model_generation` makes a real generation call. With a live but busy daemon,
any test that drives a pipeline on the default Ollama judge path blocks (heuristic configs skip the probe). CI has no Ollama so it never showed.

## Sweep
A socket-level recorder (connect/connect_ex to the configured Ollama host, recording
the calling socr frames) plus a 60 s per-test alarm, run over the whole suite with
Ollama up and its GPU busy. 278 tests in 54 files connected: 277 through the judge
probe, 1 through `qwen.is_available` -> `check_ollama_model`. Once the judge probe was
pinned, 16 more (3 files) surfaced behind it via `_available_engines_for_agentic` ->
`<engine>.is_available` -> `check_ollama_model`.

## Fix
- `tests/conftest.py`: autouse `_no_live_ollama` guard (`ollama_connection_guard`).
  It refuses AND records any connect to the ambient `OLLAMA_HOST` (default
  127.0.0.1:11434, ::1, plus resolved addresses), captured before the test body, and
  fails the test at teardown with the socr call chain (a probe's own `except` swallows
  the refusal, so the record is what fails). Other loopback ports, and an
  `OLLAMA_HOST` a test sets itself for its own server, are untouched. It hides nothing.
- Explicit opt-in lists, not autouse: `_JUDGE_PROBE_PINNED_MODULES` (53 modules) pins
  `UnifiedPipeline._resolve_judge_model` to ""; `_ENGINES_PINNED_MODULES`
  (test_chart_lane, test_gh498_figure_repair_through_process,
  test_gh519_visual_values_debt) pins `_available_engines_for_agentic` to
  `[PROFILE_QWEN_LOCAL]`. A new leaking file is on neither list and fails the guard.
  Modules that exercise `_resolve_judge_model` itself (gh873, gh903) are not listed.
- `tests/test_table_judge_gate.py::TestProcessFlagDifference::test_native_lane_is_witnessed_too`:
  patches `socr.engines.qwen._check_ollama_model`.
- `tests/test_gh984_ollama_connection_guard.py`: 6 guard tests (other port allowed,
  default refused, connect_ex, configured host, restore, swallowed refusal).
- Guard seen failing: before the pins, 278 errors; the pins were then derived from them.

## Results (Ollama up, GPU busy, other sessions' pytest running concurrently)
- Default OLLAMA_HOST: 6488 passed, 2 skipped, 4 xfailed; 476 s (268 s when quiet).
- OLLAMA_HOST=127.0.0.1:61999 (unreachable): 6488 passed; 267 s.
- Do not use `OLLAMA_HOST=127.0.0.1:1` as the unreachable host:
  `test_agentic_corrupt_math_default_flip_no_provider_is_additive_only` uses port 1 as
  its own dead endpoint, and the guard correctly flags it.
- Not mine: `test_gh974_review_pins::test_budget_is_shared_across_the_tables_on_one_page`
  is wall-clock based; it failed twice when suites ran concurrently, passes alone.

## Tests fixed
The 53 modules in `_JUDGE_PROBE_PINNED_MODULES` (277 tests), the 3 in
`_ENGINES_PINNED_MODULES` (16 tests), and the one test in test_table_judge_gate.py.

## Review follow-up (Astra, ACCEPT-WITH-FIXES)
Limits of the guard, also stated in the conftest comment:
- Child processes are not covered. Exec'd subprocess engines do not inherit monkeypatches,
  so unit tests must stub the subprocess launch boundary.
- A proxy connection reaches the proxy address, not the Ollama endpoint, so endpoint
  matching can be bypassed.
- The module-wide pins also silently cover FUTURE tests added to a listed module. A new
  test meant to exercise the judge probe would see "" and never reach it; put such a test
  in its own module. See the 53-module list in tests/conftest.py.

Added guard tests (10 total): fixture loopback server works under the autouse guard; a
test pointing OLLAMA_HOST at its own server is allowed; httpx (via Client, because
conftest stubs module-level `httpx.get`) and urllib connections to the ambient host are
each caught. origin/main had not moved (3498cb2), so no rebase was needed.
