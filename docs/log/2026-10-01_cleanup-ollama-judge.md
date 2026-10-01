# 2026-10-01 Behaviour-preserving cleanup: ollama utils, judge, providers

Branch `chore/cleanup-ollama-judge` from origin/main c0c67e4. Source review:
`~/.local/state/socr-housekeeping/cleanup/clean-pipeline.md` (item numbers below).
No behaviour change intended; nothing here alters outputs, events, error strings or routing.

## Done

- **2** `_call_within(fn, timeout) -> (finished, value_or_exc)` in `ollama_utils.py`,
  shared by `_get_tags` (re-raises the exception, `None` on timeout, as before) and
  `_resolve_within` (`OSError` -> `None`, timeout -> `None`). Checked by hand: ConnectError
  propagates, overrun -> `None`, success returns the value. One micro-difference: a
  non-`OSError` raised by `getaddrinfo` (e.g. `ValueError`) still returns `None`, but no longer
  prints a thread traceback to stderr.
- **3** `_THINK` deleted; `ollama_judge._post_generate` uses `ollama_utils.PROBE_THINK`
  (both `False`); the GH-903 rationale now lives on that constant.
- **4** orphaned OLLAMA_HOST comment block deleted.
- **5** `is_available` and `judge()` docstrings cut to what the code does; the nonexistent
  `_probe_failure_reason` reference is gone.
- **6** `zero_cap_pinned_forbids_cloud` docstring no longer claims a default `qwen-cloud` rung.
- **7 (partial)** `_resolve_judge_model`: `_cached()` closure replaces the two duplicated cache
  checks; `_probe_judge_candidate(model) -> (available, "<model>: <reason>")` is used by the
  explicit-override branch (exceptions propagate, as before) and the ladder (exceptions still
  caught by the ladder's own try/except with its own string). History paragraphs cut to one line.
  `OllamaVisionJudge` is still imported lazily inside the helper, so patching
  `socr.judge.ollama_judge.OllamaVisionJudge` keeps working.
- **11** `_probe_cloud(model)` in `qwen.py`; `cloud_model_available` kept.
- **13** stale comments/docstrings trimmed in `ollama_utils.py` (CONNECT_PROBE_TIMEOUT_SEC,
  `host_reachable`, `probe_generate`, `probe_failure_reason`).
- **14** `_TIMEOUT_MSG` and `_listed_model_names(resp)`; same exception types raised inside
  the existing `try`, so each `except` arm maps identically.
- **15 (partial)** type hints in `ollama_utils.py` (`reachable`, `runner`, `list[tuple]`) and
  `providers.py` (`PipelineConfig` under `TYPE_CHECKING`, one `type: ignore` removed).
- **16** `noqa: F401` and the unused `CONNECT_PROBE_TIMEOUT_SEC` import dropped; no importer in
  `src/` or `tests/`.
- **18** `_surface_doc_event` used by `_refuse_cloud_pinned_qwen_rung` only. Log lines render
  identically (`agentic: <label>: <reason>`), console text and `AuditEvent` fields identical.
  `_build_page_judge` / `_report_unservable_engines` emissions untouched (not identical).
- **19** comment trims in `_build_page_judge` only; fitz page cache untouched.
- **20** providers.py module docstring grammar and GH-905 note, `profile_by_model` docstring,
  `import os` moved to module scope.

## Skipped

- **7, `_UNRESOLVED = object()` sentinel**: tests assign `pipe._judge_model_cache = False` to mean
  "unresolved" (`tests/test_gh903_judge_model_retired.py:512`,
  `tests/test_gh154_remote_call_entry_points.py:333,345,364`). With an `object()` sentinel,
  `False is not _UNRESOLVED`, so `False` would be returned as a cached model: a behaviour change
  for those tests, and the #903 file must stay unchanged. Kept `False`.
- **15, orchestrator type hints** (`plan`, `NativeTableFirstWork`): they sit on
  `_plan_native_table_first`, which collides with PR #926.
- **1, 8, 9, 10, 12 (behaviour part), 17, mixin split**: out of scope as instructed.

## Tests

See the commit message / final report for the counts. Tests touched: none.
