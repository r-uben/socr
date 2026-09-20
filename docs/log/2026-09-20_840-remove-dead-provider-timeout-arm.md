# 2026-09-20 — GH-840: delete the phantom `agentic_provider_timeout` override arm

## What changed

- `src/socr/pipeline/orchestrator.py`: `provider_timeout` in `_phase_agentic` no
  longer reads `getattr(self.config, "agentic_provider_timeout", None)`. It is now
  a plain `provider_timeout = DEFAULT_PROVIDER_TIMEOUTS`, with a comment pointing
  at #840 so the arm is not silently re-added. `agentic_provider_timeout` was never
  a declared `PipelineConfig` field, had no CLI flag or config-file key, and the
  `getattr` could only ever return `None` on a shipped run — the override was
  unreachable in production (flagged, not fixed, by the #797 decision log).
- `tests/test_gh797_agentic_soft_deadline_wiring.py`: the two wiring tests used
  `pipe.config.agentic_provider_timeout = ...` as their only lever to vary the
  bound reaching the engine seam. Re-pointed at a lever that still exists:
  `_phase_agentic` does `from socr.pipeline.agentic import
  DEFAULT_PROVIDER_TIMEOUTS` fresh on every call, so the tests now
  `monkeypatch.setattr(agentic_module, "DEFAULT_PROVIDER_TIMEOUTS", ...)`, scoped
  to each leg via `monkeypatch.context()` so the unpatched ("default") leg still
  observes the real calibrated table. Each override dict is
  `{**DEFAULT_PROVIDER_TIMEOUTS, _ENGINE: sentinel}` so the other engines' entries
  stay intact. All three legs still hand the engine seam three DIFFERENT bounds;
  the guard was not weakened to a single-leg assertion.

Grepped repo-wide for `agentic_provider_timeout`: the only remaining hits are
dated historical decision logs (`docs/plans/agentic-local-first/logs/2026-06-14_C1*.md`,
`docs/log/2026-09-20_797-agentic-soft-deadline-wiring.md`) that correctly record
what was true when written; left untouched.

## Guard proof (mandatory)

Copied `src`, `tests`, `pyproject.toml` to `/tmp/mut840` (outside the repo), with a
`conftest.py` canary asserting `socr.__file__` resolves inside `/tmp/mut840`.
Asserted an UNCAPPED `src.count("subprocess_timeout_sec=(\n") == 1` across all of
`src/` before editing (count was 1). Deleted the `subprocess_timeout_sec=(...)`
keyword argument from the `run_provider` closure's `_run_engine_on_pages` call
(the mutation the GH-797 test docstring names).

Result: the rewritten wiring tests FAIL in the mutant, both collapsing to a
single observed value:

```
FAILED tests/test_gh797_agentic_soft_deadline_wiring.py::test_the_configured_soft_deadline_tracks_through_to_the_engine
FAILED tests/test_gh797_agentic_soft_deadline_wiring.py::test_the_agentic_loop_passes_the_deadline_into_the_engine_runner
AssertionError: _phase_agentic's run_provider closure is not passing subprocess_timeout_sec down: three differently-configured legs produced {None}
assert 1 == 3
 +  where 1 = len({None})
2 failed, 1 passed, 5 warnings in 0.45s
```

(The third, control test — which drives `_run_engine_on_pages` directly with an
explicit `subprocess_timeout_sec=` argument, unrelated to `run_provider` — still
passes, as expected.) Mutant copy and canary deleted after the run.

## Tests

- `tests/test_gh797_agentic_soft_deadline_wiring.py` (worktree, unmutated) → 3 passed
- `tests/test_gh797_agentic_soft_deadline_wiring.py tests/test_gh172_cli_subprocess_timeout.py` → 5 passed
- Whole suite (`PYTHONPATH=<worktree>/src pytest -q`) → 5722 passed, 4 xfailed
- Collected-test reconciliation against `origin/main` (87cc94b, same commit this
  branch was cut from, checked via a separate worktree pinned to that SHA):
  both collect exactly 5726 tests (5722 + 4 xfailed here; no test added or
  removed by this change, only two existing bodies edited).
- `uvx ruff@0.16.0 format --check .` → clean (one file reformatted during work,
  re-checked clean after)

## Work location

Done in a scratch worktree at `/tmp/wt-840` (branch
`fix/840-remove-dead-provider-timeout-arm`, cut from `origin/main`), per the
ticket's instruction not to touch the shared `~/repos/tools/socr` checkout, which
sits on an unrelated branch. Not pushed; no PR opened.
