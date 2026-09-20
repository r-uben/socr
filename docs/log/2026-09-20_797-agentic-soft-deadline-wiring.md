# 2026-09-20 — GH-797: pin the CLI soft deadline at the shipping boundary

Test-only ticket. Leftover from PR #796 (parent #172): the per-provider soft
deadline was wired in production but unguarded by any test.

## What changed

- **New:** `tests/test_gh797_agentic_soft_deadline_wiring.py` (3 tests).
- No source file touched. The ticket's hypothesis held: the wiring is correct,
  only the guard was missing.

## The hole, confirmed by mutation

`grep -rln subprocess_timeout_sec tests/` returned nothing before this commit.
Two production hops carry the deadline, and both were unpinned:

1. `_phase_agentic`'s `run_provider` closure →
   `_run_engine_on_pages(..., subprocess_timeout_sec=provider_timeout.get(...))`
2. `_run_engine_on_pages` → `engine.process_pages(..., subprocess_timeout=...)`

Mutant A (delete hop 1) and mutant B (delete hop 2) were each built from copies of
`src/` and `tests/` outside the repo, with a `pytest_configure` canary asserting
`os.path.realpath(socr.__file__).startswith("/private/tmp/mut797…/")`, and an
UNCAPPED `src.count(anchor) == 1` assertion before the edit.

- Mutant A: the two new wiring tests fail; the new argument-control test passes;
  `test_gh172_cli_subprocess_timeout.py`, `test_gh159_provider_identity.py`,
  `test_r174b_orchestrator_agentic_lane.py`, `test_pp2_agentic_fuse.py` — 30
  pre-existing tests — all stay green.
- Mutant B: the end-to-end test and the argument-control test fail; the
  runner-level test passes (that hop is intact); `test_gh172…` stays green.

## Pin a DIFFERENCE, not a value

Each test runs the same production path two or three times in one process,
changing ONLY the configured per-provider timeout, and asserts the downstream
observation tracks it. No page status, `audit_passed`, failure mode or document
status is asserted anywhere in the file, so the CI-vs-workstation provider
divergence that reverted #253 cannot reach these assertions. The two sentinel
values are arbitrary and documented as such; the third leg's expectation is read
from `DEFAULT_PROVIDER_TIMEOUTS`, never repeated as a literal.

## Hermeticity finding (worth carrying forward)

`_available_engines_for_agentic` + `_resolve_judge_model` are not sufficient on a
`process()`-level test. `PipelineConfig.primary_engine` defaults to `AUTO`, and
`process()` then calls `resolve_auto_engine()`, which builds engines from the
registry **directly — not through `get_engine`** — and shells out to the `ollama`
CLI. Patching `orch.get_engine` does not intercept it. Measured ~6s per
`process()` call against an unreachable ollama host; pinning `primary_engine`
took the file from 21s to 0.4s. `tests/test_gh159_provider_identity.py`'s
`test_the_agentic_loop_hands_the_profile_down_to_the_engine_runner` has the same
live probe (pre-existing; not touched here).

Verified hermetic three ways: normal env, `OLLAMA_HOST` pointed at a dead port,
and `env -i` with `ollama` off `PATH`. A `socket.socket.connect`/`connect_ex`/
`getaddrinfo` trace recorded zero network calls.

## Deviation from the ticket

`agentic_provider_timeout` is **not a declared `PipelineConfig` field** — the
orchestrator reads it with `getattr(self.config, …, None)`, so on shipping configs
the override arm is unreachable and `DEFAULT_PROVIDER_TIMEOUTS` always wins. The
tests set the attribute on the instance to exercise the override arm. Flagged, not
fixed: adding the field is a source change and this ticket is test-only.

## Tests

- `pytest tests/test_gh797_agentic_soft_deadline_wiring.py -q` → 3 passed
- `+ tests/test_gh172_cli_subprocess_timeout.py` → 5 passed
- 10-file orchestrator/agentic selection → 156 passed
- `uvx ruff@0.16.0 format --check .` → 749 files already formatted
