# 2026-09-29 - GH-905: drop the retired cloud qwen rung, move the math model local

## Problem (measured 2026-09-29, main@bda9660)

Ollama Cloud retired `qwen3.5:cloud` on 2026-09-25 (410 Gone on every call);
`ollama list` / `/api/tags` still list it. Two non-judge defaults still used it:

1. `PROFILE_QWEN_CLOUD` (cloud OCR rung). `engines/qwen.py::cloud_model_available()`
   gated it on `check_ollama_model` (an `ollama list` LISTING), so the rung was emitted
   and attempted on every escalated page. On an 11-page run it was reached 13 times in
   11/22 runs, each failing instantly (`[qwen] CLI exited 1: ... '410 Gone'`).
2. `PipelineConfig.math_model = "qwen3.5:cloud"`: every corrupt-font equation-crop
   recovery failed.

The 410 appeared on the console only. No page JSON, audit_log, manifest or metadata had it.

## What changed

1. **`math_model` default** -> named constant `core/config.py::DEFAULT_MATH_MODEL =
   "qwen3-vl:30b-a3b-instruct"` (local, free, June benchmark: flawless LaTeX). Policy
   check: `_corrupt_math_model_disabled_reason` gates on the substring "cloud", so the
   local default runs under `--strict-local` / `--max-cost-per-page 0`; an explicit
   `--math-model <x>:cloud` is still accepted and still gated (tests pin both). The
   `--clean-equation-model` path is untouched (it already defaulted local, and its own
   cloud policy gate at orchestrator ~15974 is unchanged). CLI help, config and
   orchestrator comments, README and `docs/MODELS.md` updated; `qwen3.5:cloud` rows are
   marked historical.
2. **Default ladder** is now local qwen -> marker -> gemini.
   `_available_engines_for_agentic` no longer emits `PROFILE_QWEN_CLOUD` and no longer
   calls `cloud_model_available`. No replacement cloud model is picked (none is measured).
   **Explicit opt-in that already exists and still works:** `--qwen-model <tag>:cloud`
   (or `qwen_model:` in YAML) pins the model on the local rung; `resolve_qwen_intent`
   passes it verbatim. I did not add a new opt-in mechanism.
   **`PROFILE_QWEN_CLOUD` is kept, model string unchanged** (`"qwen3.5:cloud"`), with a
   HISTORICAL comment. Reason: `profile_by_id("qwen-cloud")` must still resolve
   pre-retirement manifests/sidecars (replay, resume, provenance) and `profile_by_model`
   still prices a historical `qwen3.5:cloud` judge name. It cannot reach the default
   ladder any more, and if a caller names it anyway the probe below reads it as
   unavailable. Changing the string would have made old manifests resolve to a model
   they never ran.
3. **Cloud availability check is a real generation.** `cloud_model_available` now uses
   `probe_model_generation` (1-token, `think:false`, `run_killable`-bounded, spawn-free
   reachability pre-check first). #906's probe body moved to a neutral module
   (`socr/core/ollama_utils.py`: `host_reachable`, `probe_generate`,
   `probe_failure_reason`, `probe_model_generation`, `PROBE_THINK`,
   `DEFAULT_PROBE_TIMEOUT_SEC`, `CONNECT_PROBE_TIMEOUT_SEC`), because
   `test_package_layering.py` forbids importing `_`-prefixed names across packages and
   `engines -> judge` would need `_probe_generate`. No allowlist entry added.
   `OllamaVisionJudge.is_available` now delegates to `probe_model_generation`, passing
   its own module-level `_host_reachable` / `run_killable` so #906's tests that patch
   those names on `socr.judge.ollama_judge` keep working unchanged.
   `DEFAULT_JUDGE_TIMEOUT_SEC` now equals `DEFAULT_PROBE_TIMEOUT_SEC` (same 120.0, the
   46s cold-load rationale moved with it). `check_ollama_model`'s docstring now says it
   is a listing and must not gate a `:cloud` tag.
4. **No-silent-failure.** The mechanism that should have recorded it already exists: the
   manifest journal (`manifest.py` builds each entry's `reason` from `skip_reason`, then
   `judge_reason`, then `failure_mode`), and the agentic loop stamps `skip_reason =
   att.reason` on every unaccepted, textless attempt. It missed the 410 because a failing
   CLI does not raise: `base.py` returns a `PageOutput(status=ERROR, error="CLI exited
   1: ...")`, the judge answers only `"empty/error output"`, and that generic string was
   the whole `reason`; the provider's `error` was dropped. Fix in the loop that stamps
   `skip_reason` (`orchestrator.py`, `_phase_agentic`): append the output's own `error`
   to `skip_reason` when the attempt is an ERROR with no text. No new persisted record
   kind, so no new emit site and no resume implication (journal entries are recomputed
   from `ps.attempts`).

## Files

`src/socr/core/ollama_utils.py`, `src/socr/judge/ollama_judge.py`,
`src/socr/engines/qwen.py`, `src/socr/core/config.py`, `src/socr/core/providers.py`,
`src/socr/pipeline/orchestrator.py`, `src/socr/cli.py`, `README.md`, `docs/MODELS.md`,
`tests/test_gh905_retired_cloud_model.py` (new), `tests/test_b2_routing.py`,
`tests/test_equation_latex.py`, `tests/test_orchestrator.py`.

## Tests

- New `tests/test_gh905_retired_cloud_model.py` (12): math_model default and policy
  (local passes strict-local, explicit cloud still gated), CLI help, 410 on the probe ->
  unavailable while the listing says present, 200 -> available, difference pin (only the
  generation status varies), unreachable host -> unavailable with no generation,
  `PROFILE_QWEN_CLOUD` still resolves by id, journal carries the CLI error text (plus a
  difference pin: only the error text varies, the journal reason varies with it).
- `test_b2_routing.py::TestCloudRungReachable` rewritten: the ladder is identical whether
  the cloud probe says yes or no and never contains `qwen-cloud`; the probe is never
  consulted. `test_equation_latex.py`: the "math_model must not pollute the clean-equation
  path" guard now sets a cloud `math_model` explicitly (the default is local now).
- Hermetic: run with `OLLAMA_HOST=http://127.0.0.1:9`. The journal tests pin the ladder,
  judge, crop VLM and engines (#841) and assert a difference, not a machine-measured tuple.

## Mutation checks

Copies at `/tmp/socr-mut-905-{a,b,c}` (src + tests + pyproject; canary asserting
`socr.__file__` inside the copy passed; anchors asserted `count == 1` uncapped; copies
deleted):

- (a) re-add the cloud rung to the default ladder: killed by
  `test_default_ladder_has_no_cloud_qwen_rung_even_if_the_probe_says_yes` and
  `test_no_cloud_rung_when_local_model_absent`.
- (b) revert `cloud_model_available` to the listing: killed by
  `test_a_410_on_the_cloud_probe_is_unavailable_even_though_it_is_listed` and
  `test_cloud_probe_difference_pin_only_the_generation_status_varies`.
- (c) drop the failure persistence: killed by
  `test_a_rung_cli_failure_is_recorded_in_the_manifest_journal` and
  `test_journal_reason_differs_exactly_by_the_provider_error`.

## Follow-ups / deviations

- Timeout attempts (`reason == "provider timeout"`, output error "qwen: timed out after
  Ns") now also carry that error in the journal `reason` ("provider timeout: qwen: timed
  out after Ns"). Intentional (same rule), but any external reader matching the journal
  reason by equality would see the change.
- `is_cloud_qwen` / `execution_overrides` / the cloud branch of `_run_engine_on_pages`
  are now reachable only by a caller that hands the cloud profile to the runner
  directly (tests do). Left in place rather than deleted so the profile stays coherent;
  a later cleanup can remove them once nothing needs the historical profile.
- `test_orchestrator.py::test_agentic_corrupt_math_remote_model_policy_is_visible` relied
  on the cloud default to exercise the remote-model policy; it now names a `:cloud`
  `math_model` explicitly.

## Verification

- Full suite with `OLLAMA_HOST=http://127.0.0.1:9`, from the worktree root (pytest's
  `pythonpath=["src"]` resolves `socr` to the worktree; a canary confirmed it):
  first run 3 failed / 5842 passed / 4 xfailed (the `test_orchestrator` test above);
  after the fix the full suite is **5845 passed, 4 xfailed** (439s).
- `uvx ruff@0.16.0 format --check .`: clean.

## Round 2 (review: ACCEPT-WITH-FIXES)

1. **Privacy gap fixed.** A pinned `--qwen-model x:cloud` rode on `PROFILE_QWEN_LOCAL`
   (tier local, $0), so the strict-local tier filter and the zero-cap filter (both read the
   profile) let it reach Ollama Cloud. New in `core/providers.py`: `is_cloud_model` (the one
   "cloud" casefold predicate, now used by the corrupt-math, clean-equation, judge-ladder
   and equation-lane sites instead of four inline copies) and `cloud_pinned_qwen_refusal`
   (resolves the model that will actually run through `resolve_qwen_intent`, refuses under
   strict-local or `zero_cap_pinned_forbids_cloud`). `_phase_agentic` calls
   `_refuse_cloud_pinned_qwen_rung` right after building `available`; the refusal is
   surfaced as a console line, a log warning and a document-level (page 0)
   `qwen_cloud_pin_refused` audit event, the surface `judge_degraded_to_heuristic` uses.
   Scope: the agentic ladder. The non-agentic single-engine path does not build this
   ladder and is unchanged.
2. **Journal error capped/deduped.** `_skip_reason_with_provider_error`: flattened to one
   line, capped at named `_SKIP_REASON_ERROR_MAX_CHARS = 500` (the bound `engines/base.py`
   already applies to stderr), not appended when the reason is `REASON_PROVIDER_TIMEOUT`
   or already contains the error. This supersedes the round-1 note about timeout reasons
   changing: they are unchanged now.
3. `probe_generate` docstring fixed.

Tests added (7, in `test_gh905_retired_cloud_model.py`): strict on/off difference pin on
the predicate and on the real `_phase_agentic` (rung calls > 0 vs 0), typed-vs-defaulted
zero cap, local/unpinned pin unaffected, only the qwen rung dropped plus audit event,
cap+flatten, timeout not duplicated.

Mutations (copies in /tmp, canary passed, anchors count==1): gate removed -> 2 tests fail
(`test_refusal_drops_only_the_qwen_rung...`, `test_strict_local_stops_pages_reaching...`);
cap removed -> `test_a_long_multiline_provider_error_is_flattened_and_capped` fails.

## Round 3 (cubic on PR #911)

Test count for `test_gh905_retired_cloud_model.py`: 12 (round 1) + 7 (round 2) + 5 (round 3)
= 24 (the round-2 line above said 8; it was 7).

- **P1 (fixed).** A cloud pin was dropped on a host with no local qwen pull because
  `QwenEngine.is_available()` probes the local build, before any policy or probe ran.
  `engines/qwen.py`: `pinned_cloud_qwen_model(config)` (the cloud model the qwen rung
  would run on a local/auto backend) and `pinned_cloud_model_available(config)` (a
  `probe_model_generation` on THAT model, reason returned). `_available_engines_for_agentic`
  now, for QWEN with a cloud pin: policy first (`cloud_pinned_qwen_refusal` non-empty ->
  rung kept in `available` so `_refuse_cloud_pinned_qwen_rung` drops and surfaces it,
  never probed); otherwise probe the pinned model; a 410 or any failure leaves the rung
  out and stores the reason, which `_refuse_cloud_pinned_qwen_rung` surfaces (console,
  log, and a page-0 `qwen_cloud_pin_unavailable` audit event). Local pins and no pin keep
  the local probe.
- **P2 (fixed).** `host_reachable` now runs `getaddrinfo` in a daemon thread joined with
  the budget and connects with only the time left, so one total deadline covers resolve +
  connect, no process spawned. Reuses `CONNECT_PROBE_TIMEOUT_SEC`; no new constant. Test
  stubs `getaddrinfo` to sleep 5s and asserts `host_reachable(..., timeout=0.2)` returns
  False in under 2s. Also fixed a stray "`the caller`" in that docstring.
- **P3 (fixed).** Reworded the `test_default_config_uses_local_instruct_model` docstring;
  README routing sentence is now native -> local qwen -> marker -> gemini;
  `--clean-equation-model` / `--qwen-model` help say "any name containing 'cloud'"
  (matching `is_cloud_model`); `TestCloudRungReachable` docstring rewritten.
- **Tests (5):** cloud pin present when local absent and its own probe OK; 410 -> absent,
  reason surfaced, local probe held constant (difference pin), event emitted;
  strict_local -> refused without probing; local/no pin still use the local probe; slow
  resolver bounded.
- **Mutations** (copies in /tmp, canary passed, anchors count==1): revert the pin branch
  to local-only availability -> 3 tests fail; revert `host_reachable` to plain
  `create_connection` -> the slow-resolver test fails.
- **#910 hermeticity.** The new tests patch `get_engine` and `probe_model_generation`; none
  reaches `check_ollama_model`. Proven by running the file with a temporary autouse guard
  that raises on any `subprocess.run` of the ollama CLI (24 passed, guard removed after).
  The full suite was run with the DEFAULT OLLAMA_HOST, per the #910 instruction.
