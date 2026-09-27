# 2026-09-26 — GH-903: retired page judge, generation-based availability probe

## What changed

Ollama Cloud retired `qwen3.5:cloud` on 2026-09-25 (`POST /api/generate` -> 410
Gone, `error: "qwen3.5:397b was retired at ..."`), but `/api/tags` kept listing
it, so `_resolve_judge_model`'s tags-based probe kept selecting a judge that
raised on every call. Measured across #901's rotated-page runs and the pinned
archive re-OCR — 296 log/output files mention the 410.

Two rounds:

### Round 1 (commit `1ae4e36`)

1. `think: false` on every judge request (page judge AND table judge/cell
   adjudicator).
2. New default page-judge candidate: `qwen3.8:27b` (local), named constant.
3. Generation-based availability probe replacing `/api/tags`, with a fixed
   10s budget.

Review verdict: ACCEPT-WITH-FIXES. A real `socr process` run on the owner's
Mac found round 1's probe budget itself wrong, and its scope too wide.

### Round 2 (this commit, on top of `1ae4e36`)

**1. Cold start.** With `qwen3.8:27b` unloaded (not resident in GPU memory),
a real run printed `VLM judge unavailable (... qwen3-vl:8b: HTTP 404 ...) ->
heuristic judge` — the FIRST candidate's 10s probe timed out on a real cold
load (measured ~46s), which fell through the whole ladder (the other two
candidates were never pulled on that box) and memoized `None` for the entire
run, and for a whole `socr batch`. Fix:
- The probe's timeout is now `self.timeout` — the SAME budget the judge call
  itself already uses (`OllamaVisionJudge.timeout`, default
  `DEFAULT_JUDGE_TIMEOUT_SEC = 120.0`, a new named constant extracted from
  the previously-inline `120.0` default) — not a separate, shorter one. No
  new number was invented; the existing judge-call budget is reused.
  Documented with the 46s cold-load measurement, not the misleading
  `check_ollama_model` 10s reference round 1 cited (that function checks
  engine pulls via a 10s `ollama list` subprocess — unrelated timeout budget,
  removed from the comment).
- An HTTP error status (`httpx.HTTPStatusError`, e.g. 410/404) is DEFINITIVE
  unavailability — the daemon answered and said no.
- A timeout (`httpx.TimeoutException`) is NOT proof of unavailability — it
  only proves the probe's budget was exceeded — and is reported distinctly
  (`"timed out after Ns"` vs `"HTTP {status}: {body}"`), so an operator (and
  the tests) can tell "this candidate is gone" apart from "this candidate
  needed longer than `timeout` to warm up".

**2. Surfacing.** `_resolve_judge_model` now records EVERY candidate's
failure reason, not just the last one tried (`self._judge_unavailable_reason
= "; ".join(reasons)`, e.g. `"qwen3.8:27b: timed out after 120s;
minicpm-v:8b: HTTP 404: ...; qwen3-vl:8b: HTTP 404: ..."`). The
`judge_degraded_to_heuristic` audit event's `data["unavailable_reason"]`
carries the same joined string.

**3. Scope.** `think: false` in `table_rung_ollama.py::_build_payload` is
REVERTED. The table judge ladder (`glm-5.3-flash:cloud`, rung 1) and the P1
cell-transcription adjudicator (`kimi-k2.6:cloud`,
`TABLE_JUDGE_ADJUDICATOR_MODEL_DEFAULT`) are cloud thinking models whose
accuracy was measured (the GH-356 bake-off; the P1 adjudicator design,
`docs/log/2026-09-02_gh359-ladder-terminals-design.md`) WITH reasoning on.
Turning it off is an unmeasured accuracy change this ticket does not make —
**the table judge / cell adjudicator are deliberately left untouched,
pending a separate accuracy-measured ticket if the owner wants to pursue
it.** `think: false` now applies ONLY to the page judge
(`socr.judge.ollama_judge`), whose retired/replacement candidates this
ticket is actually about.

## Files

- `src/socr/judge/ollama_judge.py` — `think: false` on `_post_generate` and
  the probe (page judge only); `is_available()` uses `self.timeout` (not a
  separate constant) as its budget; `DEFAULT_JUDGE_TIMEOUT_SEC = 120.0`
  extracted as a named default; `_probe_failure_reason` distinguishes an
  HTTP status (definitive) from a timeout (inconclusive, reports duration);
  removed the round-1 `probe_timeout` constructor param and
  `PROBE_TIMEOUT_SEC` constant entirely (no longer a separate budget).
- `src/socr/judge/table_rung_ollama.py` — `think: false` REVERTED in
  `_build_payload`; docstring explains why (owner ruling, round 2).
- `src/socr/pipeline/orchestrator.py` — `_resolve_judge_model` collects
  every candidate's `unavailable_reason` into a list and joins them (was:
  kept only the last).
- `tests/test_gh903_judge_model_retired.py` — rewritten: cold-load tests
  (short budget times out, full judge-timeout budget succeeds, the
  difference pin between them), an HTTP-error-is-definitive-regardless-of-
  budget test, every-candidate-reason tests (ladder + audit event), and a
  guard that the table judge payload does NOT send `think`.
- `docs/log/2026-09-26_903-judge-model-retired.md` — this file, rewritten
  for round 2.

(`docs/MODELS.md` and the other round-1 files are unchanged in round 2.)

## Verification

- Judge-scope subset: `~/venvs/socr/bin/pytest
  tests/test_judge_wiring_gh133.py tests/test_gh903_judge_model_retired.py
  tests/test_gh154_remote_call_entry_points.py tests/test_table_rung_ollama.py
  tests/test_gh873_judge_vllm_backend.py tests/test_gh172_judge_killable.py
  -q` — 143 passed.
- Full suite: `~/venvs/socr/bin/pytest tests/ -q` — **5823 passed, 4
  xfailed**, 358s (round 1 was 5819 passed; +4 from the new round-2 tests
  net of the one removed think:false-on-table-judge test).
- `uvx ruff@0.16.0 format --check .` — clean.

## Mutation check (per CLAUDE.md), round 2

All in copies at `/tmp/socr-mut-903-c` (`src` + `tests` + `pyproject.toml`,
`socr.__file__` canary confirmed inside the copy before each test run;
copies deleted after):

- **(A) drop `think: false`** from `_post_generate` (page judge): killed —
  `test_page_judge_generate_call_sends_think_false` failed. 1 failed, 12
  passed.
- **(B) revert the probe to tags-only**: killed — 14 of 29 tests across
  `test_gh903_judge_model_retired.py` + `test_judge_wiring_gh133.py` failed.
- **(C) collapse "every candidate's reason" back to last-reason-only**
  (`self._judge_unavailable_reason = reasons[-1] if reasons else ""`):
  killed — exactly the two tests written for this
  (`test_every_candidates_reason_is_recorded_not_just_the_last`,
  `test_degradation_audit_event_carries_every_candidates_reason`) failed, 11
  passed.
- **(D) reintroduce `think: false`** in `table_rung_ollama._build_payload`
  (scope reversion): killed —
  `test_table_judge_chat_payload_does_not_send_think` failed. 1 failed, 12
  passed.

## Deviations / follow-ups

- Round 1's `_build_page_judge` no longer re-probes `is_available()` a
  second time after `_resolve_judge_model` already verified the same model
  (removed as redundant, since a probe is now a real generation call, not a
  cheap GET). Unchanged in round 2 and round 3.
- All #901-era real-page measurements taken since 2026-09-25 were made
  against the broken judge and still need to be redone — out of scope here.
- If the owner later wants `think: false` (or off) measured for the table
  judge / cell adjudicator, that is a new ticket with its own accuracy
  comparison — not folded into this one per the round-2 ruling.

## Round 3 (this commit, on top of `54ef144`): cubic findings on PR #906

PR #906 (branch `fix/903-judge-model-retired`, pushed at `54ef144`, CI green)
got 5 cubic findings; the coordinator verified all 5 as valid.

**P2-a (real regression from round 1).** An explicit `--judge-model` bypasses
the candidate LADDER in `_resolve_judge_model` (an operator named the exact
model; the ladder must never substitute a different one on failure) but,
before this round, it never probed AT ALL -- it returned the string verbatim.
Round 1 removed `_build_page_judge`'s own second `is_available()` call as a
redundant re-probe of an already-verified ladder candidate; that reasoning
never covered the explicit-override branch, which had NO probe to be
redundant with. The net effect: an unpulled or retired explicit override
started being treated as an active judge and failing on every page, instead
of degrading to heuristics with a reason. Fix: the explicit-override branch
now probes through the same `OllamaVisionJudge(model=...).is_available()`,
memoized on the same `_judge_model_cache` (one probe per run, not one per
page via `_run_fingerprint`). The vLLM pair (#873) is unaffected -- it
already had, and keeps, its own separate probe/memoization.

**P2-b.** `is_available()`'s `httpx` `timeout=` is a per-READ inactivity
timeout, not a total wall-clock deadline -- the same gap #172 closed for
`judge()` itself. A peer that trickles a byte before every read interval
never trips it, so the probe (and everything that consults
`_resolve_judge_model`, i.e. the whole per-page loop) could wedge
indefinitely. Fix: `is_available()` now runs its probe body
(`_probe_generate`, a new top-level, picklable function) through
`run_killable`/`CallSpec` -- the exact same killable-process boundary
`judge()` uses -- with the same budget constant (`self.timeout`, unchanged
from round 2). `_probe_generate` classifies HTTP-status/connection failures
itself and returns a plain `{"available": bool, "reason": str}` dict (
`run_killable` collapses any child exception crossing the pipe into a bare
`RuntimeError` string, which would lose the response body
`_probe_failure_reason` needs -- classifying inside the child, before the
pipe, keeps it); it re-raises a genuine `httpx.TimeoutException`, which
`run_killable` reclassifies as `KillableTimeoutError` (a `TimeoutError`
subclass) exactly as it does for `judge()`, and `is_available()` reports that
as `"timed out after Ns"`, inconclusive, per round 2's classification.

**P2-c.** Added `test_judge_model_default_is_qwen3_8_27b`, pinning
`UnifiedPipeline.JUDGE_MODEL_DEFAULT == "qwen3.8:27b"` and
`_JUDGE_MODEL_CANDIDATES[0] == JUDGE_MODEL_DEFAULT` by NAME, not just
position.

**P3-a/P3-b.** The `_judge_unavailable_reason` class-level comment
(orchestrator.py ~786) and the `_resolve_judge_model` docstring (~10096)
both still described round-1/round-2 semantics (last-candidate-only reason;
`qwen3.5:cloud` as the first candidate). Both updated to describe the
current (round 2/3) behaviour, with the superseded claims kept as explicitly
labelled history (GH-154 round 5 predates this ticket).

### Files (round 3)

- `src/socr/judge/ollama_judge.py` -- new top-level `_probe_generate`
  (picklable probe body, classifies HTTP/connection failures itself,
  re-raises a real timeout); `is_available()` rewritten to run it through
  `run_killable`; `_probe_failure_reason` no longer takes a `timeout` arg or
  classifies `httpx.TimeoutException` (that case never reaches it now --
  `run_killable`'s own `TimeoutError` is caught first, in `is_available()`).
- `src/socr/pipeline/orchestrator.py` -- explicit-override branch in
  `_resolve_judge_model` now probes and memoizes; stale comments (P3-a,
  P3-b) corrected.
- `tests/test_judge_wiring_gh133.py` -- added an autouse
  `_probe_run_killable_is_synchronous` fixture (fakes `run_killable` to call
  the probe body in-process, reclassifying a real `httpx.TimeoutException` as
  `KillableTimeoutError` the same way the real boundary would) so every
  existing `httpx.post`-stubbing test in this file keeps working un-changed;
  rewrote `test_explicit_judge_model_is_never_discarded` (now
  `test_explicit_judge_model_bypasses_the_ladder_not_the_probe`) for the new
  P2-a behaviour.
- `tests/test_gh903_judge_model_retired.py` -- same autouse fixture; added
  P2-a tests (410 degrades with reason, healthy override selected, probed
  once) and the P2-c default-pinning test.
- `tests/test_gh172_judge_killable.py` -- new
  `test_probe_is_bounded_and_typed_against_a_trickling_peer`, reusing the
  existing `trickle_server` fixture, proving the REAL `run_killable`
  boundary (no in-process fake) bounds `is_available()` the same way it
  bounds `judge()`; corrected a stale comment that `is_available()` still
  used `httpx.get`.

### Verification (round 3)

- Focused run: `~/venvs/socr/bin/pytest tests/test_gh903_judge_model_retired.py
  tests/test_judge_wiring_gh133.py tests/test_gh172_judge_killable.py -q` --
  36 passed (includes the real trickling-server test, ~5s total).
- Broader judge-adjacent run: `tests/test_gh154_remote_call_entry_points.py
  tests/test_gh873_judge_vllm_backend.py tests/test_table_rung_ollama.py
  tests/test_gh172_killable_boundary.py tests/test_gh849_peer_timeout_no_cascade.py`
  -- 121 passed.
- Full suite: `~/venvs/socr/bin/pytest tests/ -q` -- **5828 passed, 4
  xfailed**, 0 failed (1899s; slower than round 2's 358s -- no root cause
  identified in the code touched here, all `_resolve_judge_model`-adjacent
  test files were audited and confirmed to patch either the pipeline method
  directly or the whole `OllamaVisionJudge`/`is_available`, so none of them
  newly cross the `run_killable` boundary for real; the machine is shared
  with other concurrent agent sessions per this repo's own operating note,
  which is the more likely explanation, but this was not proven).
- `uvx ruff@0.16.0 format --check .` -- clean.

### Mutation check (per CLAUDE.md), round 3

Copies at `/tmp/socr-mut-903-r3a`/`r3b` (`src` + `tests` + `pyproject.toml`,
`socr.__file__` canary confirmed inside each copy; both deleted after):

- **(1) remove the explicit-override probe** (revert to the round-1/2
  unconditional `return self.config.judge_model`): killed -- 3 tests failed
  (`test_explicit_override_that_410s_degrades_to_heuristic_with_reason`,
  `test_explicit_override_probe_is_memoized`,
  `test_explicit_judge_model_bypasses_the_ladder_not_the_probe`), 30 passed.
- **(2) call the probe body directly, without the `run_killable` deadline
  wrapper**: killed -- `test_probe_is_bounded_and_typed_against_a_trickling_peer`
  hung indefinitely (confirmed still running after 12s wall-clock against a
  peer that answers within ~5s when the fix is in place; killed by hand
  rather than waited out), while every other test in the same file (2
  tests) still passed -- proving the kill is specific to the deadline
  boundary, not a broken fixture.
- Round 1/round 2 mutants ((A) drop page-judge `think:false`, (B) revert
  probe to tags-only, (C) collapse per-candidate reasons to last-only, (D)
  reintroduce `think:false` on the table judge) were spot-re-checked (A)
  against the round-3 tree and still kill correctly; not exhaustively
  re-run, since none of the round-3 changes touch that code.
