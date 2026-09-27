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
  cheap GET). Unchanged in round 2.
- All #901-era real-page measurements taken since 2026-09-25 were made
  against the broken judge and still need to be redone — out of scope here.
- If the owner later wants `think: false` (or off) measured for the table
  judge / cell adjudicator, that is a new ticket with its own accuracy
  comparison — not folded into this one per the round-2 ruling.
