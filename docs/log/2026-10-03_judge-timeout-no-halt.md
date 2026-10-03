# 2026-10-03 judge-timeout-no-halt (issue #987, branch fix/judge-timeout-no-halt)

## Problem

Measured in `~/.local/state/socr-housekeeping/withheld/` (11 re-OCR'd papers): 8 document halts
(`PARTIAL_SAVE_VLM_TIMEOUT`), every one preceded by a page-JUDGE timeout (`qwen3.8:27b`, 120 s),
never an OCR-rung timeout. About 57 table blocks lost to pages that got no model after the halt.

## Measurement (no models run)

- `_attempts_show_timeout` armed the halt on `judge_outcome == JUDGE_OUTCOME_TIMEOUT` (#713 r3) and
  on the substring "timeout", which a judge reason ("judge raised: page judge timeout ...") also
  contains. The canary (`probe_ollama_idle`) then ran on the OCR model.
- The judge (27B) and the OCR model (30B-A3B) take turns on one GPU. Ollama evicts a model when
  another needs the memory and the next request to the evicted one queues behind a reload.
  `docs/log/2026-09-16_221.md` already recorded a cold load of ~37.5 s against the 30 s canary
  default (`_CROP_DEADLINE_FLOOR_S`), and excused it because "the model is normally warm after a
  crop". That premise is false right after a judge call on a different model. So a 30 s canary
  cannot cover a cold load, and the failure follows a judge timeout by construction.

## Change

- `pipeline/orchestrator.py::_attempts_show_timeout`: attempts typed `JUDGE_OUTCOME_TIMEOUT` are
  skipped. A page whose OCR rung also timed out still arms the halt (that rung is its own attempt).
  The page itself still fails closed as before.
- `tables/extract.py`: `canary_deadline() = _CROP_DEADLINE_FLOOR_S + CANARY_LOAD_ALLOWANCE_S`, the
  allowance being `DEFAULT_READER_TIMEOUT_S` (120 s, the reader's existing read budget; no new
  number). Default `generation_timeout` of both canaries now uses it. A wedge never answers, so only
  the verdict on a real wedge is delayed, once.
- Halt surfacing unchanged.

## Tests

New `tests/test_gh987_judge_timeout_no_halt.py` (hermetic: ladder + `probe_ollama_idle` patched):
difference pin (judge timeout on p2 -> pages 3-4 run, no halt, canary not asked; provider timeout on
p2 -> halts after p2), mixed-attempt masking pin, cold-load default pin, slow-but-alive canary pin
(scaled constants). Updated to the new contract: `test_gh222_probe_host.py` (position test),
`test_gh713_round3_supersession_identity.py`, `test_gh221_generation_canary.py` (default timeout).

Mutations (external copy, `socr.__file__` canary asserted inside the copy):
- old predicate restored: 2 failed (difference pin, masking pin).
- `canary_deadline()` back to the floor: 2 failed (cold-load default, slow-but-alive).

## Notes

- #984's guard was not on this base; the full suite ran with `OLLAMA_HOST=http://127.0.0.1:1`.
- Not changed: a judge that is wedged still costs 120 s per page; this ticket only stops it halting
  the document.

## Review round 1 (Astra, P1): judge circuit breaker

Suppressing judge timeouts entirely let a wedged judge cost a full judge deadline on every remaining
page. Added: after a judge timeout `_judge_circuit_breaker` sends one 1-token `probe_model_generation`
to the judge model (timeout `canary_deadline()`, existing constants). Probe fails -> the judge chain
switches (`SwitchablePageJudge`) to the heuristic judge, the path a missing judge takes, with ONE
document-level `judge_wedged_degraded_to_heuristic` event. Probe passes -> judge stays active. Not a
count (#851). Ollama judges only (a vLLM judge has no such probe). `agentic_judge_model` provenance
still names the VLM; the event is the record of the mid-document switch.

Pins: wedged vs alive difference (judge calls 1 vs 4, 1 event vs 0), OCR halt unaffected.

### Mutations, external copy (src+tests+pyproject)

`socr.__file__` asserted inside the copy by a canary test; uncapped `count(anchor) == 1` asserted
before each edit. M0 is the unmutated control (8 = 7 tests + the canary).

```
M0_none                       8 passed
M1_judge_timeout_arms_halt    3 failed (difference pin, masking pin, wedged-vs-alive), 5 passed
M2_canary_floor_only          2 failed (cold-load default, slow-but-alive), 6 passed
M3_breaker_never_trips        1 failed (test_wedged_judge_is_cut_off_after_one_probe_slow_judge_is_not), 7 passed
M4_breaker_trips_when_alive   1 failed (same test), 7 passed
M5_no_probe_after_timeout     1 failed (same test), 7 passed
```

Rebased onto origin/main 790ce1fa (#984 guard present); full suite run with the default OLLAMA_HOST.

## Review round 2 (Astra rejected 52d52498): the breaker fails closed, it does not degrade

Round 1 switched a wedged judge to the heuristic judge. That is WEAKER: heuristics do not check
fidelity to the page image, so pages accepted that way ship COMPLETED with audit_passed=True. Replaced.

- `SwitchablePageJudge` removed. `CircuitBreakerPageJudge` (agentic.py) wraps the VLM LEAF (inside the
  deterministic native-table verifier, so pages that verifier settles without a model are untouched).
  While the breaker is open it raises the same `PageJudgeTimeoutError` a real deadline raises,
  instantly. `route_page` types it `JUDGE_OUTCOME_TIMEOUT` and the page fails closed exactly as
  today, minus the wait. Event renamed `judge_wedged_circuit_open` (one, page 0).
- Probe exceptions count as wedged (fail closed).
- vLLM judges: `VLLMVisionJudge.is_available` only lists `/models`, which answers on a wedge, so it
  is not used. The breaker probes with `probe_openai_server_idle(url, model=...)` (listing + a
  1-token generation, the same OpenAI-compatible canary the OCR side uses). No trip-on-timeout
  fallback was needed.
- Measured through `process()` (tests/test_gh987_judge_breaker_e2e.py): the wedged run's page
  sidecars equal the slow-judge run's, key for key (excluding input_checksum/timings_s), only the
  judge call count differs (1 vs 4). On this harness both end as status=warning, audit_passed=false,
  disposition demoted_native (flagged native fallback), failure_mode none; document AUDIT_FAILED.
  The `page_judge_timeout` failure mode itself is not what these pages carry here, because the
  native fallback outranks it with no provider/credential; the pin is therefore the equality with
  today's per-page behaviour, not a pinned tuple (CLAUDE.md, no-provider trap).
- Resume pinned: a control run whose judge accepts leaves SUCCESS pages and a second run skips all of
  them (0 OCR calls); the wedged run's pages are WARNING, so a second run with a healthy judge
  re-reads all four and they become SUCCESS. Note: a document recorded completed is skipped at the
  DOCUMENT level without `--reprocess` (pre-existing, same for ordinary judge timeouts); the page
  ledger is what the test exercises, via `reprocess=True`.

### Mutations, external copy (src+tests+pyproject), socr.__file__ canary, count(anchor)==1 asserted

```
M0_none                                           11 passed
M1_judge_timeout_arms_halt                        2 failed, 9 passed
M2_canary_floor_only                              2 failed, 9 passed
M3_breaker_never_opens                            5 failed, 6 passed (all five e2e pins)
M4_breaker_opens_when_alive                       3 failed, 8 passed
M5_no_probe_after_timeout                         5 failed, 6 passed
M6_probe_exception_counts_alive                   1 failed, 10 passed (test_a_probe_that_raises_counts_as_wedged)
M7_open_breaker_accepts_instead_of_failing_closed 4 failed, 7 passed
```

## Review round 3 (cubic on 4e3cfb99)

- openai/vLLM probes (`/models` precondition and generation canary) now run under
  `call_with_total_deadline`, as the Ollama sibling does; a trickling server no longer holds the
  document loop (trickle-server pin on the canary; the `/models` GET cannot reach a loopback server
  under the suite's hermetic `httpx.get` stub, so it is pinned by a call-site spy instead).
- The cold-load allowance (`canary_deadline()`) is Ollama-only: `probe_openai_server_idle` defaults to
  `_CROP_DEADLINE_FLOOR_S`, and the breaker's vLLM probe passes no `generation_timeout`.
- `judge_wedged_circuit_open` is in `_RESUME_REPLAYED`. Only a page's own events reach its sidecar,
  so a page-0 event would have made the allowlist entry a no-op: the event now carries the TRIGGER
  page's number (still exactly one per document; it is in `audit_log.json`).
  `test_resume_restore_kinds` expects it (41 kinds).
- `PageJudgeCircuitOpenError(PageJudgeTimeoutError)`: routing still types it a judge timeout, the
  log says "circuit open", `_TimeoutJudge` re-raises it unchanged (no re-probe, no doubled
  "page judge timeout: page judge timeout:" text).
- Comments: halt-site comment in `_phase_agentic` and the `probe_ollama_idle` / `probe_openai_server_idle`
  docstrings corrected.

### Mutations, external copy (src+tests+pyproject), `socr.__file__` canary, `count(anchor)==1` asserted

```
M0_none                                        43 passed
M1a_openai_canary_unwrapped                    2 failed (trickle pin, call-site spy)
M1b_openai_models_unwrapped                    1 failed (call-site spy)
M2a_openai_default_gets_allowance              1 failed (allowance Ollama-only pin)
M2b_breaker_openai_probe_gets_allowance        1 failed (vLLM breaker e2e)
M3a_kind_not_replayed                          2 failed (e2e replay pin, test_resume_restore_kinds)
M3b_event_on_page_0                            1 failed (e2e replay pin)
M4_circuit_open_wrapped_as_ordinary_timeout    1 failed (circuit-open log/type pin)
```
