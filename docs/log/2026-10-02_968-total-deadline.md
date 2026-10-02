# GH-968: every Ollama HTTP call gets a TOTAL wall-clock deadline

Branch `fix/968-total-deadline`, cut from `origin/main@c2859b8`. Static analysis:
`~/.local/state/socr-housekeeping/gh968/findings.md`.

## Defect

httpx `timeout=` and urllib `timeout=` are per-read / per-socket-op inactivity limits.
A peer that trickles a byte (or keepalive) never trips them, so a call configured with
`timeout=600` can hang the main thread, and the document, indefinitely. The prime site was
`table_rung_ollama._post_chat`, shared by rung 1, the adjudicator and cell transcribe.

## Change

* `ollama_utils.call_with_total_deadline(fn, timeout, *, label="")`: runs `fn` in a daemon
  thread (`_call_within`, kept as the primitive and as the alias `_get_tags` already uses),
  abandons it on overrun, re-raises what `fn` raised.
* Overrun raises `TotalDeadlineExceeded(httpx.ReadTimeout, TimeoutError)`. Choice justified:
  the httpx callers (`_post_chat` -> rung `except httpx.HTTPError`, probes `_PROBE_ERRORS`)
  catch `httpx.HTTPError`; the urllib callers (`latex_for_image`, `latex_for_crop`) catch
  `(URLError, TimeoutError, ValueError, OSError)`. One class that is both needs zero caller
  changes. `is_availability_exception` treats it as an outage (`httpx.TransportError` /
  `TimeoutError`), so a hung rung gets `unavailable=True` exactly as a read timeout did.
  That sets the page's retry marker (`table_judge_retry_pending`, orchestrator.py
  ~7585-7599) when the ladder stays UNVERIFIED; an accepted fallback rung does not set it. It does NOT disable later rung calls: only `refusal=True` trips the
  per-run breaker, so later pages still try the rung (bounded by the rule below).
* Wrapped sites, deadline = the call's existing configured timeout, no new number:
  `table_rung_ollama._post_chat` (covers rung 1, adjudicator, cell transcribe) and
  `ollama_rung_reachable`; `extract._ollama_generation_canary` and the `/api/tags` probe in
  `probe_ollama_idle`; `equation_latex.latex_for_crop`; `math/recover.latex_for_image`;
  `gemini_api.OllamaFigureEngine.is_available` (3.0) and `describe_figure` (120.0, the
  pre-existing literals).
* Surfacing: the exception message names the call ("ollama /api/chat (<model>) exceeded
  total deadline of Ns"), and a warning is logged. It flows into `RungResult.error` and so
  into the `table_ladder_unverified` event (pinned by the gate test). No new event kind.
* Each site fails as it did on a timeout: rung -> `ok=False, unavailable`, ladder moves on or
  ends UNVERIFIED; canary/probe -> False; equation lanes -> `""`; figure engine -> error
  description.

## Review fix: abandoned workers are bounded (Astra, P1)

An abandoned call keeps its thread, socket and buffered response. `call_with_total_deadline`
now records the abandoned thread per label (label = endpoint; `_post_chat` labels include
host and model). While it is alive, a new call with the same label raises
`TotalDeadlineExceeded("... previous call still outstanding; not retried")` and starts no
thread (the #851 rule: never stack a second call on an unresponsive peer). When the stray
finishes the label is free again, so later pages recover. Cap = one per label, no new
constant. `_get_tags` keeps the untracked primitive (single short probe).
Test hygiene: environment proxies are removed for loopback, timing bounds are
`MARGIN`x the deadline (4x) with the raw-call-still-running-at-3x difference pin kept.
New pins: repeated overruns leave at most one live worker; calls resume after the stray
ends; a hung page, then a fail-fast page, then a recovered page.

## Review round 2 (Astra rejected 3eee3b1)

* P1: the worker was registered after `join()`, so concurrent same-label callers all started
  a thread and overwrote each other's entries; the overwritten strays escaped the cap. Now
  the check and the registration happen under the lock BEFORE `thread.start()`; a call that
  finishes in time deregisters itself. Consequence, accepted: two genuinely concurrent calls
  with the same label are serialized (the second fails fast as "still outstanding"). Ollama
  serializes per model anyway, and the page loop is sequential.
* P2: labels now carry host and model at every site (`recover`, `equation_latex`,
  `gemini_api` x2, canary, tags probes, `_post_chat`, `ollama_rung_reachable`), so a hung
  endpoint cannot block a different host or model.
* Tests: 8 concurrent same-label callers start exactly 1 worker and 7 fail fast; endpoint
  isolation (hung host/model A leaves host/model B working, behaviourally, via a faked
  `urlopen`); a label-content test over all 8 sites; the recovery pin now runs all three
  pages through `_run_table_judge_gate` and asserts the fail-fast page carries the
  `table_ladder_unverified` event naming "still outstanding" and `table_judge_retry_pending`
  True, no second request, and page 3 reaches the endpoint with no UNVERIFIED event.
* Mutants (external copy, canary, anchor count 1): baseline 23 passed; register-after-join
  -> concurrent test fails; recover label without host -> 2 fail; without model -> 2 fail;
  equation label without host -> 1 fails; gemini label without host -> 1 fails. Survivor:
  "never deregister on success" (23 passed): equivalent, since `is_alive()` already ignores a
  finished worker; deregistration only keeps the dict small.

## Review round 3 (cubic)

* P2 security: labels carried the configured host verbatim, so `http://user:pass@host`
  would leak credentials into timeout warnings and `RungResult.error`. New helper
  `ollama_utils.safe_host_label` keeps scheme, host, port and path, drops userinfo, query
  and fragment; every site label uses it. Pinned: no label contains the user, password or
  `@`; hosts differing only in userinfo give identical labels at the six sites that build
  the label from the configured host.
  Pre-existing, NOT fixed here: `resolve_ollama_host` mangles a userinfo host (a bare-IPv6
  bracket heuristic fires on the extra colons), so the two sites that resolve first
  (`probe_ollama_idle`, `ollama_rung_reachable`) get a garbled but credential-free label,
  and a userinfo host likely does not work for requests there either.
* P3: `OllamaFigureEngine.is_available` treated any exception as "daemon unavailable". A
  `TotalDeadlineExceeded` (overrun, or fail-fast behind a draining probe) now returns the
  last DEFINITIVE answer instead of a fresh False (timeouts are never proof of
  unavailability, as in `probe_failure_reason`). Limit: before any definitive answer the
  remembered value is False, so a first-ever probe that times out still reads False (same as
  before this ticket).
* Mutants: helper keeps userinfo -> 3 fail; gemini chat label uses the raw host -> 1 fails;
  fail-fast returns False -> 1 fails. A `fail_fast` flag on the exception had no consumer,
  survived its mutant, and was removed.

## Not done (filed as #974)

* Per-page ladder budget and a console line per rung call.
* Non-daemon abandoned threads (escalation `orchestrator.py:6047`, judge `:8457`,
  `extract.py:678`, `agentic.py:258`).

## Tests: `tests/test_gh968_total_deadline.py` (16)

Loopback trickle server (accepts, then one byte per 0.1 s, huge Content-Length) with
`DEADLINE=0.5`. Difference pin: raw `httpx.post` is still running at 3x the deadline;
`_post_chat` fails in under 1.5x. Real-trickle tests also cover the canary and both urllib
lanes. One spy test per wrapped site. Gate test: `build_ollama_rung` against the trickle
server through `_run_table_judge_gate` yields exactly one `table_ladder_unverified`
event naming the call. Every hang-capable call runs in a joined daemon thread, so a
regression fails instead of hanging CI. No Ollama, no provider (rungs injected, binding and
adjudicator stubbed; conftest's `_post_chat` stub is replaced only inside this file).

## Mutations (external copy of src + tests + pyproject, `socr.__file__` canary, uncapped anchor count == 1)

| mutant | result |
| --- | --- |
| baseline (unmutated) | 17 passed (16 + canary) |
| wrapper runs `fn()` inline, no deadline | 6 failed |
| `_post_chat` calls `httpx.post` unwrapped | 3 failed (trickle, site spy, gate) |

## Suite

Commit c587745a: 6349 passed, 2 skipped, 4 xfailed, 1 failed (`test_timings.py::test_native_page_records_extract_not_route`,
a load-sensitive timing test; passes in isolation, 13 passed).
Commit a201ae08 (review fixes): 6353 passed, 2 skipped, 4 xfailed, 0 failed. `ruff@0.16.0 format --check .` clean.

Review-fix mutations (external copy, `socr.__file__` canary, anchor count 1): baseline 20 passed;
fail-fast check removed -> 3 failed; stray never registered -> 3 failed; `is_alive()` dropped
(no recovery) -> 2 failed.
