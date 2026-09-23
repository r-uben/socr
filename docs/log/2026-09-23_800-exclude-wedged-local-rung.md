# GH-800 — drop a still-wedged local rung after a rescued timeout (2026-09-23)

Branch `fix/800-exclude-wedged-local-rung`.

## Gap

#799 gated the cascade halt on `not decision.accepted`, so a page whose local rung timed out
and whose cloud rung rescued it no longer truncated the document — correct. But nothing else
happened: the ladder is built once per document (`_phase_agentic`, `ladder` from
`_build_ladder_and_escalation_profile`), so every later page walked into the same wedged local
backend, paid a full provider timeout, then reached the rung that works. Slow, and paid for,
silently, for the whole document. Efficiency and spend, not content: the text was always recovered.

## What

`UnifiedPipeline._exclude_wedged_local_rungs(state, page_num, decision, ladder)`, called in the
page loop on `_had_timeout and decision.accepted` — the branch #799 left empty. It removes a rung
only if all of these hold:

- the rung is `TIER_LOCAL`;
- its own attempt on this page was a **provider** timeout: `reason == REASON_PROVIDER_TIMEOUT`
  (a named constant in `agentic.py`, now also used where that reason is written). Not the
  `"timeout"` substring `_attempts_show_timeout` scans — the review of #890 showed a judge
  timeout on the rung's output records `"judge raised: page judge timeout …"`, which contains
  the word but says nothing about the local backend (the judge may be a cloud model);
- `_probe_backend_idle()` says the backend is still unresponsive now (a slow page on a healthy
  machine keeps its rung);
- at least one rung survives (the rescuing rung is never local-and-timed-out, so this holds by
  construction; the check is a belt for the "never strand the document" rule).

Recorded as a `local_rung_excluded_after_rescue` event with the excluded and remaining rung ids,
plus a console line. The halt path for unaccepted pages is unchanged.

Resume re-derives the ladder from scratch, so a later run re-probes rather than inheriting the
exclusion. That is deliberate: a backend that recovers between runs gets its rung back.

## Tests

`tests/test_gh800_exclude_wedged_local_rung.py` — four tests over a 3-page document with two
DISTINCT provider identities (`PROFILE_QWEN_LOCAL`, `PROFILE_QWEN_CLOUD`), the thing #799's tests
could not distinguish. The provider timeout is real: the per-provider deadline is shrunk and the
local stand-in blocks past it. One test guards against a vacuous pass by requiring page 1 to reach
both rungs. The difference pin varies only the backend probe: wedged → local attempted on page 1
only; healthy → local attempted on all 3. Every page is still read (no #227 regression). Engines
pinned per #841.

Mutations seen to fail (out-of-repo copy, import canary): never calling the exclusion → 2 failures;
ignoring the probe → 1 failure. Timing margins are named constants with the reason for each.

## Review (PR #890) — accepted with follow-ups, both handled

1. **Judge timeouts could drop a healthy rung.** My first version matched the `"timeout"`
   substring, and I wrote that this was "the same provider-half signal" — it is not: a judge
   timeout contains the word too. Fixed by exact comparison with `REASON_PROVIDER_TIMEOUT`;
   `test_a_judge_timeout_never_costs_a_local_rung_its_place` pins the difference and fails with
   the substring restored.
2. **`state.agentic_ladder` is a pre-loop snapshot** and still lists an excluded rung. Left as is,
   deliberately: it records the ladder the document *started* with, and the
   `local_rung_excluded_after_rescue` event carries the page, the excluded ids and the remaining
   ids, so which rungs later pages used is recoverable from the audit trail. Overwriting the
   snapshot would lose the starting ladder instead.
