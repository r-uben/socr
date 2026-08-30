# 2026-08-30 — TICKET-A2: CLI1 rung, ollama-cloud table judge (GH-353)

## What changed

New `src/socr/judge/table_rung_ollama.py`: `build_ollama_rung(model, host,
timeout)` returns an A1 `RungCallable` bound to one ollama model/host/timeout.
POSTs `/api/chat` (the GH-356 bake-off's confirmed integration path —
`/api/generate` is what `OllamaVisionJudge` uses for the page-level judge,
and `ollama run --images` does not exist on 0.32.15) with:

- `messages: [{role: "user", content: <prompt>, images: [<base64 crop>]}]`
- `format: "json"`, `stream: false`
- prompt built via `socr.judge.table_prompt.build_table_judge_prompt`, with
  `prior_findings` (A1 `Finding` objects) converted to the dict shape that
  module expects, so a tiebreak call carries the prior rung's findings.

Transport isolated behind two module-local seams tests can mock without
touching `httpx` globally: `_build_payload` (pure, asserted on directly) and
`_post_chat(host, payload, timeout) -> str` (the one function that calls
`httpx.post`). `httpx.HTTPError` (covers `TimeoutException`, `ConnectError`,
and `HTTPStatusError` via `raise_for_status`) from `_post_chat` becomes
`RungResult(ok=False)` with the exception recorded in `error` — no exception
propagates, no synthesized FAIL. A successful response is classified through
A1's `rung_result_from_output`, which already treats malformed/fenced JSON
per the A1 contract.

`host` resolves once, at `build_ollama_rung` call time, through the existing
GH-222 resolver (`socr.tables.extract.resolve_ollama_host`) rather than a
second host-resolution rule — explicit arg, then `OLLAMA_HOST`, then
localhost.

Did NOT reuse `OllamaVisionJudge.is_available()` (live `/api/tags` probe) per
the ticket — irrelevant to this rung's contract, and known blind to a wedged
GPU besides ([[ollama-tags-probe-blind-to-wedge]] fact from a prior ticket).

## Files

- `src/socr/judge/table_rung_ollama.py` (new)
- `tests/test_table_rung_ollama.py` (new, 12 tests)

## Tests

`~/venvs/socr/bin/pytest tests/test_table_rung_ollama.py -q` — 12 passed.
Also ran `tests/test_table_prompt.py tests/test_table_verdict.py
tests/test_ladder_config.py` (the three completed dependencies) — 52 passed,
unaffected.

Coverage: exact outgoing payload (model/format/images/stream=False), PASS and
FAIL round-trips, prior-findings injection reaching the rendered prompt,
malformed-JSON → ¬S1, timeout → ¬S1 with a `time.sleep` spy proving no retry
sleep, connection error → ¬S1, HTTP status error → ¬S1, host resolution
(explicit arg wins over `OLLAMA_HOST`, `OLLAMA_HOST` wins over default), and
a sentinel test that calls the real (unmocked) `_post_chat` to prove the
autouse `httpx.post` guard actually intercepts a live-network attempt rather
than being a no-op that never fires.

## Lint

`uvx ruff@0.16.0 format --check src/socr/judge/table_rung_ollama.py
tests/test_table_rung_ollama.py` — clean (ran the exact CI invocation on the
whole repo too; the only other reformat finding, `tests/test_table_ladder.py`,
belongs to the concurrent A4 ticket and was left untouched).

## Deviations / follow-ups

None. B1 (the gate) constructs this rung once via `build_ollama_rung` using
the G1 config fields (`table_judge_rung1_model`, `table_judge_rung1_host`,
`table_judge_timeout_sec`) and injects it into A4's ladder state machine.
