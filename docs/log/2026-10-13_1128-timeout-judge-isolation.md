# #1128 — page-judge timeout worker isolation

The worker behind `_TimeoutJudge` judged the live `PageOutput`. After the
deadline the caller moved on, and the worker still wrote status, failure
mode, table fields and text.

`_TimeoutJudge` lives on `UnifiedPipeline` in `orchestrator.py` (not
`agentic.py`). `route_page` and promotion (`_promotion_inner_judge`) both
call it. #1133 did not touch this.

## Fix

The worker judges a deep copy. Fields are copied back when the worker
finished inside the deadline, whether it returned or raised. A late
result is discarded and the attempt carries `late page-judge result
discarded` in `audit_notes`. `_run` returns `("ok", verdict)` or
`("err", exc)` so `result(timeout=)` raising TimeoutError is always
our deadline — lateness is never re-read from `future.done()`.
Circuit-open is re-raised as-is after adopt; an inner TimeoutError is
still wrapped as `PageJudgeTimeoutError`.

## Other deadline sites

None of these share a live `PageOutput` with an abandoned worker. Each
already drops the late return value:

- `route_page` provider `submit_daemon` — timeout records a fresh
  `_error_output`
- escalation `submit_daemon` — timeout keeps the incumbent
- crop `_read_with_deadline` — timeout raises; the late string is unused
- `killable.run` — the parent only adopts a result received before the
  deadline

`hpc_pipeline`'s pool is parallel documents, not a killable judge.

## Measure (before any edit)

Same shape as the reviewer: contradicted table 99.9 / 88.8, witness
41.5 / 17.2, deadline 0.1 s. Witness slept 0.12 s so the gate could not
finish in time. 15 `process()` runs on 5567dbda: **15 / 15** shipped
page SUCCESS, no mark, `judge_outcome=page_judge_timeout`. That is the
timeout path itself (the witness never marked the live object before
flush), not a late wipe. N for the hermetic test is that batch: 15.

After isolation the same 15 runs are still 15 / 15 timeout-SUCCESS
unmarked — isolation does not invent a mark the gate never delivered.
A late-clearing judge (sleep 0.2 s, then rewrite the text) wiped the
live object on the unfixed path; after the fix that wipe is **0 / 15**.

## Tests

- `test_late_mutation_never_reaches_shipped_page` — 15 `process()` runs,
  identical fragment, late wipe absent
- `test_on_time_result_is_adopted` — wrapped vs `timeout_sec=None`,
  mutations match
- `test_late_discard_is_journaled` — note on the live object, text
  unchanged

Mutant `judge_live_object` (`probe = output`) fails the late-mutation
test.

## What the issue got wrong

The class is in `orchestrator.py`, not `agentic.py`. The reviewer's
2 / 5 SUCCESS-unmarked count did not reproduce as a mix on this host
with a 0.12 s delayed witness: every run timed out before the gate
wrote. The isolation bug is real; it is the late write, not the
timeout-SUCCESS page itself.

Full suite (round 1): 7493 passed, 2 skipped, 4 xfailed. Ruff 0.16.0
format `--check` clean.

## Round 2

Astra: do not merge (1 P1, 2 P2).

1. **Adopt on-time exceptions.** Source evidence writes
   `table_label_unverified` then the leaf raises (HTTP 404 or breaker
   already open). Adopt-on-return-only dropped those writes: wrapped
   `process()` shipped SUCCESS with no label, passthrough shipped
   WARNING + `alpha` + the audit note. Adopt whenever the worker
   finished inside the deadline. Pin: wrapped == passthrough for both
   leaves.
2. **Shipping test handshake.** Sleeping after `process()` does not
   force the late write before assemble, so `probe = output` survived
   all 15 runs. The judge now waits until the caller is past the
   deadline, mutates, then signals back. Promotion can call `assess`
   again; the handshake resets per call so that second assess cannot
   run on-time on a leftover event.
3. **Discard-note race.** A worker finishing between `result()` timing
   out and `done()` made `inner_raised=True` and left `audit_notes`
   empty. Lateness is the `result()` TimeoutError itself. Pin:
   `CompletionBetweenTimeoutAndHandler`.

Mutants under `exec/1128/mutants-r2/` (external copy, `socr.__file__`
asserted, `src.count(anchor) == 1`):

- `adopt_only_on_ok` — fails `test_on_time_exception_ships_as_passthrough`
- `judge_live_object` (`probe = output`) — fails the shipping test
- `skip_note_if_done` — fails the completion-between-timeout test

### Follow-ups (not this ticket)

Pre-existing shared state the review listed; left alone:

- document events (`_record_judge_event` → `state.events.append`)
- provider page flags (e.g. `native_table_structure_failed`)
- judge PDF caches (`_fitz_doc_cache`) and temp PNGs (`socr_judge_*.png`)

Full suite (round 2): 7496 passed, 2 skipped, 4 xfailed.
Ruff 0.16.0 format `--check`: clean.

## Round 3 (#1135 folded in)

Astra r2 P1: when source evidence finishes and the leaf stalls, discarding
the copy also discarded `table_label_unverified`. Same mechanism as #1135:
`SourceEvidenceTableJudge` ran inside `_TimeoutJudge`, so a 0.1 s deadline
skipped a 0.12 s witness and shipped SUCCESS unmarked (15/15 on the r1
measure).

### Chain

Outside → inside, on the live output until the leaf:

`SourceEvidenceTableJudge` (Tesseract witness) →
`NativeTableVerifierJudge` (local, deterministic: PyMuPDF geometry +
`verify_native_table`; no model call) →
`_TimeoutJudge` (copy isolation) → circuit breaker → VLM / heuristic.

A phase wrap still applies only when the built chain has no leaf deadline
(replaced `_build_page_judge` in tests; the `se_inside_deadline` mutant).

### #1126 promotion

`_promotion_inner_judge` still peels only SE. Production
`SE(Native(_TimeoutJudge(leaf)))` keeps Native + the leaf deadline.
`test_promotion_inner_judge_peels_production_page_judge` now peels
`_build_page_judge` directly (no extra outer Timeout) and asserts the
deadline is still in the chain. Wiring tests that named
`Native._inner is Heuristic` now walk through `_TimeoutJudge`.

### Measure (r3)

Same reviewer shape, production `_build_page_judge`, deadline 0.1 s,
witness delayed 0.12 s. **0 / 15** SUCCESS unmarked. Pages floored
(ERROR, unverifiable-table mark).

### Tests

- r1/r2 kept
- `test_evidence_written_before_leaf_deadline_survives_stall` — Astra
  repro: label survives a stalled leaf
- `test_contradicted_table_never_ships_success_unmarked` — #1135
  acceptance

Mutants under `exec/1128/mutants-r3/`:

- `judge_live_object` — shipping test
- `adopt_only_on_ok` — on-time exception
- `skip_note_if_done` — discard-note race
- `se_inside_deadline` — #1135 contradicted-table test

Full suite (round 3): 7498 passed, 2 skipped, 4 xfailed.
Ruff 0.16.0 format `--check`: clean.

## Round 4 (#1128 + #1135)

Astra r3: do not merge (P1 Tesseract unbounded after the reorder, P2
#1135 test never stalled the leaf).

### P1 — witness timeout

`classical_ocr_with_state` now passes
`timeout=TESSERACT_WITNESS_TIMEOUT_SEC` to `pytesseract.image_to_string`.
Expire is `RuntimeError("Tesseract process timeout")` on this pytesseract
(0.3.13 has no `TesseractTimeoutError`); the existing `except Exception`
maps it to `WITNESS_EXEC_ERROR` (could-not-read, table unverifiable).

Derivation (2026-10-13): 181 / 181 pages from
`witness_detail_out.json` against `witness-pdfs/`, rendered at
`_EVIDENCE_OCR_DPI` (150). min 0.095 s, p50 0.752 s, p95 1.568 s,
p99 1.789 s, max 2.326 s. Timeout is `ceil(p99 * 2.0)` = 4 s (same
headroom name as `_CROP_WALL_CLOCK_MULTIPLIER`); that is already above
`ceil(max)`. Not the 300 s provider / 600 s table-judge budgets.

`test_tesseract_witness_timeout_fails_closed_without_hanging`: stub
raises on a finite `timeout` and sleeps 30 s otherwise. Production
returns `exec_error` inside a 2 s watchdog. Mutant
`tesseract_no_timeout` hangs until the watchdog fails the test.

### P2 — #1135 test stalls the leaf

`test_contradicted_table_never_ships_success_unmarked` now uses
production `_build_page_judge`, a 0.12 s contradicted witness (so
`se_inside_deadline` still misses it), and an event-gated stalled
`HeuristicPageJudge` inside the existing 0.1 s `_TimeoutJudge`. Asserts
the leaf ran, a page-judge timeout fired, and the page is
floored/marked, never SUCCESS unmarked. Same-process refuse-now leaf
ships the same rejection without a timeout.

Mutants under `exec/1128/mutants-r4/` (external copy, `socr.__file__`
asserted, `src.count(anchor) == 1`):

- `judge_live_object` — shipping test
- `adopt_only_on_ok` — on-time exception
- `skip_note_if_done` — discard-note race
- `se_inside_deadline` — rewritten #1135 test
- `tesseract_no_timeout` — witness hang watchdog

Full suite (round 4): 7499 passed, 2 skipped, 4 xfailed.
Ruff 0.16.0 format `--check`: clean.

## Hygiene (#1138 / #1139 softs / cubic on #1136)

Test-only. No production edit.

- **#1138 item 1.** `test_inner_timeout_error_is_wrapped_and_adopted`: a leaf raises builtin `TimeoutError` inside the deadline. It becomes `PageJudgeTimeoutError("page judge timeout: …")`, `on_timeout` fires, adopted writes stay. Mutant `inner_timeout_if_false` (`if isinstance(payload, TimeoutError)` → `if False`) re-raises the bare `TimeoutError`.
- **#1139 softs.** Figure-step test now parametrises `unknown_via` like the table pin, and asserts the uncovered-text mark so apply actually ran. Dropped the implied `!=` on the two modes.
- **cubic stub leak.** The tesseract probe runs in a subprocess with a 2 s outer timeout; the child is killed on expire, so a no-timeout mutant cannot leave a sleeping stub in this process's `sys.modules`. Still hermetic without pytesseract. `tesseract_no_timeout` still fails it.

Full suite: 7506 passed, 2 skipped, 4 xfailed. Ruff 0.16.0 format
`--check` clean. Mutants: `inner_timeout_if_false`,
`tesseract_no_timeout`, `remove_figure_step`, `swap_table_and_figure`.


