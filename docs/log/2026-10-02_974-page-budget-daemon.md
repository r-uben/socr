# GH-974: daemon deadline workers, per-page ladder budget

Branch `fix/974-page-budget-daemon`, cut from `origin/main@f667473` (ancestry verified).
Follow-ups from #968 / #975 (`docs/log/2026-10-02_968-total-deadline.md`).

## Part 1: abandoned non-daemon threads

`ThreadPoolExecutor` workers are never daemon (GH-172), so a deadline site that abandons its
worker keeps the interpreter alive until the hung call returns. New
`socr.core.daemon_call.submit_daemon(fn, *args)`: one daemon thread, returns a real
`concurrent.futures.Future`. Same shape the sites used (`future.result(timeout=)`, `done()`,
`cancel()`), so timeout behaviour is unchanged.

Sites converted (each lost its `ex.shutdown(wait=False)`):

* `orchestrator._escalate_table_page` (escalation pool). The #851 abandoned-`Future`
  evidence (`_AbandonedEscalation.future.done()`) is untouched; `tests/test_gh851_...` now
  records futures by wrapping `orchestrator.submit_daemon` instead of
  `ThreadPoolExecutor.submit`, and still passes with the same call-count differences.
* `orchestrator._TimeoutJudge.assess`.
* `tables/extract.TableCropExtractor._read_with_deadline`.
* `pipeline/agentic.route_page` provider timeout.

Not converted: `hpc_pipeline` (`with ThreadPoolExecutor`, joins its workers by design).

The GH-172 prose comments at the converted sites were replaced; `test_gh172_abandoned_worker_exit.py`
keeps measuring the stdlib behaviour (the reason for the change) with a header pointing here.

Pins (`tests/test_gh974_page_budget_daemon.py`, child interpreters): a stdlib-pool control blocks
for the full hang (8 s); `route_page` and `_read_with_deadline` with a hung call exit in under half
of it. A static guard asserts no `ThreadPoolExecutor(` construction remains in the three modules
(covers the escalation pool and `_TimeoutJudge`, which a child cannot cheaply drive).

## Part 2: per-page ladder budget + console line

`judge/ladder_budget.PageLadderBudget`, created fresh by `_run_table_judge_gate` (now a thin
wrapper over `_run_table_judge_gate_unbudgeted`, so no re-indent of the 450-line body) and cleared
in `finally`, so an exhausted budget cannot leak to the next page.

* Covers all three call kinds: reader rungs (wrapped at the `run_table_ladder` call; refusal and
  identity bookkeeping keep the original callables), the blind-cell adjudicator (wrapped when built),
  and `_transcribe_cell_token` (cell transcribe, GH-367).
* Checked BEFORE each call. A call that starts inside the budget keeps its own per-call deadline,
  so a page can overrun by at most one call's timeout. Not shortened: that would mean threading a
  deadline through every transport.
* Skipped calls return the result each caller already treats as "no verdict"
  (`RungResult(ok=False, unavailable=False)`, `BlindCellResult(ok=False)`, `None` token), so the
  table ends UNVERIFIED through the existing terminals (page WARNING / document AUDIT_FAILED via the
  disposition, `table_ladder_unverified` events). `unavailable=False` on purpose: the budget is
  deterministic for that page; latching it would reprocess the page on every resume.
* Exactly one `table_ladder_budget_exhausted` event per page (detail names the budget). Added to
  `_RESUME_REPLAYED` (restore count 39 -> 40) so a resumed page keeps the record. Not added to
  `TABLE_DISTRUST_KINDS` / `CREDENTIAL_BLOCKING_EVENT_KINDS`: the page is already untrusted through
  the `table_ladder_unverified` event it always co-occurs with.
* Console: one dim line per call that ran, `pN: table ladder <rung> (<model>) <elapsed>s`, plus one
  yellow line when the budget is spent; silent under `quiet`.

### Budget value and derivation

`PipelineConfig.table_judge_page_budget_sec: float | None = None`. `None` derives
`table_judge_timeout_sec * (len(rungs) + 1)`: one worst-case call per ladder stage (each reader rung,
plus one for the adjudicator/transcriber stage). Defaults: 600 s x (2 + 1) = 1800 s. Reasoning: the
per-call 600 s is the measured floor from the GH-356 bake-off; the page budget says a page may spend
one worst-case call per stage in total, however many tables it has. No new magic number; it moves
with the timeout. Not added to the run fingerprint (an override changes only pathological pages;
adding a key would invalidate every existing resume).

## Tests

`tests/test_gh974_page_budget_daemon.py` (13): Part 1 as above; budget: 4 slow rungs, budget 0.5 s,
0.3 s each -> calls `[r0, r1]`, elapsed < budget + one call + slack, exactly one budget event,
one `table_ladder_unverified`, disposition set; page 2 on the same pipeline ACCEPTED with no new
budget event and `_ladder_budget is None`. Difference pin: same page with a 60 s budget runs all
rungs, no event. Default/override derivation; console line per call with rung, model, elapsed;
quiet silent; adjudicator and cell transcribe skip (fake clock); wiring: `evaluate_cell_guard`
receives a budget-wrapped adjudicator. `test_resume_restore_kinds` updated (+1 kind, 40).

Mutations (external copy of src + tests + pyproject, canary `socr.__file__` inside the copy,
uncapped anchor count == 1 asserted before each edit; baseline 22 passed):

| mutant | result |
| --- | --- |
| `daemon=False` | 3 failed |
| budget never exhausted | 2 failed |
| event on every skip | 1 failed |
| budget leaks across pages (reuse + no clear) | 1 failed |
| reader rungs not wrapped | 2 failed |
| cell transcribe unbudgeted | 1 failed |
| adjudicator not wrapped | 1 failed |
| no console report | 1 failed |

## Limits

* Budget overrun is bounded by one call's timeout, not zero.
* #975's other follow-up (abandoned-only fail-fast in `call_with_total_deadline`) is unchanged; still
  no parallel caller.

## Suite

Full suite, default OLLAMA_HOST, nohup: 6419 passed, 2 skipped, 4 xfailed, 0 failed.
`uvx ruff@0.16.0 format --check .`: 832 files already formatted.

## Review round 1 (Astra, PR #978: ACCEPT-WITH-FIXES)

* **Resume (P2).** I earlier left the budget out of the run fingerprint. That was wrong for an
  EXPLICIT budget: a cached terminal (e.g. a WITHHELD page whose sibling table hit the budget)
  would survive a raised budget. Now `table_judge_page_budget_sec` joins the fingerprint extras
  only when the ladder is on AND it is explicitly set; the default (None) adds no key, so existing
  fingerprints and resumes are byte-for-byte unchanged, and the derived default is covered by
  `table_judge_timeout_sec` (already fingerprinted) and the source digest.
* **Pins (P2)**, `tests/test_gh974_review_pins.py` (8), each a difference between two runs that
  change only the budget:
  multi-table page (two ruled grids on one page: calls `[r0, r1]` total at 0.5 s, `[r0,r1,r2]*2`
  at 60 s, 2 unverified events, 1 budget event); same page with equal rung counts (4 vs 4)
  replaces the old 4-versus-3 comparison, which is deleted; exhausted adjudicator (guard chain asked
  with budget, not asked without; page UNVERIFIED, one event); `process()` on the committed fixture:
  tight budget gives page 1 WARNING (sidecar `winning_output.status`), document AUDIT_FAILED naming
  page 1, page 2 unaffected, one budget event on page 1 only, versus SUCCESS/SUCCESS when loose;
  exhausted cell transcriber (shifted table + high PASS: `transcribe_cell` not called, table stays
  UNVERIFIED, page WARNING, document AUDIT_FAILED; control reaches the transcriber); resume: a
  changed explicit budget reprocesses, the same budget and the default both still resume; the
  fingerprint extra has no budget key by default, has it when set, and not with the ladder off.
* **Mutants** (external copy, canary, uncapped anchor count 1; baseline 29 passed): fingerprint
  clause removed 3 failed; key always present 2 failed; ignores the ladder flag 2 failed;
  adjudicator unwrapped 2 failed; transcriber unbudgeted 2 failed; budget never exhausted 7 failed;
  per-table budget reset 3 failed.

* **Structure fix found by the suite.** The first review-round suite run failed 4 tests: the new config field was unclassified in test_cli_flag_agentic_status_gh142 (now classified); and tests that call `UnifiedPipeline._run_table_judge_gate(MagicMock(), ...)` or inspect its source broke because it had become a thin wrapper. The budget is now a decorator (`_under_page_ladder_budget`) on the original method, which keeps its name and body, and `_ladder_budget = None` is a class-level default. Final suite: 6379 passed, 2 skipped, 4 xfailed, 0 failed; ruff format clean (833 files).
