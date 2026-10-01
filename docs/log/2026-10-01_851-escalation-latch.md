# GH-851: the escalation latch requires evidence of a wedge

Branch `fix/851-escalation-latch-evidence`, cut from `origin/main@bcc23ea`. No prior or
withdrawn attempt exists in `docs/log` or `docs/plans` (grep for 851 hits only unrelated
numbers and `2026-09-20_855-...`, which explicitly parks the latch for #843/#851).

## Defect

`_escalate_table_page` returned `True` on the first outer-deadline expiry, and
`_phase_agentic` stored it in `_escalation_degraded`, a set-once document-scoped latch.
One slow read removed escalation from every later page, and the only record was a
`table_escalation_timeout` event whose detail claimed "lane disabled".

## Design

The #849 mechanism (`KillableTimeoutError.killed`: killed versus peer answered late)
does not transfer. It is a property of a `run_killable` child. The escalation site is a
`ThreadPoolExecutor` around `run_provider` for a non-local per-page profile; there is no
child to ask, and `probe_model_generation` addresses an Ollama host only (a Gemini or
cloud-CLI rung has nothing to probe). The equivalent observable that does exist is the
abandoned `Future`: a wedge stays unfinished, a slow call finishes.

* A timeout records `_AbandonedEscalation(future, page_num, started)` in a per-document
  list owned by `_phase_agentic`, emits `table_escalation_timeout` (detail no longer says
  "lane disabled"), and returns `False`.
* The next page that qualifies for escalation drops finished entries
  (`Future.done()`). If any remain, the page is NOT attempted: it emits
  `table_escalation_withheld` (page-scoped, names the abandoned page and its age),
  keeps its incumbent, and returns `True`. A second call is never stacked on an
  unresponsive provider, so at most one abandoned call is outstanding.
* No count and no duration constant: the evidence is observed, not thresholded.
* `_escalation_degraded` is gone. `_lane_live` is now only "lane configured"; health is
  decided per page inside the function. This also makes the #855 coupling (scoring
  versus latch) structurally impossible.

## Surfacing and resume

`table_escalation_withheld` is added to `tables_trust.TABLE_DISTRUST_KINDS` and
`manifest.CREDENTIAL_BLOCKING_EVENT_KINDS` beside `table_escalation_timeout`, so the page
stays untrusted and the manifest carries it. Both kinds were absent from
`_RESUME_REPLAYED`, so a resumed terminal page lost them (the #252 / GH-353 shape); both
are now replayed.

## Tests

`tests/test_gh851_escalation_latch_evidence.py` (5). Two end-to-end `process()` runs of a
2-page document differ only in whether page 1's abandoned call has finished when page 2
qualifies: provider calls are `[1, 2]` versus `[1]`, page-1 events are identical, and
page 2 differs by exactly `table_escalation_withheld`. A direct test shows a never-returning
call withholds pages 2 and 3 with one provider call. Hermetic: ladder, judge, engine call
and `_resolve_judge_model` patched; engines pinned.

Updated for the changed contract: `test_gh96_escalation_lane` and
`test_gh160_escalation_cost_cap` (timeout returns `False`), and the GH-855 harness double
(a withheld page returns `True` without a provider call instead of a set latch).

Mutations (external copy of src + tests + pyproject, canary asserting `socr.__file__`
inside the copy, uncapped anchor count == 1 asserted before each edit):

| mutant | result |
| --- | --- |
| outstanding check disabled | 3 fail |
| every abandoned call treated as outstanding (the old latch) | 2 fail |
| abandoned call not recorded | 3 fail |

## Limits

While a call is outstanding, every qualifying page is withheld, up to the engine's
subprocess bound (e.g. 300 s for qwen). Those pages are visible, not silent, but the
window is real. The late result of an abandoned call is still discarded, as before. The
late-write race on `ps.native_table_structure_failed` noted on #851 is not touched here.

## Review round 1 (Astra, PR #937): status and resume

P1a: the first commit left a withheld page SUCCESS (only an event + CLI print, which
`--quiet` hides). P1b: the page stayed terminal SUCCESS / `audit_passed=True`, so resume
skipped it and the escalation was never retried.

Mechanism reused, not invented: the table ladder's "needed a verdict, did not get one"
state. `_mark_escalation_not_received(ps)` (called on a page's own timeout and on a
withheld page) sets:

* `ps.table_ladder_disposition = TABLE_UNVERIFIED` when none is set (REJECTED / WITHHELD
  are kept). `_apply_ladder_disposition_guard` then demotes the FINALIZED copy to
  WARNING / `table_unverified`; `best_output.audit_passed` is never flipped in place, and
  the page text is unchanged (asserted). Document status, CLI buckets and
  `tables_trust` already read the disposition: measured result is page WARNING and
  document AUDIT_FAILED, naming the pages, versus SUCCESS for a healthy run.
* `ps.table_ladder_incomplete = True`: forfeits the content-terminal resume exception, so
  a REJECTED/WITHHELD page that also lost its escalation is retried.
* `ps.table_escalation_retry_pending = True` (new in-run field): the DOCUMENT-level gate
  (`_resume_skippable`) runs before the per-page ledger and skips an identical-config
  partial document whole, which is what actually defeated the page-level marker (found
  by the resume test: second run returned SKIPPED with zero calls). The flag rides into
  the root-index entry like `equation_lane_retry_pending`, and the gate refuses the skip
  only when `_escalation_retry_blocks_resume()` says the lane could act now (agentic,
  escalation enabled, an escalation rung in the tier/cost-filtered ladder).
  With the lane off, strict-local or no rung the marker is inert, so a permanently dead
  provider does not force endless reprocessing.

Tests: the `sleep(0.05)` is gone; the slow case waits on the abandoned `Future` itself
(recorded from `ThreadPoolExecutor.submit`) and asserts `done()`. New: page-and-document
status difference pin (healthy vs slow vs wedge); wedge run then healthy resume must
escalate pages 1 and 2; control (healthy then healthy) skips, so the gate is the cause.
The fixture previously rewrote the PDF per run, changing its checksum and making every
resume look like a changed input; it is now written once.

Mutations (external copy, canary, uncapped anchor == 1): outstanding check off (5 fail),
old latch (3 fail), no document latch (1 fail: resume), no UNVERIFIED mark (2 fail:
status, resume), abandoned call not recorded (5 fail).

Follow-ups, pre-existing and not touched: the late `native_table_structure_failed` write
from an abandoned worker thread, and worker lifetime across documents in batch mode (an
abandoned thread outlives its document; `_escalation_abandoned` is per document). A
permanently wedged provider reopens the document on every resume while the lane is
reachable; that is the equation-lane latch's accepted trade-off.

## Verification

Full suite, default OLLAMA_HOST, nohup, one complete run: 6060 passed, 2 skipped, 4 xfailed,
1 failed. The failure was `tests/test_resume_restore_kinds.py` (pins the exact replay set;
updated to add the two kinds, 37 -> 39) and passes after the update. `ruff@0.16.0 format
--check .` clean (802 files).

(Review round 1 verification: full suite 6064 passed, 2 skipped, 4 xfailed; ruff format clean.)
