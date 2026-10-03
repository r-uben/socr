# 2026-10-03 GH-1001: a halted document is not skippable on resume

**Cause.** `_resume_skippable` skips a PARTIAL document whose checksum and fingerprint match ("re-running the
identical config cannot improve a partial result"). A PARTIAL_SAVE_VLM_TIMEOUT halt is transient (a wedged
backend), so a plain re-run never reached the per-page ledger and the unprocessed pages stayed unprocessed.

**Fix.** Same latch mechanism as the lane retry latches: `_phase_assemble` records `halt_retry_pending` on the
root index entry whenever `state.pp2_halt_reason` is set (one write via `_LatchedDocMetadata`), and
`_resume_skippable` returns False for an entry carrying it. Every other skip rule is untouched. The per-page
ledger (#995) then reuses only terminal pages. The latch is cleared naturally: the next run's entry is
rewritten without it once the document finishes without a halt.

**Tests.** `tests/test_gh1001_halted_doc_resumable.py`: halt at page 3 of 4; same output dir re-run plainly
(no `--reprocess`) processes p3, reuses p1/p2 through the ledger. Pinned as a difference in one process: the
same halted dir with the latch key stripped from the root index (the pre-fix index shape) is skipped whole.
Mutant (external src+tests copy, `socr.__file__` canary, anchor count 1): the gate line replaced; the test
fails. The `--reprocess` caveat in `result.py` and the #995 log was removed.
