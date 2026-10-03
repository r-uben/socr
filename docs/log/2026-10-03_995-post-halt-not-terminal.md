# 2026-10-03 GH-995: pages after a PARTIAL_SAVE halt are not terminal SUCCESS

**Root cause.** `_phase_agentic` `break`s at the top of the loop once the backend is degraded, so later
pages are never processed. Assemble then flushes a sidecar for EVERY page with `terminal=True`, and
`finalized_page_record` synthesises the native-text winner (SUCCESS, native_prose, no events) for the
unprocessed ones. `_load_terminal_page` trusts that (terminal + fingerprint + checksum + SUCCESS), so a
`--reprocess` run (which lifts the document-level skip, not the per-page ledger) skipped pages no table or
model pass had seen. Measured in #994: bernanke_kuttner p21, Barrot_Sauvagnat p29/p30.

**Fix.**
- `PageState.not_processed_after_halt`, set for every page at/after the halt point (except pages the
  pre-pass resumed and unloadable pages).
- `FailureMode.PAGE_NOT_PROCESSED_AFTER_HALT`; `_apply_halt_unprocessed_guard` (manifest) demotes SUCCESS to
  WARNING and names the mode. Status-only: text and `audit_passed` untouched, so the final `.md` is
  byte-identical.
- `_flush_page_sidecar` forces `terminal=False` for flagged pages (covers every writer).
- `result.error` now lists the unprocessed pages after `PARTIAL_SAVE_VLM_TIMEOUT` (document/metadata/CLI).

**Tests.** `tests/test_gh995_post_halt_not_terminal.py`: same document with and without a halt at p1
(difference pinned, no local absolute); resume with `reprocess=True` must get zero ledger hits for pages
after the halt. Mutant (external copy of src+tests, canary on `socr.__file__`, anchor count == 1, flag
assignment replaced by `pass`): both tests fail (ledger hits [2, 4]). Full suite: 6645 passed, 2 skipped,
4 xfailed. `uvx ruff@0.16.0 format --check .` clean.

**Note.** A PARTIAL doc with an unchanged fingerprint is still skipped whole by the document gate unless
`--reprocess` is passed; that rule is unchanged and out of scope here.

**Review follow-up (cubic P3s).** Comment wording narrowed to the per-page passes and notes the plain-re-run limitation (#1001). Trusted-native pages past the halt are flagged even if an earlier run left them terminal: this run never restored or processed them (the pre-pass is OCR-only), so flagging is conservative and correct; pinned by test_native_pages_past_halt_flagged_even_if_previously_terminal. The halt test now asserts the page list in result.error.
