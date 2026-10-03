# #1004: judge timeout on a page with known-bad native text

## What shipped

Parts 1 and 2 of the issue. Part 3 (re-judge on resume) is NOT implemented.

- `FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT` and `SelectionProvenance.NATIVE_UNTRUSTED_JUDGE_TIMEOUT`
  (declared before `NATIVE_FALLBACK`, matching the cascade's return order that the R7 guard enforces).
  Public disposition is the same `DEMOTED_NATIVE` pair as `NATIVE_FALLBACK`; the new fact rides on the
  failure mode and the provenance tag (the #713 pattern).
- `manifest.judge_timeout_candidate(p)`: the model candidate (non-native, has text) whose TYPED
  `judge_outcome == JUDGE_OUTCOME_TIMEOUT`, and which no attempt COMPLETED a refusal of
  (`superseding_rejection`). Reason text is never read.
- `manifest.native_untrusted_judge_timeout(p)`: `needs_ocr_enhancement` and that candidate, excluding
  structure-class pages and the four native-table-defect flags. Those keep the #713 / D3 / table-distrust
  endings unchanged.
- In `_select_page_output_tagged`'s native fallback return: the page ships native text, WARNING,
  `audit_passed` untouched (it selects the winner), `failure_mode = NATIVE_UNTRUSTED_JUDGE_TIMEOUT`.
  The unjudged model bytes are never shipped.
- `_phase_assemble`: `judge_timeout_native_pages` is PARTITIONED out of `native_fallback_pages`, so a page is
  counted once (the #293 shape). Own event `native_untrusted_judge_timeout`, own CLI line, document
  `AUDIT_FAILED` through `pages_ok`. The four retained-native buckets also exclude it.

## Tests

`tests/test_gh1004_judge_timeout_untrusted_native.py` (7): same PageState with TIMEOUT vs COMPLETED
(only the cause differs: failure mode, provenance, event kind), with vs without `needs_ocr_enhancement`,
a later completed rejection of the same bytes supersedes the timeout, a structure-class page never takes
the new ending, document AUDIT_FAILED and one event per page. Hermetic: `_available_engines_for_agentic`
patched, `_resolve_judge_model` returns "". Provenance counts updated in
`test_p6_disposition_contract.py` (21 -> 22) and `test_r7_winner_kind_tags.py` (count 22, new equivalence
group with `NATIVE_FALLBACK`).

Mutants (external copy of src and tests, `socr.__file__` canary asserted by the file's own test, anchor
counted exactly once before editing): (a) `judge_timeout_native = False` -> 2 tests fail;
(b) gate accepts any non-audit-passed candidate instead of typed TIMEOUT -> 3 tests fail.

## Part 3: what it would take

A native WARNING page is never terminal, so a re-run reprocesses it from scratch (re-OCR). Re-judging the
old bytes needs: (1) persisting the timed-out candidate text plus its identity in the page sidecar (today
only the shipped winner is frozen); (2) a pre-route step in the per-page loop that loads it, verifies
checksum and `_run_fingerprint`, and calls the page judge on those bytes with a bounded, configurable
retry count; (3) shipping them only on an accepting verdict, then re-running the table and credential
gates on them as if freshly routed; (4) resume/golden/byte-identity tests for the new sidecar field.
That touches the sidecar schema and the loop's resume gate, so it is its own ticket.
